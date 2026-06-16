"""ComfyUI nodes wrapping DiffSynth-Studio's Z-Image i2L v2 (Image-to-LoRA).

Design notes
------------
- v2 uses the *Diffusion Templates* API (`diffsynth.diffusion.template.TemplatePipeline`),
  which is different from v1's `ZImageUnit_Image2LoRAEncode/Decode`. This package targets v2.
- ALL heavy imports (torch, diffsynth, modelscope, folder_paths) are intentionally lazy,
  done inside the node methods. This lets the module import and the nodes register on any
  machine (e.g. a Mac with no CUDA / no diffsynth) so you can verify loading, while the
  actual work runs only when a node executes on a CUDA box.
"""

import os


# ---------------------------------------------------------------------------
# Helpers (lazy imports inside so the module loads without torch/PIL present)
# ---------------------------------------------------------------------------

def _images_to_pils(image):
    """ComfyUI IMAGE tensor [B,H,W,C] float 0-1 -> list[PIL.Image]."""
    import numpy as np
    from PIL import Image
    arr = (image.detach().clamp(0, 1).cpu().numpy() * 255.0).round().astype("uint8")
    return [Image.fromarray(a).convert("RGB") for a in arr]


def _pil_to_image_tensor(pil_image):
    """PIL.Image -> ComfyUI IMAGE tensor [1,H,W,C] float 0-1."""
    import numpy as np
    import torch
    arr = np.array(pil_image.convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(arr)[None, ]


def _gray_negatives(pil_images):
    """Neutral gray (128) counterparts of the reference images, for asymmetric CFG."""
    import numpy as np
    from PIL import Image
    return [Image.fromarray(np.zeros_like(np.array(i)) + 128) for i in pil_images]


def _resolve_dtype(name):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _comfy_pbar_cmd():
    """A tqdm-shaped callable that drives ComfyUI's per-node progress bar.

    DiffSynth's sampling loop does `for ... in enumerate(progress_bar_cmd(timesteps))`, and by
    default `progress_bar_cmd=tqdm`, which only prints to the server console. We pass this
    instead so each step ticks `comfy.utils.ProgressBar`, filling the green bar on the node.
    Falls back to tqdm if comfy.utils isn't importable (e.g. outside a running ComfyUI).
    """
    try:
        import comfy.utils
    except Exception:
        from tqdm import tqdm
        return tqdm

    class _Bar:
        def __init__(self, iterable=None, total=None, *args, **kwargs):
            self.iterable = iterable
            if total is None and hasattr(iterable, "__len__"):
                total = len(iterable)
            self.pbar = comfy.utils.ProgressBar(total) if total else None

        def __iter__(self):
            if self.iterable is None:
                return
            for x in self.iterable:
                yield x
                if self.pbar is not None:
                    self.pbar.update(1)

        def update(self, n=1):
            if self.pbar is not None:
                self.pbar.update(n)

        def close(self):
            pass

    return _Bar


def _check_v2_available():
    """Raise a clear, actionable error if the v2 template API isn't importable."""
    try:
        from diffsynth.diffusion.template import TemplatePipeline  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "DiffSynth-Studio with the v2 Diffusion Templates API is required, but "
            "`from diffsynth.diffusion.template import TemplatePipeline` failed.\n"
            "Install the latest from git (the PyPI build may lag):\n"
            "    git clone https://github.com/modelscope/DiffSynth-Studio.git\n"
            "    cd DiffSynth-Studio && pip install -e .\n"
            f"Underlying import error: {e!r}"
        ) from e


# ---------------------------------------------------------------------------
# Loader: builds the base Z-Image pipeline + the i2L v2 TemplatePipeline
# ---------------------------------------------------------------------------

class ZImageI2LV2Loader:
    """Load the base Z-Image generation pipeline and the i2L v2 template once.

    Outputs both because Generate needs both, while Extract needs only the template.
    Models auto-download from ModelScope into MODELSCOPE_CACHE (default ~/.cache/modelscope/hub).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cuda", "mps", "cpu"], {"default": "cuda"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                # On by default: 24 GB cards OOM during DiT sampling with everything
                # resident. Streams weights CPU<->GPU. Turn off on >=32 GB for speed.
                "low_vram": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Blank = use the MODELSCOPE_CACHE env var / default cache location.
                "modelscope_cache": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("ZIMAGE_PIPE", "ZIMAGE_I2L_TEMPLATE")
    RETURN_NAMES = ("pipe", "template")
    FUNCTION = "load"
    CATEGORY = "ZImage-i2L"

    def load(self, device, dtype, low_vram=True, modelscope_cache=""):
        _check_v2_available()
        import torch
        from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
        from diffsynth.diffusion.template import TemplatePipeline

        if modelscope_cache.strip():
            os.environ["MODELSCOPE_CACHE"] = modelscope_cache.strip()

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' selected but CUDA is not available. "
                "On a non-CUDA box this pipeline is not supported by the upstream code."
            )

        torch_dtype = _resolve_dtype(dtype)

        # Low-VRAM: keep base-pipeline weights on CPU and stream them to the GPU only for
        # computation (vram_limit=0). This frees the text encoder (~8 GB) etc. during DiT
        # sampling so the stack fits in 24 GB. TemplatePipeline has no vram_limit, so we use
        # its lazy_loading instead. Disable on >=32 GB GPUs for speed.
        offload = low_vram and device == "cuda"
        if offload:
            vram_config = dict(
                offload_dtype=torch_dtype, offload_device="cpu",
                onload_dtype=torch_dtype, onload_device="cpu",
                preparing_dtype=torch_dtype, preparing_device="cuda",
                computation_dtype=torch_dtype, computation_device="cuda",
            )
            vram_limit = 0
        else:
            vram_config = {}
            vram_limit = None

        # Coarse milestone progress: from_pretrained exposes no per-file hook, so we can only
        # tick after each major stage. Watch the console for actual download/load bytes.
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(2)
        except Exception:
            pbar = None

        print("[ZImageI2LV2] Loading base Z-Image pipeline (first run downloads models)...")
        pipe = ZImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=[
                ModelConfig(model_id="Tongyi-MAI/Z-Image", origin_file_pattern="transformer/*.safetensors", **vram_config),
                ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
                ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
            ],
            tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
            vram_limit=vram_limit,
        )
        # Required so predicted LoRAs can be hot-loaded onto the DiT at generation time.
        pipe.enable_lora_hot_loading(pipe.dit)
        if pbar is not None:
            pbar.update(1)

        print("[ZImageI2LV2] Loading i2L v2 template (DiffSynth-Studio/ZImage-i2L-v2)...")
        template = TemplatePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=[ModelConfig(model_id="DiffSynth-Studio/ZImage-i2L-v2")],
            lazy_loading=low_vram,
        )
        if pbar is not None:
            pbar.update(1)
        return (pipe, template)


# ---------------------------------------------------------------------------
# Extract: reference images -> predicted LoRA state dict (no sampling)
# ---------------------------------------------------------------------------

class ZImageI2LV2ExtractLoRA:
    """Predict a LoRA from one or more reference images via the i2L v2 hypernetwork.

    Note: extraction is a deterministic forward pass (no diffusion sampling), so there is
    no seed input here — matching v2's `template.call_single_side(inputs=...)` API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("ZIMAGE_I2L_TEMPLATE",),
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("ZIMAGE_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "extract"
    CATEGORY = "ZImage-i2L"

    def extract(self, template, images):
        import torch
        pils = _images_to_pils(images)
        if not pils:
            raise ValueError("ExtractLoRA received no images.")
        with torch.no_grad():
            lora = template.call_single_side(inputs=[{"image": pils}])["lora"]
        return (lora,)


# ---------------------------------------------------------------------------
# Save: write a predicted LoRA to models/loras as safetensors
# ---------------------------------------------------------------------------

class ZImageI2LV2SaveLoRA:
    """Save a predicted LoRA into ComfyUI's loras folder for reuse anywhere."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": ("ZIMAGE_LORA",),
                "filename": ("STRING", {"default": "zimage_i2l_v2_style"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "save"
    CATEGORY = "ZImage-i2L"
    OUTPUT_NODE = True

    def save(self, lora, filename):
        import folder_paths
        from safetensors.torch import save_file

        name = filename.strip() or "zimage_i2l_v2_style"
        if not name.endswith(".safetensors"):
            name += ".safetensors"
        out_dir = folder_paths.get_folder_paths("loras")[0]
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, name)
        save_file(lora, out_path)
        print(f"[ZImageI2LV2] Saved LoRA -> {out_path}")
        return (name,)


# ---------------------------------------------------------------------------
# Generate: full styled image with asymmetric CFG (gray-image negative branch)
# ---------------------------------------------------------------------------

class ZImageI2LV2Generate:
    """Generate a styled image directly, preserving the paper's asymmetric CFG.

    Reference images drive the positive branch; their neutral-gray counterparts drive the
    negative branch (built automatically). This keeps full v2 quality rather than handing a
    saved LoRA to ComfyUI's stock loader (which would apply it to both CFG branches).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("ZIMAGE_PIPE",),
                "template": ("ZIMAGE_I2L_TEMPLATE",),
                "images": ("IMAGE",),
                "prompt": ("STRING", {"default": "A cat is sitting on a stone", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "sigma_shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "ZImage-i2L"

    def generate(self, pipe, template, images, prompt, seed, cfg_scale, num_inference_steps, sigma_shift):
        import torch
        pils = _images_to_pils(images)
        if not pils:
            raise ValueError("Generate received no reference images.")

        kwargs = dict(
            prompt=prompt,
            seed=int(seed),
            cfg_scale=float(cfg_scale),
            num_inference_steps=int(num_inference_steps),
            template_inputs=[{"image": pils}],
            negative_template_inputs=[{"image": _gray_negatives(pils)}],
            progress_bar_cmd=_comfy_pbar_cmd(),  # show progress on the node, not just console
        )
        # sigma_shift is a Z-Image pipeline knob (v1 i2L example used 8). Omit if zeroed
        # so we don't pass an unexpected kwarg when the user disables it.
        if sigma_shift and sigma_shift > 0:
            kwargs["sigma_shift"] = float(sigma_shift)

        with torch.no_grad():
            image = template(pipe, **kwargs)

        return (_pil_to_image_tensor(image),)


# ---------------------------------------------------------------------------
# Load Images From Folder: folder path -> batched IMAGE for the Generate/Extract input
# ---------------------------------------------------------------------------

class ZImageI2LV2LoadImagesFromFolder:
    """Load every image in a folder as one IMAGE batch.

    Convenience for feeding a set of style references without wiring N Load Image nodes.
    Files are taken in sorted filename order. Because a ComfyUI IMAGE output is a single
    uniformly-sized tensor batch, images that differ from the first image's dimensions are
    resized to match it (the i2L encoders resize internally anyway, so style is unaffected).
    """

    _EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
            },
            "optional": {
                # 0 = load all. i2L works well with ~1-8 references.
                "limit": ("INT", {"default": 0, "min": 0, "max": 1000}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "load_folder"
    CATEGORY = "ZImage-i2L"

    def load_folder(self, folder_path, limit=0):
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        raw = folder_path.strip()
        if not raw:
            raise ValueError("folder_path is empty.")
        # Accept an absolute path, ~user path, or a path relative to ComfyUI's input dir
        # (so e.g. "00_raw" or "styles/cat" resolves under ComfyUI/input).
        candidates = [os.path.expanduser(raw)]
        try:
            import folder_paths
            candidates.append(os.path.join(folder_paths.get_input_directory(), raw))
        except Exception:
            pass
        path = next((c for c in candidates if os.path.isdir(c)), None)
        if path is None:
            raise ValueError(
                f"folder_path is not a directory: {folder_path!r}. "
                f"Tried: {candidates}. Use an absolute path, or one relative to ComfyUI/input."
            )

        files = sorted(
            f for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in self._EXTS
            and os.path.isfile(os.path.join(path, f))
        )
        if limit and limit > 0:
            files = files[:limit]
        if not files:
            raise ValueError(f"No images found in folder: {path}")

        tensors = []
        ref_size = None
        for fn in files:
            img = Image.open(os.path.join(path, fn))
            img = ImageOps.exif_transpose(img).convert("RGB")
            if ref_size is None:
                ref_size = img.size
            elif img.size != ref_size:
                img = img.resize(ref_size, Image.LANCZOS)
            tensors.append(torch.from_numpy(np.array(img).astype("float32") / 255.0))

        batch = torch.stack(tensors, dim=0)  # [B, H, W, C]
        print(f"[ZImageI2LV2] Loaded {len(files)} image(s) from {path} -> batch {tuple(batch.shape)}")
        return (batch, len(files))


# ---------------------------------------------------------------------------
# Atomic building blocks: split the all-in-one Generate into composable pieces.
# Generate(refs, prompt, ...) == ExtractLoRA(refs) [+ ExtractLoRA(GrayImages(refs))]
#                                + ApplyLoRA(pipe, ...) + Sample(SamplerConfig, prompt)
# ---------------------------------------------------------------------------

class ZImageI2LV2GrayImages:
    """Neutral-gray (128) copies of the input images.

    Feed these through Extract LoRA to get the negative-branch LoRA used for the paper's
    asymmetric CFG (reference LoRA on the positive branch, gray LoRA on the negative).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("gray_images",)
    FUNCTION = "make"
    CATEGORY = "ZImage-i2L/atomic"

    def make(self, images):
        import torch
        return (torch.full_like(images, 128.0 / 255.0),)


class ZImageI2LV2SamplerConfig:
    """Bundle the diffusion sampling parameters into one config object for Sample."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "sigma_shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
            },
        }

    RETURN_TYPES = ("ZIMAGE_SAMPLER_CFG",)
    RETURN_NAMES = ("sampler_cfg",)
    FUNCTION = "build"
    CATEGORY = "ZImage-i2L/atomic"

    def build(self, seed, cfg_scale, num_inference_steps, sigma_shift, width, height):
        return ({
            "seed": int(seed),
            "cfg_scale": float(cfg_scale),
            "num_inference_steps": int(num_inference_steps),
            "sigma_shift": float(sigma_shift),
            "width": int(width),
            "height": int(height),
        },)


class ZImageI2LV2ApplyLoRA:
    """Associate a predicted LoRA (and optional negative/gray LoRA) with the pipeline.

    Outputs a bundle consumed by Sample. The positive LoRA drives the conditional branch;
    the optional negative LoRA drives the unconditional branch (asymmetric CFG).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("ZIMAGE_PIPE",),
                "lora": ("ZIMAGE_LORA",),
            },
            "optional": {
                "negative_lora": ("ZIMAGE_LORA",),
            },
        }

    RETURN_TYPES = ("ZIMAGE_PIPE_LORA",)
    RETURN_NAMES = ("pipe_lora",)
    FUNCTION = "apply"
    CATEGORY = "ZImage-i2L/atomic"

    def apply(self, pipe, lora, negative_lora=None):
        return ({"pipe": pipe, "lora": lora, "negative_lora": negative_lora},)


class ZImageI2LV2Sample:
    """Run the diffusion sampling: pipe + LoRA bundle + sampler config + prompt -> IMAGE.

    This is the atomic equivalent of Generate's final step. It calls
    pipe(prompt=..., lora=<positive>, negative_lora=<negative>, **sampler_cfg).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe_lora": ("ZIMAGE_PIPE_LORA",),
                "sampler_cfg": ("ZIMAGE_SAMPLER_CFG",),
                "prompt": ("STRING", {"default": "a cat is sitting on a stone", "multiline": True}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "ZImage-i2L/atomic"

    def sample(self, pipe_lora, sampler_cfg, prompt, negative_prompt=""):
        import torch
        pipe = pipe_lora["pipe"]

        cfg = dict(sampler_cfg)
        sigma_shift = cfg.pop("sigma_shift", None)
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            lora=pipe_lora.get("lora"),
            negative_lora=pipe_lora.get("negative_lora"),
            progress_bar_cmd=_comfy_pbar_cmd(),  # show progress on the node, not just console
            **cfg,  # seed, cfg_scale, num_inference_steps, width, height
        )
        if sigma_shift and sigma_shift > 0:
            kwargs["sigma_shift"] = float(sigma_shift)

        with torch.no_grad():
            image = pipe(**kwargs)
        return (_pil_to_image_tensor(image),)


NODE_CLASS_MAPPINGS = {
    "ZImageI2LV2Loader": ZImageI2LV2Loader,
    "ZImageI2LV2LoadImagesFromFolder": ZImageI2LV2LoadImagesFromFolder,
    "ZImageI2LV2ExtractLoRA": ZImageI2LV2ExtractLoRA,
    "ZImageI2LV2SaveLoRA": ZImageI2LV2SaveLoRA,
    "ZImageI2LV2Generate": ZImageI2LV2Generate,
    # atomic building blocks
    "ZImageI2LV2GrayImages": ZImageI2LV2GrayImages,
    "ZImageI2LV2SamplerConfig": ZImageI2LV2SamplerConfig,
    "ZImageI2LV2ApplyLoRA": ZImageI2LV2ApplyLoRA,
    "ZImageI2LV2Sample": ZImageI2LV2Sample,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageI2LV2Loader": "Z-Image i2L v2 — Loader",
    "ZImageI2LV2LoadImagesFromFolder": "Z-Image i2L v2 — Load Images From Folder",
    "ZImageI2LV2ExtractLoRA": "Z-Image i2L v2 — Extract LoRA",
    "ZImageI2LV2SaveLoRA": "Z-Image i2L v2 — Save LoRA",
    "ZImageI2LV2Generate": "Z-Image i2L v2 — Generate",
    "ZImageI2LV2GrayImages": "Z-Image i2L v2 — Make Gray Images",
    "ZImageI2LV2SamplerConfig": "Z-Image i2L v2 — Sampler Config",
    "ZImageI2LV2ApplyLoRA": "Z-Image i2L v2 — Apply LoRA",
    "ZImageI2LV2Sample": "Z-Image i2L v2 — Sample",
}
