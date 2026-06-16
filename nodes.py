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

        template = TemplatePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=[ModelConfig(model_id="DiffSynth-Studio/ZImage-i2L-v2")],
            lazy_loading=low_vram,
        )
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


NODE_CLASS_MAPPINGS = {
    "ZImageI2LV2Loader": ZImageI2LV2Loader,
    "ZImageI2LV2LoadImagesFromFolder": ZImageI2LV2LoadImagesFromFolder,
    "ZImageI2LV2ExtractLoRA": ZImageI2LV2ExtractLoRA,
    "ZImageI2LV2SaveLoRA": ZImageI2LV2SaveLoRA,
    "ZImageI2LV2Generate": ZImageI2LV2Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageI2LV2Loader": "Z-Image i2L v2 — Loader",
    "ZImageI2LV2LoadImagesFromFolder": "Z-Image i2L v2 — Load Images From Folder",
    "ZImageI2LV2ExtractLoRA": "Z-Image i2L v2 — Extract LoRA",
    "ZImageI2LV2SaveLoRA": "Z-Image i2L v2 — Save LoRA",
    "ZImageI2LV2Generate": "Z-Image i2L v2 — Generate",
}
