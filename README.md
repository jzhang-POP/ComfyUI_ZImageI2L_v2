# ComfyUI_ZImageI2L_v2

A thin ComfyUI wrapper around **DiffSynth-Studio's Z-Image i2L v2** (Image-to-LoRA).

Given reference images, the i2L v2 hypernetwork *predicts* a LoRA in a single forward pass
(no per-style training), then applies it to the Z-Image generator. This package targets the
**v2** model on ModelScope: [`DiffSynth-Studio/ZImage-i2L-v2`](https://modelscope.cn/models/DiffSynth-Studio/ZImage-i2L-v2).

> v2 uses the new *Diffusion Templates* API (`TemplatePipeline` / `call_single_side`), which
> is different from v1's `ZImageUnit_Image2LoRAEncode/Decode`. Existing v1 community nodes do
> **not** run v2 by swapping a model id.

## Nodes

| Node | In | Out | What it does |
|---|---|---|---|
| **Loader** | device, dtype, (cache dir) | `pipe`, `template` | Loads base Z-Image pipeline + i2L v2 template once. Models auto-download from ModelScope. |
| **Extract LoRA** | `template`, `IMAGE` | `lora` | `template.call_single_side(...)` → predicted LoRA (deterministic; no seed). |
| **Save LoRA** | `lora`, filename | filename | Writes `.safetensors` into `models/loras`. |
| **Generate** | `pipe`, `template`, `IMAGE`, prompt, seed, cfg, steps, sigma_shift | `IMAGE` | Full styled image with **asymmetric CFG** (auto gray-image negative branch). |

Two usage patterns:
- **Loader → Extract → Save** — get a reusable LoRA file.
- **Loader → Generate** — best-quality styled image in one shot (keeps asymmetric CFG).

## Requirements / setup (on the CUDA box, e.g. RunPod RTX 5090)

The 5090 is Blackwell (sm_120) and needs a recent PyTorch (cu128+). Most current ComfyUI
pod images already ship one; verify with the env check below.

```bash
# 1. Install DiffSynth-Studio from git (NOT PyPI — v2 needs diffsynth.diffusion.template)
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio && pip install -e . && cd ..

# 2. This node's light deps
pip install -r ComfyUI/custom_nodes/ComfyUI_ZImageI2L_v2/requirements.txt

# 3. (optional) point the model cache somewhere with space
export MODELSCOPE_CACHE=/workspace/modelscope_cache

# 4. Env check — must print True and import cleanly
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda); \
from diffsynth.diffusion.template import TemplatePipeline; print('v2 API OK')"
```

Then restart ComfyUI. First **Loader** run downloads the models (Z-Image + Z-Image-Turbo
encoders/VAE + ZImage-i2L-v2 ≈ tens of GB, one-time) into the ModelScope cache.

## Notes / known gaps
- `device` defaults to `cuda`; `mps`/`cpu` are exposed but the upstream pipeline is CUDA-oriented.
- `sigma_shift` defaults to 8 (matches the v1 i2L example); set 0 to omit the kwarg.
- Out of scope (for now): multi-style fusion UI, ControlNet/inpaint composition, CPU-offload tuning.

## Credits
Built on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and Tongyi-MAI's
Z-Image. Paper: *Compressing Image Style Training into a Single Model Forward* (arXiv 2606.13809).
Apache-2.0.
