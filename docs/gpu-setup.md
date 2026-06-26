# GPU Setup — RTX 5070 (Blackwell) and other NVIDIA GPUs

This guide fixes the most common problem when launching training:
**`torch.cuda.is_available()` returns `False` (or `True`, but the model
keeps running on CPU)**.

## TL;DR

```bash
# Supported hardware: any NVIDIA GPU with driver >= 525.
# RTX 5070 / 5080 / 5090 (Blackwell, sm_120) require CUDA 12.8+ wheels.
# RTX 30xx / 40xx work with cu121 or cu124, but cu128 works too.

pip uninstall -y torch torchvision torch-geometric torch-scatter torch-sparse torch-cluster
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install torch_geometric
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Expected output on the 5070:

```
2.6.0+cu128  12.8  True  NVIDIA GeForce RTX 5070
```

## Force GPU from the config

By default the scripts use `device: auto` (CUDA if present, otherwise CPU with
a warning). To make a training run **fail when no GPU is available** instead of
silently falling back to CPU (which is what masked slow runs before the fix),
add this to the model's YAML:

```yaml
training:
  device: cuda    # auto | cuda | cpu
```

With `device: cuda`, if `torch.cuda.is_available()` is `False` the script
raises a `RuntimeError` with concrete instructions (CPU-only build, stale
wheels, outdated driver, …) instead of carrying on with the CPU.

## Diagnosis

If you still don't see the GPU after installing cu128, check the first-boot
log. `select_device()` now prints:

```
Device: cuda:0 (NVIDIA GeForce RTX 5070, sm_120, 12.0 GB, torch CUDA build=12.8)
```

If you see something like this:

```
Device: cpu (CUDA unavailable).
  torch version: 2.5.0+cpu | CUDA build: None
```

→ The installed wheel is **CPU-only**. Reinstall using the cu128 index-url
above.

If you see:

```
Blackwell GPU (sm_120) detected with torch CUDA build 12.1. Operations will
likely fail at runtime: install cu128 wheels.
```

→ You have stale wheels. PyTorch detects the GPU but has no kernels for
sm_120 — the first real `.cuda()` call will crash. Reinstall with cu128
anyway.

## Fallback: Colab

If the 5070 can't be enabled in time (driver, permissions, etc.), upload the
cached snapshots to Drive and train on Colab:

```python
# On Colab
!pip install torch_geometric
# torch already ships with cu121 on Colab; enough for T4/L4/A100.
```

The snapshot cache (`data/processed/snapshots/snapshots_<hash>.pt`) is
portable: if you upload the `.pt` to Drive and mount Drive in Colab,
`build_graph_dataset` gets a **cache HIT** and the preprocessing phase
disappears — the bottleneck becomes the data loader, not feature
construction.

## Fallback: CPU

Works, but is ~10–20× slower. Useful for debugging the pipeline, not for the
final training run. Make sure to cache the snapshots first (a single build
serves all subsequent re-runs).
