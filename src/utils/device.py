"""Device selection (CPU/CUDA) with verbose diagnostics.

Centralizes the logic each script used to repeat (``torch.device("cuda" if
torch.cuda.is_available() else "cpu")``) and adds:

- Support for forcing the device from config (``training.device``).
- Actionable log messages when CUDA is requested but unavailable, with
  concrete hints (install cu128 wheels for Blackwell, reinstall if the
  build is CPU-only, etc.).
- Detection of the GPU generation to warn when an RTX Blackwell (sm_120)
  is running with stale wheels that mark it as available but have no
  kernels for it.
"""

from __future__ import annotations

import torch

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def select_device(config: dict | None = None) -> torch.device:
    """Return the ``torch.device`` to use and log detailed diagnostics.

    Device resolution:

    1. If ``config['training']['device']`` is set to ``"cuda"`` or
       ``"cpu"``, that choice is used. ``"cuda"`` requested but not
       available raises ``RuntimeError`` with installation instructions,
       instead of silently falling back to CPU (which is what was masking
       slow training runs on the user's machine).
    2. ``"auto"`` or absent → ``cuda`` if available, otherwise ``cpu``,
       with a warning explaining why CPU was chosen.

    Args:
        config: Configuration dictionary (may be ``None``).

    Returns:
        ``torch.device`` ready to pass to models and tensors.

    Raises:
        RuntimeError: If ``device='cuda'`` was explicitly requested but
            ``torch.cuda.is_available()`` is ``False``.
    """
    requested = "auto"
    if config is not None:
        requested = (config.get("training", {}) or {}).get("device", "auto")
    requested = str(requested).lower()

    cuda_available = torch.cuda.is_available()
    torch_version = torch.__version__
    cuda_build = torch.version.cuda  # None if torch is CPU-only

    if requested == "cpu":
        logger.info("Device: cpu (forced via training.device='cpu')")
        return torch.device("cpu")

    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "training.device='cuda' but torch.cuda.is_available() is "
                f"False.\n"
                f"  torch version: {torch_version}\n"
                f"  CUDA build:    {cuda_build}\n"
                "  Typical causes:\n"
                "    1. PyTorch installed as CPU-only (cuda_build=None). "
                "Reinstall from https://pytorch.org/get-started/locally/\n"
                "    2. RTX 5070 / Blackwell (sm_120) with cu121 or older "
                "wheels. You need cu128: see docs/gpu-setup.md\n"
                "    3. Outdated or missing NVIDIA driver.\n"
                "  Pass training.device='auto' (or remove the key) to "
                "allow fallback to CPU."
            )
        device = torch.device("cuda")
        _log_cuda_info(device)
        return device

    # requested == "auto" (or any other value)
    if cuda_available:
        device = torch.device("cuda")
        _log_cuda_info(device)
        return device

    logger.warning(
        "Device: cpu (CUDA unavailable).\n"
        "  torch version: %s | CUDA build: %s\n"
        "  If you have an NVIDIA GPU and this surprises you:\n"
        "    - cuda_build=None means the installed wheel is CPU-only.\n"
        "    - If you have an RTX 5070 (Blackwell), you need cu128 wheels. "
        "See docs/gpu-setup.md.\n"
        "    - To fail hard instead of falling back to CPU, add "
        "training.device: cuda to the YAML.",
        torch_version, cuda_build,
    )
    return torch.device("cpu")


def _log_cuda_info(device: torch.device) -> None:
    """Log GPU info and warn if the generation seems incompatible."""
    idx = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(idx)
    cap_major, cap_minor = torch.cuda.get_device_capability(idx)
    total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
    cuda_build = torch.version.cuda

    logger.info(
        "Device: cuda:%d (%s, sm_%d%d, %.1f GB, torch CUDA build=%s)",
        idx, name, cap_major, cap_minor, total_mem_gb, cuda_build,
    )

    # Specific warning: Blackwell (sm_120+) with CUDA build < 12.8 usually
    # reports "is_available()=True" but fails on kernels at the first real
    # .cuda() call.
    if cap_major >= 12:
        try:
            major, minor = (int(p) for p in (cuda_build or "0.0").split(".")[:2])
        except ValueError:
            major, minor = 0, 0
        if (major, minor) < (12, 8):
            logger.warning(
                "Blackwell GPU (sm_%d%d) detected with torch CUDA build %s. "
                "Operations will likely fail at runtime: install cu128 "
                "wheels (see docs/gpu-setup.md).",
                cap_major, cap_minor, cuda_build,
            )
