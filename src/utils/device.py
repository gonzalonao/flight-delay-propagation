"""Selección de dispositivo (CPU/CUDA) con diagnóstico verboso.

Centraliza la lógica que cada script repetía (``torch.device("cuda" if
torch.cuda.is_available() else "cpu")``) y añade:

- Soporte para forzar dispositivo desde config (``training.device``).
- Mensajes de log accionables cuando se solicita CUDA pero no está
  disponible, con pistas concretas (instalar wheels cu128 para Blackwell,
  reinstalar si la build es CPU-only, etc.).
- Detección de la generación de la GPU para advertir cuando una RTX
  Blackwell (sm_120) está corriendo con wheels antiguos que la marcan
  como disponible pero no tienen kernels para ella.
"""

from __future__ import annotations

import torch

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def select_device(config: dict | None = None) -> torch.device:
    """Devuelve el ``torch.device`` a usar y loguea diagnóstico detallado.

    Resolución del dispositivo:

    1. Si ``config['training']['device']`` está definido como ``"cuda"`` o
       ``"cpu"``, se usa esa elección. ``"cuda"`` solicitado pero no
       disponible lanza ``RuntimeError`` con instrucciones de instalación,
       en lugar de caer silenciosamente a CPU (que es lo que estaba
       enmascarando los entrenamientos lentos en máquina del usuario).
    2. ``"auto"`` o ausente → ``cuda`` si está disponible, si no ``cpu``,
       con un warning explicando por qué se eligió CPU.

    Args:
        config: Diccionario de configuración (puede ser ``None``).

    Returns:
        ``torch.device`` listo para pasar a modelos y tensores.

    Raises:
        RuntimeError: Si ``device='cuda'`` se solicitó explícitamente pero
            ``torch.cuda.is_available()`` es ``False``.
    """
    requested = "auto"
    if config is not None:
        requested = (config.get("training", {}) or {}).get("device", "auto")
    requested = str(requested).lower()

    cuda_available = torch.cuda.is_available()
    torch_version = torch.__version__
    cuda_build = torch.version.cuda  # None si torch es CPU-only

    if requested == "cpu":
        logger.info("Dispositivo: cpu (forzado vía training.device='cpu')")
        return torch.device("cpu")

    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "training.device='cuda' pero torch.cuda.is_available() es "
                f"False.\n"
                f"  torch version: {torch_version}\n"
                f"  CUDA build:    {cuda_build}\n"
                "  Causas típicas:\n"
                "    1. PyTorch instalado como CPU-only (cuda_build=None). "
                "Reinstala desde https://pytorch.org/get-started/locally/\n"
                "    2. RTX 5070 / Blackwell (sm_120) con wheels cu121 o "
                "anteriores. Necesitas cu128: ver GPU_SETUP.md\n"
                "    3. Driver NVIDIA desactualizado o no instalado.\n"
                "  Pasa training.device='auto' (o quita la clave) para "
                "permitir fallback a CPU."
            )
        device = torch.device("cuda")
        _log_cuda_info(device)
        return device

    # requested == "auto" (o cualquier otro valor)
    if cuda_available:
        device = torch.device("cuda")
        _log_cuda_info(device)
        return device

    logger.warning(
        "Dispositivo: cpu (CUDA no disponible).\n"
        "  torch version: %s | CUDA build: %s\n"
        "  Si tienes una GPU NVIDIA y esto te sorprende:\n"
        "    - cuda_build=None significa que el wheel instalado es CPU-only.\n"
        "    - Si tienes RTX 5070 (Blackwell), necesitas wheels cu128. "
        "Ver GPU_SETUP.md.\n"
        "    - Para fallar duro en lugar de caer a CPU, añade "
        "training.device: cuda al YAML.",
        torch_version, cuda_build,
    )
    return torch.device("cpu")


def _log_cuda_info(device: torch.device) -> None:
    """Loguea info de la GPU y advierte si la generación parece incompatible."""
    idx = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(idx)
    cap_major, cap_minor = torch.cuda.get_device_capability(idx)
    total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
    cuda_build = torch.version.cuda

    logger.info(
        "Dispositivo: cuda:%d (%s, sm_%d%d, %.1f GB, torch CUDA build=%s)",
        idx, name, cap_major, cap_minor, total_mem_gb, cuda_build,
    )

    # Aviso específico: Blackwell (sm_120+) con CUDA build < 12.8 suele dar
    # "is_available()=True" pero fallar con kernels al primer .cuda() real.
    if cap_major >= 12:
        try:
            major, minor = (int(p) for p in (cuda_build or "0.0").split(".")[:2])
        except ValueError:
            major, minor = 0, 0
        if (major, minor) < (12, 8):
            logger.warning(
                "GPU Blackwell (sm_%d%d) detectada con torch CUDA build %s. "
                "Es probable que las operaciones fallen en runtime: instala "
                "wheels cu128 (ver GPU_SETUP.md).",
                cap_major, cap_minor, cuda_build,
            )
