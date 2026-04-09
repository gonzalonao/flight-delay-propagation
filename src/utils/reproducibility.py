"""Utilidades para garantizar reproducibilidad de experimentos.

Fija las semillas aleatorias de todas las librerías relevantes para
asegurar resultados consistentes entre ejecuciones.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fija la semilla aleatoria en Python, NumPy y PyTorch.

    Args:
        seed: Valor de la semilla aleatoria.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Operaciones deterministas (puede reducir rendimiento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
