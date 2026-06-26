"""Utilities to guarantee experiment reproducibility.

Sets the random seeds of all relevant libraries to ensure consistent
results across runs.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set the random seed in Python, NumPy and PyTorch.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic operations (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
