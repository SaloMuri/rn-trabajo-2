# src/creditrisk/seeding.py
"""Fijar y documentar todas las semillas aleatorias."""

import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """Fija las semillas de Python, NumPy y PyTorch (CPU y GPU) para reproducibilidad."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False