"""Espacio de búsqueda y muestreo aleatorio de configuraciones del MLP.

Búsqueda aleatoria (Bergstra y Bengio, 2012): cada configuración se sortea de forma
independiente; con la misma semilla se obtienen siempre las mismas configuraciones.
"""
from typing import Any, Dict, List

import numpy as np


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Sortea uniformemente en escala logarítmica y redondea a 2 cifras significativas."""
    return float(f"{float(np.exp(rng.uniform(np.log(low), np.log(high)))):.2g}")


def sample_mlp_configs(space: Dict[str, Any], n: int, seed: int) -> List[Dict[str, Any]]:
    """Devuelve `n` configuraciones distintas del MLP sorteadas del espacio `space`.

    `space` define listas de opciones (hidden_dims, activation, dropout, batch_size,
    use_batchnorm, optimizer) y rangos log-uniformes (weight_decay y lr por optimizador).
    """
    rng = np.random.default_rng(seed)
    out: List[Dict[str, Any]] = []
    seen = set()
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 100 * n:
            raise RuntimeError("No se pudieron sortear tantas configuraciones distintas.")
        opt = space["optimizer"][int(rng.integers(len(space["optimizer"])))]
        lr_low, lr_high = space["lr_range"][opt]
        cfg = {
            "hidden_dims": list(space["hidden_dims"][int(rng.integers(len(space["hidden_dims"])))]),
            "activation": space["activation"][int(rng.integers(len(space["activation"])))],
            "dropout": float(space["dropout"][int(rng.integers(len(space["dropout"])))]),
            "use_batchnorm": bool(space["use_batchnorm"][int(rng.integers(len(space["use_batchnorm"])))]),
            "batch_size": int(space["batch_size"][int(rng.integers(len(space["batch_size"])))]),
            "optimizer": opt,
            "lr": _log_uniform(rng, lr_low, lr_high),
            "weight_decay": _log_uniform(rng, *space["weight_decay_range"]),
        }
        key = repr(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out
