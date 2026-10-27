"""Carga de configuración desde uno o más YAML, con overrides por línea de comandos (sin editar código).

Ejemplo:
    cfg = load_config(["configs/base.yaml", "configs/lgd.yaml"], overrides=["mlp.lr=0.01"])

Los YAML se fusionan en orden (el último gana, sección por sección) y cada override
"ruta.con.puntos=valor" se aplica al final. El valor se interpreta como YAML, así que
`mlp.hidden_dims=[128,64]`, `mlp.dropout=0.3` o `mlp.use_batchnorm=false` funcionan.
"""
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional

import yaml


def deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Fusiona `new` sobre `base` de forma recursiva (devuelve una copia)."""
    out = deepcopy(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def apply_override(cfg: Dict[str, Any], override: str) -> None:
    """Aplica en sitio un override "a.b.c=valor" sobre `cfg`."""
    if "=" not in override:
        raise ValueError(f"Override inválido (se espera ruta=valor): {override!r}")
    path, raw = override.split("=", 1)
    keys = path.strip().split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"'{k}' en {path!r} no es una sección")
    node[keys[-1]] = yaml.safe_load(raw)


def load_config(
    paths: Iterable[str], overrides: Optional[Iterable[str]] = None
) -> Dict[str, Any]:
    """Carga y fusiona YAML en orden; luego aplica los overrides."""
    cfg: Dict[str, Any] = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    for o in overrides or []:
        apply_override(cfg, o)
    return cfg
