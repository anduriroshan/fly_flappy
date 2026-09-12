"""Device selection helpers."""
from __future__ import annotations

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Return a torch.device from a config string.

    Accepts: 'auto', 'cpu', 'cuda', 'cuda:N'. When 'auto', prefers CUDA if
    available, then MPS (Apple), then CPU. Falls back gracefully — a laptop
    with no GPU still gets a usable device.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        mem = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        return f"CUDA:{idx} — {name} ({mem:.1f} GB)"
    return f"{device.type.upper()}"
