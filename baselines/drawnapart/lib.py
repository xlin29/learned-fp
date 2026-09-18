"""DRAWNAPART baseline — torch-side helpers for the evaluators: seeding,
state-dict loading, embed_numpy, and the data helpers re-exported from data.py."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path

import numpy as np
import torch


# ---------- Constants -------------------------------------------------------

TRACE_LEN = 1024
TRACES_PER_SAMPLE = 7  # paper: 4 ndjson files x 7 traces = 28 traces per device
SAMPLES_PER_DEVICE = 4

_PID_RE = re.compile(r"__pid-([a-f0-9]+)__")


# ---------- Seeding -----------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed the full torch + numpy + python stack for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------- Filesystem / parsing utilities ----------------------------------

from .data import (  # noqa: F401  (re-exported for the torch-side scripts)
    list_devices, _open_text, _hash, extract_pid, build_pid_map,
    load_whitelist, load_ndjson_traces, load_samples_for_device, DPSample,
    discover_samples_strict_4x7, trace_to_img, split_stratified,
)


# ---------- Model -----------------------------------------------------------

from .released_model import ReleasedDPCNN, build_from_arch  # noqa: F401


def torch_load_state_dict(path: Path, device: torch.device):
    """torch.load wrapper that prefers weights_only on newer PyTorch."""
    try:
        sd = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location=device)
    return sd


@torch.no_grad()
def embed_numpy(net: ReleasedDPCNN, X: np.ndarray, device: torch.device, batch: int = 512) -> np.ndarray:
    net.eval().to(device)
    Xt = torch.from_numpy(X).float()
    outs = [net.embedding(Xt[i: i + batch].to(device)).cpu() for i in range(0, Xt.size(0), batch)]
    return torch.cat(outs, 0).numpy()


__all__ = [
    "TRACE_LEN", "TRACES_PER_SAMPLE", "SAMPLES_PER_DEVICE",
    "set_seed",
    "list_devices", "_open_text", "_hash",
    "extract_pid", "build_pid_map", "load_whitelist",
    "load_ndjson_traces", "load_samples_for_device",
    "DPSample", "discover_samples_strict_4x7",
    "trace_to_img", "split_stratified",
    "ReleasedDPCNN", "build_from_arch",
    "torch_load_state_dict", "embed_numpy",
]
