"""DRAWNAPART baseline — data loading and splitting, shared by the Keras
trainer, the converter and the torch evaluators (no torch import here)."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm


# ---------- Constants -------------------------------------------------------

TRACE_LEN = 1024
TRACES_PER_SAMPLE = 7  # paper: 4 ndjson files x 7 traces = 28 traces per device
SAMPLES_PER_DEVICE = 4

_PID_RE = re.compile(r"__pid-([a-f0-9]+)__")


def list_devices(root: Path) -> List[Path]:
    return sorted(p for p in root.glob("dev_*") if p.is_dir())


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def _hash(arr: np.ndarray) -> str:
    return hashlib.sha1(arr.astype(np.float32, copy=False).tobytes()).hexdigest()


def extract_pid(folder_name: str) -> Optional[str]:
    m = _PID_RE.search(folder_name)
    return m.group(1) if m else None


def build_pid_map(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in list_devices(root):
        pid = extract_pid(p.name)
        if pid:
            out[pid] = p.name
    return out


def load_whitelist(path: Path) -> Set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict):
        return set(data.keys())
    raise ValueError(f"Unrecognised whitelist format in {path}")


# ---------- Trace parsing ---------------------------------------------------

def load_ndjson_traces(
    fp: Path, dedup: bool = True, min_len: int = TRACE_LEN
) -> List[np.ndarray]:
    """Parse one .ndjson(.gz) file into a list of float32 timing traces.

    Paper data format: each line is one JSON object with traces[i].times_ms
    holding a length-1024 array.
    """
    traces: List[np.ndarray] = []
    seen: Set[str] = set()
    with _open_text(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            raw = obj.get("traces", [])
            if not isinstance(raw, list):
                continue
            for t in raw:
                if not isinstance(t, dict):
                    continue
                times = t.get("times_ms")
                if not isinstance(times, list) or len(times) < min_len:
                    continue
                arr = np.array(times[:min_len], dtype=np.float32)
                if dedup:
                    h = _hash(arr)
                    if h in seen:
                        continue
                    seen.add(h)
                traces.append(arr)
    return traces


def load_samples_for_device(
    dev_dir: Path, min_len: int = TRACE_LEN, traces_per_sample: int = TRACES_PER_SAMPLE
) -> List[List[np.ndarray]]:
    """Group a device's traces by ndjson file (1 file = 1 sample = 7 traces).

    Used for return-visit / cold-start eval where each ndjson file is treated
    as one "sample" and queries draw k samples per device.
    """
    dpdir = dev_dir / "drawnapart"
    if not dpdir.is_dir():
        return []
    files = sorted(set(dpdir.rglob("*.ndjson")) | set(dpdir.rglob("*.ndjson.gz")))
    samples: List[List[np.ndarray]] = []
    for fp in files:
        arrs = load_ndjson_traces(fp, dedup=True, min_len=min_len)
        if len(arrs) >= traces_per_sample:
            samples.append(arrs[:traces_per_sample])
    return samples


# ---------- Sample discovery ------------------------------------------------

@dataclass
class DPSample:
    device: str
    mtime: float
    traces: List[np.ndarray]  # exactly TRACES_PER_SAMPLE entries, each (TRACE_LEN,)


def discover_samples_strict_4x7(
    root: Path,
    min_trace_len: int = TRACE_LEN,
    dedup: bool = True,
    debug: bool = False,
    whitelist: Optional[Set[str]] = None,
    dump_devices: bool = False,
    dump_path: Optional[Path] = None,
    min_file_bytes: int = 0,
    samples_per_device: int = SAMPLES_PER_DEVICE,
) -> Dict[str, List[DPSample]]:
    """Keep devices with exactly `samples_per_device` ndjson files, each
    yielding >= TRACES_PER_SAMPLE valid traces. `min_file_bytes > 0` warns
    about ndjson files smaller than that."""
    all_paths = list_devices(root)
    if whitelist is not None:
        all_paths = [p for p in all_paths if p.name in whitelist]
        print(f"[whitelist] {len(whitelist)} requested; {len(all_paths)} on disk")

    if min_file_bytes > 0:
        small: List[Tuple[Path, int]] = []
        for dev in all_paths:
            dpdir = dev / "drawnapart"
            if not dpdir.is_dir():
                continue
            for fp in sorted(set(dpdir.rglob("*.ndjson")) | set(dpdir.rglob("*.ndjson.gz"))):
                try:
                    sz = fp.stat().st_size
                    if sz < min_file_bytes:
                        small.append((fp, sz))
                except Exception:
                    pass
        if small:
            print(f"[size-check] {len(small)} ndjson file(s) < {min_file_bytes // 1024} KB:")
            for fp, sz in sorted(small):
                print(f"  {fp}  ({sz / 1024:.1f} KB)")
        else:
            print(f"[size-check] All ndjson files >= {min_file_bytes // 1024} KB. OK.")

    dev_map_all: Dict[str, List[DPSample]] = {}
    for dev in tqdm(all_paths, desc="scan devices", unit="dev"):
        dpdir = dev / "drawnapart"
        if not dpdir.is_dir():
            continue
        files = sorted(set(dpdir.rglob("*.ndjson")) | set(dpdir.rglob("*.ndjson.gz")))
        samples: List[DPSample] = []
        for fp in files:
            try:
                ts = fp.stat().st_mtime
            except Exception:
                ts = 0.0
            arrs = load_ndjson_traces(fp, dedup=dedup, min_len=min_trace_len)
            if len(arrs) >= TRACES_PER_SAMPLE:
                samples.append(DPSample(dev.name, ts, arrs[:TRACES_PER_SAMPLE]))
        if samples:
            samples.sort(key=lambda s: s.mtime)
            dev_map_all[dev.name] = samples

    dev_map = {d: ss for d, ss in dev_map_all.items() if len(ss) == samples_per_device}

    if whitelist is not None:
        ok = set(dev_map.keys())
        nok = whitelist - set(dev_map_all.keys())
        bad = set(dev_map_all.keys()) - ok
        print(f"[whitelist] satisfying whitelist + 4x7 : {len(ok)}")
        print(f"[whitelist] in whitelist, no DP data   : {len(nok)}")
        print(f"[whitelist] has DP but != {samples_per_device} samples : {len(bad)}")
    elif debug:
        print(f"[debug] any DP samples : {len(dev_map_all)}")
        print(f"[debug] kept ({samples_per_device}x{TRACES_PER_SAMPLE}) : {len(dev_map)}")

    if dump_devices and dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(sorted(dev_map.keys()), indent=2), encoding="utf-8")
        print(f"[devices] saved to {dump_path}")

    return dev_map


# ---------- Preprocessing / splits ------------------------------------------

def trace_to_img(x: np.ndarray) -> np.ndarray:
    """Z-score 1024-dim trace then reshape to (1, 32, 32). Paper §VI-A."""
    x = x.astype(np.float32).reshape(-1)[:TRACE_LEN]
    mu = float(x.mean())
    sd = float(x.std())
    if sd < 1e-8:
        sd = 1.0
    return ((x - mu) / sd).reshape(1, 32, 32).astype(np.float32)


def split_stratified(
    y: np.ndarray, seed: int, mem_frac: float = 0.80, train_frac: float = 0.80
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class 80/20 mem/test, then 80/20 train/val within mem."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = idx.size
        n_test = max(1, min(int(round((1 - mem_frac) * n)), n - 2))
        mem = idx[: n - n_test]
        te = idx[n - n_test:]
        n_tr = max(1, min(int(round(train_frac * mem.size)), mem.size - 1))
        train.extend(mem[:n_tr].tolist())
        val.extend(mem[n_tr:].tolist())
        test.extend(te.tolist())
    return (
        np.array(train, np.int64),
        np.array(val, np.int64),
        np.array(test, np.int64),
    )

