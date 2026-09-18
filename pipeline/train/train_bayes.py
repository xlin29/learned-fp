#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_bayes.py — Per-key CNN trainer with Bayesian (GP) hyperparameter search.
    Aligned to DRAWNAPART's budget: 79 trials × 5 epochs / trial.

    Paper:    §5.2 (DRAWNAPART matched comparison) + Appendix C (hyperparameters)
"""

import argparse
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args


# -------------------- config --------------------
EXCLUDED_KEYS = {"raw_flags"}
REQUIRED_TRACES_PER_KEY = 100
VISITS = ("visit1",)
H = W = 100


# -------------------- json helpers --------------------
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _dumps(obj, **kwargs) -> str:
    if "default" not in kwargs:
        kwargs["default"] = _json_default
    return json.dumps(obj, **kwargs)


# -------------------- utils --------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: PYTHONHASHSEED must be exported before python starts for strict effect,
    # but we keep this for logging consistency.
    os.environ["PYTHONHASHSEED"] = str(seed)


_SESSION_PREFIX_RE = re.compile(r"^S\d+_")


def _strip_ext_and_meta(name: str) -> Optional[str]:
    low = name.lower()
    if "hash" in low or low.endswith(".meta.json"):
        return None
    return name[:-len(".json.gz")] if low.endswith(".json.gz") else Path(name).stem


def extract_key_from_filename(name: str) -> Optional[str]:
    base = _strip_ext_and_meta(name)
    if not base:
        return None
    s = _SESSION_PREFIX_RE.sub("", base)
    if not s:
        return None
    toks = s.split("_")
    if not toks or not toks[0]:
        return None
    if toks[0] == "raw" and len(toks) >= 2 and toks[1]:
        return f"raw_{toks[1]}"
    return toks[0]


def list_device_folders(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.glob("dev_*") if p.is_dir()])


def list_key_rgba_sorted_all_visits(dev_dir: Path, key_string: str) -> List[Path]:
    items = []
    for v in VISITS:
        vdir = dev_dir / v
        if not vdir.is_dir():
            continue
        for p in vdir.rglob("*.rgba"):
            k = extract_key_from_filename(p.name)
            if k != key_string:
                continue
            try:
                mt = p.stat().st_mtime
            except Exception:
                mt = 0.0
            items.append((p, mt, p.name))
    items.sort(key=lambda x: (x[1], x[2]))
    return [p for p, _, _ in items]


def collect_device_key_lists_all_vis(base_dir: Path, keys: List[str]) -> Dict[str, Dict[str, List[Path]]]:
    """
    Keep first REQUIRED_TRACES_PER_KEY traces per key per device.
    Keep device only if every key has >= REQUIRED_TRACES_PER_KEY.
    """
    out: Dict[str, Dict[str, List[Path]]] = {}
    for dev in list_device_folders(base_dir):
        if not any((dev / v).is_dir() for v in VISITS):
            continue
        perkey: Dict[str, List[Path]] = {}
        ok = True
        for k in keys:
            lst = list_key_rgba_sorted_all_visits(dev, k)
            if len(lst) < REQUIRED_TRACES_PER_KEY:
                ok = False
                break
            perkey[k] = lst[:REQUIRED_TRACES_PER_KEY]
        if ok:
            out[dev.name] = perkey
    return out


def build_shared_split_all_vis(
    device_key_lists: Dict[str, Dict[str, List[Path]]],
    keys: List[str],
    seed: int,
    test_frac: float = 0.20,
) -> Tuple[Dict[str, Dict[str, List[Path]]], Dict[str, Dict[str, List[Path]]]]:
    """
    Outer split: cal/test, aligned across keys per device.
    """
    rng = np.random.default_rng(seed)
    cal_plan, test_plan = {k: {} for k in keys}, {k: {} for k in keys}

    for dname, perkey in device_key_lists.items():
        N = min(len(perkey[k]) for k in keys)
        if N < 2:
            continue
        perm = rng.permutation(N)
        n_te = max(1, int(round(N * test_frac)))
        n_tr = max(1, N - n_te)
        if n_tr + n_te > N:
            n_tr = N - n_te

        tr_idx = sorted(perm[:n_tr].tolist())
        te_idx = sorted(perm[n_tr:].tolist())

        for k in keys:
            files = perkey[k][:N]
            cal_plan[k][dname] = [files[i] for i in tr_idx]
            test_plan[k][dname] = [files[i] for i in te_idx]

    return cal_plan, test_plan


def split_cal_into_train_val(
    cal_plan: Dict[str, Dict[str, List[str]]],
    keys: List[str],
    seed: int,
    val_frac: float = 0.20,
) -> Tuple[dict, dict]:
    """
    Inner split inside cal: train/val, aligned across keys per device.
    Input cal_plan uses string paths in meta json format (recommended) or Path objects.
    Returns train_plan, val_plan as dicts like cal_plan.
    """
    rng = np.random.default_rng(seed)
    train_plan = {k: {} for k in keys}
    val_plan = {k: {} for k in keys}

    devs = None
    for k in keys:
        ds = set(cal_plan.get(k, {}).keys())
        devs = ds if devs is None else (devs & ds)
    devs = sorted(devs or [])

    for d in devs:
        N = min(len(cal_plan[k][d]) for k in keys)
        if N < 2:
            continue
        perm = rng.permutation(N)
        n_val = max(1, int(round(N * val_frac)))
        n_val = min(n_val, N - 1)

        val_idx = sorted(perm[:n_val].tolist())
        tr_idx = sorted(perm[n_val:].tolist())

        for k in keys:
            files = cal_plan[k][d][:N]
            train_plan[k][d] = [files[i] for i in tr_idx]
            val_plan[k][d] = [files[i] for i in val_idx]

    return train_plan, val_plan


def _write_text_atomic(path: Path, text: str, encoding: str = "utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def _try_acquire_lock(lock_path: Path) -> bool:
    """
    Simple lock via atomic create. Returns True if acquired.
    """
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _wait_for_file(path: Path, timeout_s: int = 3600, poll_s: float = 2.0):
    t0 = time.time()
    while True:
        if path.exists() and path.stat().st_size > 0:
            return
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"[fatal] timeout waiting for {path}")
        time.sleep(poll_s)


# -------------------- model --------------------
class PixelCNN(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32, blocks: int = 3, ksize: int = 3, dropout: float = 0.2):
        super().__init__()
        c0 = int(base_channels)
        b = max(2, min(int(blocks), 4))  # safe for 100x100
        k = int(ksize)
        pad = k // 2

        layers: List[nn.Module] = []
        in_ch = 4
        for bi in range(b):
            out_ch = c0 * (2 ** bi)
            layers += [
                nn.Conv2d(in_ch, out_ch, k, padding=pad),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, k, padding=pad),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch

        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# -------------------- dataset + loaders --------------------
def load_rgba_as_tensor(path: Path) -> Optional[np.ndarray]:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 4:
        return None
    n = (raw.size // 4) * 4
    if n == 0:
        return None
    arr = raw[:n].reshape(-1, 4).astype(np.int32, copy=False)
    a0 = (arr[:, 3] == 0)
    if a0.any():
        arr[a0, 0:3] = 0
    if arr.shape[0] != H * W:
        return None
    img = arr.reshape(H, W, 4).transpose(2, 0, 1) / 255.0
    return img.astype(np.float32)


class PathsDataset(Dataset):
    def __init__(self, paths: List[Path], labels: List[int], mean: Optional[np.ndarray], std: Optional[np.ndarray]):
        self.paths = paths
        self.labels = labels
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_rgba_as_tensor(self.paths[i])
        if img is None:
            raise RuntimeError(f"Failed to load RGBA: {self.paths[i]}")
        if self.mean is not None and self.std is not None:
            img = (img - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-6)
        return torch.from_numpy(img), int(self.labels[i])


def compute_pixnorm_from_paths(paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    s1 = np.zeros((4,), dtype=np.float64)
    s2 = np.zeros((4,), dtype=np.float64)
    npx = 0
    for p in paths:
        img = load_rgba_as_tensor(p)
        if img is None:
            continue
        px = img.reshape(4, -1)
        s1 += px.sum(axis=1)
        s2 += (px**2).sum(axis=1)
        npx += px.shape[1]
    mean = (s1 / max(npx, 1)).astype(np.float32)
    var = (s2 / max(npx, 1) - mean**2).astype(np.float32)
    std = np.sqrt(np.clip(var, 1e-8, None)).astype(np.float32)
    return mean, std


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {}
        if not (0.0 < self.decay < 1.0):
            self.shadow = None
            return
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()

    def update(self, model: nn.Module):
        if not self.shadow:
            return
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module):
        if not self.shadow:
            return
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    v.copy_(self.shadow[k])


# -------------------- training --------------------
def train_model_ce(
    num_classes: int,
    train_paths: List[Path],
    train_labels: List[int],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    wd: float,
    label_smoothing: float,
    ema_decay: float,
    warmup_epochs: int,
    base_channels: int,
    blocks: int,
    ksize: int,
    dropout: float,
) -> nn.Module:
    ds = PathsDataset(train_paths, train_labels, mean, std)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, num_workers=0, drop_last=False)

    model = PixelCNN(
        num_classes=num_classes,
        base_channels=int(base_channels),
        blocks=int(blocks),
        ksize=int(ksize),
        dropout=float(dropout),
    ).to(device)

    opt = AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))
    sched = CosineAnnealingLR(opt, T_max=max(1, int(epochs) - max(1, int(warmup_epochs))))
    ema = EMA(model, float(ema_decay))

    use_amp = (device.type == "cuda")
    amp_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    base_lr = float(lr)
    for ep in range(1, int(epochs) + 1):
        model.train()
        if ep <= max(1, int(warmup_epochs)):
            scale = float(ep) / float(max(1, int(warmup_epochs)))
            for pg in opt.param_groups:
                pg["lr"] = base_lr * scale

        for xb, yb in dl:
            xb = xb.to(device)
            yb = torch.as_tensor(yb, dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)

            if use_amp:
                with amp_ctx:
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb, label_smoothing=float(label_smoothing))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=float(label_smoothing))
                loss.backward()
                opt.step()

            ema.update(model)

        if ep > max(1, int(warmup_epochs)):
            sched.step()

    if ema.shadow is not None:
        ema.copy_to(model)
    return model


@torch.no_grad()
def eval_top1_softmax(
    model: nn.Module,
    paths: List[Path],
    labels: List[int],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    ds = PathsDataset(paths, labels, mean, std)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0, drop_last=False)
    model.eval().to(device)
    corr = 0
    tot = 0
    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        corr += int((pred == yb).sum().item())
        tot += int(yb.numel())
    return corr / float(max(1, tot))


# -------------------- vote helpers --------------------
@torch.no_grad()
def logits_for_paths(model: nn.Module, paths: List[Path], mean: np.ndarray, std: np.ndarray, device: torch.device, bs: int = 256) -> np.ndarray:
    out = []
    m = torch.from_numpy(mean.reshape(1, 4, 1, 1)).to(device)
    s = torch.from_numpy(std.reshape(1, 4, 1, 1)).to(device)
    batch = []
    for p in paths:
        img = load_rgba_as_tensor(p)
        if img is None:
            continue
        t = torch.from_numpy(img).unsqueeze(0).to(device)
        t = (t - m) / (s + 1e-6)
        batch.append(t)
        if len(batch) == int(bs):
            xb = torch.cat(batch, dim=0)
            out.append(model(xb).detach().cpu().numpy())
            batch = []
    if batch:
        xb = torch.cat(batch, dim=0)
        out.append(model(xb).detach().cpu().numpy())
    if not out:
        return np.zeros((0, model.head[-1].out_features), dtype=np.float32)
    return np.concatenate(out, axis=0)


def fuse_traces_to_prob(logits_mat: np.ndarray, mode: str) -> Optional[np.ndarray]:
    if logits_mat.size == 0 or logits_mat.ndim != 2:
        return None
    if mode == "prob_mean":
        p = F.softmax(torch.from_numpy(logits_mat), dim=1).numpy()
        return p.mean(axis=0).astype(np.float32)
    lm = logits_mat.mean(axis=0, keepdims=True)
    p = F.softmax(torch.from_numpy(lm), dim=1).numpy()[0]
    return p.astype(np.float32)


def make_nonoverlap_blocks(paths: List[str], block: int) -> List[List[Path]]:
    block = max(1, int(block))
    L = len(paths)
    out: List[List[Path]] = []
    for start in range(0, L - block + 1, block):
        out.append([Path(paths[i]) for i in range(start, start + block)])
    return out


def base_rate_from_cal_plan(cal_plan: dict, keys: List[str], idx2dev: List[str], topk_list: List[int]) -> Dict[str, float]:
    counts = np.zeros(len(idx2dev), dtype=np.float64)
    for d_idx, d in enumerate(idx2dev):
        for k in keys:
            counts[d_idx] += len(cal_plan.get(k, {}).get(d, []))
    counts_sorted = np.sort(counts)[::-1]
    tot = float(counts_sorted.sum())
    out = {}
    for k in topk_list:
        kk = min(k, len(counts_sorted))
        out[f"top{k}"] = float(counts_sorted[:kk].sum() / tot) if tot > 0 else float("nan")
    return out


# -------------------- CLI modes --------------------
def mode_train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    base_dir = Path(args.base_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = results_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # discover keys
    discovered = set()
    for dev in list_device_folders(base_dir):
        if not any((dev / v).is_dir() for v in VISITS):
            continue
        for v in VISITS:
            vdir = dev / v
            if not vdir.is_dir():
                continue
            for p in vdir.rglob("*.rgba"):
                k = extract_key_from_filename(p.name)
                if k:
                    discovered.add(k)
    discovered = sorted(k for k in discovered if k not in EXCLUDED_KEYS)

    arg = (args.keys or "").strip()
    m = re.fullmatch(r"auto(?:0*(\d+))?$", arg, flags=re.IGNORECASE)
    if arg.upper() == "ALL":
        keys_all = discovered
    elif m:
        n = int(m.group(1)) if m.group(1) else None
        keys_all = discovered[: (n if n is not None else len(discovered))]
    else:
        req = [k.strip() for k in arg.split(",") if k.strip()]
        keys_all = [k for k in req if (k in set(discovered)) and (k not in EXCLUDED_KEYS)]

    if not keys_all:
        print("[exit] no usable keys after selection")
        return

    if not (0 <= args.key_index < len(keys_all)):
        raise SystemExit(f"--key-index {args.key_index} out of range [0..{len(keys_all)-1}]")
    key = keys_all[args.key_index]
    print(f"[key] training key[{args.key_index}] = {key}")
    print(f"[keys] global set for aligned split: {keys_all}")

    # build outer split
    dkl = collect_device_key_lists_all_vis(base_dir, keys_all)
    if not dkl:
        print("[exit] no devices with required traces across all keys")
        return
    cal_plan_raw, test_plan_raw = build_shared_split_all_vis(dkl, keys_all, seed=args.seed, test_frac=float(args.test_frac))

    # strict intersection on devices in cal across all keys
    devices_kept = None
    for k in keys_all:
        ds = set(cal_plan_raw.get(k, {}).keys())
        devices_kept = ds if devices_kept is None else (devices_kept & ds)
    devices_kept = sorted(devices_kept or [])
    if not devices_kept:
        print("[exit] no devices present in calibration across all keys")
        return

    idx2dev_path = meta_dir / "devices_kept.json"
    sig_path = meta_dir / "plan_signature.json"
    cal_plan_path = meta_dir / "cal_plan.json"
    test_plan_path = meta_dir / "test_plan.json"
    train_plan_path = meta_dir / "train_plan.json"
    val_plan_path = meta_dir / "val_plan.json"
    done_path = meta_dir / ".meta_done.json"
    lock_path = meta_dir / ".meta_lock"

    sig = {
        "seed": int(args.seed),
        "keys": keys_all,
        "test_frac": float(args.test_frac),
        "val_frac_in_cal": float(args.val_frac_in_cal),
        "excluded_keys": sorted(list(EXCLUDED_KEYS)),
        "visits": list(VISITS),
        "required_traces_per_key": int(REQUIRED_TRACES_PER_KEY),
        "tuning_budget": {"search_trials": int(args.search_trials), "search_epochs": int(args.search_epochs)},
        "note": "meta is lock-guarded; other tasks wait for meta done",
    }

    cal_plan_dump = {k: {d: [str(p) for p in lst] for d, lst in cal_plan_raw[k].items()} for k in keys_all}
    test_plan_dump = {k: {d: [str(p) for p in lst] for d, lst in test_plan_raw[k].items()} for k in keys_all}

    # --- meta initialization guarded by lock ---
    if done_path.exists():
        # meta finished already; sanity check
        prev_sig = json.loads(sig_path.read_text()) if sig_path.exists() else None
        if prev_sig != sig:
            raise SystemExit("[fatal] plan_signature mismatch vs existing meta/plan_signature.json")
        prev_idx2dev = json.loads(idx2dev_path.read_text())
        if prev_idx2dev != devices_kept:
            raise SystemExit("[fatal] devices_kept mismatch vs existing meta/devices_kept.json")
    else:
        if _try_acquire_lock(lock_path):
            try:
                # re-check inside lock
                if not done_path.exists():
                    _write_text_atomic(idx2dev_path, _dumps(devices_kept, indent=2))
                    _write_text_atomic(sig_path, _dumps(sig, indent=2))
                    _write_text_atomic(cal_plan_path, _dumps(cal_plan_dump, indent=2))
                    _write_text_atomic(test_plan_path, _dumps(test_plan_dump, indent=2))

                    train_plan, val_plan = split_cal_into_train_val(
                        cal_plan_dump,
                        keys_all,
                        seed=args.seed,
                        val_frac=float(args.val_frac_in_cal),
                    )
                    _write_text_atomic(train_plan_path, _dumps(train_plan, indent=2))
                    _write_text_atomic(val_plan_path, _dumps(val_plan, indent=2))

                    _write_text_atomic(done_path, _dumps({"ok": True, "time": time.time()}, indent=2))
            finally:
                # best-effort remove lock
                try:
                    lock_path.unlink(missing_ok=True)  # py3.8+ supports missing_ok
                except TypeError:
                    if lock_path.exists():
                        lock_path.unlink()
        else:
            _wait_for_file(done_path, timeout_s=3600, poll_s=2.0)

    # Ensure required files exist before proceeding (important for arrays)
    _wait_for_file(done_path, timeout_s=3600, poll_s=1.0)
    _wait_for_file(train_plan_path, timeout_s=3600, poll_s=1.0)
    _wait_for_file(val_plan_path, timeout_s=3600, poll_s=1.0)

    # load canonical order for labels
    idx2dev = json.loads(idx2dev_path.read_text())
    dev2idx = {d: i for i, d in enumerate(idx2dev)}
    C = len(idx2dev)

    train_plan = json.loads(train_plan_path.read_text())
    val_plan = json.loads(val_plan_path.read_text())

    # build per-key train/val path lists
    train_paths: List[Path] = []
    train_labels: List[int] = []
    val_paths: List[Path] = []
    val_labels: List[int] = []
    cal_paths_full: List[Path] = []
    cal_labels_full: List[int] = []

    for d in idx2dev:
        y = dev2idx[d]
        for p in train_plan.get(key, {}).get(d, []):
            train_paths.append(Path(p)); train_labels.append(y)
        for p in val_plan.get(key, {}).get(d, []):
            val_paths.append(Path(p)); val_labels.append(y)
        for p in cal_plan_dump.get(key, {}).get(d, []):
            cal_paths_full.append(Path(p)); cal_labels_full.append(y)

    if not train_paths or not val_paths:
        raise SystemExit("[fatal] empty train/val after inner split; check val_frac_in_cal and REQUIRED_TRACES_PER_KEY")

    # normalization uses TRAIN only during tuning
    mean_tr, std_tr = compute_pixnorm_from_paths(train_paths)

    # ---- Bayesian search (DP-aligned budget) ----
    space = [
        Integer(16, 64, name="base_channels"),
        Integer(2, 4, name="blocks"),
        Categorical([3, 5], name="ksize"),
        Real(0.0, 0.5, name="dropout"),
        Real(1e-4, 3e-3, prior="log-uniform", name="lr"),
        Real(1e-6, 1e-3, prior="log-uniform", name="wd"),
        Real(0.0, 0.10, name="label_smoothing"),
        Categorical([128, 256, 512], name="batch_size"),
        Categorical([0.0, 0.99, 0.999], name="ema_decay"),
        Integer(1, 3, name="warmup_epochs"),
    ]

    @use_named_args(space)
    def objective(base_channels, blocks, ksize, dropout, lr, wd, label_smoothing, batch_size, ema_decay, warmup_epochs):
        try:
            model = train_model_ce(
                num_classes=C,
                train_paths=train_paths,
                train_labels=train_labels,
                mean=mean_tr,
                std=std_tr,
                device=device,
                epochs=int(args.search_epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                wd=float(wd),
                label_smoothing=float(label_smoothing),
                ema_decay=float(ema_decay),
                warmup_epochs=int(warmup_epochs),
                base_channels=int(base_channels),
                blocks=int(blocks),
                ksize=int(ksize),
                dropout=float(dropout),
            )
            acc = eval_top1_softmax(
                model,
                val_paths,
                val_labels,
                mean_tr,
                std_tr,
                device=device,
                batch_size=max(128, int(batch_size)),
            )
            return -float(acc)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and device.type == "cuda":
                torch.cuda.empty_cache()
                return 1.0
            raise

    print(f"[search] key={key} trials={args.search_trials} epochs_per_trial={args.search_epochs} (VAL only)")
    res = gp_minimize(objective, space, n_calls=int(args.search_trials), random_state=int(args.seed))
    best_val = -float(res.fun)

    # cast best params to plain python types to avoid any downstream surprises
    best = dict(zip([s.name for s in space], res.x))
    best = {k: (_json_default(v) if isinstance(v, (np.integer, np.floating, np.ndarray)) else v) for k, v in best.items()}
    print("[search] best_val_top1=", best_val)
    print("[search] best_params=", best)

    # ---- final train on FULL CAL (no test) with best params ----
    mean_cal, std_cal = compute_pixnorm_from_paths(cal_paths_full)
    final_model = train_model_ce(
        num_classes=C,
        train_paths=cal_paths_full,
        train_labels=cal_labels_full,
        mean=mean_cal,
        std=std_cal,
        device=device,
        epochs=int(args.epochs),
        batch_size=int(best["batch_size"]),
        lr=float(best["lr"]),
        wd=float(best["wd"]),
        label_smoothing=float(best["label_smoothing"]),
        ema_decay=float(best["ema_decay"]),
        warmup_epochs=int(best["warmup_epochs"]),
        base_channels=int(best["base_channels"]),
        blocks=int(best["blocks"]),
        ksize=int(best["ksize"]),
        dropout=float(best["dropout"]),
    )

    out_models = results_dir / "models"
    out_pixnorm = results_dir / "pixnorm"
    out_hparams = results_dir / "hparams"
    out_models.mkdir(parents=True, exist_ok=True)
    out_pixnorm.mkdir(parents=True, exist_ok=True)
    out_hparams.mkdir(parents=True, exist_ok=True)

    torch.save(final_model.state_dict(), out_models / f"fp_rawcnn_{key}.pt")
    (out_pixnorm / f"fp_rawcnn_pixnorm_{key}.json").write_text(
        _dumps({"pix_mean": mean_cal.tolist(), "pix_std": std_cal.tolist()}, indent=2),
        encoding="utf-8",
    )
    (out_hparams / f"fp_rawcnn_best_{key}.json").write_text(
        _dumps(
            {
                "key": key,
                "best_val_top1": float(best_val),
                "best_params": best,
                "search_trials": int(args.search_trials),
                "search_epochs": int(args.search_epochs),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] trained key={key} (tuned, no leak) saved -> {out_models}/")


def mode_vote(args):
    set_seed(args.seed)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    results_dir = Path(args.results_dir)
    models_dir = results_dir / "models"
    pixnorm_dir = results_dir / "pixnorm"
    hparams_dir = results_dir / "hparams"
    meta_dir = results_dir / "meta"

    try:
        idx2dev = json.loads((meta_dir / "devices_kept.json").read_text())
        sig = json.loads((meta_dir / "plan_signature.json").read_text())
        cal_plan = json.loads((meta_dir / "cal_plan.json").read_text())
        test_plan = json.loads((meta_dir / "test_plan.json").read_text())
    except Exception as e:
        print(f"[fatal] missing meta files in {meta_dir}: {e}", file=sys.stderr)
        sys.exit(2)

    plan_keys = sig.get("keys", [])
    req = (args.keys or "").strip()
    m = re.fullmatch(r"auto(?:0*(\d+))?$", req, flags=re.IGNORECASE)
    if req.upper() == "ALL":
        keys = plan_keys
    elif m:
        n = int(m.group(1)) if m.group(1) else None
        keys = plan_keys[: (n if n is not None else len(plan_keys))]
    else:
        want = [k.strip() for k in req.split(",") if k.strip()]
        keys = [k for k in plan_keys if k in set(want)]
    if not keys:
        print("[fatal] no keys selected", file=sys.stderr)
        sys.exit(3)

    C = len(idx2dev)
    dev2idx = {d: i for i, d in enumerate(idx2dev)}

    models: Dict[str, nn.Module] = {}
    norms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for k in keys:
        mpath = models_dir / f"fp_rawcnn_{k}.pt"
        jpath = pixnorm_dir / f"fp_rawcnn_pixnorm_{k}.json"
        hpath = hparams_dir / f"fp_rawcnn_best_{k}.json"
        if not mpath.exists() or not jpath.exists() or not hpath.exists():
            print(f"[fatal] missing trained artifacts for key={k}", file=sys.stderr)
            sys.exit(4)

        hp = json.loads(hpath.read_text())
        bp = hp["best_params"]

        model = PixelCNN(
            num_classes=C,
            base_channels=int(bp["base_channels"]),
            blocks=int(bp["blocks"]),
            ksize=int(bp["ksize"]),
            dropout=float(bp["dropout"]),
        ).to(device)

        try:
            state = torch.load(mpath, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(mpath, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()

        jp = json.loads(jpath.read_text())
        mean = np.array(jp["pix_mean"], dtype=np.float32)
        std = np.array(jp["pix_std"], dtype=np.float32)
        models[k] = model
        norms[k] = (mean, std)

    block = max(1, int(args.n_test_per_key))
    per_query: List[Tuple[str, int]] = []

    for d in idx2dev:
        if any(d not in test_plan.get(k, {}) for k in keys):
            continue
        blocks_per_key = [len(make_nonoverlap_blocks(test_plan[k][d], block)) for k in keys]
        B = min(blocks_per_key) if blocks_per_key else 0
        for bi in range(B):
            per_query.append((d, bi))

    if not per_query:
        print("[fatal] no test queries after blocking", file=sys.stderr)
        sys.exit(5)

    try:
        topk_list = sorted({int(t) for t in args.topk.split(",") if t.strip().isdigit()})
    except Exception:
        topk_list = [1, 5, 10, 20]
    if not topk_list:
        topk_list = [1, 5, 10, 20]

    per_trace_mode = str(args.per_key_trace_agg)
    fuse_mode = str(args.fuse_mode)

    ens_correct = {k: 0 for k in topk_list}
    perkey_correct = {key: {k: 0 for k in topk_list} for key in keys}
    perkey_total = {key: 0 for key in keys}

    for (dev_name, bi) in per_query:
        gt = dev2idx.get(dev_name, None)
        if gt is None:
            continue

        perkey_probs: Dict[str, np.ndarray] = {}
        perkey_logits_mean: Dict[str, np.ndarray] = {}

        for k in keys:
            paths_blocks = make_nonoverlap_blocks(test_plan[k][dev_name], block)
            paths_k = paths_blocks[bi]
            model = models[k]
            mean, std = norms[k]
            logits_mat = logits_for_paths(model, paths_k, mean, std, device=device, bs=int(args.batch_size))
            if logits_mat.shape[0] == 0:
                continue
            perkey_logits_mean[k] = logits_mat.mean(axis=0).astype(np.float32)
            p_k = fuse_traces_to_prob(logits_mat, mode=per_trace_mode)
            if p_k is not None:
                perkey_probs[k] = p_k

        if perkey_probs:
            if fuse_mode == "prob_mean":
                prob_sum = np.zeros((C,), dtype=np.float32)
                used = 0
                for p in perkey_probs.values():
                    prob_sum += p
                    used += 1
                prob_avg = prob_sum / max(1, used)
                order = np.argsort(-prob_avg)
            else:
                logit_sum = np.zeros((C,), dtype=np.float32)
                for lm in perkey_logits_mean.values():
                    logit_sum += lm
                prob = F.softmax(torch.from_numpy(logit_sum.reshape(1, -1)), dim=1).numpy()[0]
                order = np.argsort(-prob)

            for kk in topk_list:
                if gt in order[:kk]:
                    ens_correct[kk] += 1

        for k, p in perkey_probs.items():
            perkey_total[k] += 1
            order_k = np.argsort(-p)
            for kk in topk_list:
                if gt in order_k[:kk]:
                    perkey_correct[k][kk] += 1

    total = len(per_query)
    ens_acc = {f"top{k}": (ens_correct[k] / max(1, total)) for k in topk_list}
    perkey_acc = {
        key: {f"top{k}": (perkey_correct[key][k] / max(1, perkey_total[key])) for k in topk_list}
        for key in keys
    }
    base = base_rate_from_cal_plan(cal_plan, keys, idx2dev, topk_list)

    summary = {
        "keys": keys,
        "num_classes": len(idx2dev),
        "num_queries_test": total,
        "test_query_block_size": block,
        "per_key_trace_agg": per_trace_mode,
        "fuse_mode": fuse_mode,
        "base_rates_from_cal": base,
        "ensemble_topk_accuracy": ens_acc,
        "per_key_topk": perkey_acc,
        "per_key_eval_counts": perkey_total,
        "note": "test queries are non-overlapping blocks; no test trace reused across queries",
    }

    (results_dir / "fp_vote_summary.json").write_text(_dumps(summary, indent=2), encoding="utf-8")
    (results_dir / "fp_vote_per_key.json").write_text(_dumps(perkey_acc, indent=2), encoding="utf-8")
    print(_dumps(summary, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description="FP per-key CNN with DP-matched tuning budget and no-leak RS evaluation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train one key with Bayesian tuning on cal train/val, then final train on full cal.")
    pt.add_argument("--base-dir", required=True)
    pt.add_argument("--results-dir", required=True)
    pt.add_argument("--keys", default="AUTO9")
    pt.add_argument("--key-index", type=int, required=True)

    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--gpu", action="store_true")

    pt.add_argument("--test-frac", type=float, default=0.20)
    pt.add_argument("--val-frac-in-cal", type=float, default=0.20)

    pt.add_argument("--search-trials", type=int, default=79)
    pt.add_argument("--search-epochs", type=int, default=5)

    pt.add_argument("--epochs", type=int, default=30)

    pv = sub.add_parser("vote", help="Vote on test only; test queries are non-overlapping blocks.")
    pv.add_argument("--base-dir", required=True, help="accepted; not used by vote")
    pv.add_argument("--results-dir", required=True)
    pv.add_argument("--keys", default="AUTO9")
    pv.add_argument("--seed", type=int, default=42)
    pv.add_argument("--gpu", action="store_true")
    pv.add_argument("--batch-size", type=int, default=256)
    pv.add_argument("--topk", default="1,5,10,20")

    pv.add_argument("--n-test-per-key", type=int, default=5, help="non-overlapping block size")
    pv.add_argument("--per-key-trace-agg", choices=["logit_mean", "prob_mean"], default="logit_mean")
    pv.add_argument("--fuse-mode", choices=["logit_sum", "prob_mean"], default="logit_sum")

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "train":
        mode_train(args)
    elif args.cmd == "vote":
        mode_vote(args)
    else:
        raise SystemExit("unknown command")


if __name__ == "__main__":
    main()