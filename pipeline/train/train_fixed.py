#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_fixed.py — Per-key CNN trainer with fixed default hyperparameters.
    Used for the full-population gallery; no hyperparameter search.

    Paper:    §5.4 (scalability — 3,303 devices)
"""

import argparse, json, os, re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# -------------------- config --------------------
EXCLUDED_KEYS = {"raw_flags"}
REQUIRED_TRACES_PER_KEY = 100
VISITS = ("visit1",)
H = W = 100


# -------------------- utils --------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    For each device keep first REQUIRED_TRACES_PER_KEY traces per key from VISITS.
    Only keep device if every key has at least REQUIRED_TRACES_PER_KEY traces.
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
    test_frac: float = 0.20
) -> Tuple[Dict[str, Dict[str, List[Path]]], Dict[str, Dict[str, List[Path]]]]:
    """
    Shared trace-level split per device.
    For each device generate a permutation over N where N is min traces across keys.
    Split indices into calibration and test so alignment across keys is preserved.
    """
    rng = np.random.default_rng(seed)
    cal_plan, test_plan = {k: {} for k in keys}, {k: {} for k in keys}

    for dname, perkey in device_key_lists.items():
        N = min(len(perkey[k]) for k in keys)
        if N < 2:
            continue
        perm = rng.permutation(N)
        n_te = max(1, int(round(N * test_frac)))
        # FIX: clamp n_te so at least 1 sample remains for calibration
        n_te = min(n_te, N - 1)
        n_tr = N - n_te

        tr_idx = sorted(perm[:n_tr].tolist())
        te_idx = sorted(perm[n_tr:].tolist())

        for k in keys:
            files = perkey[k][:N]
            cal_plan[k][dname] = [files[i] for i in tr_idx]
            test_plan[k][dname] = [files[i] for i in te_idx]

    return cal_plan, test_plan


# -------------------- model --------------------
class PixelCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        c = 32
        self.backbone = nn.Sequential(
            nn.Conv2d(4, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c, 2*c, 3, padding=1), nn.BatchNorm2d(2*c), nn.ReLU(inplace=True),
            nn.Conv2d(2*c, 2*c, 3, padding=1), nn.BatchNorm2d(2*c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(2*c, 4*c, 3, padding=1), nn.BatchNorm2d(4*c), nn.ReLU(inplace=True),
            nn.Conv2d(4*c, 4*c, 3, padding=1), nn.BatchNorm2d(4*c), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*c, 256), nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(256, num_classes)
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
def train_one_key(
    key: str,
    cal_plan_key: Dict[str, List[Path]],
    idx2dev: List[str],
    device: torch.device,
    out_models: Path,
    out_pixnorm: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    wd: float,
    label_smoothing: float,
    ema_decay: float,
    warmup_epochs: int,
    dropout: float,
):
    """
    IMPORTANT: class mapping is derived from idx2dev (meta/devices_kept.json) for consistency with vote.
    """
    dev2idx = {d: i for i, d in enumerate(idx2dev)}
    num_classes = len(idx2dev)

    train_paths: List[Path] = []
    train_labels: List[int] = []
    for d in idx2dev:
        if d not in cal_plan_key:
            continue
        y = dev2idx[d]
        for p in cal_plan_key[d]:
            train_paths.append(p)
            train_labels.append(y)

    mean, std = compute_pixnorm_from_paths(train_paths)

    # filter broken files early
    clean_paths, clean_labels = [], []
    for p, y in zip(train_paths, train_labels):
        img = load_rgba_as_tensor(p)
        if img is not None:
            clean_paths.append(p)
            clean_labels.append(y)

    ds = PathsDataset(clean_paths, clean_labels, mean, std)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, num_workers=0, drop_last=False)

    model = PixelCNN(num_classes=num_classes, dropout=float(dropout)).to(device)
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

        losses = []
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
            losses.append(float(loss.item()))

        if ep > max(1, int(warmup_epochs)):
            sched.step()

        print(f"[train {key}] epoch {ep:02d}/{epochs} loss={float(np.mean(losses)):.4f}")

    if ema.shadow is not None:
        ema.copy_to(model)

    out_models.mkdir(parents=True, exist_ok=True)
    out_pixnorm.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_models / f"rawcnn_allvis_{key}.pt")
    (out_pixnorm / f"rawcnn_pixnorm_allvis_{key}.json").write_text(
        json.dumps({"pix_mean": mean.tolist(), "pix_std": std.tolist()}, indent=2),
        encoding="utf-8",
    )


# -------------------- voting helpers --------------------
@torch.no_grad()
def logits_for_paths(
    model: nn.Module,
    paths: List[Path],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    bs: int = 256
) -> np.ndarray:
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
    """
    logits_mat shape [T, C]
    mode:
      logit_mean -> mean logits over traces then softmax
      prob_mean  -> mean softmax probs over traces
    """
    if logits_mat.size == 0 or logits_mat.ndim != 2:
        return None
    if mode == "prob_mean":
        p = F.softmax(torch.from_numpy(logits_mat), dim=1).numpy()
        return p.mean(axis=0).astype(np.float32)
    lm = logits_mat.mean(axis=0, keepdims=True)
    p = F.softmax(torch.from_numpy(lm), dim=1).numpy()[0]
    return p.astype(np.float32)

def make_nonoverlap_blocks(paths: List[str], block: int) -> List[List[Path]]:
    """
    Partition a path list into non-overlapping blocks of size=block.
    Drops the tail if shorter than block, so each query uses same information amount.
    """
    block = max(1, int(block))
    out: List[List[Path]] = []
    L = len(paths)
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
    print(f"[keys] global set for intersection and split: {keys_all}")

    # build plans
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
    train_plan_path = meta_dir / "train_plan.json"  # the original job runner also waited on this
    val_plan_path = meta_dir / "val_plan.json"       # the original job runner also waited on this
    meta_done_path = meta_dir / ".meta_done.json"

    sig = {
        "seed": int(args.seed),
        "keys": keys_all,
        "test_frac": float(args.test_frac),
        "excluded_keys": sorted(list(EXCLUDED_KEYS)),
        "visits": list(VISITS),
        "required_traces_per_key": int(REQUIRED_TRACES_PER_KEY),
        "note": "no tuning; fixed hyperparams; vote uses non-overlapping test blocks",
    }

    # write plans once (task 0), then all array tasks reuse them
    if not idx2dev_path.exists():
        idx2dev_path.write_text(json.dumps(devices_kept, indent=2))
        sig_path.write_text(json.dumps(sig, indent=2))
        cal_plan_path.write_text(json.dumps(
            {k: {d: [str(p) for p in lst] for d, lst in cal_plan_raw[k].items()} for k in keys_all},
            indent=2
        ))
        test_plan_path.write_text(json.dumps(
            {k: {d: [str(p) for p in lst] for d, lst in test_plan_raw[k].items()} for k in keys_all},
            indent=2
        ))
        # write train_plan.json and val_plan.json, which the original job runner's wait-loop expected
        # (these are aliases — train_plan == cal_plan, val_plan is empty placeholder)
        train_plan_path.write_text(json.dumps(
            {"note": "alias of cal_plan.json"},
            indent=2
        ))
        val_plan_path.write_text(json.dumps(
            {"note": "placeholder; no separate val split in this pipeline"},
            indent=2
        ))
        # FIX: write .meta_done.json LAST so other tasks know all files are ready
        meta_done_path.write_text(json.dumps(
            {"status": "ok", "keys": keys_all, "num_devices": len(devices_kept)},
            indent=2
        ))
        print(f"[meta] wrote all meta files including .meta_done.json")
    else:
        prev = json.loads(idx2dev_path.read_text())
        if prev != devices_kept:
            raise SystemExit("[fatal] devices_kept mismatch vs existing meta/devices_kept.json")
        prev_sig = json.loads(sig_path.read_text()) if sig_path.exists() else None
        if prev_sig != sig:
            raise SystemExit("[fatal] plan_signature mismatch vs existing meta/plan_signature.json")

    idx2dev = json.loads(idx2dev_path.read_text())

    # per-key cal_plan for training
    cal_plan_key = {d: cal_plan_raw[key][d] for d in idx2dev if d in cal_plan_raw[key]}

    train_one_key(
        key=key,
        cal_plan_key=cal_plan_key,
        idx2dev=idx2dev,
        device=device,
        out_models=results_dir / "models",
        out_pixnorm=results_dir / "pixnorm",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        label_smoothing=args.label_smoothing,
        ema_decay=args.ema,
        warmup_epochs=args.warmup_epochs,
        dropout=args.dropout,
    )
    print(f"[done] trained key={key}")

def mode_vote(args):
    set_seed(args.seed)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    results_dir = Path(args.results_dir)
    models_dir = results_dir / "models"
    pixnorm_dir = results_dir / "pixnorm"
    meta_dir = results_dir / "meta"

    try:
        idx2dev = json.loads((meta_dir / "devices_kept.json").read_text())
        sig = json.loads((meta_dir / "plan_signature.json").read_text())
        cal_plan = json.loads((meta_dir / "cal_plan.json").read_text())
        test_plan = json.loads((meta_dir / "test_plan.json").read_text())
    except Exception as e:
        print(f"[fatal] missing or unreadable meta files in {meta_dir}: {e}", file=sys.stderr)
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
        print("[fatal] no keys selected to evaluate", file=sys.stderr)
        sys.exit(3)

    C = len(idx2dev)
    dev2idx = {d: i for i, d in enumerate(idx2dev)}

    models: Dict[str, nn.Module] = {}
    norms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for k in keys:
        mpath = models_dir / f"rawcnn_allvis_{k}.pt"
        jpath = pixnorm_dir / f"rawcnn_pixnorm_allvis_{k}.json"
        if not mpath.exists() or not jpath.exists():
            print(f"[fatal] missing model or pixnorm for key={k}", file=sys.stderr)
            sys.exit(4)
        try:
            state = torch.load(mpath, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(mpath, map_location=device)

        model = PixelCNN(num_classes=C, dropout=float(args.dropout)).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()

        jp = json.loads(jpath.read_text())
        mean = np.array(jp["pix_mean"], dtype=np.float32)
        std = np.array(jp["pix_std"], dtype=np.float32)
        models[k] = model
        norms[k] = (mean, std)

    # parse topk
    try:
        topk_list = sorted({int(t) for t in args.topk.split(",") if t.strip().isdigit()})
    except Exception:
        topk_list = [1, 5, 10, 20]
    if not topk_list:
        topk_list = [1, 5, 10, 20]

    block = max(1, int(args.n_test_per_key))  # block size
    per_trace_mode = str(args.per_key_trace_agg)
    fuse_mode = str(args.fuse_mode)

    # build non-overlapping test queries aligned across keys
    per_query: List[Tuple[str, int]] = []  # (device, block_id)
    for d in idx2dev:
        if any(d not in test_plan.get(k, {}) for k in keys):
            continue
        blocks_per_key = [len(make_nonoverlap_blocks(test_plan[k][d], block)) for k in keys]
        B = min(blocks_per_key) if blocks_per_key else 0
        for bi in range(B):
            per_query.append((d, bi))

    if not per_query:
        print("[fatal] no test queries after non-overlapping blocking", file=sys.stderr)
        sys.exit(5)

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

        # fuse across keys
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
                used = 0
                for lm in perkey_logits_mean.values():
                    logit_sum += lm
                    used += 1
                prob = F.softmax(torch.from_numpy(logit_sum.reshape(1, -1)), dim=1).numpy()[0]
                order = np.argsort(-prob)

            for kk in topk_list:
                if gt in order[:kk]:
                    ens_correct[kk] += 1

        # per-key accuracy
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

    (results_dir / "fp_vote_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results_dir / "fp_vote_per_key.json").write_text(json.dumps(perkey_acc, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

def build_parser():
    p = argparse.ArgumentParser(description="Train per-key CNNs and vote with no-leak non-overlapping test blocks.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train one key (job arrays with --key-index).")
    pt.add_argument("--base-dir", required=True)
    pt.add_argument("--results-dir", required=True)
    pt.add_argument("--keys", default="AUTO9", help="comma list, ALL, or AUTO or AUTO<N>")
    pt.add_argument("--key-index", type=int, required=True)

    pt.add_argument("--epochs", type=int, default=30)
    pt.add_argument("--batch-size", type=int, default=256)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--wd", type=float, default=1e-4)
    pt.add_argument("--label-smoothing", type=float, default=0.05)
    pt.add_argument("--ema", type=float, default=0.999)
    pt.add_argument("--warmup-epochs", type=int, default=1)
    pt.add_argument("--dropout", type=float, default=0.2)

    pt.add_argument("--test-frac", type=float, default=0.20)
    pt.add_argument("--gpu", action="store_true")
    pt.add_argument("--seed", type=int, default=42)

    pv = sub.add_parser("vote", help="Vote on test using non-overlapping blocks.")
    pv.add_argument("--base-dir", default="", help="accepted; not used by vote")
    pv.add_argument("--results-dir", required=True)
    pv.add_argument("--keys", default="AUTO9", help="ALL, AUTO<N>, or comma list subset of saved keys")
    pv.add_argument("--seed", type=int, default=42)
    pv.add_argument("--batch-size", type=int, default=256)
    pv.add_argument("--gpu", action="store_true")
    pv.add_argument("--topk", default="1,5,10,20")
    pv.add_argument("--dropout", type=float, default=0.2)

    # vote knobs
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