#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_per_browser.py — Per-browser accuracy splits for the active-defense gallery.

    Paper:    §6.4 / Table 4 — per-browser accuracy breakdown
"""

import argparse, json, os, re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- config --------------------
EXCLUDED_KEYS = {"raw_flags"}
REQUIRED_TRACES_PER_KEY = 100
VISITS = None  # auto-discover all visit* directories per device
H = W = 100




# -------------------- browser classification --------------------
_BROWSER_RULES = [
    ("Edge",             lambda u: "edg/" in u or "edge/" in u),
    ("Opera",            lambda u: "opr/" in u or (" opera" in u)),
    ("Samsung Internet", lambda u: "samsungbrowser" in u),
    ("Chrome iOS",       lambda u: "crios/" in u),
    ("Firefox iOS",      lambda u: "fxios/" in u),
    ("Yandex",           lambda u: "yabrowser" in u),
    ("Brave",            lambda u: "brave/" in u),
    ("Firefox",          lambda u: "firefox/" in u),
    ("Chrome",           lambda u: "chrome/" in u and "safari/" in u),
    ("Safari",           lambda u: "safari/" in u and "chrome/" not in u),
]

def classify_browser(ua):
    u = (ua or "").lower()
    for name, pred in _BROWSER_RULES:
        if pred(u): return name
    return "Other"

def load_ua_from_device(dev_dir):
    try:
        for fp in sorted(dev_dir.glob("fpjs_*.json")):
            try:
                d = json.loads(fp.read_text())
            except Exception:
                continue
            ua = d.get("userAgent")
            if ua: return ua
    except Exception:
        pass
    return None

# -------------------- json helpers --------------------
def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if torch.is_tensor(o): return o.detach().cpu().tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def _dumps(obj, **kwargs) -> str:
    if "default" not in kwargs: kwargs["default"] = _json_default
    return json.dumps(obj, **kwargs)


# -------------------- utils --------------------
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

_SESSION_PREFIX_RE = re.compile(r'^S\d+_')

def _strip_ext_and_meta(name: str) -> Optional[str]:
    low = name.lower()
    if "hash" in low or low.endswith(".meta.json"): return None
    return name[:-len(".json.gz")] if low.endswith(".json.gz") else Path(name).stem

def extract_key_from_filename(name: str) -> Optional[str]:
    base = _strip_ext_and_meta(name)
    if not base: return None
    s = _SESSION_PREFIX_RE.sub('', base)
    if not s: return None
    toks = s.split('_')
    if not toks or not toks[0]: return None
    if toks[0] == 'raw' and len(toks) >= 2 and toks[1]:
        return f'raw_{toks[1]}'
    return toks[0]

def list_device_folders(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.glob("dev_*") if p.is_dir()])

_PID_RE = re.compile(r'__pid-([a-f0-9]+)__')

def extract_pid(folder_name: str) -> Optional[str]:
    m = _PID_RE.search(folder_name)
    return m.group(1) if m else None

def build_pid_map(base_dir: Path) -> Dict[str, str]:
    out = {}
    for p in list_device_folders(base_dir):
        pid = extract_pid(p.name)
        if pid: out[pid] = p.name
    return out

def _get_visit_dirs(dev_dir: Path) -> List[Path]:
    """Auto-discover all visit* subdirectories for a device."""
    return sorted([d for d in dev_dir.iterdir() if d.is_dir() and d.name.startswith("visit")])

def list_key_rgba_sorted_all_visits(dev_dir: Path, key_string: str) -> List[Path]:
    items = []
    for vdir in _get_visit_dirs(dev_dir):
        for p in vdir.rglob("*.rgba"):
            k = extract_key_from_filename(p.name)
            if k != key_string: continue
            try: mt = p.stat().st_mtime
            except Exception: mt = 0.0
            items.append((p, mt, p.name))
    items.sort(key=lambda x: (x[1], x[2]))
    return [p for p, _, _ in items]

def collect_device_key_lists_all_vis(base_dir: Path, keys: List[str], min_traces: int = REQUIRED_TRACES_PER_KEY) -> Dict[str, Dict[str, List[Path]]]:
    out = {}
    for dev in list_device_folders(base_dir):
        if not _get_visit_dirs(dev): continue
        perkey = {}; ok = True
        for k in keys:
            lst = list_key_rgba_sorted_all_visits(dev, k)
            if len(lst) < min_traces:
                ok = False; break
            perkey[k] = lst[:REQUIRED_TRACES_PER_KEY]
        if ok:
            out[dev.name] = perkey
    return out

def build_shared_split_all_vis(device_key_lists, keys, seed, test_frac=0.20):
    rng = np.random.default_rng(seed)
    cal_plan, test_plan = {k: {} for k in keys}, {k: {} for k in keys}
    for dname, perkey in device_key_lists.items():
        N = min(len(perkey[k]) for k in keys)
        if N < 2: continue
        perm = rng.permutation(N)
        n_te = max(1, int(round(N * test_frac)))
        n_tr = max(1, N - n_te)
        if n_tr + n_te > N: n_tr = N - n_te
        tr_idx = sorted(perm[:n_tr].tolist())
        te_idx = sorted(perm[n_tr:].tolist())
        for k in keys:
            files = perkey[k][:N]
            cal_plan[k][dname] = [files[i] for i in tr_idx]
            test_plan[k][dname] = [files[i] for i in te_idx]
    return cal_plan, test_plan



def build_chronological_split_all_vis(device_key_lists, keys, max_query_pool=20):
    """Chronological 80/20 split.
    Assumes each device_key_lists[dev][key] is already sorted oldest-first
    (the case when list_key_rgba_sorted_all_visits is used, since it sorts by mtime).

    For each device with N samples:
        q = min(max_query_pool, max(1, ceil(0.2 * N)))
        enroll = first (N - q) samples    (past)
        query  = last q samples           (future)
    Returns (cal_plan, test_plan) in the same shape as build_shared_split_all_vis.
    """
    import math
    cal_plan = {k: {} for k in keys}
    test_plan = {k: {} for k in keys}
    for dname, perkey in device_key_lists.items():
        N = min(len(perkey[k]) for k in keys)
        if N < 2: continue
        q = min(int(max_query_pool), max(1, int(math.ceil(0.2 * N))))
        enroll_n = N - q
        if enroll_n < 1: continue
        for k in keys:
            files = perkey[k][:N]
            cal_plan[k][dname]  = files[:enroll_n]
            test_plan[k][dname] = files[enroll_n:]
    return cal_plan, test_plan
def resolve_keys(req_str, plan_keys):
    req = (req_str or "").strip()
    m = re.fullmatch(r'auto(?:0*(\d+))?$', req, flags=re.IGNORECASE)
    if req.upper() == "ALL": return plan_keys
    elif m:
        n = int(m.group(1)) if m.group(1) else None
        return plan_keys[: (n if n is not None else len(plan_keys))]
    else:
        want = [k.strip() for k in req.split(",") if k.strip()]
        return [k for k in plan_keys if k in set(want)]


# -------------------- model --------------------
class PixelCNN(nn.Module):
    def __init__(self, num_classes, base_channels=32, blocks=3, ksize=3, dropout=0.2,
                 last_block_pool=True):
        super().__init__()
        c0 = int(base_channels); b = max(2, min(int(blocks), 4))
        k = int(ksize); pad = k // 2
        layers = []; in_ch = 4
        for bi in range(b):
            out_ch = c0 * (2 ** bi)
            block = [
                nn.Conv2d(in_ch, out_ch, k, padding=pad), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, k, padding=pad), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            ]
            if last_block_pool or bi < b - 1:
                block.append(nn.MaxPool2d(2))
            layers += block
            in_ch = out_ch
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(in_ch, 256), nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)), nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.head(self.backbone(x))


# -------------------- data --------------------
def load_rgba_as_tensor(path: Path) -> Optional[np.ndarray]:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 4: return None
    n = (raw.size // 4) * 4
    if n == 0: return None
    arr = raw[:n].reshape(-1, 4).astype(np.int32, copy=False)
    a0 = (arr[:, 3] == 0)
    if a0.any(): arr[a0, 0:3] = 0
    if arr.shape[0] != H * W: return None
    img = arr.reshape(H, W, 4).transpose(2, 0, 1) / 255.0
    return img.astype(np.float32)


# -------------------- inference --------------------
def _torch_load_safe(path, map_location):
    try: return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError: return torch.load(path, map_location=map_location)

@torch.no_grad()
def logits_for_paths(model, paths, mean, std, device, bs=256):
    out = []; m = torch.from_numpy(mean.reshape(1,4,1,1)).to(device)
    s = torch.from_numpy(std.reshape(1,4,1,1)).to(device); batch = []
    for p in paths:
        img = load_rgba_as_tensor(p)
        if img is None: continue
        t = torch.from_numpy(img).unsqueeze(0).to(device)
        t = (t - m) / (s + 1e-6); batch.append(t)
        if len(batch) == bs:
            out.append(model(torch.cat(batch, dim=0)).detach().cpu().numpy()); batch = []
    if batch: out.append(model(torch.cat(batch, dim=0)).detach().cpu().numpy())
    if not out: return np.zeros((0, model.head[-1].out_features), dtype=np.float32)
    return np.concatenate(out, axis=0)

@torch.no_grad()
def feats_for_paths(model, paths, mean, std, device, bs=256):
    """256-D penultimate features: backbone -> flatten -> Linear(in_ch,256) -> ReLU."""
    out = []; m = torch.from_numpy(mean.reshape(1,4,1,1)).to(device)
    s = torch.from_numpy(std.reshape(1,4,1,1)).to(device); batch = []
    for p in paths:
        img = load_rgba_as_tensor(p)
        if img is None: continue
        t = torch.from_numpy(img).unsqueeze(0).to(device)
        t = (t - m) / (s + 1e-6); batch.append(t)
        if len(batch) == bs:
            xb = torch.cat(batch, dim=0)
            hb = model.backbone(xb); zb = torch.flatten(hb, 1)
            zb = model.head[2](model.head[1](zb))
            out.append(zb.detach().cpu().numpy()); batch = []
    if batch:
        xb = torch.cat(batch, dim=0)
        hb = model.backbone(xb); zb = torch.flatten(hb, 1)
        zb = model.head[2](model.head[1](zb))
        out.append(zb.detach().cpu().numpy())
    if not out: return np.zeros((0, 256), dtype=np.float32)
    return np.concatenate(out, axis=0)

def fuse_traces_to_prob(logits_mat, mode):
    if logits_mat.size == 0 or logits_mat.ndim != 2: return None
    if mode == "prob_mean":
        p = F.softmax(torch.from_numpy(logits_mat), dim=1).numpy()
        return p.mean(axis=0).astype(np.float32)
    lm = logits_mat.mean(axis=0, keepdims=True)
    return F.softmax(torch.from_numpy(lm), dim=1).numpy()[0].astype(np.float32)

def make_nonoverlap_blocks(paths, block):
    block = max(1, int(block)); L = len(paths); out = []
    for start in range(0, L - block + 1, block):
        out.append([Path(paths[i]) for i in range(start, start + block)])
    return out

def base_rate_from_cal_plan(cal_plan, keys, idx2dev, topk_list):
    counts = np.zeros(len(idx2dev), dtype=np.float64)
    for d_idx, d in enumerate(idx2dev):
        for k in keys: counts[d_idx] += len(cal_plan.get(k, {}).get(d, []))
    cs = np.sort(counts)[::-1]; tot = float(cs.sum())
    return {f"top{k}": float(cs[:min(k,len(cs))].sum() / tot) if tot > 0 else float("nan") for k in topk_list}


# -------------------- load models --------------------
def load_bayes_v1_models(keys, models_dir, pixnorm_dir, hparams_dir, num_classes, device):
    """Load Bayesian v1: fp_rawcnn_{key}.pt + hparams for arch."""
    models, norms = {}, {}
    for k in keys:
        mpath = models_dir / f"fp_rawcnn_{k}.pt"
        jpath = pixnorm_dir / f"fp_rawcnn_pixnorm_{k}.json"
        hpath = hparams_dir / f"fp_rawcnn_best_{k}.json"
        if not mpath.exists() or not jpath.exists() or not hpath.exists():
            print(f"[fatal] missing artifacts for key={k} (bayes)", file=sys.stderr); sys.exit(4)
        hp = json.loads(hpath.read_text()); bp = hp["best_params"]
        model = PixelCNN(num_classes=num_classes, base_channels=int(bp["base_channels"]),
                         blocks=int(bp["blocks"]), ksize=int(bp["ksize"]),
                         dropout=float(bp["dropout"])).to(device)
        model.load_state_dict(_torch_load_safe(mpath, map_location=device), strict=True)
        model.eval()
        jp = json.loads(jpath.read_text())
        norms[k] = (np.array(jp["pix_mean"], dtype=np.float32), np.array(jp["pix_std"], dtype=np.float32))
        models[k] = model
    return models, norms


def load_fixed_models(keys, models_dir, pixnorm_dir, num_classes, device, dropout=0.2):
    """Load fixed-hyperparam: rawcnn_allvis_{key}.pt, arch c=32/3/3."""
    models, norms = {}, {}
    for k in keys:
        mpath = models_dir / f"rawcnn_allvis_{k}.pt"
        jpath = pixnorm_dir / f"rawcnn_pixnorm_allvis_{k}.json"
        if not mpath.exists() or not jpath.exists():
            print(f"[fatal] missing artifacts for key={k} (fixed)", file=sys.stderr); sys.exit(4)
        # train_and_vote_noleak_nobay.py has no MaxPool at the end of the last block;
        # pass last_block_pool=False so inference matches training arch.
        model = PixelCNN(num_classes=num_classes, base_channels=32, blocks=3, ksize=3,
                         dropout=float(dropout), last_block_pool=False).to(device)
        model.load_state_dict(_torch_load_safe(mpath, map_location=device), strict=True)
        model.eval()
        jp = json.loads(jpath.read_text())
        norms[k] = (np.array(jp["pix_mean"], dtype=np.float32), np.array(jp["pix_std"], dtype=np.float32))
        models[k] = model
    return models, norms


def load_models_auto(args, keys, results_dir, num_classes, device):
    """Unified loader: dispatches based on --model-type (default: bayes)."""
    models_dir = results_dir / "models"
    pixnorm_dir = results_dir / "pixnorm"
    hparams_dir = results_dir / "hparams"
    mt = getattr(args, "model_type", "bayes")
    if mt == "fixed":
        dropout = getattr(args, "dropout", 0.2)
        return load_fixed_models(keys, models_dir, pixnorm_dir, num_classes, device, dropout=dropout)
    else:
        return load_bayes_v1_models(keys, models_dir, pixnorm_dir, hparams_dir, num_classes, device)


# ================== MODE: vote (seen, non-overlapping blocks) ==================

# -------------------- subsampled per-browser evaluation --------------------
def _subsample_eval_common(members, protos_per_key, pos_map, feats_lookup,
                            dev_browser_map, target_N, n_resamples,
                            topk_list, ks_list, keys, rng):
    import numpy as np
    br2members = {}
    for m in members:
        br = dev_browser_map.get(m[0], "Unknown")
        br2members.setdefault(br, []).append(m)
    out = {}
    for ks in ks_list:
        out[ks] = {}
        for br, mlist in br2members.items():
            if len(mlist) < target_N: continue
            per_resample = {kk: [] for kk in topk_list}
            used_counts = []
            for s in range(n_resamples):
                idxs = rng.choice(len(mlist), size=target_N, replace=False)
                sampled = [mlist[i] for i in idxs]
                sampled_gts = [m[0] for m in sampled]
                local_idx = {g: i for i, g in enumerate(sampled_gts)}
                gallery_pos = [pos_map[g] for g in sampled_gts]
                mini_protos = {k: protos_per_key[k][gallery_pos] for k in keys}
                correct = {kk: 0 for kk in topk_list}; total = 0
                for m in sampled:
                    gt = local_idx[m[0]]
                    feats = feats_lookup(m)
                    perkey_scores = {}
                    for k in keys:
                        F_all = feats.get(k)
                        if F_all is None or F_all.shape[0] < ks: continue
                        idx = rng.choice(F_all.shape[0], size=ks, replace=False)
                        perkey_scores[k] = (F_all[idx] @ mini_protos[k].T).mean(axis=0)
                    if not perkey_scores: continue
                    total += 1
                    S_avg = np.mean(np.stack(list(perkey_scores.values()), axis=0), axis=0)
                    order = np.argsort(-S_avg)
                    for kk in topk_list:
                        if gt in order[:kk]: correct[kk] += 1
                for kk in topk_list:
                    per_resample[kk].append(correct[kk] / max(1, total))
                used_counts.append(total)
            out[ks][br] = {
                "n_browser_total": len(mlist),
                "target_N": target_N,
                "n_resamples": n_resamples,
                "n_used_mean": float(np.mean(used_counts)) if used_counts else 0,
                **{f"top{kk}_mean": float(np.mean(per_resample[kk])) for kk in topk_list},
                **{f"top{kk}_std":  float(np.std(per_resample[kk]))  for kk in topk_list},
            }
    return out

def mode_vote(args):
    set_seed(args.seed)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir); meta_dir = results_dir / "meta"
    out_dir = Path(getattr(args, "output_dir", None) or results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir_for_ua = Path(getattr(args, "base_dir", "") or "")

    try:
        idx2dev = json.loads((meta_dir / "devices_kept.json").read_text())
        sig = json.loads((meta_dir / "plan_signature.json").read_text())
        cal_plan = json.loads((meta_dir / "cal_plan.json").read_text())
        test_plan = json.loads((meta_dir / "test_plan.json").read_text())
    except Exception as e:
        print(f"[fatal] missing meta: {e}", file=sys.stderr); sys.exit(2)

    keys = resolve_keys(args.keys, sig.get("keys", []))
    if not keys: print("[fatal] no keys", file=sys.stderr); sys.exit(3)

    C = len(idx2dev); dev2idx = {d: i for i, d in enumerate(idx2dev)}
    models, norms = load_models_auto(args, keys, results_dir, C, device)

    dev_browser = {}
    if base_dir_for_ua and base_dir_for_ua.is_dir():
        for d in idx2dev:
            ua = load_ua_from_device(base_dir_for_ua / d)
            dev_browser[d] = classify_browser(ua or "")
    else:
        for d in idx2dev: dev_browser[d] = "Unknown"
    from collections import Counter as _Ct
    _bc = _Ct(dev_browser.values())
    print("[per-browser] vote closed-set distribution: " + ", ".join(f"{b}={n}" for b,n in _bc.most_common()))
    block = max(1, int(args.n_test_per_key))
    per_query = []
    for d in idx2dev:
        if any(d not in test_plan.get(k, {}) for k in keys): continue
        B = min(len(make_nonoverlap_blocks(test_plan[k][d], block)) for k in keys)
        for bi in range(B): per_query.append((d, bi))
    if not per_query: print("[fatal] no test queries", file=sys.stderr); sys.exit(5)

    topk_list = sorted({int(t) for t in args.topk.split(",") if t.strip().isdigit()}) or [1,5,10,20]
    per_trace_mode = str(args.per_key_trace_agg); fuse_mode = str(args.fuse_mode)
    ens_correct = {k: 0 for k in topk_list}
    perkey_correct = {key: {k: 0 for k in topk_list} for key in keys}
    perkey_total = {key: 0 for key in keys}
    ens_correct_by_br = {}
    total_by_br = {}

    for (dev_name, bi) in per_query:
        gt = dev2idx.get(dev_name)
        if gt is None: continue
        perkey_probs, perkey_logits_mean = {}, {}
        for k in keys:
            paths_k = make_nonoverlap_blocks(test_plan[k][dev_name], block)[bi]
            lm = logits_for_paths(models[k], paths_k, norms[k][0], norms[k][1], device=device, bs=int(args.batch_size))
            if lm.shape[0] == 0: continue
            perkey_logits_mean[k] = lm.mean(axis=0).astype(np.float32)
            p_k = fuse_traces_to_prob(lm, mode=per_trace_mode)
            if p_k is not None: perkey_probs[k] = p_k
        if perkey_probs:
            if fuse_mode == "prob_mean":
                order = np.argsort(-np.mean(list(perkey_probs.values()), axis=0))
            else:
                ls = np.zeros((C,), dtype=np.float32)
                for lm in perkey_logits_mean.values(): ls += lm
                order = np.argsort(-F.softmax(torch.from_numpy(ls.reshape(1,-1)), dim=1).numpy()[0])
            br = dev_browser.get(dev_name, "Unknown")
            total_by_br[br] = total_by_br.get(br, 0) + 1
            if br not in ens_correct_by_br:
                ens_correct_by_br[br] = {kk: 0 for kk in topk_list}
            for kk in topk_list:
                if gt in order[:kk]:
                    ens_correct[kk] += 1
                    ens_correct_by_br[br][kk] += 1
        for k, p in perkey_probs.items():
            perkey_total[k] += 1; order_k = np.argsort(-p)
            for kk in topk_list:
                if gt in order_k[:kk]: perkey_correct[k][kk] += 1

    total = len(per_query)
    per_browser_acc = {br: {"n": total_by_br[br],
                            **{f"top{kk}": ens_correct_by_br[br][kk] / max(1, total_by_br[br]) for kk in topk_list}}
                        for br in total_by_br}
    summary = {
        "mode": "vote_seen", "keys": keys, "num_classes": C, "num_queries_test": total,
        "test_query_block_size": block, "per_key_trace_agg": per_trace_mode, "fuse_mode": fuse_mode,
        "base_rates_from_cal": base_rate_from_cal_plan(cal_plan, keys, idx2dev, topk_list),
        "ensemble_topk_accuracy": {f"top{k}": ens_correct[k]/max(1,total) for k in topk_list},
        "per_key_topk": {key: {f"top{k}": perkey_correct[key][k]/max(1,perkey_total[key]) for k in topk_list} for key in keys},
        "per_key_eval_counts": perkey_total,
        "per_browser": per_browser_acc,
        "note": "seen devices; non-overlapping test blocks; no test trace reused",
    }
    (out_dir / "fp_vote_per_browser_summary.json").write_text(_dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "fp_vote_per_key_per_browser.json").write_text(
        _dumps({key: {f"top{k}": perkey_correct[key][k]/max(1,perkey_total[key]) for k in topk_list} for key in keys}, indent=2), encoding="utf-8")
    print(_dumps(summary, indent=2))


# ================== MODE: vote_temporal (k-shot, PID matching, per-k filtering) ==================
def mode_vote_temporal(args):
    set_seed(args.seed)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    t0_dir = Path(args.t0_dir)
    t130_dir = Path(args.t130_dir)
    results_dir = Path(args.results_dir)
    meta_dir = results_dir / "meta"
    # [per-browser] separate output dir (falls back to results_dir)
    out_dir = Path(getattr(args, "output_dir", None) or results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        sig = json.loads((meta_dir / "plan_signature.json").read_text())
        idx2dev_train = json.loads((meta_dir / "devices_kept.json").read_text())
        cal_plan_t0 = json.loads((meta_dir / "cal_plan.json").read_text())
    except Exception as e:
        print(f"[fatal] cannot read training meta: {e}", file=sys.stderr); sys.exit(2)

    C_train = len(idx2dev_train)
    keys = resolve_keys(args.keys, sig.get("keys", []))
    if not keys: print("[fatal] no keys", file=sys.stderr); sys.exit(3)

    kshots = sorted({int(s) for s in args.kshots.split(",") if s.strip().isdigit()})
    if not kshots: kshots = [1, 2, 3, 4]
    min_k = min(kshots)

    models, norms = load_models_auto(args, keys, results_dir, C_train, device)

    # Per-k whitelist
    per_k_wl: Optional[Dict[int, set]] = None
    if args.per_k_whitelist:
        wl_path = Path(args.per_k_whitelist)
        if not wl_path.exists():
            print(f"[fatal] --per-k-whitelist not found: {wl_path}", file=sys.stderr); sys.exit(2)
        raw = json.loads(wl_path.read_text())
        per_k_wl = {int(k): set(v) for k, v in raw.items()}
        print(f"[per-k-wl] loaded: " + ", ".join(f"k={k}:{len(v)}" for k, v in sorted(per_k_wl.items())))

    # PID maps
    t0_devices_all = set()
    for k in keys:
        ds = set(cal_plan_t0.get(k, {}).keys())
        t0_devices_all = ds if not t0_devices_all else (t0_devices_all & ds)
    t0_pid2folder = {}
    for fname in t0_devices_all:
        pid = extract_pid(fname)
        if pid: t0_pid2folder[pid] = fname
    print(f"[temporal] T0 devices with PID: {len(t0_pid2folder)}")

    t130_pid2folder = {}
    for p in list_device_folders(t130_dir):
        pid = extract_pid(p.name)
        if pid: t130_pid2folder[pid] = p.name
    print(f"[temporal] T+130d devices with PID: {len(t130_pid2folder)}")

    overlap_pids = sorted(set(t0_pid2folder.keys()) & set(t130_pid2folder.keys()))
    if not overlap_pids:
        print("[fatal] no overlap PIDs", file=sys.stderr); sys.exit(4)
    print(f"[temporal] overlap PIDs: {len(overlap_pids)}")

    dkl_t130 = collect_device_key_lists_all_vis(t130_dir, keys, min_traces=min_k)

    overlap_pairs = []
    for pid in overlap_pids:
        t0_f = t0_pid2folder[pid]
        t130_f = t130_pid2folder[pid]
        if t130_f not in dkl_t130: continue
        overlap_pairs.append((pid, t0_f, t130_f))

    if not overlap_pairs:
        print("[fatal] no overlap devices with sufficient traces", file=sys.stderr); sys.exit(4)

    dev_min_traces = {}
    for (pid, t0_f, t130_f) in overlap_pairs:
        dev_min_traces[pid] = min(len(dkl_t130[t130_f][k]) for k in keys)

    # [per-browser] build pid -> browser map using T130 UA (query-time), fallback T0
    dev_browser = {}
    for (pid, t0_f, t130_f) in overlap_pairs:
        ua = load_ua_from_device(t130_dir / t130_f) or load_ua_from_device(t0_dir / t0_f)
        dev_browser[pid] = classify_browser(ua or "")
    from collections import Counter as _Ct
    _bc = _Ct(dev_browser.values())
    print(f"[per-browser] distribution across {len(dev_browser)} devices: " + ", ".join(f"{b}={n}" for b,n in _bc.most_common()))
    print(f"[temporal] {len(overlap_pairs)} overlap devices (>= {min_k} traces/key), kshots={kshots}")

    (out_dir / "temporal_devices_kept.json").write_text(
        json.dumps([t0_f for (pid, t0_f, t130_f) in overlap_pairs], indent=2), encoding="utf-8")

    fp_eligible_by_k = {}
    for ks in kshots:
        fp_eligible_by_k[str(ks)] = [t0_f for (pid, t0_f, t130_f) in overlap_pairs
                                      if dev_min_traces[pid] >= ks]
    (out_dir / "fp_temporal_eligible_by_k.json").write_text(
        json.dumps(fp_eligible_by_k, indent=2), encoding="utf-8")
    for ks in kshots:
        print(f"  k={ks}: {len(fp_eligible_by_k[str(ks)])} FP-CNN eligible devices")

    # Enrollment prototypes
    protos_all = {}
    for k in keys:
        rows = []
        for (pid, t0_f, t130_f) in overlap_pairs:
            paths = [Path(p) for p in cal_plan_t0[k].get(t0_f, [])]
            Fd = feats_for_paths(models[k], paths, norms[k][0], norms[k][1], device=device, bs=args.batch_size)
            rows.append(Fd.mean(axis=0) if Fd.size else np.zeros((256,), dtype=np.float32))
        P = np.stack(rows, axis=0)
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        protos_all[k] = P

    t130_feats = {}
    for (pid, t0_f, t130_f) in overlap_pairs:
        for k in keys:
            paths = [Path(p) for p in dkl_t130[t130_f][k]]
            F = feats_for_paths(models[k], paths, norms[k][0], norms[k][1], device=device, bs=args.batch_size)
            if F.size:
                F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
            t130_feats[(t130_f, k)] = F

    topk_list = sorted({int(t) for t in args.topk.split(",") if t.strip().isdigit()}) or [1, 5, 10, 20]
    rng = np.random.default_rng(args.seed)

    results_by_kshot = {}
    for ks in kshots:
        eligible = [(pid, t0_f, t130_f) for (pid, t0_f, t130_f) in overlap_pairs
                    if dev_min_traces[pid] >= ks]

        if per_k_wl is not None and ks in per_k_wl:
            dp_set = per_k_wl[ks]
            before = len(eligible)
            eligible = [(pid, t0_f, t130_f) for (pid, t0_f, t130_f) in eligible if t0_f in dp_set]
            print(f"  k={ks}: trace-eligible={before} -> after DP whitelist={len(eligible)}")

        if not eligible:
            print(f"  k={ks}: 0 eligible, skipping")
            results_by_kshot[ks] = {"k": ks, "num_devices_eval": 0, "num_devices_eligible": 0,
                                    "ensemble_topk": {}, "per_key_topk": {}, "base_rates_random": {}}
            continue

        D_ks = len(eligible)
        pid2idx_ks = {pid: i for i, (pid, _, _) in enumerate(eligible)}

        protos_ks = {}
        for k in keys:
            idx_map = [overlap_pairs.index(e) for e in eligible]
            protos_ks[k] = protos_all[k][idx_map]

        ens_correct = {kk: 0 for kk in topk_list}
        perkey_correct = {k: {kk: 0 for kk in topk_list} for k in keys}
        total = 0
        # [per-browser] accumulators per browser
        ens_correct_by_br = {}
        total_by_br = {}

        for (pid, t0_f, t130_f) in eligible:
            gt = pid2idx_ks[pid]; perkey_scores = {}
            for k in keys:
                F_all = t130_feats.get((t130_f, k))
                if F_all is None or F_all.shape[0] < ks: continue
                idx = np.arange(ks)  # chronological: earliest k of time-sorted features
                perkey_scores[k] = (F_all[idx] @ protos_ks[k].T).mean(axis=0)
            if not perkey_scores: continue
            total += 1
            br = dev_browser.get(pid, "Unknown")
            total_by_br[br] = total_by_br.get(br, 0) + 1
            if br not in ens_correct_by_br:
                ens_correct_by_br[br] = {kk: 0 for kk in topk_list}
            for k, Sv in perkey_scores.items():
                order_k = np.argsort(-Sv)
                for kk in topk_list:
                    if gt in order_k[:kk]: perkey_correct[k][kk] += 1
            S_avg = np.mean(np.stack(list(perkey_scores.values()), axis=0), axis=0)
            order = np.argsort(-S_avg)
            for kk in topk_list:
                if gt in order[:kk]:
                    ens_correct[kk] += 1
                    ens_correct_by_br[br][kk] += 1

        ens_acc = {f"top{kk}": ens_correct[kk]/max(1,total) for kk in topk_list}
        perkey_acc = {k: {f"top{kk}": perkey_correct[k][kk]/max(1,total) for kk in topk_list} for k in keys}
        base_ks = {f"top{kk}": float(min(kk, D_ks))/float(D_ks) for kk in topk_list}
        per_browser_acc = {
            br: {
                "n": total_by_br[br],
                **{f"top{kk}": ens_correct_by_br[br][kk] / max(1, total_by_br[br]) for kk in topk_list},
            }
            for br in total_by_br
        }
        results_by_kshot[ks] = {
            "k": ks, "num_devices_eval": total, "num_devices_eligible": D_ks,
            "base_rates_random": base_ks, "ensemble_topk": ens_acc, "per_key_topk": perkey_acc,
            "per_browser": per_browser_acc,
        }
        print(f"  k={ks}: {D_ks} eligible, {total} eval'd, ensemble {ens_acc}")

    subsample_N = int(getattr(args, "subsample_gallery", 0) or 0)
    n_resamples = int(getattr(args, "n_resamples", 50))
    subsample_results = {}
    if subsample_N > 0:
        pos_map = {pid: i for i,(pid,_,_) in enumerate(overlap_pairs)}
        members_t = [(pid, t130_f) for (pid, t0_f, t130_f) in overlap_pairs]
        def feats_lookup_t(m):
            pid, t130_f = m
            return {k: t130_feats.get((t130_f, k)) for k in keys}
        subsample_results = _subsample_eval_common(
            members=members_t, protos_per_key=protos_all, pos_map=pos_map,
            feats_lookup=feats_lookup_t, dev_browser_map=dev_browser,
            target_N=subsample_N, n_resamples=n_resamples,
            topk_list=topk_list, ks_list=kshots, keys=keys, rng=rng)
        print(f"[subsample] temporal at N={subsample_N}, resamples={n_resamples}")
        for ks in subsample_results:
            for br, v in sorted(subsample_results[ks].items(), key=lambda x: -x[1].get("top1_mean",0)):
                t1m = v.get("top1_mean", 0) * 100
                t1s = v.get("top1_std", 0) * 100
                ncoh = v.get("n_browser_total")
                print(f"  ks={ks} {br:<18}  top1={t1m:.2f}% +/- {t1s:.2f}%  (cohort={ncoh})")
    summary = {
        "mode": "vote_temporal_kshot",
        "t0_dir": str(t0_dir), "t130_dir": str(t130_dir),
        "keys": keys, "num_devices_overlap_total": len(overlap_pairs),
        "per_k_whitelist": str(args.per_k_whitelist) if args.per_k_whitelist else None,
        "kshots": kshots, "results_by_kshot": results_by_kshot,
        "subsample_results": subsample_results, "subsample_N": subsample_N,
        "note": "enrollment=ALL T0 cal; query=k from T+130d; per-k filtering; PID matching; cosine prototype; per-browser breakdown included",
    }
    (out_dir / "vote_temporal_per_browser_summary.json").write_text(_dumps(summary, indent=2), encoding="utf-8")
    print(_dumps(summary, indent=2))


# ================== MODE: vote_unseen (k-shot, new devices) ==================
def mode_vote_unseen(args):
    set_seed(args.seed)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    base_dir = Path(args.base_dir)
    results_dir = Path(args.results_dir)
    meta_dir = results_dir / "meta"
    out_dir = Path(getattr(args, "output_dir", None) or results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        sig = json.loads((meta_dir / "plan_signature.json").read_text())
        idx2dev_train = json.loads((meta_dir / "devices_kept.json").read_text())
    except Exception as e:
        print(f"[fatal] cannot read meta: {e}", file=sys.stderr); sys.exit(2)

    C_train = len(idx2dev_train)
    keys = resolve_keys(args.keys, sig.get("keys", []))
    if not keys: print("[fatal] no keys", file=sys.stderr); sys.exit(3)

    kshots = sorted({int(s) for s in args.kshots.split(",") if s.strip().isdigit()})
    if not kshots: kshots = [1, 5, 10, 20]
    min_k = min(kshots)

    min_traces = max(args.min_traces, min_k + 1)
    print(f"[unseen] min_traces_per_key={min_traces}, kshots={kshots}")

    models, norms = load_models_auto(args, keys, results_dir, C_train, device)

    dkl = collect_device_key_lists_all_vis(base_dir, keys, min_traces=min_traces)
    if not dkl: print("[fatal] no devices with sufficient traces", file=sys.stderr); sys.exit(4)
    print(f"[unseen] devices collected (>= {min_traces} traces/key): {len(dkl)}")

    if getattr(args, "chronological_split", False):
        cal_plan, test_plan = build_chronological_split_all_vis(dkl, keys, max_query_pool=20)
        print("[chronological] unseen split: first (N - min(20, 0.2N)) enroll, last q query")
    else:
        cal_plan, test_plan = build_shared_split_all_vis(dkl, keys, seed=args.seed, test_frac=0.20)
    all_devices = sorted(set.intersection(*[set(cal_plan[k].keys()) for k in keys]))
    if not all_devices: print("[fatal] no devices after intersection", file=sys.stderr); sys.exit(5)
    print(f"[unseen] devices after intersection: {len(all_devices)}")

    dev_min_test = {}
    for d in all_devices:
        dev_min_test[d] = min(len(test_plan[k].get(d, [])) for k in keys)
    dev_browser_u = {}
    for d in all_devices:
        ua = load_ua_from_device(base_dir / d)
        dev_browser_u[d] = classify_browser(ua or "")
    from collections import Counter as _Ct
    _bc = _Ct(dev_browser_u.values())
    print("[per-browser] vote_unseen distribution: " + ", ".join(f"{b}={n}" for b,n in _bc.most_common()))

    protos_all = {}
    for k in keys:
        rows = []
        for d in all_devices:
            paths = cal_plan[k].get(d, [])
            Fd = feats_for_paths(models[k], [Path(p) for p in paths], norms[k][0], norms[k][1], device=device, bs=args.batch_size)
            rows.append(Fd.mean(axis=0) if Fd.size else np.zeros((256,), dtype=np.float32))
        P = np.stack(rows, axis=0)
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        protos_all[k] = P

    test_feats = {}
    for d in all_devices:
        for k in keys:
            paths = [Path(p) for p in test_plan[k].get(d, [])]
            F = feats_for_paths(models[k], paths, norms[k][0], norms[k][1], device=device, bs=args.batch_size)
            if F.size:
                F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
            test_feats[(d, k)] = F

    topk_list = sorted({int(t) for t in args.topk.split(",") if t.strip().isdigit()}) or [1, 5, 10, 20]
    rng = np.random.default_rng(args.seed)

    results_by_kshot = {}
    for ks in kshots:
        eligible = [d for d in all_devices if dev_min_test[d] >= ks]
        if not eligible:
            print(f"  k={ks}: 0 eligible, skipping")
            results_by_kshot[ks] = {"k": ks, "num_devices_eval": 0, "ensemble_topk": {}, "per_key_topk": {}}
            continue

        D_ks = len(eligible)
        dev2idx_ks = {d: i for i, d in enumerate(eligible)}

        protos_ks = {}
        for k in keys:
            idx_map = [all_devices.index(d) for d in eligible]
            protos_ks[k] = protos_all[k][idx_map]

        ens_correct = {kk: 0 for kk in topk_list}
        perkey_correct = {k: {kk: 0 for kk in topk_list} for k in keys}
        total = 0
        ens_correct_by_br = {}
        total_by_br = {}

        for d in eligible:
            gt = dev2idx_ks[d]; perkey_scores = {}
            for k in keys:
                F_all = test_feats.get((d, k))
                if F_all is None or F_all.shape[0] < ks: continue
                if getattr(args, "chronological_split", False):
                    idx = np.arange(ks)  # chronological: earliest k
                else:
                    idx = rng.choice(F_all.shape[0], size=ks, replace=False)
                perkey_scores[k] = (F_all[idx] @ protos_ks[k].T).mean(axis=0)
            if not perkey_scores: continue
            total += 1
            for k, Sv in perkey_scores.items():
                order_k = np.argsort(-Sv)
                for kk in topk_list:
                    if gt in order_k[:kk]: perkey_correct[k][kk] += 1
            S_avg = np.mean(np.stack(list(perkey_scores.values()), axis=0), axis=0)
            order = np.argsort(-S_avg)
            br = dev_browser_u.get(d, "Unknown")
            total_by_br[br] = total_by_br.get(br, 0) + 1
            if br not in ens_correct_by_br:
                ens_correct_by_br[br] = {kk: 0 for kk in topk_list}
            for kk in topk_list:
                if gt in order[:kk]:
                    ens_correct[kk] += 1
                    ens_correct_by_br[br][kk] += 1

        ens_acc = {f"top{kk}": ens_correct[kk]/max(1,total) for kk in topk_list}
        perkey_acc = {k: {f"top{kk}": perkey_correct[k][kk]/max(1,total) for kk in topk_list} for k in keys}
        base_ks = {f"top{kk}": float(min(kk, D_ks))/float(D_ks) for kk in topk_list}
        per_browser_acc = {br: {"n": total_by_br[br],
                                **{f"top{kk}": ens_correct_by_br[br][kk] / max(1, total_by_br[br]) for kk in topk_list}}
                            for br in total_by_br}
        results_by_kshot[ks] = {
            "k": ks, "num_devices_eval": total, "num_devices_eligible": D_ks,
            "base_rates_random": base_ks, "ensemble_topk": ens_acc, "per_key_topk": perkey_acc,
            "per_browser": per_browser_acc,
        }
        print(f"  k={ks}: {D_ks} eligible, {total} eval'd, ensemble {ens_acc}")

    subsample_N = int(getattr(args, "subsample_gallery", 0) or 0)
    n_resamples = int(getattr(args, "n_resamples", 50))
    subsample_results_u = {}
    if subsample_N > 0:
        pos_map_u = {d: i for i,d in enumerate(all_devices)}
        members_u = [(d,) for d in all_devices]
        def feats_lookup_u(m):
            d = m[0]
            return {k: test_feats.get((d, k)) for k in keys}
        subsample_results_u = _subsample_eval_common(
            members=members_u, protos_per_key=protos_all, pos_map=pos_map_u,
            feats_lookup=feats_lookup_u, dev_browser_map=dev_browser_u,
            target_N=subsample_N, n_resamples=n_resamples,
            topk_list=topk_list, ks_list=kshots, keys=keys, rng=rng)
        print(f"[subsample] unseen at N={subsample_N}, resamples={n_resamples}")
        for ks in subsample_results_u:
            for br, v in sorted(subsample_results_u[ks].items(), key=lambda x: -x[1].get("top1_mean",0)):
                t1m = v.get("top1_mean", 0) * 100
                t1s = v.get("top1_std", 0) * 100
                ncoh = v.get("n_browser_total")
                print(f"  ks={ks} {br:<18}  top1={t1m:.2f}% +/- {t1s:.2f}%  (cohort={ncoh})")
    summary = {
        "mode": "vote_unseen_kshot",
        "unseen_base_dir": str(base_dir), "keys": keys,
        "num_devices_total": len(all_devices), "min_traces_per_key": min_traces,
        "kshots": kshots, "results_by_kshot": results_by_kshot,
        "subsample_results": subsample_results_u, "subsample_N": subsample_N,
        "note": "unseen devices; per-k filtering; enrollment=all CAL; query=k from TEST; frozen encoders; cosine prototype",
    }
    (out_dir / "vote_unseen_per_browser_summary.json").write_text(_dumps(summary, indent=2), encoding="utf-8")
    print(_dumps(summary, indent=2))


# -------------------- CLI --------------------
def _add_model_type_args(parser):
    parser.add_argument("--model-type", choices=["bayes", "fixed"], default="bayes",
                        help="bayes: fp_rawcnn_* + hparams; fixed: rawcnn_allvis_*, arch c=32/3/3")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout for fixed model type (ignored for bayes)")

def build_parser():
    p = argparse.ArgumentParser(description="Evaluate FP models (bayes or fixed): seen / temporal / unseen.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("vote", help="Seen identification on T0 test split.")
    pv.add_argument("--base-dir", required=True, help="accepted; not used by vote")
    pv.add_argument("--results-dir", required=True)
    pv.add_argument("--keys", default="AUTO9")
    pv.add_argument("--seed", type=int, default=42)
    pv.add_argument("--gpu", action="store_true")
    pv.add_argument("--batch-size", type=int, default=256)
    pv.add_argument("--topk", default="1,5,10,20")
    pv.add_argument("--output-dir", default=None,
                    help="Directory for per-browser outputs. Defaults to --results-dir.")
    pv.add_argument("--n-test-per-key", type=int, default=5)
    pv.add_argument("--per-key-trace-agg", choices=["logit_mean", "prob_mean"], default="logit_mean")
    pv.add_argument("--fuse-mode", choices=["logit_sum", "prob_mean"], default="logit_sum")
    _add_model_type_args(pv)

    pt = sub.add_parser("vote_temporal", help="Temporal k-shot: T0 enrollment → T+130d k-shot query.")
    pt.add_argument("--chronological-split", action="store_true",
                    help="accepted; not used by vote_temporal")
    pt.add_argument("--t0-dir", required=True, help="T0 dataset root")
    pt.add_argument("--t130-dir", required=True, help="T+130d dataset root")
    pt.add_argument("--results-dir", required=True, help="Trained model folder")
    pt.add_argument("--keys", default="AUTO9")
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--gpu", action="store_true")
    pt.add_argument("--batch-size", type=int, default=256)
    pt.add_argument("--topk", default="1,5,10,20")
    pt.add_argument("--kshots", default="1,2,3,4", help="comma-separated k values")
    pt.add_argument("--subsample-gallery", type=int, default=0)
    pt.add_argument("--n-resamples", type=int, default=50)
    pt.add_argument("--output-dir", default=None,
                    help="Directory for per-browser outputs. Defaults to --results-dir.")
    pt.add_argument("--per-k-whitelist", default=None,
                    help="Path to dp_temporal_eligible_by_k.json for fair DP comparison")
    _add_model_type_args(pt)

    pu = sub.add_parser("vote_unseen", help="Unseen k-shot within T+130d new devices.")
    pu.add_argument("--base-dir", required=True, help="T+130d new device root")
    pu.add_argument("--results-dir", required=True, help="Trained model folder")
    pu.add_argument("--keys", default="AUTO9")
    pu.add_argument("--seed", type=int, default=42)
    pu.add_argument("--gpu", action="store_true")
    pu.add_argument("--batch-size", type=int, default=256)
    pu.add_argument("--chronological-split", action="store_true",
                    help="Use chronological 80/20 (past enroll, future query) instead of random seed split")
    pu.add_argument("--topk", default="1,5,10,20")
    pu.add_argument("--kshots", default="1,5,10,20", help="comma-separated k values")
    pu.add_argument("--subsample-gallery", type=int, default=0)
    pu.add_argument("--n-resamples", type=int, default=50)
    pu.add_argument("--output-dir", default=None,
                    help="Directory for per-browser outputs. Defaults to --results-dir.")
    pu.add_argument("--min-traces", type=int, default=10, help="minimum traces per key per device")
    _add_model_type_args(pu)

    return p

def main():
    args = build_parser().parse_args()
    {"vote": mode_vote, "vote_temporal": mode_vote_temporal, "vote_unseen": mode_vote_unseen}[args.cmd](args)

if __name__ == "__main__":
    main()