#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_defense.py — Defense-identification post-processing — characterizes each defended
    browser group along the paper Table 4 axes.

    Paper:    §6 / Table 4 — anti-fingerprinting defense identification
"""

import argparse, json, os, re, sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── constants ─────────────────────────────────────────────────────────────────
REQUIRED_TRACES_PER_KEY = 100
H = W = 100
EXCLUDED_KEYS = {"raw_flags"}
_SESSION_PREFIX_RE = re.compile(r'^S\d+_')

# ── defense group classification ─────────────────────────────────────────────
_SID_TO_DEFENSE = [
    ("ChromeCanvasBlocker",               "Canvas Blocker (Chrome)"),
    ("ChromeCanvasFingerprintDefender",    "Canvas FP Defender (Chrome)"),
    ("ChromeCanvasFingeprintDefender",     "Canvas FP Defender (Chrome)"),
    ("CanvasFingerprintDefender",          "Canvas FP Defender (Chrome)"),
    ("FingerprintSpoofer",                 "Fingerprint Spoofer (Chrome)"),
    ("CanvasBlockerFirefox",              "CanvasBlocker (Firefox)"),
    ("FirefoxCanvasBlocker",              "CanvasBlocker (Firefox)"),
    ("firefoxCanvasBlocker",              "CanvasBlocker (Firefox)"),
    ("CanvasBlocker",                      "Canvas Blocker (Chrome)"),
    ("Brave",                              "Brave"),
    ("FirefoxStandard",                    "Firefox (Standard)"),
    ("Firefox",                            "Firefox (Strict)"),
    ("SafariStandard",                     "Safari (Standard)"),
    ("Safari",                             "Safari (Private)"),
    ("SamsungInternet",                    "Samsung Internet"),
    ("Tor15",                              "Tor"),
    ("Tor",                                "Tor"),
]

DEFENSE_BROWSERS  = ["Brave", "Firefox (Strict)", "Firefox (Standard)", "Safari (Private)", "Safari (Standard)", "Samsung Internet", "Tor"]
DEFENSE_EXTS      = ["Canvas Blocker (Chrome)", "Canvas FP Defender (Chrome)",
                     "Fingerprint Spoofer (Chrome)", "CanvasBlocker (Firefox)"]
DEFENSE_ALL       = DEFENSE_BROWSERS + DEFENSE_EXTS
DEFENSE_NOISE     = [d for d in DEFENSE_ALL if d != "Tor"]


def classify_defense(sid_value: str) -> str:
    """Map a sid string to a display group name; an unknown sid is its own
    group, so undefended browsers appear as baseline rows."""
    if not sid_value or sid_value == "unnamed":
        return "unnamed"
    for pattern, group in _SID_TO_DEFENSE:
        if sid_value == pattern or sid_value.startswith(pattern):
            return group
    # fallback: use the sid itself as the group name
    return sid_value


def discover_sid_groups(base_dir: Path) -> List[str]:
    """Scan base_dir and return an ordered list of all unique sid groups.

    Known defense groups (DEFENSE_ALL) are listed first in their original
    order.  Any additional sid groups found in the directory (e.g. Chrome,
    FirefoxStandard, SafariStandard) are appended in alphabetical order.
    """
    found_groups = set()
    for dev in base_dir.glob("dev_*"):
        if not dev.is_dir():
            continue
        sid = extract_sid(dev.name)
        if _is_generated_sid(sid):   # skip auto-generated hex session IDs
            continue
        group = classify_defense(sid)
        if group and group != "unnamed":
            found_groups.add(group)

    # preserve known order for recognised defenses, then alpha for the rest
    ordered = [g for g in DEFENSE_ALL if g in found_groups]
    extras  = sorted(found_groups - set(DEFENSE_ALL))
    return ordered + extras


# ── is_antifp_device ─────────────────────────────────────────────────────────
def _is_generated_sid(sid: str) -> bool:
    """True if sid looks like an auto-generated session ID.

    All human-readable browser/defense sids (Chrome, Brave, FirefoxStandard,
    SamsungInternet, ...) start with an uppercase letter.
    Auto-generated IDs (e.g. 0123456789abcdef01234567, 0abc123xyz89) start
    with a digit or lowercase letter.
    """
    return bool(sid) and not sid[0].isupper()

def is_antifp_device(folder_name: str) -> bool:
    """True for devices with a human-readable named sid (browser/defense group).

    Excludes:
      - empty / 'unnamed' sids
      - sids starting with a digit or lowercase letter (auto-generated IDs)
    """
    sid = extract_sid(folder_name)
    if not sid or sid == "unnamed":
        return False
    if _is_generated_sid(sid):
        return False
    return True


# ── helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def _strip_ext_and_meta(name):
    low = name.lower()
    if "hash" in low or low.endswith(".meta.json"): return None
    return name[:-len(".json.gz")] if low.endswith(".json.gz") else Path(name).stem

def extract_key_from_filename(name):
    base = _strip_ext_and_meta(name)
    if not base: return None
    s = _SESSION_PREFIX_RE.sub('', base)
    if not s: return None
    toks = s.split('_')
    if not toks or not toks[0]: return None
    if toks[0] == 'raw' and len(toks) >= 2 and toks[1]:
        return f'raw_{toks[1]}'
    return toks[0]

def _get_visit_dirs(dev_dir):
    return sorted([d for d in dev_dir.iterdir()
                   if d.is_dir() and d.name.startswith("visit")])

def list_key_rgba_sorted_all_visits(dev_dir, key_string):
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

def list_device_folders(base_dir):
    return sorted([p for p in Path(base_dir).glob("dev_*") if p.is_dir()])

def collect_device_key_lists_all_vis(base_dir, keys, min_traces=2, verbose=False):
    out = {}
    rejected = []
    for dev in list_device_folders(base_dir):
        visit_dirs = _get_visit_dirs(dev)
        if not visit_dirs:
            rejected.append((dev.name, "no visit* subdirectories"))
            continue
        perkey = {}; fail_key = None; fail_count = None
        for k in keys:
            lst = list_key_rgba_sorted_all_visits(dev, k)
            if len(lst) < min_traces:
                fail_key   = k
                fail_count = len(lst)
                break
            perkey[k] = lst[:REQUIRED_TRACES_PER_KEY]
        if fail_key is not None:
            rejected.append((dev.name,
                             f"key '{fail_key}' has {fail_count} traces "
                             f"(need >= {min_traces})"))
        else:
            out[dev.name] = perkey
    if verbose and rejected:
        print(f"  [rejected devices: {len(rejected)}]")
        for rname, reason in rejected:
            print(f"    REJECTED: {rname}  ({reason})")
    return out

def collect_antifp_flexible(base_dir, keys, min_traces=1):
    out = {}
    device_keys = {}
    for dev in list_device_folders(base_dir):
        if not _get_visit_dirs(dev): continue
        if not is_antifp_device(dev.name): continue
        perkey = {}
        for k in keys:
            lst = list_key_rgba_sorted_all_visits(dev, k)
            if len(lst) >= min_traces:
                perkey[k] = lst[:REQUIRED_TRACES_PER_KEY]
        if perkey:
            out[dev.name] = perkey
            device_keys[dev.name] = list(perkey.keys())
    return out, device_keys

def build_shared_split_all_vis(device_key_lists, keys, seed, test_frac=0.20):
    rng = np.random.default_rng(seed)
    cal_plan  = {k: {} for k in keys}
    test_plan = {k: {} for k in keys}
    for dname, perkey in device_key_lists.items():
        dev_keys = [k for k in keys if k in perkey]
        if not dev_keys: continue
        N = min(len(perkey[k]) for k in dev_keys)
        if N < 2: continue
        perm  = rng.permutation(N)
        n_te  = max(1, int(round(N * test_frac)))
        n_tr  = max(1, N - n_te)
        if n_tr + n_te > N: n_tr = N - n_te
        tr_idx = sorted(perm[:n_tr].tolist())
        te_idx = sorted(perm[n_tr:].tolist())
        for k in dev_keys:
            files = perkey[k][:N]
            cal_plan[k][dname]  = [files[i] for i in tr_idx]
            test_plan[k][dname] = [files[i] for i in te_idx]
    return cal_plan, test_plan

def build_chronological_split_all_vis(device_key_lists, keys, max_query_pool=20):
    """Chronological 80/20 split: first (n-q) enroll, last q query."""
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



def load_rgba_as_tensor(path):
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

class PixelCNN(nn.Module):
    def __init__(self, num_classes, base_channels=32, blocks=3, ksize=3, dropout=0.2):
        super().__init__()
        c0 = int(base_channels); b = max(2, min(int(blocks), 4))
        k = int(ksize); pad = k // 2
        layers = []; in_ch = 4
        for bi in range(b):
            out_ch = c0 * (2 ** bi)
            layers += [
                nn.Conv2d(in_ch, out_ch, k, padding=pad), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, k, padding=pad), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(in_ch, 256), nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)), nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.head(self.backbone(x))

def _torch_load_safe(path, map_location):
    try: return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError: return torch.load(path, map_location=map_location)

@torch.no_grad()
def feats_for_paths(model, paths, mean, std, device, bs=256):
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

# ── SID extraction ────────────────────────────────────────────────────────────
_SID_RE = re.compile(r'__sid-(.+)$')
_PID_RE = re.compile(r'__pid-(.+?)__sid-')

def extract_sid(folder_name: str) -> str:
    m = _SID_RE.search(folder_name)
    return m.group(1) if m else "unnamed"

def extract_pid(folder_name: str) -> str:
    m = _PID_RE.search(folder_name)
    return m.group(1) if m else "?"

# ── feature extraction helper ─────────────────────────────────────────────────
def get_avg_feat(models, norms, keys, cal_paths_by_key, device, bs):
    vecs = []
    for k in keys:
        paths = [Path(p) for p in cal_paths_by_key.get(k, [])]
        if not paths: continue
        F = feats_for_paths(models[k], paths, norms[k][0], norms[k][1],
                            device=device, bs=bs)
        if F.size:
            vecs.append(F.mean(axis=0))
    if not vecs:
        return np.zeros(256, dtype=np.float32)
    avg = np.mean(vecs, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg)
    return avg / (norm + 1e-9)

def score_query(models, norms, keys, test_paths_by_key, protos, ks, device, bs):
    """Score a query device against all gallery prototypes (deterministic)."""
    perkey_scores = {}
    for k in keys:
        paths = test_paths_by_key.get(k, [])
        if len(paths) < ks: continue
        F_all = feats_for_paths(models[k], [Path(p) for p in paths],
                                norms[k][0], norms[k][1], device=device, bs=bs)
        if F_all.shape[0] < ks: continue
        F_all = F_all / (np.linalg.norm(F_all, axis=1, keepdims=True) + 1e-9)
        perkey_scores[k] = (F_all @ protos[k].T).mean(axis=0)
    if not perkey_scores:
        return None
    return np.mean(np.stack(list(perkey_scores.values()), axis=0), axis=0)

# ── evaluation loop ───────────────────────────────────────────────────────────
def run_eval(antifp_devices, dev2idx, all_devices,
             protos_all, keys, cal_plan, test_plan,
             models, norms, kshots, topk_list, device, bs,
             neighbor_depth=5, base_dir=None):
    """Run identification for each anti-FP device against the gallery."""
    browsers = sorted(set(antifp_devices.values()))
    results_by_browser = defaultdict(
        lambda: defaultdict(
            lambda: {f"correct_top{kk}": 0 for kk in topk_list} | {"total": 0}
        )
    )
    per_device_results = []

    _ua_cache = {}
    def _get_ua(dev_name):
        if dev_name not in _ua_cache:
            if base_dir is not None:
                _ua_cache[dev_name] = _get_useragent_from_fpjs(
                    Path(base_dir) / dev_name)
            else:
                _ua_cache[dev_name] = None
        return _ua_cache[dev_name]

    for d, browser in antifp_devices.items():
        if d not in dev2idx:
            print(f"  [skip-not-in-gallery] {d}")
            continue
        gt = dev2idx[d]

        test_paths = {k: test_plan[k].get(d, []) for k in keys}
        dev_min_test = min(
            len(test_paths.get(k, [])) for k in keys
            if cal_plan[k].get(d)
        ) if any(cal_plan[k].get(d) for k in keys) else 0

        for ks in kshots:
            if dev_min_test < ks:
                continue
            S = score_query(models, norms, keys, test_paths,
                            protos_all, ks, device, bs)
            if S is None:
                continue
            order = np.argsort(-S)
            rank = int(np.where(order == gt)[0][0]) + 1

            results_by_browser[browser][ks]["total"] += 1
            for kk in topk_list:
                if gt in order[:kk]:
                    results_by_browser[browser][ks][f"correct_top{kk}"] += 1

            gt_score = round(float(S[gt]), 6)

            n_nb = min(neighbor_depth, len(order))
            neighbors = []
            for ni in range(n_nb):
                gal_idx = int(order[ni])
                gal_dev = all_devices[gal_idx]
                nb = {
                    "rank":     ni + 1,
                    "device":   gal_dev,
                    "hardware": extract_pid(gal_dev),
                    "sid":      extract_sid(gal_dev),
                    "defense":  classify_defense(extract_sid(gal_dev)),
                    "score":    round(float(S[gal_idx]), 6),
                    "is_self":  gal_idx == gt,
                }
                ua = _get_ua(gal_dev)
                if ua:
                    nb["useragent"] = ua
                neighbors.append(nb)

            entry = {
                "device": d, "browser": browser, "kshot": ks,
                "gt_rank": rank,
                "gt_score": gt_score,
                **{f"top{kk}_correct": bool(gt in order[:kk]) for kk in topk_list},
                "neighbors": neighbors,
            }
            ua = _get_ua(d)
            if ua:
                entry["useragent"] = ua
            per_device_results.append(entry)

    return results_by_browser, per_device_results

def aggregate(results_by_browser, browsers, kshots, topk_list, N_gallery):
    base_rates = {f"top{kk}": kk / N_gallery for kk in topk_list}
    summary = {}
    for browser in browsers:
        summary[browser] = {}
        for ks in kshots:
            rec = results_by_browser[browser][ks]
            total = rec["total"]
            if total == 0:
                summary[browser][ks] = {"num_queries": 0}
                continue
            entry = {"num_queries": total}
            for kk in topk_list:
                acc  = rec[f"correct_top{kk}"] / total
                lift = acc / base_rates[f"top{kk}"] if base_rates[f"top{kk}"] > 0 else float("nan")
                entry[f"top{kk}_acc"]  = round(acc * 100, 2)
                entry[f"top{kk}_lift"] = round(lift, 1)
            summary[browser][ks] = entry

    overall = {}
    for ks in kshots:
        total_all = sum(results_by_browser[b][ks]["total"] for b in browsers)
        if total_all == 0:
            overall[ks] = {"num_queries": 0}
            continue
        entry = {"num_queries": total_all}
        for kk in topk_list:
            correct_all = sum(results_by_browser[b][ks][f"correct_top{kk}"]
                              for b in browsers)
            acc  = correct_all / total_all
            lift = acc / base_rates[f"top{kk}"]
            entry[f"top{kk}_acc"]  = round(acc * 100, 2)
            entry[f"top{kk}_lift"] = round(lift, 1)
        overall[ks] = entry
    return summary, overall, base_rates

def print_table(title, summary, overall, browsers, kshots, topk_list, N):
    # Since score_query() uses all available traces (deterministic), results are
    # identical across k values — k only gates eligibility. Print only k=min.
    ks_report = min(kshots)
    print(f"\n{'='*70}")
    print(f"  {title}  |  N={N}  |  base_rate_top1={1/N*100:.3f}%")
    print(f"  (k-shot results are identical across k; showing k={ks_report})")
    print(f"{'='*70}")
    for browser in browsers:
        rec = summary[browser].get(ks_report, {})
        if rec.get("num_queries", 0) == 0:
            print(f"  [{browser}]  no queries")
            continue
        parts = [f"Top-{kk}={rec[f'top{kk}_acc']:.1f}% ({rec[f'top{kk}_lift']:.0f}×)"
                 for kk in topk_list]
        print(f"  [{browser}]  {'  |  '.join(parts)}  (n={rec['num_queries']})")
    rec_ov = overall.get(ks_report, {})
    if rec_ov.get("num_queries", 0) > 0:
        parts = [f"Top-{kk}={rec_ov[f'top{kk}_acc']:.1f}% ({rec_ov[f'top{kk}_lift']:.0f}×)"
                 for kk in topk_list]
        print(f"\n  [Overall]  {'  |  '.join(parts)}  (n={rec_ov['num_queries']})")
    print(f"{'='*70}")

# ── defense-grouped summary ──────────────────────────────────────────────────
def print_defense_summary(per_device_results, topk_list, kshots, N_gallery,
                          all_sid_groups=None,
                          all_per_device_k1=None, base_dir=None):
    base_rates = {kk: kk / N_gallery for kk in topk_list}

    defense_results = defaultdict(lambda: defaultdict(list))
    for entry in per_device_results:
        sid = extract_sid(entry["device"])
        group = classify_defense(sid)
        k = entry["kshot"]
        defense_results[group][k].append(entry)

    # fall back to all discovered groups if not supplied
    if all_sid_groups is None:
        all_sid_groups = sorted(defense_results.keys())

    def _pct_lift(n_correct, n_total, topn):
        if n_total == 0: return "  —   ", "     "
        acc = n_correct / n_total
        lft = acc / base_rates[topn] if base_rates[topn] > 0 else 0
        return f"{acc*100:5.1f}%", f"({lft:5.0f}x)"

    ks_report = min(kshots)  # results identical across k; report only min

    def _print_section(groups, title):
        print(f"\n{'─'*90}")
        print(f"  {title}")
        print(f"{'─'*90}")
        print(f"  {'Defense/Browser':<30} {'#dev':>4}  " +
              "  ".join(f"{'Top-'+str(kk):>14}" for kk in topk_list) +
              f"  {'MeanRank':>8}")
        print(f"  {'-'*83}")
        for g in groups:
            if g not in defense_results:
                print(f"  {g:<30}  NO DATA")
                continue
            entries = defense_results[g].get(ks_report, [])
            n = len(entries)
            if n == 0:
                print(f"  {g:<30}  NO DATA")
                continue
            mr = sum(e["gt_rank"] for e in entries) / n
            parts = []
            for kk in topk_list:
                nc = sum(1 for e in entries if e["gt_rank"] <= kk)
                pct_s, lft_s = _pct_lift(nc, n, kk)
                parts.append(f"{pct_s} {lft_s}")
            print(f"  {g:<30} {n:>4}  " + "  ".join(parts) + f"  {mr:7.1f}")

    known_defense_groups = [g for g in all_sid_groups if g in set(DEFENSE_ALL)]
    baseline_groups      = [g for g in all_sid_groups if g not in set(DEFENSE_ALL)]

    print(f"\n{'='*90}")
    print(f"  DEFENSE-GROUPED SUMMARY  (gallery N={N_gallery})")
    base_str = "  ".join(f"Top-{kk}={base_rates[kk]*100:.2f}%" for kk in topk_list)
    print(f"  Base rates: {base_str}")
    print(f"{'='*90}")

    defense_browsers_found = [g for g in DEFENSE_BROWSERS if g in known_defense_groups]
    defense_exts_found     = [g for g in DEFENSE_EXTS     if g in known_defense_groups]

    if defense_browsers_found:
        _print_section(defense_browsers_found, "Privacy Browsers (Defended)")
    if defense_exts_found:
        _print_section(defense_exts_found, "Browser Extensions (Defended)")
    if baseline_groups:
        _print_section(baseline_groups, "Baseline (No Defense)")

    # ── per-device detail (k=1) ──
    print(f"\n{'─'*90}")
    print(f"  Per-Device Detail (k=1)")
    print(f"{'─'*90}")
    print(f"  {'Hardware':<30} {'Group':<30} {'Rank':>5} " +
          "  ".join(f"{'T'+str(kk):>4}" for kk in topk_list))
    print(f"  {'-'*86}")

    k1_entries = []
    for entry in per_device_results:
        if entry["kshot"] != 1: continue
        dev = entry["device"]
        sid = extract_sid(dev)
        group = classify_defense(sid)
        hw = extract_pid(dev)
        # sort key: known defense order first, then alphabetical
        sort_idx = all_sid_groups.index(group) if group in all_sid_groups else 999
        k1_entries.append((sort_idx, group, hw, entry["gt_rank"], entry))

    k1_entries.sort(key=lambda x: (x[0], x[3]))
    for _, group, hw, rank, entry in k1_entries:
        marks = []
        for kk in topk_list:
            marks.append("✓" if entry["gt_rank"] <= kk else "✗")
        mark_str = "  ".join(f"{m:>4}" for m in marks)
        ua = ""
        if base_dir:
            ua_val = _get_useragent_from_fpjs(Path(base_dir) / entry["device"])
            if ua_val:
                ua = f"  UA: {ua_val}"
        print(f"  {hw:<30} {group:<30} {rank:>5} {mark_str}{ua}")
        for nb in entry.get("neighbors", []):
            if nb["is_self"]:
                continue
            nb_def = nb["defense"] or "(none)"
            nb_ua = ""
            if base_dir:
                nb_ua_val = _get_useragent_from_fpjs(Path(base_dir) / nb["device"])
                if nb_ua_val:
                    nb_ua = f"  UA: {nb_ua_val}"
            print(f"    #{nb['rank']:>2}  {nb['hardware']:<25} [{nb_def:<28}] "
                  f"score={nb['score']:.4f}{nb_ua}")

    # ── aggregate: noise-based defenses (excl Tor) ──
    print(f"\n{'─'*90}")
    print(f"  Aggregate: All Noise-Based Defenses (excl. Tor)")
    print(f"{'─'*90}")
    all_entries = []
    for g in DEFENSE_NOISE:
        if ks_report in defense_results.get(g, {}):
            all_entries.extend(defense_results[g][ks_report])
    if all_entries:
        n = len(all_entries)
        mr = sum(e["gt_rank"] for e in all_entries) / n
        parts = []
        for kk in topk_list:
            nc = sum(1 for e in all_entries if e["gt_rank"] <= kk)
            pct_s, lft_s = _pct_lift(nc, n, kk)
            parts.append(f"Top-{kk}={pct_s} {lft_s}")
        print(f"  n={n:>3}  " + "  ".join(parts) + f"  MeanRank={mr:.1f}")

    # ── all-device top-N breakdown (k=1) ──
    if all_per_device_k1 is not None:
        k1_all = [e for e in all_per_device_k1 if e.get("kshot", 1) == 1]

        defended_k1   = [e for e in k1_all
                         if is_antifp_device(e["device"])
                         and classify_defense(extract_sid(e["device"])) not in ("Tor", *baseline_groups)]
        tor_k1        = [e for e in k1_all
                         if classify_defense(extract_sid(e["device"])) == "Tor"]
        baseline_k1   = [e for e in k1_all
                         if classify_defense(extract_sid(e["device"])) in baseline_groups]
        undefended_k1 = [e for e in k1_all
                         if not is_antifp_device(e["device"])]

        print(f"\n{'─'*90}")
        print(f"  All Devices Top-N (k=1)  —  full gallery N={N_gallery}")
        print(f"{'─'*90}")

        segments = [
            ("All devices",           k1_all),
            ("  Undefended",          undefended_k1),
            ("  Baseline (no def.)",  baseline_k1),
            ("  Defended (excl Tor)", defended_k1),
            ("  Tor",                 tor_k1),
        ]
        for label, entries in segments:
            n = len(entries)
            if n == 0: continue
            mr = sum(e["gt_rank"] for e in entries) / n
            parts = []
            for kk in topk_list:
                nc = sum(1 for e in entries if e["gt_rank"] <= kk)
                pct_s, lft_s = _pct_lift(nc, n, kk)
                parts.append(f"Top-{kk}={pct_s} {lft_s}")
            print(f"  {label:<25}  n={n:>4}  " + "  ".join(parts) + f"  MeanRank={mr:.1f}")

    print(f"\n{'='*90}")


# ── defense-grouped JSON summary ──────────────────────────────────────────────
def build_defense_grouped_json(per_device_results, topk_list, kshots, N_gallery,
                               all_sid_groups=None):
    base_rates = {kk: kk / N_gallery for kk in topk_list}

    defense_results = defaultdict(lambda: defaultdict(list))
    for entry in per_device_results:
        sid = extract_sid(entry["device"])
        group = classify_defense(sid)
        defense_results[group][entry["kshot"]].append(entry)

    groups_to_report = all_sid_groups if all_sid_groups else sorted(defense_results.keys())

    summary = {}
    for group in groups_to_report:
        if group not in defense_results:
            summary[group] = {}
            continue
        summary[group] = {}
        for k in kshots:
            entries = defense_results[group].get(k, [])
            n = len(entries)
            if n == 0:
                summary[group][k] = {"num_queries": 0}
                continue
            rec = {"num_queries": n}
            for kk in topk_list:
                nc = sum(1 for e in entries if e["gt_rank"] <= kk)
                acc = nc / n
                lft = acc / base_rates[kk] if base_rates[kk] > 0 else 0
                rec[f"top{kk}_acc"]  = round(acc * 100, 2)
                rec[f"top{kk}_lift"] = round(lft, 1)
            rec["mean_rank"] = round(sum(e["gt_rank"] for e in entries) / n, 2)
            summary[group][k] = rec

        k1_entries = defense_results[group].get(1, [])
        devices_detail = []
        for e in sorted(k1_entries, key=lambda x: x["gt_rank"]):
            dev_info = {
                "hardware":  extract_pid(e["device"]),
                "device_id": e["device"],
                "rank":      e["gt_rank"],
                "gt_score":  e.get("gt_score"),
            }
            if e.get("useragent"):
                dev_info["useragent"] = e["useragent"]
            devices_detail.append(dev_info)
        summary[group]["devices"] = devices_detail

    return summary

# ── userAgent extraction from fpjs JSON ──────────────────────────────────────
def _get_useragent_from_fpjs(dev_dir):
    fpjs_files = sorted(dev_dir.glob("fpjs*.json"))
    if not fpjs_files:
        return None
    try:
        data = json.loads(fpjs_files[0].read_text(encoding="utf-8"))
        return data.get("userAgent")
    except Exception:
        return None

# ── anti-FP device info listing ───────────────────────────────────────────────
def print_antifp_device_info(dkl_antifp, antifp_device_keys, keys, base_dir,
                             all_sid_groups=None):
    print(f"\n{'='*120}")
    print(f"  DEVICE INFO  ({len(dkl_antifp)} devices)")
    print(f"{'='*120}")

    by_group = defaultdict(list)
    for d in sorted(dkl_antifp.keys()):
        sid = extract_sid(d)
        group = classify_defense(sid)
        by_group[group].append(d)

    groups_to_show = all_sid_groups if all_sid_groups else sorted(by_group.keys())

    idx = 0
    for group in groups_to_show:
        devs = by_group.get(group, [])
        if not devs:
            continue
        for d in devs:
            idx += 1
            hw = extract_pid(d)
            sid = extract_sid(d)
            avail_keys = antifp_device_keys.get(d, [])
            n_keys = len(avail_keys)
            trace_counts = [f"{k}={len(dkl_antifp[d][k])}" for k in avail_keys]
            missing = [k for k in keys if k not in avail_keys]
            trace_str = ", ".join(trace_counts)
            if missing:
                trace_str += f"  (MISSING: {', '.join(missing)})"
            ua = _get_useragent_from_fpjs(Path(base_dir) / d) or "N/A"
            print(f"\n  {idx:>3}. [{group}] {hw}")
            print(f"       SID:       {sid}")
            print(f"       Keys({n_keys}):  {trace_str}")
            print(f"       UserAgent: {ua}")

    print(f"\n  {'Group':<30} {'Count':>5}")
    print(f"  {'-'*36}")
    total = 0
    for g in groups_to_show:
        c = len(by_group.get(g, []))
        if c > 0:
            print(f"  {g:<30} {c:>5}")
            total += c
    print(f"  {'TOTAL':<30} {total:>5}")
    print(f"{'='*120}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir",    required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--batch-size",  type=int, default=256)
    parser.add_argument('--chronological-split', action='store_true', help='Use chronological 80/20 (past enroll, future query)')
    parser.add_argument("--topk",        default="1,3,5")
    parser.add_argument("--kshots",      default="1,5,10,20")
    parser.add_argument("--min-traces",  type=int, default=2)
    parser.add_argument("--antifp-min-traces", type=int, default=1)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--gpu",         action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"[postprocess] device={device}")

    base_dir    = Path(args.base_dir)
    results_dir = Path(args.results_dir)
    meta_dir    = results_dir / "meta"

    topk_list = sorted({int(t) for t in args.topk.split(",")  if t.strip().isdigit()})
    kshots    = sorted({int(s) for s in args.kshots.split(",") if s.strip().isdigit()})
    min_k     = min(kshots)

    # ── discover all sid groups in the directory ──────────────────────────────
    all_sid_groups = discover_sid_groups(base_dir)

    # ── load meta ──
    try:
        sig     = json.loads((meta_dir / "plan_signature.json").read_text())
        idx2dev = json.loads((meta_dir / "devices_kept.json").read_text())
    except Exception as e:
        print(f"[fatal] cannot read meta: {e}", file=sys.stderr); sys.exit(2)

    C_train  = len(idx2dev)
    keys_all = sig.get("keys", [])
    keys     = [k for k in keys_all if k not in EXCLUDED_KEYS]
    if not keys:
        print("[fatal] no keys", file=sys.stderr); sys.exit(3)
    print(f"[postprocess] keys={keys}  C_train={C_train}")

    # ── load models ──
    models_dir  = results_dir / "models"
    pixnorm_dir = results_dir / "pixnorm"
    models, norms = {}, {}
    for k in keys:
        mpath = models_dir  / f"rawcnn_allvis_{k}.pt"
        jpath = pixnorm_dir / f"rawcnn_pixnorm_allvis_{k}.json"
        if not mpath.exists() or not jpath.exists():
            print(f"[fatal] missing model key={k}", file=sys.stderr); sys.exit(4)
        model = PixelCNN(num_classes=C_train, base_channels=32, blocks=3,
                         ksize=3, dropout=args.dropout).to(device)
        model.load_state_dict(_torch_load_safe(mpath, map_location=device), strict=True)
        model.eval()
        jp = json.loads(jpath.read_text())
        norms[k] = (np.array(jp["pix_mean"], dtype=np.float32),
                    np.array(jp["pix_std"],  dtype=np.float32))
        models[k] = model
    print(f"[postprocess] loaded {len(models)} models")

    # ── collect gallery devices ──
    min_traces_gallery = max(args.min_traces, min_k + 1)
    print(f"\n[postprocess] collecting gallery devices "
          f"(min_traces={min_traces_gallery}, all {len(keys)} keys required)...")
    dkl_gallery = collect_device_key_lists_all_vis(
        base_dir, keys, min_traces=min_traces_gallery, verbose=True)
    n_total = len(list_device_folders(base_dir))
    print(f"[postprocess] scanned {n_total} dev_* folders → "
          f"{len(dkl_gallery)} kept, {n_total - len(dkl_gallery)} rejected")

    # ── collect antifp/baseline devices ──
    print(f"\n[postprocess] collecting query devices "
          f"(min_traces={args.antifp_min_traces}, flexible keys)...")
    dkl_antifp, antifp_device_keys = collect_antifp_flexible(
        base_dir, keys, min_traces=args.antifp_min_traces)

    print(f"[postprocess] query devices found: {len(dkl_antifp)}")
    print_antifp_device_info(dkl_antifp, antifp_device_keys, keys, base_dir,
                             all_sid_groups=all_sid_groups)

    # ── merge gallery + query devices ──
    dkl_full = dict(dkl_gallery)
    for d, perkey in dkl_antifp.items():
        if d not in dkl_full:
            merged = {k: perkey.get(k, []) for k in keys}
            dkl_full[d] = merged

    # ── 80/20 split (chronological or random) ──
    if getattr(args, 'chronological_split', False):
        cal_plan, test_plan = build_chronological_split_all_vis(dkl_full, keys, max_query_pool=20)
        print('[postprocess] chronological split: past enroll, future query')
    else:
        cal_plan, test_plan = build_shared_split_all_vis(dkl_full, keys, seed=args.seed)
        print('[postprocess] random shared 80/20 split')

    all_devices_full = sorted({d for k in keys for d in cal_plan[k]})
    print(f"\n[postprocess] total devices after split: {len(all_devices_full)}")

    antifp_devices = {d: classify_defense(extract_sid(d))
                      for d in dkl_antifp
                      if any(cal_plan[k].get(d) for k in keys)}

    browsers = sorted(set(antifp_devices.values()))

    # ── build FULL gallery prototypes ──
    print("\n[postprocess] building full gallery prototypes...")
    protos_full = {}
    for k in keys:
        rows = []
        for d in all_devices_full:
            paths = [Path(p) for p in cal_plan[k].get(d, [])]
            Fd = feats_for_paths(models[k], paths, norms[k][0], norms[k][1],
                                 device=device, bs=args.batch_size)
            rows.append(Fd.mean(axis=0) if Fd.size else np.zeros(256, dtype=np.float32))
        P = np.stack(rows, axis=0)
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        protos_full[k] = P
    dev2idx_full = {d: i for i, d in enumerate(all_devices_full)}
    N_full = len(all_devices_full)
    print(f"[postprocess] full gallery N={N_full}")

    # ── build ANTI-FP-ONLY gallery prototypes ──
    antifp_enrolled = sorted(antifp_devices.keys())
    print(f"\n[postprocess] building anti-FP-only gallery "
          f"({len(antifp_enrolled)} devices)...")
    protos_af = {}
    for k in keys:
        rows = []
        for d in antifp_enrolled:
            paths = [Path(p) for p in cal_plan[k].get(d, [])]
            Fd = feats_for_paths(models[k], paths, norms[k][0], norms[k][1],
                                 device=device, bs=args.batch_size)
            rows.append(Fd.mean(axis=0) if Fd.size else np.zeros(256, dtype=np.float32))
        P = np.stack(rows, axis=0)
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        protos_af[k] = P
    dev2idx_af = {d: i for i, d in enumerate(antifp_enrolled)}
    N_af = len(antifp_enrolled)

    # ══ EVAL 1: full gallery ══
    print("\n[postprocess] === Eval 1: full gallery ===")
    res_full, perdev_full = run_eval(
        antifp_devices, dev2idx_full, all_devices_full,
        protos_full, keys, cal_plan, test_plan,
        models, norms, kshots, topk_list, device, args.batch_size,
        base_dir=base_dir)

    summary_full, overall_full, base_full = aggregate(
        res_full, browsers, kshots, topk_list, N_full)
    print_table("Query Devices vs Full Gallery", summary_full, overall_full,
                browsers, kshots, topk_list, N_full)

    # ══ EVAL 2: anti-FP-only gallery ══
    print("\n[postprocess] === Eval 2: query-devices-only gallery ===")
    res_af, perdev_af = run_eval(
        antifp_devices, dev2idx_af, antifp_enrolled,
        protos_af, keys, cal_plan, test_plan,
        models, norms, kshots, topk_list, device, args.batch_size,
        base_dir=base_dir)

    summary_af, overall_af, base_af = aggregate(
        res_af, browsers, kshots, topk_list, N_af)
    print_table("Query Devices vs Query-Only Gallery", summary_af, overall_af,
                browsers, kshots, topk_list, N_af)

    # ══ DEFENSE-GROUPED SUMMARY ══
    print("\n[postprocess] collecting undefended device results for comparison...")
    all_k1_results = list(perdev_full)
    undef_devices = [d for d in all_devices_full if not is_antifp_device(d)]
    n_undef = len(undef_devices)
    for i, d in enumerate(undef_devices):
        if (i + 1) % 100 == 0 or (i + 1) == n_undef:
            print(f"  [{i+1}/{n_undef}] undefended devices evaluated")
        gt = dev2idx_full[d]
        test_paths = {k: test_plan[k].get(d, []) for k in keys}
        S = score_query(models, norms, keys, test_paths,
                        protos_full, 1, device, args.batch_size)
        if S is None: continue
        order = np.argsort(-S)
        rank = int(np.where(order == gt)[0][0]) + 1
        all_k1_results.append({
            "device": d, "browser": "undefended", "kshot": 1,
            "gt_rank": rank,
            "gt_score": round(float(S[gt]), 6),
            **{f"top{kk}_correct": bool(gt in order[:kk]) for kk in topk_list},
        })

    print_defense_summary(perdev_full, topk_list, kshots, N_full,
                          all_sid_groups=all_sid_groups,
                          all_per_device_k1=all_k1_results,
                          base_dir=base_dir)

    # ── build all-devices top-N summary for JSON ──
    all_devices_topn = {}
    k1_all = [e for e in all_k1_results if e.get("kshot", 1) == 1]
    baseline_groups_main = [g for g in all_sid_groups if g not in set(DEFENSE_ALL)]
    segments_json = {
        "all":        k1_all,
        "undefended": [e for e in k1_all if not is_antifp_device(e["device"])],
        "baseline":   [e for e in k1_all if classify_defense(extract_sid(e["device"])) in baseline_groups_main],
        "defended":   [e for e in k1_all
                       if is_antifp_device(e["device"])
                       and classify_defense(extract_sid(e["device"])) not in ("Tor", *baseline_groups_main)],
        "tor":        [e for e in k1_all if classify_defense(extract_sid(e["device"])) == "Tor"],
    }
    base_rates_full_topn = {kk: kk / N_full for kk in topk_list}
    for seg_name, entries in segments_json.items():
        n = len(entries)
        if n == 0:
            all_devices_topn[seg_name] = {"n": 0}
            continue
        rec = {"n": n, "mean_rank": round(sum(e["gt_rank"] for e in entries) / n, 2)}
        for kk in topk_list:
            nc = sum(1 for e in entries if e["gt_rank"] <= kk)
            acc = nc / n
            rec[f"top{kk}_acc"]  = round(acc * 100, 2)
            rec[f"top{kk}_lift"] = round(acc / base_rates_full_topn[kk], 1)
        all_devices_topn[seg_name] = rec

    # ── save ──
    defense_grouped    = build_defense_grouped_json(
        perdev_full, topk_list, kshots, N_full, all_sid_groups=all_sid_groups)
    defense_grouped_af = build_defense_grouped_json(
        perdev_af,  topk_list, kshots, N_af,  all_sid_groups=all_sid_groups)

    output = {
        "N_gallery_full":      N_full,
        "N_gallery_antifp":    N_af,
        "antifp_device_count": len(antifp_devices),
        "kshots":              kshots,
        "topk":                topk_list,

        "base_rates_full":     base_full,
        "by_browser_full":     summary_full,
        "overall_antifp_full": overall_full,
        "per_device_full":     perdev_full,

        "base_rates_antifp":   base_af,
        "by_browser_antifp":   summary_af,
        "overall_antifp_af":   overall_af,
        "per_device_antifp":   perdev_af,

        "by_defense_full":     defense_grouped,
        "by_defense_antifp":   defense_grouped_af,

        "all_devices_topn":    all_devices_topn,
    }

    out_path = results_dir / "antifp_bypass_by_browser.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n[postprocess] saved → {out_path}")


if __name__ == "__main__":
    main()