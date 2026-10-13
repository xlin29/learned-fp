#!/usr/bin/env python3
"""
Browser Fingerprint Device Matcher
===================================
Scans pid-xxx folders, reads fpjs_*.json from each dev_ subfolder,
and determines whether multiple sessions for the same prolificPid
came from the same physical machine.

Features:
  - Supports two JSON formats (fingerprint / reorderedResult)
  - Hash-only fallback for plugins, webGlExtensions
  - SAME_DEVICE: copies entire pid folder
  - MIXED: copies only same-device dev_ folders into pid folder
  - VisitorId consistency check across all fpjs files
  - DrawnApart subfolder statistics
  - All stats written to a separate report file

Usage:
    python fingerprint_matcher.py <base_dir> [--threshold 0.80] \
        [--output results.csv] [--copy-same DIR] [--report report.txt]
"""

import json, os, sys, csv, shutil, argparse, re, glob
from collections import defaultdict
from itertools import combinations


# =============================================================================
# Helpers
# =============================================================================

def safe_get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def normalize_data(data):
    if "fingerprint" in data:
        fp = data["fingerprint"]
        return fp.get("components", {}), data.get("userAgent", ""), fp.get("visitorId", "")
    elif "reorderedResult" in data:
        fp = data["reorderedResult"]
        return fp.get("components", {}), fp.get("userAgent", "") or data.get("userAgent", ""), fp.get("visitorId", "")
    return {}, data.get("userAgent", ""), ""


def extract_visitor_id(data):
    _, _, vid = normalize_data(data)
    return vid


def extract_browser_version(ua):
    for pat, name in [(r'Chrome/([\d.]+)', "Chrome"), (r'Firefox/([\d.]+)', "Firefox"), (r'Safari/([\d.]+)', "Safari")]:
        m = re.search(pat, ua or "")
        if m:
            return (name, m.group(1))
    return ("Unknown", "")


def parse_dev_folder_name(folder_name):
    info = {}
    for p in folder_name.split("__"):
        if p.startswith("dev_"):
            info["deviceId"] = p
        elif p.startswith("pid-"):
            info["prolificPid"] = p.replace("pid-", "")
        elif p.startswith("sid-"):
            info["sessionId"] = p.replace("sid-", "")
    return info


def extract_year_from_fpjs_filename(filename):
    m = re.search(r'fpjs_(\d{4})-', filename)
    return int(m.group(1)) if m else None


def find_first_fpjs(dev_folder_path):
    files = sorted(glob.glob(os.path.join(dev_folder_path, "fpjs_*.json")))
    return files[0] if files else None


def find_all_fpjs(dev_folder_path):
    return sorted(glob.glob(os.path.join(dev_folder_path, "fpjs_*.json")))


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_stable_features(data):
    components, _, _ = normalize_data(data)
    f = {}

    # --- Tier 1: Very strong hardware identifiers ---
    f["gpu_renderer"] = safe_get(components, "webGlBasics", "value", "rendererUnmasked", default="")
    f["gpu_vendor"] = safe_get(components, "webGlBasics", "value", "vendorUnmasked", default="")
    f["audio"] = safe_get(components, "audio", "value", default=None)
    mv = safe_get(components, "math", "value", default={})
    f["math"] = tuple(sorted(mv.items())) if isinstance(mv, dict) and mv else None

    # --- Tier 2: Strong system identifiers ---
    f["screen_resolution"] = tuple(safe_get(components, "screenResolution", "value", default=[]))
    f["screen_frame"] = tuple(safe_get(components, "screenFrame", "value", default=[]))
    f["device_memory"] = safe_get(components, "deviceMemory", "value", default=None)
    f["hardware_concurrency"] = safe_get(components, "hardwareConcurrency", "value", default=None)
    f["color_depth"] = safe_get(components, "colorDepth", "value", default=None)
    f["architecture"] = safe_get(components, "architecture", "value", default=None)

    # --- Tier 3: System/environment identifiers ---
    f["platform"] = safe_get(components, "platform", "value", default="")
    f["timezone"] = safe_get(components, "timezone", "value", default="")
    langs = safe_get(components, "languages", "value", default=[])
    f["languages"] = json.dumps(langs, sort_keys=True)
    f["datetime_locale"] = safe_get(components, "dateTimeLocale", "value", default="")
    fonts = safe_get(components, "fonts", "value", default=[])
    f["fonts"] = tuple(sorted(fonts)) if isinstance(fonts, list) else ()
    fp = safe_get(components, "fontPreferences", "value", default={})
    f["font_preferences"] = tuple(sorted(fp.items())) if isinstance(fp, dict) and fp else None

    # --- Tier 4: Moderate identifiers ---
    pr = components.get("plugins", {})
    if isinstance(pr.get("value"), list):
        f["plugins"] = tuple(sorted(p.get("name", "") for p in pr["value"] if isinstance(p, dict)))
        f["plugins_hash"] = None
    elif "hash" in pr:
        f["plugins"] = ()
        f["plugins_hash"] = pr["hash"]
    else:
        f["plugins"] = ()
        f["plugins_hash"] = None

    t = safe_get(components, "touchSupport", "value", default={})
    f["touch_support"] = (
        safe_get(t, "maxTouchPoints", default=None),
        safe_get(t, "touchEvent", default=None),
        safe_get(t, "touchStart", default=None),
    )
    f["color_gamut"] = safe_get(components, "colorGamut", "value", default="")
    f["hdr"] = safe_get(components, "hdr", "value", default=None)

    # NOTE: Canvas is extracted for data completeness but NOT used in scoring.
    # Canvas fingerprints are too volatile across browser versions.
    cr = components.get("canvas", {})
    if isinstance(cr.get("value"), dict):
        f["canvas_text"] = cr["value"].get("text", "")
        f["canvas_hash"] = None
    elif "hash" in cr:
        f["canvas_text"] = ""
        f["canvas_hash"] = cr["hash"]
    else:
        f["canvas_text"] = ""
        f["canvas_hash"] = None

    wr = components.get("webGlExtensions", {})
    if isinstance(wr.get("value"), dict):
        exts = wr["value"].get("extensions", [])
        f["webgl_extensions"] = tuple(sorted(exts)) if isinstance(exts, list) else ()
        f["webgl_extensions_hash"] = None
    elif "hash" in wr:
        f["webgl_extensions"] = ()
        f["webgl_extensions_hash"] = wr["hash"]
    else:
        f["webgl_extensions"] = ()
        f["webgl_extensions_hash"] = None

    f["webgl_version"] = safe_get(components, "webGlBasics", "value", "version", default="")
    f["webgl_shading_lang"] = safe_get(components, "webGlBasics", "value", "shadingLanguageVersion", default="")
    fv = safe_get(components, "vendorFlavors", "value", default=[])
    f["vendor_flavors"] = tuple(sorted(fv)) if isinstance(fv, list) else ()

    comp_map = {
        "session_storage": "sessionStorage", "local_storage": "localStorage",
        "indexed_db": "indexedDB", "open_database": "openDatabase",
        "pdf_viewer": "pdfViewerEnabled", "reduced_motion": "reducedMotion",
        "reduced_transparency": "reducedTransparency", "forced_colors": "forcedColors",
        "inverted_colors": "invertedColors", "monochrome": "monochrome",
        "contrast": "contrast",
    }
    for k, ck in comp_map.items():
        f[k] = safe_get(components, ck, "value", default=None)

    return f


# =============================================================================
# Similarity Scoring
# =============================================================================

# Canvas is intentionally excluded - too volatile across browser updates.
# Total weight pool: 95.5
FEATURE_WEIGHTS = {
    # Tier 1 - Hardware (very strong)
    "gpu_renderer":          10.0,
    "gpu_vendor":             3.0,
    "audio":                 10.0,
    "math":                   8.0,
    # Tier 2 - System hardware
    "screen_resolution":      5.0,
    "screen_frame":           3.0,
    "device_memory":          4.0,
    "hardware_concurrency":   4.0,
    "color_depth":            2.0,
    "architecture":           2.0,
    # Tier 3 - System/environment
    "platform":               3.0,
    "timezone":               3.0,
    "languages":              3.0,
    "datetime_locale":        2.0,
    "fonts":                  7.0,
    "font_preferences":       6.0,
    # Tier 4 - Moderate
    "plugins":                2.0,
    "touch_support":          3.0,
    "color_gamut":            1.0,
    "hdr":                    1.0,
    "webgl_extensions":       4.0,
    "webgl_version":          1.0,
    "webgl_shading_lang":     1.0,
    "vendor_flavors":         2.0,
    "session_storage":        0.5,
    "local_storage":          0.5,
    "indexed_db":             0.5,
    "open_database":          0.5,
    "pdf_viewer":             0.5,
    "reduced_motion":         0.5,
    "reduced_transparency":   0.5,
    "forced_colors":          0.5,
    "inverted_colors":        0.5,
    "monochrome":             0.5,
    "contrast":               0.5,
}

# Fields that may be hash-only in some JSON formats
HASH_FIELDS = {
    "plugins": "plugins_hash",
    "webgl_extensions": "webgl_extensions_hash",
}


def compare_features(feat_a, feat_b):
    """
    Compare two feature dicts. Returns (similarity_score, details).
    Handles mixed full-value vs hash-only fields gracefully.
    """
    total_weight = 0.0
    matched_weight = 0.0
    details = []

    for feature, weight in FEATURE_WEIGHTS.items():
        val_a, val_b = feat_a.get(feature), feat_b.get(feature)

        # Hash fallback for fields that may be hash-only
        if feature in HASH_FIELDS:
            hk = HASH_FIELDS[feature]
            ha, hb = feat_a.get(hk), feat_b.get(hk)
            if ha is not None and hb is not None:
                total_weight += weight
                if ha == hb:
                    matched_weight += weight
                    details.append((feature, weight, True, f"hash match ({ha})"))
                else:
                    details.append((feature, weight, False, f"hash mismatch: {ha} vs {hb}"))
                continue
            elif ha is not None or hb is not None:
                details.append((feature, weight, None, "skipped: hash vs full value"))
                continue

        # Skip if both empty/None
        if val_a is None and val_b is None:
            continue
        if val_a == "" and val_b == "":
            continue
        if val_a == () and val_b == ():
            continue

        total_weight += weight

        if val_a == val_b:
            matched_weight += weight
            details.append((feature, weight, True, "exact match"))
        else:
            # Set-based comparison for fonts, extensions, plugins
            if feature in ("fonts", "webgl_extensions", "plugins") and \
               isinstance(val_a, tuple) and isinstance(val_b, tuple) and val_a and val_b:
                sa, sb = set(val_a), set(val_b)
                j = len(sa & sb) / len(sa | sb)
                if j > 0.8:
                    matched_weight += weight * j
                details.append((feature, weight, False, f"Jaccard={j:.3f}"))

            # Font preferences: per-key comparison
            elif feature == "font_preferences" and \
                 isinstance(val_a, tuple) and isinstance(val_b, tuple):
                da, db = dict(val_a), dict(val_b)
                ak = set(da.keys()) | set(db.keys())
                if ak:
                    mc = sum(1 for k in ak
                             if k in da and k in db and
                             isinstance(da[k], (int, float)) and
                             isinstance(db[k], (int, float)) and
                             abs(da[k] - db[k]) < 0.01)
                    r = mc / len(ak)
                    matched_weight += weight * r
                    details.append((feature, weight, r > 0.99, f"ratio={r:.3f}"))

            # Audio: near-exact float comparison
            elif feature == "audio" and val_a is not None and val_b is not None:
                try:
                    if abs(float(val_a) - float(val_b)) < 1e-10:
                        matched_weight += weight
                        details.append((feature, weight, True, "near-exact"))
                    else:
                        details.append((feature, weight, False,
                                        f"diff={abs(float(val_a)-float(val_b)):.15f}"))
                except (ValueError, TypeError):
                    details.append((feature, weight, False, "type error"))

            # Default: exact mismatch
            else:
                details.append((feature, weight, False,
                                f"mismatch: {repr(val_a)[:50]} vs {repr(val_b)[:50]}"))

    score = matched_weight / total_weight if total_weight > 0 else 0.0
    return score, details


# =============================================================================
# Union-Find for clustering same-device sessions
# =============================================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self):
        clusters = defaultdict(list)
        for i in range(len(self.parent)):
            clusters[self.find(i)].append(i)
        return list(clusters.values())


# =============================================================================
# Directory scanning
# =============================================================================

def scan_base_dir(base_dir):
    """
    Scan the base directory for pid-xxx folders.
    Only loads sessions from pid folders with >=2 dev_ subfolders.
    """
    records = []
    pid_folders = sorted([
        d for d in os.listdir(base_dir)
        if d.startswith("pid-") and os.path.isdir(os.path.join(base_dir, d))
    ])

    total_pid_count = len(pid_folders)
    print(f"Found {total_pid_count} pid folders total.")

    skipped_single = 0
    skipped_no_fpjs = 0
    multi_session_count = 0

    for pid_folder in pid_folders:
        pid_path = os.path.join(base_dir, pid_folder)
        pid = pid_folder.replace("pid-", "")

        dev_folders = sorted([
            d for d in os.listdir(pid_path)
            if d.startswith("dev_") and os.path.isdir(os.path.join(pid_path, d))
        ])

        if len(dev_folders) < 2:
            skipped_single += 1
            continue

        multi_session_count += 1
        print(f"  {pid_folder}: {len(dev_folders)} sessions")

        for dev_folder in dev_folders:
            dev_path = os.path.join(pid_path, dev_folder)
            fpjs_file = find_first_fpjs(dev_path)

            if fpjs_file is None:
                print(f"    [WARN] No fpjs_*.json in {dev_folder}, skipping.")
                skipped_no_fpjs += 1
                continue

            try:
                with open(fpjs_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                print(f"    [WARN] Failed to parse {fpjs_file}: {e}")
                continue

            features = extract_stable_features(data)
            folder_info = parse_dev_folder_name(dev_folder)
            _, user_agent, visitor_id = normalize_data(data)
            fpjs_basename = os.path.basename(fpjs_file)
            year = extract_year_from_fpjs_filename(fpjs_basename)

            records.append({
                "features": features,
                "metadata": {
                    "prolificPid": pid,
                    "deviceId": folder_info.get("deviceId", ""),
                    "sessionId": folder_info.get("sessionId", ""),
                    "userAgent": user_agent,
                    "visitorId": visitor_id,
                    "pid_folder": pid_folder,
                    "dev_folder": dev_folder,
                    "dev_path": dev_path,
                    "fpjs_file": fpjs_basename,
                    "year": year,
                },
            })

    print(f"\n  Total pid folders:           {total_pid_count}")
    print(f"  PIDs with >=2 sessions:      {multi_session_count}")
    print(f"  PIDs with only 1 session:    {skipped_single}")
    if skipped_no_fpjs:
        print(f"  Dev folders with no fpjs:    {skipped_no_fpjs}")

    return records


# =============================================================================
# Analysis helpers
# =============================================================================

def analyze_year_spans(records):
    """Analyze which PIDs have sessions spanning from 2025 to 2026."""
    pid_years = defaultdict(set)
    for rec in records:
        y = rec["metadata"]["year"]
        if y is not None:
            pid_years[rec["metadata"]["prolificPid"]].add(y)

    result = {
        "pid_years": pid_years,
        "spans_2025_2026": [],
        "only_2025": [],
        "only_2026": [],
        "other": [],
        "no_year": [],
    }
    for pid, years in sorted(pid_years.items()):
        if {2025, 2026}.issubset(years):
            result["spans_2025_2026"].append(pid)
        elif years == {2025}:
            result["only_2025"].append(pid)
        elif years == {2026}:
            result["only_2026"].append(pid)
        else:
            result["other"].append(pid)

    all_pids = set(r["metadata"]["prolificPid"] for r in records)
    result["no_year"] = [p for p in sorted(all_pids) if p not in pid_years]
    return result


def check_visitor_id_consistency(base_dir, pids_with_dev_paths):
    """
    For each PID, read ALL fpjs_*.json from same-device dev_ folders,
    extract visitorId, check consistency.

    pids_with_dev_paths: dict { pid: [dev_path, ...] }
    """
    consistent, inconsistent = [], []

    for pid in sorted(pids_with_dev_paths.keys()):
        all_vids = {}
        for dev_path in pids_with_dev_paths[pid]:
            for fpath in find_all_fpjs(dev_path):
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        vid = extract_visitor_id(json.load(fh))
                    if vid:
                        all_vids[os.path.relpath(fpath, base_dir)] = vid
                except Exception:
                    pass

        unique = set(all_vids.values())
        if len(unique) <= 1:
            consistent.append({
                "pid": pid,
                "visitorId": unique.pop() if unique else "N/A",
                "num_files": len(all_vids),
            })
        else:
            vtf = defaultdict(list)
            for fp, v in all_vids.items():
                vtf[v].append(fp)
            inconsistent.append({
                "pid": pid,
                "num_unique": len(unique),
                "num_files": len(all_vids),
                "groups": dict(vtf),
            })

    return consistent, inconsistent


def check_drawnapart(pids_with_dev_paths):
    """
    For each PID, count how many same-device dev_ folders contain
    a 'drawnapart' subfolder. Returns list of PIDs with >=2.
    """
    results = []
    for pid in sorted(pids_with_dev_paths.keys()):
        da_count = 0
        da_devs = []
        for dev_path in pids_with_dev_paths[pid]:
            da_path = os.path.join(dev_path, "drawnapart")
            if os.path.isdir(da_path):
                da_count += 1
                da_devs.append(os.path.basename(dev_path))
        if da_count >= 2:
            results.append({"pid": pid, "count": da_count, "dev_folders": da_devs})
    return results


# =============================================================================
# Copy logic
# =============================================================================

def copy_pid_folders(base_dir, dest_dir, full_copy_pids, partial_copy):
    """
    full_copy_pids: set of PIDs to copy entirely (SAME_DEVICE)
    partial_copy: dict { pid: [dev_folder_name, ...] } for MIXED
    """
    os.makedirs(dest_dir, exist_ok=True)
    copied, failed = 0, 0

    # Full copies (SAME_DEVICE)
    for pid in sorted(full_copy_pids):
        pf = f"pid-{pid}"
        src = os.path.join(base_dir, pf)
        dst = os.path.join(dest_dir, pf)
        if not os.path.isdir(src):
            print(f"  [WARN] Source not found: {src}")
            failed += 1
            continue
        if os.path.exists(dst):
            print(f"  [WARN] Already exists: {dst}")
            failed += 1
            continue
        try:
            shutil.copytree(src, dst)
            print(f"  [COPIED] {pf} (full)")
            copied += 1
        except Exception as e:
            print(f"  [ERROR] {pf}: {e}")
            failed += 1

    # Partial copies (MIXED - only same-device dev_ folders)
    for pid in sorted(partial_copy.keys()):
        pf = f"pid-{pid}"
        src_pid = os.path.join(base_dir, pf)
        dst_pid = os.path.join(dest_dir, pf)
        if not os.path.isdir(src_pid):
            print(f"  [WARN] Source not found: {src_pid}")
            failed += 1
            continue
        if os.path.exists(dst_pid):
            print(f"  [WARN] Already exists: {dst_pid}")
            failed += 1
            continue
        try:
            os.makedirs(dst_pid)
            dev_folders = partial_copy[pid]
            for df in dev_folders:
                s = os.path.join(src_pid, df)
                d = os.path.join(dst_pid, df)
                if os.path.isdir(s):
                    shutil.copytree(s, d)
            print(f"  [COPIED] {pf} (partial: {len(dev_folders)} dev folders)")
            copied += 1
        except Exception as e:
            print(f"  [ERROR] {pf}: {e}")
            failed += 1

    return copied, failed


# =============================================================================
# Report writer
# =============================================================================

class ReportWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.lines = []

    def write(self, text=""):
        self.lines.append(text)

    def section(self, title):
        self.write(f"\n{'='*70}")
        self.write(title)
        self.write(f"{'='*70}")

    def save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"\nReport saved -> {self.filepath}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Match browser fingerprints across sessions to identify same device."
    )
    parser.add_argument("base_dir",
                        help="Base directory containing pid-xxx folders")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Similarity threshold (default: 0.80)")
    parser.add_argument("--output", default="fingerprint_results.csv",
                        help="Output CSV file (default: fingerprint_results.csv)")
    parser.add_argument("--copy-same", default=None, metavar="DIR",
                        help="Copy SAME_DEVICE pid folders to this directory")
    parser.add_argument("--report", default=None, metavar="FILE",
                        help="Write stats report to this file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed match info for every pair")
    args = parser.parse_args()

    # Default report path
    if args.report is None:
        args.report = args.output.replace(".csv", "_report.txt")

    rpt = ReportWriter(args.report)

    # --- Scan ---
    print(f"Scanning: {args.base_dir}\n")
    records = scan_base_dir(args.base_dir)
    n = len(records)
    print(f"\nLoaded {n} sessions total.\n")

    if n < 2:
        print("Not enough sessions to compare.")
        sys.exit(0)

    # --- Year-span analysis ---
    year_stats = analyze_year_spans(records)

    rpt.section("YEAR-SPAN ANALYSIS")
    rpt.write(f"  Sessions spanning 2025 -> 2026:  {len(year_stats['spans_2025_2026'])} PIDs")
    rpt.write(f"  Sessions only in 2025:           {len(year_stats['only_2025'])} PIDs")
    rpt.write(f"  Sessions only in 2026:           {len(year_stats['only_2026'])} PIDs")
    if year_stats['other']:
        rpt.write(f"  Sessions in other years:         {len(year_stats['other'])} PIDs")
    if year_stats['no_year']:
        rpt.write(f"  No year info in filename:        {len(year_stats['no_year'])} PIDs")
    if year_stats['spans_2025_2026']:
        rpt.write(f"\n  PIDs spanning 2025-2026:")
        for pid in year_stats['spans_2025_2026']:
            rpt.write(f"    {pid}: years={sorted(year_stats['pid_years'][pid])}")

    # --- Group by pid ---
    pid_groups = defaultdict(list)
    for idx, rec in enumerate(records):
        pid_groups[rec["metadata"]["prolificPid"]].append(idx)

    all_pair_results = []
    summary_rows = []
    same_device_pids = set()           # full copy
    mixed_partial_copy = {}            # pid -> [dev_folder_names to copy]

    # Track all dev_paths that end up in same_device for later analysis
    same_device_dev_paths = defaultdict(list)  # pid -> [dev_path, ...]

    rpt.section("COMPARISON RESULTS (threshold={:.2f})".format(args.threshold))

    for pid in sorted(pid_groups.keys()):
        indices = pid_groups[pid]
        if len(indices) < 2:
            continue

        rpt.write(f"\n--- PID: {pid} ({len(indices)} sessions) ---")
        for idx in indices:
            m = records[idx]["metadata"]
            br = extract_browser_version(m["userAgent"])
            ys = str(m["year"]) if m["year"] else "?"
            rpt.write(f"  [{idx}] {m['dev_folder']}")
            rpt.write(f"       Browser: {br[0]} {br[1]} | Year: {ys} | File: {m['fpjs_file']}")

        # Pairwise comparison + union-find clustering
        local_n = len(indices)
        uf = UnionFind(local_n)
        all_same = True
        pair_scores = []

        for li, lj in combinations(range(local_n), 2):
            i, j = indices[li], indices[lj]
            score, details = compare_features(
                records[i]["features"], records[j]["features"]
            )
            same = score >= args.threshold
            if not same:
                all_same = False
            else:
                uf.union(li, lj)

            mi, mj = records[i]["metadata"], records[j]["metadata"]
            pair_scores.append(score)

            yi = str(mi["year"]) if mi["year"] else "?"
            yj = str(mj["year"]) if mj["year"] else "?"
            spans = "2025->2026" if {mi.get("year"), mj.get("year")} == {2025, 2026} else ""

            all_pair_results.append({
                "prolificPid": pid,
                "dev_a": mi["deviceId"],
                "dev_b": mj["deviceId"],
                "session_a": mi["sessionId"],
                "session_b": mj["sessionId"],
                "year_a": yi,
                "year_b": yj,
                "score": f"{score:.4f}",
                "same_device": same,
                "browser_a": " ".join(extract_browser_version(mi["userAgent"])),
                "browser_b": " ".join(extract_browser_version(mj["userAgent"])),
                "year_span": spans,
            })

            flag = "SAME" if same else "DIFF"
            st = f" [{spans}]" if spans else ""
            rpt.write(f"\n  [{flag}] {mi['deviceId']} <-> {mj['deviceId']}{st}")
            rpt.write(f"         Score: {score:.4f}")

            if args.verbose or not same:
                mismatches = [d for d in details if d[2] is False]
                skipped = [d for d in details if d[2] is None]
                if mismatches:
                    rpt.write(f"         Mismatches ({len(mismatches)}):")
                    for feat, w, _, note in sorted(mismatches, key=lambda x: -x[1]):
                        rpt.write(f"           - {feat} (w={w}): {note}")
                if skipped and args.verbose:
                    rpt.write(f"         Skipped ({len(skipped)}):")
                    for feat, w, _, note in skipped:
                        rpt.write(f"           - {feat} (w={w}): {note}")

        # Determine verdict
        if all_same:
            verdict = "SAME_DEVICE"
            same_device_pids.add(pid)
            for li in range(local_n):
                same_device_dev_paths[pid].append(
                    records[indices[li]]["metadata"]["dev_path"]
                )
            rpt.write(f"\n  >>> VERDICT: SAME_DEVICE <<<")
        else:
            # Check clusters via union-find
            clusters = uf.groups()
            same_clusters = [c for c in clusters if len(c) >= 2]

            if not same_clusters:
                verdict = "ALL_DIFFERENT"
                rpt.write(f"\n  >>> VERDICT: ALL_DIFFERENT <<<")
            else:
                verdict = "MIXED"
                # Collect dev folders from same-device clusters for partial copy
                devs_to_copy = []
                for cluster in same_clusters:
                    for li in cluster:
                        idx = indices[li]
                        devs_to_copy.append(records[idx]["metadata"]["dev_folder"])
                        same_device_dev_paths[pid].append(
                            records[idx]["metadata"]["dev_path"]
                        )
                mixed_partial_copy[pid] = devs_to_copy

                n_same = sum(1 for s in pair_scores if s >= args.threshold)
                n_diff = sum(1 for s in pair_scores if s < args.threshold)
                rpt.write(f"\n  >>> VERDICT: MIXED - {n_same} same, {n_diff} different <<<")
                rpt.write(f"      Copying {len(devs_to_copy)} same-device dev folders")

        min_s, max_s = min(pair_scores), max(pair_scores)
        avg_s = sum(pair_scores) / len(pair_scores)
        pys = year_stats["pid_years"].get(pid, set())
        yl = ("2025->2026" if {2025, 2026}.issubset(pys)
              else ",".join(str(y) for y in sorted(pys)) if pys
              else "?")

        summary_rows.append({
            "prolificPid": pid,
            "num_sessions": len(indices),
            "num_pairs": len(pair_scores),
            "min_score": f"{min_s:.4f}",
            "max_score": f"{max_s:.4f}",
            "avg_score": f"{avg_s:.4f}",
            "verdict": verdict,
            "year_span": yl,
        })

    # --- Write CSVs ---
    csv_pairs = args.output
    with open(csv_pairs, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "prolificPid", "dev_a", "dev_b", "session_a", "session_b",
            "year_a", "year_b", "score", "same_device",
            "browser_a", "browser_b", "year_span",
        ])
        w.writeheader()
        w.writerows(all_pair_results)
    print(f"Pairwise results -> {csv_pairs}")

    csv_summary = csv_pairs.replace(".csv", "_summary.csv")
    with open(csv_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "prolificPid", "num_sessions", "num_pairs",
            "min_score", "max_score", "avg_score", "verdict", "year_span",
        ])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Summary          -> {csv_summary}")

    # --- VisitorId consistency ---
    if same_device_dev_paths:
        consistent, inconsistent = check_visitor_id_consistency(
            args.base_dir, same_device_dev_paths
        )

        rpt.section("VISITOR-ID CONSISTENCY (same-device sessions)")
        rpt.write(f"  PIDs checked:                {len(same_device_dev_paths)}")
        rpt.write(f"  visitorId ALL consistent:    {len(consistent)}")
        rpt.write(f"  visitorId INCONSISTENT:      {len(inconsistent)}")

        if consistent:
            rpt.write(f"\n  Consistent PIDs:")
            for c in consistent:
                rpt.write(f"    {c['pid']}: visitorId={c['visitorId']} "
                          f"({c['num_files']} files)")

        if inconsistent:
            rpt.write(f"\n  INCONSISTENT PIDs:")
            for ic in inconsistent:
                rpt.write(f"    {ic['pid']}: {ic['num_unique']} unique visitorIds "
                          f"across {ic['num_files']} files")
                for vid, files in ic['groups'].items():
                    rpt.write(f"      visitorId={vid} ({len(files)} files):")
                    for fp in files[:3]:
                        rpt.write(f"        - {fp}")
                    if len(files) > 3:
                        rpt.write(f"        ... and {len(files)-3} more")

        csv_vid = csv_pairs.replace(".csv", "_visitorid.csv")
        with open(csv_vid, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "prolificPid", "verdict", "num_unique_visitorIds",
                "num_fpjs_files", "visitorIds",
            ])
            w.writeheader()
            for c in consistent:
                w.writerow({
                    "prolificPid": c["pid"],
                    "verdict": "CONSISTENT",
                    "num_unique_visitorIds": 1,
                    "num_fpjs_files": c["num_files"],
                    "visitorIds": c["visitorId"],
                })
            for ic in inconsistent:
                w.writerow({
                    "prolificPid": ic["pid"],
                    "verdict": "INCONSISTENT",
                    "num_unique_visitorIds": ic["num_unique"],
                    "num_fpjs_files": ic["num_files"],
                    "visitorIds": " | ".join(sorted(ic["groups"].keys())),
                })
        print(f"VisitorId report -> {csv_vid}")

    # --- DrawnApart statistics ---
    da_results = check_drawnapart(same_device_dev_paths)

    rpt.section("DRAWNAPART STATISTICS (same-device sessions)")
    rpt.write(f"  PIDs with same-device sessions:        {len(same_device_dev_paths)}")
    rpt.write(f"  PIDs with >=2 drawnapart folders:       {len(da_results)}")

    if da_results:
        rpt.write(f"\n  PIDs with >=2 drawnapart:")
        for da in da_results:
            rpt.write(f"    {da['pid']}: {da['count']} drawnapart folders")
            for df in da['dev_folders']:
                rpt.write(f"      - {df}")

    # --- Final overview ---
    total_pids = len(summary_rows)
    same_count = sum(1 for r in summary_rows if r["verdict"] == "SAME_DEVICE")
    diff_count = sum(1 for r in summary_rows if r["verdict"] == "ALL_DIFFERENT")
    mixed_count = sum(1 for r in summary_rows if r["verdict"] == "MIXED")
    cross_year = sum(1 for r in summary_rows if r["year_span"] == "2025->2026")

    rpt.section("FINAL OVERVIEW")
    rpt.write(f"  PIDs with >=2 sessions:      {total_pids}")
    rpt.write(f"  SAME_DEVICE:                 {same_count}")
    rpt.write(f"  ALL_DIFFERENT:               {diff_count}")
    rpt.write(f"  MIXED:                       {mixed_count}")
    rpt.write(f"    - MIXED same-device devs copied: {len(mixed_partial_copy)} PIDs")
    rpt.write(f"  Spanning 2025->2026:         {cross_year}")
    rpt.write(f"  DrawnApart (>=2 folders):    {len(da_results)}")

    if diff_count > 0 or mixed_count > 0:
        rpt.write(f"\n  PIDs with different devices:")
        for r in summary_rows:
            if r["verdict"] != "SAME_DEVICE":
                rpt.write(f"    {r['prolificPid']}: {r['verdict']} "
                          f"(sessions={r['num_sessions']}, "
                          f"scores={r['min_score']}~{r['max_score']}, "
                          f"years={r['year_span']})")

    # Print final overview to stdout too
    print(f"\n  PIDs with >=2 sessions:  {total_pids}")
    print(f"  SAME_DEVICE:             {same_count}")
    print(f"  ALL_DIFFERENT:           {diff_count}")
    print(f"  MIXED:                   {mixed_count}")
    print(f"  Spanning 2025->2026:     {cross_year}")
    print(f"  DrawnApart (>=2):        {len(da_results)}")

    # --- Save report ---
    rpt.save()

    # --- Copy ---
    if args.copy_same and (same_device_pids or mixed_partial_copy):
        print(f"\nCopying to {args.copy_same} ...")
        copied, failed = copy_pid_folders(
            args.base_dir, args.copy_same,
            same_device_pids, mixed_partial_copy
        )
        print(f"  Copied: {copied}  Failed: {failed}")
    elif args.copy_same:
        print("\n  No same-device PIDs to copy.")


if __name__ == "__main__":
    main()