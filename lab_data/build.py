#!/usr/bin/env python3
"""Build the `lab_data/devices/*.tar.gz` archives from the raw lab captures
(one directory per device under --raw, default .lab_data_raw/ at the repo root).

For each lab device:
  1. Rewrite the directory name from the internal label to a slug-form pair
     (paper Table 6 device + paper Table 4 browser config).
  2. Sanitize meta.json — replace the misleading prolificPid / studyId /
     sessionId fields (the values are lab labels, not real Prolific IDs)
     with explicit device_label / browser_config / category fields.
  3. Rename every .rgba file from the internal key (raw_faces, moire,
     a-randomString, …) to its paper-facing key (EmojiFace, MoireLine,
     GlyphSet, …) and drop the _bs<N>_ms<N>_ suffixes, which encode
     server-side validation salts that matter only at capture time.
  4. tar.gz the result into ./devices/<new-name>.tar.gz.

Also emits manifest.json with one row per device.

Run:
    python3 lab_data/build.py --raw <RAW_CAPTURES_DIR>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Mappings — confirmed against paper Table 4 (browser configs) and
# Table 6 (lab device list). Where an internal label is ambiguous, the user
# disambiguated:
#   HPWindows         → HP 14-dk1xxx (AMD Radeon iGPU)
#   WindowsHPi7       → HP OMEN 30L  (i7-10700K)
#   MacBookPro13      → MacBook Pro M2 Max
#   Firefox           → Firefox / ETP Strict
#   FirefoxStandard   → Firefox / Standard
#   Safari            → Safari / Private
#   SafariStandard    → Safari / Standard
#   CanvasFingerprintDefender = FingerprintDefender → Canvas FP Defender
# ---------------------------------------------------------------------------

DEVICE_LABELS: dict[str, dict] = {
    "MacbookProM4":      {"slug": "macbook-pro-m4",      "paper": "MacBook Pro (M4)"},
    "MacBookPro13":      {"slug": "macbook-pro-m2-max",  "paper": "MacBook Pro (M2 Max)"},
    "MacBookAirM1":      {"slug": "macbook-air-m1",      "paper": "MacBook Air (M1)"},
    "LenovoWindowsi7":   {"slug": "lenovo-loq-15irx10",  "paper": "Lenovo LOQ 15IRX10 (i7-13650HX)"},
    "LenovoWIndowsi7":   {"slug": "lenovo-loq-15irx10",  "paper": "Lenovo LOQ 15IRX10 (i7-13650HX)"},
    "WindowsHPi7":       {"slug": "hp-omen-30l",         "paper": "HP OMEN 30L (i7-10700K)"},
    "HPWindows":         {"slug": "hp-14-dk1xxx",        "paper": "HP 14-dk1xxx (AMD Radeon iGPU)"},
    "iPhone17":          {"slug": "iphone-17",           "paper": "iPhone 17 (A19)"},
    "iPhoneAir":         {"slug": "iphone-air",          "paper": "iPhone Air (A19 Pro)"},
    "SamsungA15":        {"slug": "galaxy-a15",          "paper": "Galaxy A15"},
    "SamsungTabA11":     {"slug": "galaxy-tab-a11-plus", "paper": "Galaxy Tab A11+"},
    "SamsungA50":        {"slug": "galaxy-a50",          "paper": "Galaxy A50"},
    "SamsungS21":        {"slug": "galaxy-s21-fe-5g",    "paper": "Galaxy S21 FE 5G"},
    "SamsungA56":        {"slug": "galaxy-a56-5g",       "paper": "Galaxy A56 5G"},
}

BROWSER_CONFIGS: dict[str, dict] = {
    "Chrome":                     {"slug": "chrome-standard",            "paper": "Chrome / Standard",                     "category": "browser-default"},
    "FirefoxStandard":            {"slug": "firefox-standard",           "paper": "Firefox / Standard",                    "category": "browser-default"},
    "Firefox":                    {"slug": "firefox-etp-strict",         "paper": "Firefox / ETP Strict",                  "category": "browser-defense"},
    "SafariStandard":             {"slug": "safari-standard",            "paper": "Safari / Standard",                     "category": "browser-default"},
    "Safari":                     {"slug": "safari-private",             "paper": "Safari / Private",                      "category": "browser-defense"},
    "Brave":                      {"slug": "brave",                      "paper": "Brave / Default",                       "category": "browser-defense"},
    "Tor":                        {"slug": "tor",                        "paper": "Tor / Default",                         "category": "browser-defense"},
    "Tor15":                      {"slug": "tor",                        "paper": "Tor / Default",                         "category": "browser-defense"},
    "SamsungInternet":            {"slug": "samsung-sat-strict",         "paper": "Samsung Internet / SAT Strict",         "category": "browser-defense"},
    "CanvasFingerprintDefender":  {"slug": "canvas-fp-defender-chrome",  "paper": "Canvas FP Defender (Chrome)",           "category": "extension"},
    "FingerprintDefender":        {"slug": "canvas-fp-defender-chrome",  "paper": "Canvas FP Defender (Chrome)",           "category": "extension"},
    "ChromeCanvasBlocker":        {"slug": "canvas-blocker-chrome",      "paper": "Canvas Blocker (Chrome)",               "category": "extension"},
    "FirefoxCanvasBlocker":       {"slug": "canvasblocker-kkapsner-ff",  "paper": "CanvasBlocker by kkapsner (Firefox)",   "category": "extension"},
    "FingerprintSpoofer":         {"slug": "fingerprint-spoofer-chrome", "paper": "Fingerprint Spoofer (Chrome)",          "category": "extension"},
}

KEY_RENAMES: dict[str, str] = {
    "raw_faces":       "EmojiFace",
    "raw_persons":     "EmojiPeople",
    "raw_travel":      "EmojiObject",
    "raw_hands":       "EmojiHand",
    "raw_randomFont":  "GlyphSet",
    "a-randomString":  "RandChar",
    "moire":           "MoireLine",
    "gradQuantSteps":  "GradStep",
    "shadowBlurProbe": "ShadowBlur",
}

# Source filename pattern. Captures S<N>, <key> (one of the 9), the
# trailing 1-9 index, and the timestamp. The optional _bs<N>_ms<N>_
# segment used to encode the per-session brightness-shift code and
# mask salt — both load-bearing only at capture time, so we strip them.
RGBA_FILENAME_RE = re.compile(
    r"^(?P<sess>S\d+)_"
    r"(?P<key>raw_faces|raw_persons|raw_travel|raw_hands|raw_randomFont"
    r"|a-randomString|moire|gradQuantSteps|shadowBlurProbe)_"
    r"(?P<idx>\d+)"
    r"(?:_bs-?\d+)?(?:_ms-?\d+)?_"
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3})\.rgba$"
)

# Dir name pattern: dev_<32hex>__pid-<X>__sid-<Y>
DEV_DIR_RE = re.compile(r"^dev_(?P<id>[0-9a-f]{32})__pid-(?P<pid>[^_]+(?:_[^_]+)*?)__sid-(?P<sid>.+)$")


def slugify_device(raw_dir_name: str) -> tuple[str, dict]:
    """Map a source device dir name to (new-slug-name, manifest-row)."""
    m = DEV_DIR_RE.match(raw_dir_name)
    if not m:
        raise ValueError(f"unparseable device dir: {raw_dir_name}")
    dev_id, pid, sid = m["id"], m["pid"], m["sid"]
    if pid not in DEVICE_LABELS:
        raise KeyError(f"unknown device label '{pid}' (dir: {raw_dir_name})")
    if sid not in BROWSER_CONFIGS:
        raise KeyError(f"unknown browser config '{sid}' (dir: {raw_dir_name})")
    dev = DEVICE_LABELS[pid]
    cfg = BROWSER_CONFIGS[sid]
    new_name = f"dev_{dev_id}__{dev['slug']}__{cfg['slug']}"
    row = {
        "device_id": f"dev_{dev_id}",
        "dir_name":  new_name,
        "device_label": dev["paper"],
        "browser_config": cfg["paper"],
        "category": cfg["category"],
    }
    return new_name, row


def rename_rgba(filename: str) -> str | None:
    """Rename one .rgba file from internal-key form to paper-key form."""
    m = RGBA_FILENAME_RE.match(filename)
    if not m:
        return None
    return f"{m['sess']}_{KEY_RENAMES[m['key']]}_{m['idx']}_{m['ts']}.rgba"


def sanitize_meta(raw: dict, manifest_row: dict) -> dict:
    """Rewrite meta.json so the misleading 'prolificPid'/'studyId'/'sessionId'
    fields don't suggest crowdsourced participant data."""
    return {
        "deviceId":       raw.get("deviceId"),
        "device_label":   manifest_row["device_label"],
        "browser_config": manifest_row["browser_config"],
        "category":       manifest_row["category"],
    }


def build_one_device(src_dir: Path, out_archive: Path, manifest_row: dict) -> dict:
    """Transform one source device dir into a tar.gz archive.

    Returns stats {visit_<N>_count, total_rgba, archive_bytes}.
    """
    new_name = manifest_row["dir_name"]
    stats: dict = {"total_rgba": 0}

    with tempfile.TemporaryDirectory(prefix="lab_data_build_") as tmp:
        staged = Path(tmp) / new_name
        staged.mkdir(parents=True)

        # meta.json (sanitized)
        try:
            raw_meta = json.loads((src_dir / "meta.json").read_text())
        except FileNotFoundError:
            raw_meta = {}
        (staged / "meta.json").write_text(
            json.dumps(sanitize_meta(raw_meta, manifest_row), indent=2)
        )

        # visit dirs
        for visit in sorted(src_dir.iterdir()):
            if not visit.is_dir() or not visit.name.startswith("visit"):
                continue
            visit_dst = staged / visit.name
            visit_dst.mkdir()
            count = 0
            for entry in visit.iterdir():
                if entry.suffix != ".rgba":
                    continue
                new_fname = rename_rgba(entry.name)
                if new_fname is None:
                    print(f"  ! skip (unrecognized): {entry.name}", file=sys.stderr)
                    continue
                shutil.copyfile(entry, visit_dst / new_fname)
                count += 1
            stats[f"{visit.name}_count"] = count
            stats["total_rgba"] += count

        # tar.gz
        out_archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out_archive, "w:gz") as tf:
            tf.add(staged, arcname=new_name)
        stats["archive_bytes"] = out_archive.stat().st_size

    return stats


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description="Package raw lab captures into lab_data/devices/*.tar.gz and rewrite manifest.json.")
    ap.add_argument("--raw", default=str(repo_root / ".lab_data_raw"),
                    help="directory of raw per-device captures (default: .lab_data_raw/ at the repo root)")
    ap.add_argument("--out", default=str(repo_root / "lab_data" / "devices"),
                    help="where the per-device archives are written (default: lab_data/devices/)")
    args = ap.parse_args()
    raw_root  = Path(args.raw)
    out_root  = Path(args.out)
    if not raw_root.exists():
        print(f"raw input not found: {raw_root}", file=sys.stderr)
        return 1
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    src_dirs = sorted(d for d in raw_root.iterdir() if d.is_dir() and d.name.startswith("dev_"))
    print(f"[build] found {len(src_dirs)} source device dirs")

    total_archive_bytes = 0
    total_rgba = 0
    for src_dir in src_dirs:
        try:
            new_name, row = slugify_device(src_dir.name)
        except (ValueError, KeyError) as e:
            print(f"[build] SKIP {src_dir.name}: {e}", file=sys.stderr)
            continue
        out_archive = out_root / f"{new_name}.tar.gz"
        stats = build_one_device(src_dir, out_archive, row)
        row.update(stats)
        manifest_rows.append(row)
        total_archive_bytes += stats["archive_bytes"]
        total_rgba += stats["total_rgba"]
        print(f"  [{len(manifest_rows):2d}/{len(src_dirs)}] {new_name}  "
              f"({stats['total_rgba']} rgba, {stats['archive_bytes']/1024:.0f} KB)")

    manifest_path = repo_root / "lab_data" / "manifest.json"
    manifest_path.write_text(json.dumps({
        "device_count":        len(manifest_rows),
        "total_rgba_samples":  total_rgba,
        "archive_total_bytes": total_archive_bytes,
        "key_paper_names":     sorted(set(KEY_RENAMES.values())),
        "category_counts":     _category_counts(manifest_rows),
        "devices":             manifest_rows,
    }, indent=2))
    print(f"[build] wrote manifest: {manifest_path}")
    print(f"[build] {len(manifest_rows)} devices · {total_rgba} samples · "
          f"{total_archive_bytes/1024/1024:.1f} MB total")
    return 0


def _category_counts(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r["category"]] = out.get(r["category"], 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
