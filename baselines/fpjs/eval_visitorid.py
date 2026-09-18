#!/usr/bin/env python3
"""FingerprintJS baseline: the visitorId counts of paper §5.5.

  enrollment collisions   enrolled devices whose visitorId is also emitted by
                          another device, and the largest such group
  temporal instability    returning devices whose return visit shares no
                          visitorId with their enrollment
  cold-start collisions   devices first seen in the return campaign whose
                          visitorId is held by an enrolled device or by another
                          new device

The canvas-pixel comparison of §5.5 is not computed here. extract_vid, load and
vids are reproduced from the scripts that produced the paper's numbers.

Input: two roots, one per campaign, each holding dev_*/fpjs_*.json (the
collector's fpjs.v1 layout or the older reorderedResult layout). A device is
its directory name unless --device-key gives a regex whose first group is the
identity (the paper's runs used __pid-([^_]+)__).

    python baselines/fpjs/eval_visitorid.py --t0-root <ENROLL_ROOT> --t1-root <RETURN_ROOT>
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path


# ---- reproduced unchanged from the scripts that produced the paper's numbers ----

def extract_vid(o):
    if not isinstance(o, dict): return None
    for h in (o.get("reorderedResult"), o.get("reordered_result"), o.get("fingerprint")):
        if isinstance(h, dict):
            v = h.get("visitorId") or h.get("visitor_id")
            if isinstance(v, str) and v.strip(): return v.strip()
    v = o.get("visitorId") or o.get("visitor_id")
    return v.strip() if isinstance(v, str) and v.strip() else None

def load(p):
    try:
        with open(p, encoding="utf-8", errors="ignore") as f: return json.load(f)
    except Exception: return None

def vids(dev_dir):
    files = glob.glob(str(Path(dev_dir) / "fpjs_*.json"))
    out = []
    for f in sorted(set(files), key=os.path.basename):
        v = extract_vid(load(f))
        if v: out.append(v)
    return out

# ---------------------------------------------------------------------------


def scan(root, key_re):
    """identity -> list of visitorIds emitted across all of that identity's files."""
    by_key = defaultdict(list)
    for d in sorted(Path(root).glob("dev_*")):
        if not d.is_dir():
            continue
        m = key_re.search(d.name)
        if not m:
            continue
        by_key[m.group(1) if m.groups() else m.group(0)] += vids(d)
    return {k: v for k, v in by_key.items() if v}


def index(by_key):
    idx = defaultdict(set)
    for k, vs in by_key.items():
        for v in set(vs):
            idx[v].add(k)
    return idx


def enrollment_collisions(t0):
    """Devices sharing a visitorId with a different identity; largest sharing group."""
    idx = index(t0)
    hit = {k for k, vs in t0.items() if any(idx[v] - {k} for v in set(vs))}
    groups = [len(ks) for ks in idx.values() if len(ks) > 1]
    return len(hit), max(groups) if groups else 0


def temporal_instability(t0, t1):
    """Returning identities whose return visitorIds share nothing with enrollment."""
    returning = [k for k in t1 if k in t0]
    unstable = [k for k in returning if not (set(t1[k]) & set(t0[k]))]
    return len(returning), len(unstable)


def coldstart_collisions(t0, t1):
    """New identities colliding with an enrolled identity or with another new one."""
    new = {k: vs for k, vs in t1.items() if k not in t0}
    t0_idx, cs_idx = index(t0), index(new)
    hit_t0 = {k for k, vs in new.items() if any(t0_idx.get(v) for v in set(vs))}
    hit_cs = {k for k, vs in new.items() if any(cs_idx.get(v, set()) - {k} for v in set(vs))}
    return len(new), len(hit_t0), len(hit_cs), len(hit_t0 | hit_cs)


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t0-root", required=True, type=Path, help="enrollment-campaign device root")
    ap.add_argument("--t1-root", required=True, type=Path, help="return-campaign device root")
    ap.add_argument("--device-key", default=r"(.*)",
                    help="regex over the dev_* directory name; group 1 is the device identity "
                         "(default: the whole name; the paper's runs used '__pid-([^_]+)__')")
    ap.add_argument("--out", type=Path, default=None, help="also write the counts as JSON")
    args = ap.parse_args()

    key_re = re.compile(args.device_key)
    t0, t1 = scan(args.t0_root, key_re), scan(args.t1_root, key_re)
    print(f"[data] enrollment identities with FingerprintJS data: {len(t0)}")
    print(f"[data] return-campaign identities with FingerprintJS data: {len(t1)}")

    n_coll, largest = enrollment_collisions(t0)
    n_ret, n_unst = temporal_instability(t0, t1)
    n_new, cs_t0, cs_cs, cs_any = coldstart_collisions(t0, t1)

    print()
    print(f"enrollment collisions   {n_coll:5d} / {len(t0):5d}  ({pct(n_coll, len(t0)):6.2f}%)   largest group {largest}")
    print(f"temporal instability    {n_unst:5d} / {n_ret:5d}  ({pct(n_unst, n_ret):6.2f}%)   returning identities with a changed visitorId")
    print(f"cold-start collisions   {cs_any:5d} / {n_new:5d}  ({pct(cs_any, n_new):6.2f}%)   "
          f"with enrolled {cs_t0} ({pct(cs_t0, n_new):.2f}%), with another new device {cs_cs} ({pct(cs_cs, n_new):.2f}%)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "enrollment": {"identities": len(t0), "colliding": n_coll, "largest_group": largest},
            "temporal": {"returning": n_ret, "changed_visitorid": n_unst},
            "coldstart": {"new": n_new, "collide_with_enrolled": cs_t0,
                          "collide_with_new": cs_cs, "collide_any": cs_any},
        }, indent=2) + "\n")
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
