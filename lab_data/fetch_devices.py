#!/usr/bin/env python3
"""Download the lab device archives that are not bundled with the repository.

Five archives ship inside the repo (enough for `pipeline/demo.sh`); the other
55, about 650 MB, are attached to a GitHub release.

Usage:
    python3 lab_data/fetch_devices.py            # fetch everything missing
    python3 lab_data/fetch_devices.py --list     # show what is missing
    python3 lab_data/fetch_devices.py --category browser-defense

Only the standard library is used, so this runs on a bare Python 3.8+.
"""

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request

REPO = "xlin29/learned-fp"
TAG = "lab-data-v1"
BASE = f"https://github.com/{REPO}/releases/download/{TAG}"

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifest.json")
DEVICES = os.path.join(HERE, "devices")


def load_rows():
    with open(MANIFEST, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = data.get("devices", data) if isinstance(data, dict) else data
    return list(rows.values()) if isinstance(rows, dict) else list(rows)


def human(n):
    return f"{n / 1048576:.1f} MB"


def ssl_context():
    """Default TLS context; uses certifi's CA bundle when it is installed, which
    covers Python builds that ship without system certificates."""
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except Exception:
        pass
    return ctx


def download(name, size, dest):
    url = f"{BASE}/{name}"
    tmp = dest + ".part"
    done = 0
    try:
        with urllib.request.urlopen(url, context=ssl_context()) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or size or 0)
            while True:
                block = resp.read(1 << 16)
                if not block:
                    break
                out.write(block)
                done += len(block)
                pct = min(100, 100 * done / (total or 1))
                sys.stdout.write(f"\r  {name[:60]:60s} {pct:5.1f}%")
                sys.stdout.flush()
    except urllib.error.HTTPError as exc:
        sys.stdout.write("\r")
        print(f"  FAILED {name}: HTTP {exc.code}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    except (urllib.error.URLError, OSError) as exc:
        sys.stdout.write("\r")
        print(f"  FAILED {name}: {getattr(exc, 'reason', exc)}")
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            print("  This Python has no CA certificates. Install them (macOS python.org "
                  "builds: run 'Install Certificates.command') or `pip install certifi`.")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, dest)
    sys.stdout.write("\r")
    print(f"  ok     {name[:60]:60s} {human(os.path.getsize(dest))}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="list the missing archives and exit")
    ap.add_argument("--category", default=None,
                    help="restrict to one manifest category "
                         "(browser-default / browser-defense / extension)")
    args = ap.parse_args()

    os.makedirs(DEVICES, exist_ok=True)
    rows = load_rows()
    if args.category:
        rows = [r for r in rows if r.get("category") == args.category]

    missing = []
    for r in rows:
        name = r["dir_name"] + ".tar.gz"
        if not os.path.exists(os.path.join(DEVICES, name)):
            missing.append((name, r.get("archive_bytes", 0)))

    if not missing:
        print(f"All {len(rows)} archives are already present in {DEVICES}.")
        return 0

    total = sum(size for _, size in missing)
    print(f"{len(missing)} archive(s) missing, {human(total)} to download "
          f"from {BASE}")
    if args.list:
        for name, size in missing:
            print(f"  {name}  {human(size)}")
        return 0

    failed = 0
    for name, size in missing:
        if not download(name, size, os.path.join(DEVICES, name)):
            failed += 1
    if failed:
        print(f"\n{failed} download(s) failed. Re-run to retry only those.")
        return 1
    what = f"all {len(rows)} {args.category} archives" if args.category else f"the full {len(rows)}-device gallery"
    print(f"\nDone. {DEVICES} now holds {what}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
