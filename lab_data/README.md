# Lab fingerprint dataset

The public in-lab portion of the LearnedFP gallery (paper §6.1 /
Appendix D Table 6): 12 browser configurations × 5 instances =
60 browser instances, on 13 physical devices. Five archives ship with the repository and the rest are one
command away — see §"Getting the full gallery". Crowdsourced captures are
withheld — see top-level README §"What's NOT released".

## Getting the full gallery

Five archives (48 MB, one per demo category) ship inside the repository so
that `pipeline/demo.sh` runs on a clean checkout with no network access.
The other 55 are attached to a GitHub release, because together they are
about 640 MB and would otherwise be cloned by everyone who only wants the
code.

```bash
python3 lab_data/fetch_devices.py            # fetch all 55 missing archives
python3 lab_data/fetch_devices.py --list     # show what is missing first
python3 lab_data/fetch_devices.py --category browser-defense
```

The script needs only the standard library, skips archives already on disk,
and can be re-run after an interrupted download. `manifest.json` describes
all 60 instances either way; the `bundled` field marks the five that come with
the repository.

Each archive expands to:

```
dev_<id>__<device>__<config>/
├── meta.json            {deviceId, device_label, browser_config, category}
├── visit1/              900 .rgba (100 sub-sessions × 9 keys, paper-named)
├── visit2/              9 .rgba   (1 sub-session × 9 keys; return visit)
├── visit3/              9 .rgba
└── visit4/              9 .rgba
```

Filenames inside each visit dir use the paper-facing key names:
`S<N>_<KeyName>_<idx>_<timestamp>.rgba`

- `raw_faces` → `EmojiFace`
- `raw_persons` → `EmojiPeople`
- `raw_travel` → `EmojiObject`
- `raw_hands` → `EmojiHand`
- `raw_randomFont` → `GlyphSet`
- `a-randomString` → `RandChar`
- `moire` → `MoireLine`
- `gradQuantSteps` → `GradStep`
- `shadowBlurProbe` → `ShadowBlur`

## Reading the archives

The evaluation pipeline reads the archives in place through Python's
stdlib `tarfile`; nothing has to be extracted. To look at one device by hand:

```bash
tar -xzf lab_data/devices/dev_015d9a38e4a54b28a78c52324ef60974__hp-omen-30l__chrome-standard.tar.gz -C /tmp
ls /tmp/dev_015d9a38*/visit1/ | head
```

## Defense coverage (paper Table 4)

Three category buckets in [`manifest.json`](manifest.json):

- `browser-default` — Chrome / Standard, Firefox / Standard, Safari / Standard
- `browser-defense` — Brave / Default, Firefox / ETP Strict, Samsung Internet / SAT Strict, Safari / Private, Tor / Default
- `extension` — Canvas FP Defender (Chrome), Canvas Blocker (Chrome), CanvasBlocker by kkapsner (Firefox), Fingerprint Spoofer (Chrome)

Five instances per configuration → 60 browser instances total, on the 13
physical devices Table 6 lists. The exact mapping
from the on-disk archive name to paper Table 4 / Table 6 is in
`manifest.json`.

## How this was produced

These archives are derived from a controlled lab capture run, not from
crowdsourced participants. The full pipeline that produced them is
preserved in [`build.py`](build.py) — it takes a raw rsync of the source
captures and emits the public layout you see here. The transformations
applied:

1. Drop derived / baseline files (`drawnapart/`, `fpjs_*.json`,
   `pixels_index.json`, `client_batch_logs.ndjson`) — those belong to
   the separately released baselines or are reconstructable from the
   `.rgba` ground truth.
2. Sanitize each `meta.json`: replace the misleading
   `prolificPid`/`studyId`/`sessionId` fields (the values were lab labels,
   not real Prolific IDs) with explicit `device_label` / `browser_config` /
   `category`.
3. Rename every `.rgba` from the internal-key form
   (`S1_raw_faces_1_bs134_ms868677983_2026-…`) to the paper-key form
   (`S1_EmojiFace_1_2026-…`), dropping the `_bs<N>_ms<N>_` suffixes which
   only encoded server-side validation salts at capture time.
4. Rename each device dir to its paper-aligned slug pair
   (`dev_<id>__<device-slug>__<browser-config-slug>`).
5. Per-device `tar.gz` so the artifact stays under repo size limits and
   a clone does not carry 50 000+ small files in its Git index.

The raw captures themselves are not distributed; with them in place the
archives regenerate byte-identically (up to gzip mtimes) with:

```bash
python3 lab_data/build.py --raw <RAW_CAPTURES_DIR>
```

DRAWNAPART and FingerprintJS captures are not part of this dataset.
