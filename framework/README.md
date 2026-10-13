# LearnedFP Framework

Collection-side of LearnedFP: server-seeded probe generation, browser
rendering, raw sample extraction, upload, freshness validation (paper §4).
Collection only: there is no identification endpoint and no trained encoder
here. One Node service serves three tabs:

- **LearnedFP on Canvas** — 9 rendering keys. The four Emoji keys and
  GlyphSet upload raw RGBA bytes; the other four upload lossless PNG, which
  the server decodes, so all nine are stored as raw RGBA (paper §4.2).
- **LearnedFP on Web Audio** — 12 probes rendered in `OfflineAudioContext`,
  raw `Float32` upload.
- **Browser Defense Probe** — the `rgba(100,150,200,0.5)` fill read twice
  (Figure 7a) plus a second semi-transparent canvas with anti-aliased edges
  read directly and through an `OffscreenCanvas` round-trip, classified by
  the rules of Figure 7b; reports the Table 4 characterization axes for the
  active defense.

## Quick start (Docker, ~2 min)

```bash
cd framework
docker compose up --build
```

Then open <http://localhost:3000>.

- Click **Run canvas probes** → captures 100 samples.
- Click **Run audio probes** → captures 60 samples.
- Click **Show images** to view the 9 RGBA keys.
- Click **Show saved data** to inspect the on-disk JSON files
  (`samples_preview.ndjson` + `validation.ndjson`).

Uploaded samples land in `./local-data/` on the host so you can verify
end-to-end without entering the container:

```bash
find local-data -type f -name "*.rgba" | head            # 9 × 100 canvas samples
find local-data -type f -name "*.f32"  | head            # 12 × 60 audio samples
cat local-data/learnedfp/save/dev_*/visit1/samples_preview.ndjson  | head
cat local-data/learnedfp/save/dev_*/visit1/validation.ndjson   | head
```

To stop and reset:

```bash
docker compose down
rm -rf local-data
```

## Bare-Node setup (alternative)

Requires Node ≥ 20 and the native deps for `canvas` / `sharp`:

- macOS: `brew install cairo pango libpng jpeg giflib librsvg`
- Debian/Ubuntu: `apt install libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev`

```bash
cd framework/server
npm ci
PERSIST_DIR=../local-data node index.js --port 3000 --open=false
```

## What it does

1. **Server seeds a per-session drawing plan for each rendering key** — canvas: `server/index.js` `/bootstrap` + `server/templates/frozen_template.js`; audio: `server/audio.js` `/audio/bootstrap` + `server/templates/audio_frozen_template.js`
2. **Browser renders each key client-side and reads back raw bytes** — canvas: `client/canvas.html` (`getImageData()`); audio: `client/audio.html` (`OfflineAudioContext.startRendering`)
3. **Raw buffer + lightweight metadata uploaded back** — canvas: `/save-batch`; audio: `/audio/save-batch`
4. **Server validates the freshness marker and persists the sample** — `server/index.js` + `server/audio.js` ingestion paths
5. **Inspect collected samples** — `client/images.html` (`/images`, `/image`), `/saved-data`, `/audio/saved-data`

The 9 canvas rendering keys (paper §4.1 / Figure 2), internal name →
paper name. The internal names are the ones the code uses everywhere —
plan keys, canvas ids, upload filenames, `validation.ndjson`:

- `raw_faces` → `EmojiFace`
- `raw_persons` → `EmojiPeople`
- `raw_travel` → `EmojiObject`
- `raw_hands` → `EmojiHand`
- `a-randomString` → `RandChar`
- `moire` → `MoireLine`
- `raw_randomFont` → `GlyphSet`
- `gradQuantSteps` → `GradStep`
- `shadowBlurProbe` → `ShadowBlur`

The 12 Web Audio probes (paper §7.1): `oscillatorMix`, `biquadChain`,
`compressorProbe`, `waveShaperProbe`, `convolverProbe`, `periodicWaveProbe`,
`analyserProbe`, `channelMixProbe`, `stereoPannerProbe`, `iirFilterProbe`,
`hrtfPannerProbe`, `delayProbe`. The "Show saved data" panel rewrites the
canvas internal names into the paper-facing names at display time; on-disk
filenames keep the internal names. The pipeline takes the key from the
filename and accepts either naming, and `lab_data/build.py` applies this
same table when it packages the lab archives, which is why the released
archives carry the paper names.

## On-disk layout after one capture

After one click of "Run canvas probes" (with default 100 samples) plus
"Run audio probes" (60 samples):

```
local-data/
├── learnedfp/save/dev_<id>/visit1/
│   ├── S1_raw_faces_1_<ts>.rgba           ← 900 × 100×100×4 buffers (9 keys × 100)
│   ├── …
│   ├── samples_preview.ndjson                 ← per-sample preview (key, ts, w, h, channels, pixelSample preview)
│   └── validation.ndjson                  ← append-only freshness-marker results, one line per sample
└── audio/save/dev_<id>/visit1/
    ├── S1_audio_oscillatorMix_<ts>.f32    ← 720 Float32 buffers (12 probes × 60): 22050 samples,
    │                                         44100 for the two panner probes (two channels)
    ├── …
    ├── samples_preview.ndjson
    ├── sample_params.json                ← per-session probe parameter plan
    └── validation.ndjson
```

`pixelSample` is the first 8 RGBA pixels (32 bytes) of each sample, shown as
RGBA tuples in the panel so the bytes can be eyeballed without
binary-decoding.

## Endpoints

**Canvas**
- `GET /` — Main page (`client/index.html`)
- `GET /bootstrap` — Issue/return the device cookie
- `GET /pp/frozen.js` — Per-session canvas probe bundle
- `POST /save-batch` — Multipart canvas RGBA upload
- `GET /images` — List saved canvas samples grouped by key
- `GET /image?path=…` — Serve a single saved sample as PNG
- `GET /saved-data` — Read `samples_preview.ndjson` + `validation.ndjson` for current device

**Audio**
- `GET /audio/bootstrap` — Audio device cookie + 12-probe config
- `GET /audio/pp/frozen.js` — Per-session audio probe bundle
- `POST /audio/save-batch` — Multipart `.f32` audio upload
- `GET /audio/saved-data` — Same as `/saved-data` but for the audio store

**Misc**
- `GET /health` — `{ ok: true, node: 'ok' }`
- `POST /clear-device` — Clear the device cookie

## Smoke test

```bash
docker compose up --build -d
bash smoke-test.sh
```

The script runs a fake canvas + audio capture against the running server and
asserts that the expected files appear on disk and the freshness marker fires.
Returns 0 on success.

`.gitignore` keeps `*.pt`, `*.npz` and `local-data/` out of the repository.
