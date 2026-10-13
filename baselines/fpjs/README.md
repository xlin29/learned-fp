# `baselines/fpjs/` — FingerprintJS baseline

The deterministic-fingerprinting baseline (upstream
<https://github.com/fingerprintjs/fingerprintjs>, BUSL-1.1 for the 4.x line
used here). This directory holds a collector and the §5.5 evaluation. The
collector loads upstream FingerprintJS from npm, serves the UMD bundle at
`/fpjs/lib/fp.umd.min.js`, and stores each capture with its
`fingerprint.visitorId`. The Express and ingestion code is original; the
library is not redistributed.

## Quick start

Node 18 or newer; no other dependencies.

```bash
cd baselines/fpjs/collector
npm install
OUT_DIR=./samples node server.js          # PORT=<n> to move it off 3000
```

Open <http://localhost:3000> (or the port you chose; the LearnedFP framework
also defaults to 3000). The page loads FingerprintJS, computes a
fingerprint, and POSTs it to `/fpjs/collect`. Captures land under
`./samples/dev_<id>/fpjs_<timestamp>.json`.

## Endpoints

- `GET /` — Collector UI (`collector/public/index.html`)
- `GET /fpjs/lib/fp.umd.min.js` — Upstream FingerprintJS UMD bundle
- `POST /fpjs/collect` — Ingestion (one JSON file per event)
- `GET /fpjs/healthz` — Liveness probe

## Evaluation (paper §5.5)

FingerprintJS reduces each visit to one `visitorId`, so its evaluation is
set arithmetic on that string. `eval_visitorid.py` computes the three
visitorId figures of paper §5.5 from two capture roots, one per campaign:

```bash
python3 baselines/fpjs/eval_visitorid.py --t0-root <ENROLL_ROOT> --t1-root <RETURN_ROOT>
```

- **enrollment collisions** — enrolled devices whose visitorId is also emitted
  by another device, and the largest such group;
- **temporal instability** — returning devices whose return visit shares no
  visitorId with their enrollment;
- **cold-start collisions** — devices first seen in the return campaign whose
  visitorId is held by an enrolled device or by another new device.

The script reads both this collector's `fpjs.v1` layout and the older
`reorderedResult` layout. The paper's exact figures need the withheld
crowdsourced captures; the canvas-pixel comparison of §5.5 is not part of
this script.

## Capture file schema (`fpjs.v1`)

```json
{
  "ts": 1714600000000,
  "deviceId": "a1b2c3d4e5f60718",
  "ua": "<User-Agent>",
  "schema": "fpjs.v1",
  "fingerprint": { "visitorId": "<sha256-hex>", "components": { /* upstream components, stored as returned */ } }
}
```

`visitorId` is the upstream FingerprintJS hash (the identifier paper §1
refers to); the evaluation uses it together with `deviceId`, `ts` and `ua`.

No participant or cohort metadata is collected. Stable device id comes
from an HttpOnly cookie issued by the server on first visit.
