# Mitigation prototype

Chromium MV3 extension implementing the canonical-snapshot canvas defense
(paper §8.2). Every script-visible canvas readback is
routed through a single canonical snapshot that suppresses
device-specific pixel structure; a display-restore path keeps
on-screen rendering at native pixels. Unlike Brave's Farbling and the per-session
noise of Firefox 144's ETP Strict (§6), the design does not perturb output per-pixel —
it removes the substrate the learning-based tracker relies on (§7
Insight 4).

## Quick install (Chromium / Chrome / Edge / Brave / Opera)

1. Open `chrome://extensions`
2. Toggle **Developer mode** (top right)
3. Click **Load unpacked** and pick this `mitigation/` directory
4. The extension's icon should appear in the toolbar; clicking it
   opens the popup with the two thresholds described below.

That's it — every canvas readback in every tab now goes through the
canonical snapshot path.

## What it does

- **`getImageData()`** — Returns the canonical snapshot bytes; the original raw pixels are cached for display-restore.
- **`toDataURL()` / `toBlob()`** — Encode the canonical snapshot, not the live canvas.
- **`OffscreenCanvas.convertToBlob()` / `transferToImageBitmap()`** — Same as above.
- **`putImageData()`** — If the bytes the page is putting back are byte-identical to the canonical snapshot we issued (i.e., no tamper), silently swap the original raw pixels back in so the user sees the true picture.
- **`getContext()` (2D)** — Wraps the returned context to share the per-canvas state.

Each frame has an extraction budget (paper §8.2): every readback in the
frame, on any canvas and through any path, advances one counter, which
resets only when the frame navigates. The fidelity a canvas delivers is
decided the first time it is read after a paint, from the counter at that
moment — up to `coarseThreshold` reads, the canonical snapshot at 8 bits
per channel; up to `blockThreshold`, quantized to 4 bits per color
channel; beyond that, a constant image — and every further read of that
same paint returns byte-identical output whichever path it takes (partial
or full `getImageData`, `toDataURL`, `toBlob`), so the downgrade never
shows up as a disagreement between extraction paths. Any drawing call,
`putImageData` or resize starts a new paint, which is re-evaluated against
the counter. The popup lets you change both thresholds per tab; the
defaults are `(3, 20)`.

## Manual smoke tests

After loading the extension, open these pages in tabs. Each page load
starts with a fresh budget, so reload a probe to re-run it; all of them
expect the default thresholds unless noted.

- `tests/probes/stability.html` — one paint read through every path:
  repeated `getImageData`, partial vs. full read, `toDataURL` decoded vs.
  raw pixels, `toDataURL` vs. `toBlob` bytes; all must agree, and the
  paint's fidelity must hold past the budget step.
- `tests/probes/modes.html` — 30 reads with a repaint before most of
  them: the normal → coarse → block downgrade across paints, and
  byte-identical re-reads of one paint across a downgrade step. Shows
  what a script receives at reads 1, 10 and 25 next to the on-screen
  canvas.
- `tests/probes/offscreen.html` — the same ladder driven through
  `OffscreenCanvas`, plus `convertToBlob` and a check that both canvas
  kinds share one budget.
- `tests/probes/alpha.html` — pixels written with `α=0` but non-zero
  RGB read back with RGB = 0.
- `tests/probes/display-restore.html` — `getImageData → putImageData`
  round-trip after the budget is spent: the extracted bytes are the
  block-mode constant, while the on-screen canvas shows the original
  pixels.

## Permissions

The manifest requests:

- `storage` — persisting the popup's threshold settings
- `tabs` + `webNavigation` — per-tab popup state

It does not request host permissions for any specific domain, only
the `<all_urls>` content-script match (which is what document_start
canvas hooking requires). It does not phone home, fetch remote rules,
or include any analytics. The full source is in `src/`; `main-world.js`
is the only code that touches canvas APIs.
