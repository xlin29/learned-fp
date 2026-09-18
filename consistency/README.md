# Metadata-consistency check

The labeling tool behind the same-device labels of paper §5.1: given two
records that share a Prolific PID across the two campaigns, it decides whether
they came from the same physical machine.

## How it scores

A pair of records is compared field by field over 35 FingerprintJS metadata
attributes, each carrying a weight. The score is the matched weight divided by
the weight actually available for that pair (fields missing on either side drop
out of both numerator and denominator). Pairs scoring at least 0.80 are
retained.

Weights follow prior longitudinal work, with hardware- and system-dependent
fields dominating the pool (total 95.5):

| Weight | Fields |
|---|---|
| 10.0 | `gpu_renderer`, `audio` |
| 8.0 | `math` |
| 7.0 | `fonts` |
| 6.0 | `font_preferences` |
| 5.0 | `screen_resolution` |
| 4.0 | `device_memory`, `hardware_concurrency`, `webgl_extensions` |
| 3.0 | `gpu_vendor`, `screen_frame`, `platform`, `timezone`, `languages`, `touch_support` |
| 2.0 | `color_depth`, `architecture`, `datetime_locale`, `plugins`, `vendor_flavors` |
| 1.0 | `color_gamut`, `hdr`, `webgl_version`, `webgl_shading_lang` |
| 0.5 | `session_storage`, `local_storage`, `indexed_db`, `open_database`, `pdf_viewer`, `reduced_motion`, `reduced_transparency`, `forced_colors`, `inverted_colors`, `monochrome`, `contrast` |

Canvas is excluded from the score (`fingerprint_matcher.py:193`): it is
extracted but never enters the decision.

## Usage

```bash
python3 consistency/fingerprint_matcher.py <base_dir> \
    --threshold 0.80 \
    --output fingerprint_results.csv \
    --report report.txt
```

`<base_dir>` holds `pid-*/` folders, each containing `dev_*/` subfolders with
`fpjs_*.json` captures, in either FingerprintJS JSON layout (`fingerprint` or
`reorderedResult`).

Per PID the tool emits one of three verdicts:

- `SAME_DEVICE` — every session pair clears the threshold
- `MIXED` — some sessions match, others do not
- `ALL_DIFFERENT` — no pair clears the threshold

`--copy-same DIR` additionally materializes the retained folders, which is how
the evaluation population was assembled. `--verbose` prints the per-field
breakdown for every pair.
