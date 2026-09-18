#!/usr/bin/env bash
# End-to-end smoke test for the LearnedFP framework.
#
# Hits every collection endpoint with a fake .rgba (canvas) and .f32 (audio)
# upload, then asserts the expected files landed on disk and that the
# freshness-marker validation log was written. Returns 0 on success.
#
# Usage:
#   docker compose up --build -d   # start the framework
#   ./smoke-test.sh                # run the test
#   docker compose down            # stop
set -euo pipefail

BASE="${BASE:-http://localhost:3000}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

red()    { printf '\033[31m%s\033[0m\n' "$*" >&2; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
note()   { printf '  %s\n' "$*"; }

fail() { red "FAIL: $*"; exit 1; }

# --------------------------- canvas surface ---------------------------
echo "[1/6] canvas: GET /health"
curl -sf "$BASE/health" >/dev/null || fail "/health unreachable"

echo "[2/6] canvas: bootstrap + frozen.js + fake .rgba upload"
curl -sf -c "$TMP/cookie.txt" "$BASE/bootstrap" >/dev/null
curl -sf -b "$TMP/cookie.txt" "$BASE/pp/frozen.js?src=visit1&ns=1" -o "$TMP/frozen.js"
test -s "$TMP/frozen.js" || fail "/pp/frozen.js empty"

# 100×100 RGBA pattern matching the rgba(100,150,200,128) fill so a reader
# can see expected pixel values in the Show saved data panel.
python3 -c "import sys;sys.stdout.buffer.write(b'\x64\x96\xc8\x80'*10000)" > "$TMP/sample.rgba"
SAVE=$(curl -sf -b "$TMP/cookie.txt" -X POST \
  "$BASE/save-batch?src=visit1&session=1" \
  -F "bin=@$TMP/sample.rgba;filename=S1_raw_faces_1.rgba")
echo "$SAVE" | grep -q '"ok":true' || fail "/save-batch did not return ok:true ($SAVE)"
note "ok"

echo "[3/6] canvas: GET /saved-data lists samples_preview.ndjson + .rgba"
SD=$(curl -sf -b "$TMP/cookie.txt" "$BASE/saved-data?src=visit1")
echo "$SD" | grep -q '"samples_preview.ndjson"' || fail "saved-data missing samples_preview.ndjson"
echo "$SD" | grep -q '"rgbaCount":1'        || fail "saved-data did not see the .rgba"
note "ok"

# --------------------------- audio surface ---------------------------
echo "[4/6] audio: bootstrap + frozen.js + fake .f32 upload"
curl -sf -b "$TMP/cookie.txt" "$BASE/audio/bootstrap" >/dev/null
curl -sf -b "$TMP/cookie.txt" "$BASE/audio/pp/frozen.js?src=visit1&ns=1" -o "$TMP/audio_frozen.js"
test -s "$TMP/audio_frozen.js" || fail "/audio/pp/frozen.js empty"

# 22050-sample Float32 zero buffer — fails marker check (no marker embedded)
# but the upload itself + validation.ndjson append must succeed.
python3 -c "import sys;sys.stdout.buffer.write(b'\x00'*88200)" > "$TMP/sample.f32"
ASAVE=$(curl -sf -b "$TMP/cookie.txt" -X POST \
  "$BASE/audio/save-batch?src=visit1&session=1" \
  -F "bin=@$TMP/sample.f32;filename=S1_audio_oscillatorMix.f32")
echo "$ASAVE" | grep -q '"ok":true' || fail "/audio/save-batch did not return ok:true ($ASAVE)"
note "ok"

echo "[5/6] audio: GET /audio/saved-data lists validation.ndjson + .f32"
ASD=$(curl -sf -b "$TMP/cookie.txt" "$BASE/audio/saved-data?src=visit1")
echo "$ASD" | grep -q '"validation.ndjson"'  || fail "audio saved-data missing validation.ndjson"
echo "$ASD" | grep -q '"f32Count":1'         || fail "audio saved-data did not see the .f32"
echo "$ASD" | grep -q '"low-match"\|"ok"'    || fail "validation entry missing marker result"
note "ok"

echo "[6/6] all expected files exist (canvas + audio)"
DEV_CV=$(echo "$SD"  | python3 -c "import sys,json;print(json.load(sys.stdin)['deviceId'])")
DEV_AU=$(echo "$ASD" | python3 -c "import sys,json;print(json.load(sys.stdin)['deviceId'])")
test "$DEV_CV" = "$DEV_AU" || fail "canvas/audio used different deviceIds — cookie sharing broken"
note "shared deviceId: $DEV_CV"

green "PASS — collection + validation pipeline works end-to-end."
