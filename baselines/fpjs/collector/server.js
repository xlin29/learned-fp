#!/usr/bin/env node
// FingerprintJS collector — minimal Express server.
//
// Serves a one-page UI that loads the upstream FingerprintJS library
// (https://github.com/fingerprintjs/fingerprintjs, BUSL-1.1; installed from
// npm, not vendored, and served as-is at /fpjs/lib/fp.umd.min.js), captures
// the visitorId and per-component values, and stores each POST to
// /fpjs/collect as <OUT_DIR>/dev_<deviceId>/fpjs_<timestamp>.json.
//
// No participant or cohort metadata is collected; the device id is an
// HttpOnly cookie, so the same browser maps to the same dev_<id>/.

'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const express = require('express');

const PORT = parseInt(process.env.PORT || '3000', 10);
const OUT_DIR = path.resolve(process.env.OUT_DIR || './samples');
const COOKIE = 'fpjs_collector_did';
const COOKIE_MAX_AGE_S = 365 * 24 * 60 * 60;

fs.mkdirSync(OUT_DIR, { recursive: true });

const app = express();
app.use(express.json({ limit: '20mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Serve the upstream FingerprintJS UMD bundle from node_modules so the
// browser loads byte-for-byte the library as published on npm.
const FPJS_UMD = require.resolve('@fingerprintjs/fingerprintjs/dist/fp.umd.min.js');
app.get('/fpjs/lib/fp.umd.min.js', (req, res) => res.sendFile(FPJS_UMD));

function parseCookies(req) {
  const out = {};
  const raw = req.headers.cookie || '';
  for (const part of raw.split(';')) {
    const i = part.indexOf('=');
    if (i < 0) continue;
    out[part.slice(0, i).trim()] = decodeURIComponent(part.slice(i + 1).trim());
  }
  return out;
}

function issueDeviceId(req, res) {
  const c = parseCookies(req);
  if (c[COOKIE] && /^[a-f0-9]{16,32}$/.test(c[COOKIE])) return c[COOKIE];
  const did = crypto.randomBytes(8).toString('hex');
  res.setHeader(
    'Set-Cookie',
    `${COOKIE}=${did}; Path=/; Max-Age=${COOKIE_MAX_AGE_S}; HttpOnly; SameSite=Lax`,
  );
  return did;
}

app.get('/fpjs/healthz', (req, res) => res.json({ ok: true }));

app.post('/fpjs/collect', (req, res) => {
  try {
    const deviceId = issueDeviceId(req, res);
    const devDir = path.join(OUT_DIR, `dev_${deviceId}`);
    fs.mkdirSync(devDir, { recursive: true });

    const ts = new Date()
      .toISOString()
      .replace(/[:.]/g, '-')
      .replace('T', '_')
      .replace('Z', '');
    const filePath = path.join(devDir, `fpjs_${ts}.json`);

    const row = {
      ts: Date.now(),
      deviceId,
      ua: req.headers['user-agent'] || '',
      schema: 'fpjs.v1',
      fingerprint: req.body || {},
    };
    fs.writeFileSync(filePath, JSON.stringify(row, null, 2), 'utf8');

    console.log(`[collect] saved=${filePath}`);
    return res.json({ ok: true, saved: filePath });
  } catch (err) {
    console.error('[collect] error:', err);
    return res.status(500).json({ ok: false, error: 'collector-failed' });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`FingerprintJS collector listening on http://localhost:${PORT}/`);
  console.log(`Samples will land under ${OUT_DIR}/dev_<id>/fpjs_*.json`);
});
