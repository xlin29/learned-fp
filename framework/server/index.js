'use strict';

const NUM_SESSIONS_LOCAL = 100;
const express = require('express');
const compression = require('compression');
const bodyParser = require('body-parser');
const minimist = require('minimist');
const open = require('open');
const path = require('path');
const fs = require('fs');
const os = require('os');
const sharp = require('sharp');
const crypto = require('crypto');
const { buildFrozenBundle } = require(path.join(__dirname, 'templates', 'frozen_template'));
const multer = require('multer');
// Disk storage: multer streams each uploaded file to a tmp file instead of
// holding the whole multipart body in RAM. Keeps peak Node memory bounded
// even when many /save-batch uploads arrive in parallel.
const MULTER_TMP_DIR = path.join(os.tmpdir(), 'learnedfp-multer');
try { fs.mkdirSync(MULTER_TMP_DIR, { recursive: true }); } catch {}
const upload = multer({
  storage: multer.diskStorage({
    destination: MULTER_TMP_DIR,
    filename: (req, file, cb) => {
      cb(null, `${Date.now()}-${crypto.randomBytes(6).toString('hex')}-${file.fieldname}`);
    }
  }),
  limits: {
    files: 1024,
    fileSize: 50 * 1024 * 1024
  }
});

// Middleware: after the response finishes (or the connection closes), unlink
// every tmp file multer created for this request. Attach this AFTER the
// `upload.fields(...)` middleware so req.files is populated.
function cleanupUploadedTmpFiles(req, res, next) {
  let cleaned = false;
  const cleanup = () => {
    if (cleaned) return;
    cleaned = true;
    const all = [
      ...((req.files && req.files.img) || []),
      ...((req.files && req.files.bin) || [])
    ];
    for (const f of all) {
      if (f && f.path) fs.promises.unlink(f.path).catch(() => {});
    }
  };
  res.on('finish', cleanup);
  res.on('close', cleanup);
  next();
}

// Save default: JSON is always written; PNG is gated by ?fmt= or this env var.
const SAVE_FORMAT = (process.env.SAVE_FORMAT || 'json').toLowerCase();

// Whether the request asks for a PNG. ?fmt=png|both|1|true overrides the env.
function needPNG(req) {
  const fmt = String((req.query.fmt || SAVE_FORMAT)).toLowerCase();
  return fmt === 'png' || fmt === 'both' || fmt === '1' || fmt === 'true';
}

const COOKIE_NAME_READABLE = 'deviceId';
const COOKIE_NAME_HTTPONLY = 'deviceId_h';
const ONE_YEAR_MS = 365 * 24 * 60 * 60 * 1000;

function sanitizeName(s, fallback = 'unnamed') {
  const t = String(s || '').replace(/[^a-zA-Z0-9._-]/g, '_');
  return t.length ? t : fallback;
}


function makeDevFolderName(deviceId) {
  return deviceId;
}

function makeParticipantDir(rootDir, deviceId, meta = {}, src = '') {
  const devFolder = makeDevFolderName(deviceId);
  const base = path.join(rootDir, devFolder);
  return src ? path.join(base, sanitizeName(src)) : base;
}

// uaDir/samples_preview.ndjson: one line per .rgba with a 32-byte pixel
// snippet. Append-only (fs.appendFile, O_APPEND) so concurrent /save-batch
// handlers do not clobber each other; the .rgba files remain the source of
// truth.
  async function upsertPixelsIndex(uaDir, baseName, meta) {
    const idxPath = path.join(uaDir, 'samples_preview.ndjson');
    const line = JSON.stringify({
      baseName,
      key: meta.key, ts: meta.ts,
      w: meta.w, h: meta.h,
      channels: meta.channels ?? 4,
      pixelSample: meta.pixelSample ?? undefined,
    }) + '\n';
    await fs.promises.appendFile(idxPath, line, 'utf8');
  }

  async function readChannelsFromIndex(uaDir, baseName) {
    const idxPath = path.join(uaDir, 'samples_preview.ndjson');
    if (!fs.existsSync(idxPath)) return 4;
    try {
      const text = await fs.promises.readFile(idxPath, 'utf8');
      for (const ln of text.split('\n')) {
        if (!ln.trim()) continue;
        try {
          const e = JSON.parse(ln);
          if (e.baseName === baseName) return e.channels || 4;
        } catch {}
      }
    } catch {}
    return 4;
  }


function parseCookie(header) {
  const out = {};
  if (!header) return out;
  header.split(';').forEach(part => {
    const [k, ...rest] = part.trim().split('=');
    if (!k) return;
    out[k] = decodeURIComponent((rest.join('=') || '').trim());
  });
  return out;
}

function isSecureReq(req) {
  return req.secure || (req.headers['x-forwarded-proto'] || '').toLowerCase() === 'https';
}

function issueDeviceIdIfNeeded(req, res, opts = {}) {
  const cookies = parseCookie(req.headers.cookie || '');
  let id = cookies[COOKIE_NAME_HTTPONLY] || cookies[COOKIE_NAME_READABLE];

  if (!id) {
    id = 'dev_' + crypto.randomUUID().replace(/-/g, '');
    const base = {
      path: '/',
      httpOnly: true,
      sameSite: 'Lax',
      secure: !!Number(process.env.COOKIE_SECURE || '0'), // 0 by default; set to 1 only when serving over HTTPS
      maxAge: ONE_YEAR_MS,
    };
    const domain = opts.domain || process.env.COOKIE_DOMAIN;
    if (domain) base.domain = domain;
    res.cookie(COOKIE_NAME_HTTPONLY, id, base);
    res.cookie(COOKIE_NAME_READABLE, id, { ...base, httpOnly: false });
  }
  return id;
}

// Resolve to an absolute path up front so saveDir / DATA_ROOT are stable
// regardless of cwd at request time.
const PERSIST_ROOT = path.resolve(process.env.PERSIST_DIR || '/data');
// PERSIST_DIR is treated as the data root itself. Deployments set it to
// /data/learnedfp; local dev defaults to /data which then gets the
// learnedfp suffix appended so paths match the deployed layout.
const DATA_ROOT = path.basename(PERSIST_ROOT) === 'learnedfp'
  ? PERSIST_ROOT
  : path.join(PERSIST_ROOT, 'learnedfp');
const CATEGORIES = require(path.join(__dirname, 'data', 'categories'));
// 3000 is the unprivileged default; production sets PORT explicitly.
const DEFAULT_PORT = Number(process.env.PORT) || 3000;
const WATERMARK_DENSITY = Number(process.env.WATERMARK_DENSITY ?? 80);

const argv = minimist(process.argv.slice(2), {
  boolean: ['open'],
  string: ['port'],
  default: { open: true, port: String(DEFAULT_PORT) },
});

/* ----------------------------- bsStore + helpers ----------------------------- */
// In-memory only; swap for Redis + TTL in production.
const bsStore = new Map();
// Index key: deviceId + src + session + category (faces / persons / ...).
function bsKey(deviceId, src, session, cat) {
  return `${deviceId}::${src}::${session}::${cat}`;
}
// Parse session and category from a raw key name.
// Supported shapes: S9_faces_..., ori_S9_faces_..., faces_S9_...
function parseSessionCatFromKey(rawKey) {
  const k = String(rawKey || '').replace(/^(?:ori|raw)_+/i, ''); // strip an ori_/raw_ prefix
  let m = k.match(/^S(\d+)_raw_([A-Za-z0-9-]+)/);               // S#_raw_<cat>
  if (!m) m = k.match(/^S(\d+)_([A-Za-z0-9-]+)/);               // S#_<cat>
  if (m) return { session: `S${m[1]}`, cat: m[2] };
  // Fallback: faces_S9_... (rare).
  m = k.match(/^([A-Za-z0-9-]+)_S(\d+)/);
  if (m) return { session: `S${m[2]}`, cat: m[1] };
  return { session: '', cat: '' };
}
// Secondary index: deviceId::session::cat -> most recent record (src-agnostic).
const bsLatest = new Map(); // key: `${deviceId}::${session}::${cat}` -> rec

// FIFO eviction cap so bsStore/bsLatest can't grow unbounded over long uptime.
// JS Map preserves insertion order, so the first 10% of keys are the oldest.
const BS_STORE_MAX = 10000;
function evictOldestIfFull(m) {
  if (m.size <= BS_STORE_MAX) return;
  const toDrop = Math.floor(BS_STORE_MAX * 0.1);
  const iter = m.keys();
  for (let i = 0; i < toDrop; i++) {
    const { value: k, done } = iter.next();
    if (done) break;
    m.delete(k);
  }
}

function putBsRecord(deviceId, src, session, cat, rec) {
  // Record under the precise key (includes src).
  evictOldestIfFull(bsStore);
  bsStore.set(bsKey(deviceId, src, session, cat), rec);
  // Refresh the src-agnostic "latest" index.
  const k2 = `${deviceId}::${session}::${cat}`;
  const old = bsLatest.get(k2);
  if (!old || (rec.ts > old.ts)) {
    evictOldestIfFull(bsLatest);
    bsLatest.set(k2, rec);
  }
}


function findBsRecord(deviceId, src, sessName, cat) {
  // Exact match only (src is part of the key) — no cross-src reuse.
  const exact = bsStore.get(bsKey(deviceId, src, sessName, cat));
  return (exact && Number.isFinite(exact.bs)) ? exact : null;
}


const app = express();
app.set('trust proxy', true);

app.use((req, res, next) => {
  issueDeviceIdIfNeeded(req, res); // ensure every request carries the same deviceId
  next();
});

// --- Basics + static assets ---
const shouldCompress = (req, res) => {
  if (req.method === 'POST' && (req.path === '/save' || req.path === '/save-bin' || req.path === '/save-batch')) {
    return false; // upload responses skip compression
  }
  return compression.filter(req, res);
};
app.use(compression({ threshold: 0, filter: shouldCompress }));

// Client lives in the sibling ../client directory; resolved from __dirname so
// the path is stable regardless of the cwd at launch (dev or Docker).
const STATIC_DIR = process.env.STATIC_DIR || path.join(__dirname, '..', 'client');
app.use(express.static(STATIC_DIR, {
  etag: false, lastModified: false,
  setHeaders: (res) => {
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
    res.setHeader('Pragma', 'no-cache'); res.setHeader('Expires', '0');
  },
}));

// Global JSON / URL-encoded body parsing (used by all routes except /save).
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));

app.get('/bootstrap', (req, res) => {
  const id = issueDeviceIdIfNeeded(req, res);
  res.setHeader('Cache-Control', 'no-store');
  res.status(200).json({ deviceId: id });
});

const saveDir = path.join(DATA_ROOT, 'save');
[DATA_ROOT, saveDir].forEach((p) => fs.mkdirSync(p, { recursive: true }));

function getLocalTimestamp() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  const ms = String(now.getMilliseconds()).padStart(3, '0');
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}T${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}-${ms}`;
}

function dataURLToBuffer(dataURL) {
  const base64 = dataURL.replace(/^data:image\/\w+;base64,/, '');
  return Buffer.from(base64, 'base64');
}

function shouldValidateForSrc(src) {
  // The framework only ships the visit1 enrollment flow, but the marker
  // validation (paper §4.2) should run on every captured emoji / randomFont
  // sample so the freshness-marker outcome is visible on disk.
  return src === 'visit1';
}

function pixelsToJsonLines(data, channels) {
  const n = Math.floor(data.length / channels);
  const out = new Array(n);
  for (let i = 0, j = 0; j < n; j++, i += channels) {
    const r = data[i], g = data[i+1], b = data[i+2];
    const a = channels === 4 ? data[i+3] : 255;
    out[j] = (r|g|b|a) === 0 ? '0' : `${r},${g},${b},${a}`;
  }
  return JSON.stringify(out);
}

/* ===== a0 guard: helpers and cache ===== */

// Count A==0 pixels in an RGBA buffer (defaults to 100x100). Very fast.
function countA0(rawBuf, width = 100, height = 100) {
  let cnt = 0;
  for (let i = 3; i < rawBuf.length; i += 4) if (rawBuf[i] === 0) cnt++;
  return cnt;
}

// Canonicalize a key by stripping trailing _bs / _ms / timestamp suffixes,
// so it lines up with the visit1 file-name prefix.
function canonicalKeyPrefix(name) {
  const s = String(name).replace(/\.(?:rgba|png)$/i, '');
  // Strip trailing _bs-?\d+ / _ms-?\d+ / _YYYY-MM-DDTHH-MM-SS-SSS groups.
  return s.replace(/(_bs-?\d+|_ms-?\d+|_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3})+$/g, '');
}

// Cache: max a0 for a given canonical key under visit1.
const V1_A0_CACHE = new Map(); // key: visit1Dir + '::' + canonical -> maxA0

async function getVisit1MaxA0(visit1Dir, canonical) {
  const cacheKey = `${visit1Dir}::${canonical}`;
  if (V1_A0_CACHE.has(cacheKey)) return V1_A0_CACHE.get(cacheKey);
  let maxA0 = 0;
  try {
    const all = fs.readdirSync(visit1Dir);
    for (const fname of all) {
      if (!fname.endsWith('.rgba')) continue;
      if (!fname.startsWith(canonical)) continue;
      try {
        const buf = await fs.promises.readFile(path.join(visit1Dir, fname));
        const a0 = countA0(buf, 100, 100);
        if (a0 > maxA0) maxA0 = a0;
      } catch {}
    }
  } catch {}
  V1_A0_CACHE.set(cacheKey, maxA0);
  return maxA0;
}


/* ===== Append validation results to a single NDJSON log =====
 *
 * One NDJSON line per record, appended to validation.ndjson. fs.appendFile
 * preserves O_APPEND ordering, so concurrent /save-batch calls interleave at
 * line granularity rather than corrupt the file.
 */
async function appendSessionValidation(uaDir, sessName, record) {
  const s = String(sessName || '').trim();
  const valPath = path.join(uaDir, 'validation.ndjson');
  const line = JSON.stringify({ sample: s || null, ...record }) + '\n';
  await fs.promises.appendFile(valPath, line, 'utf8');
  return valPath;
}


/* ===== PNG-JUDGE: pixel-only verdict (no dependency on bs) ===== */
const PNG_JUDGE = {
  enable: (process.env.PNG_JUDGE ?? '1') !== '0',
  a0GrayNonzeroMin: Number(process.env.PNG_A0_GRAY_NONZERO_MIN ?? 0.90), // background nonzero-gray coverage threshold
  uniqNonzeroMin:   Number(process.env.PNG_UNIQ_NONZERO_MIN   ?? 2),     // distinct nonzero-gray levels threshold
  top2RatioMin:     Number(process.env.PNG_TOP2_RATIO_MIN     ?? 0.10),  // second-peak share threshold
  rejectOnTampered: (process.env.PNG_JUDGE_REJECT ?? '0') === '1'        // when set, reject (HTTP 403) on a hit
};

/* ===== PNG-JUDGE: save the PNG used for the verdict (only when running PNG-Judge) ===== */
const PNG_JUDGE_SAVE = 0;

async function savePngFromRGBA(rawBuf, w = 100, h = 100, outPath) {
  const pngBuf = await sharp(rawBuf, { raw: { width: w, height: h, channels: 4 } }).png().toBuffer();
  await fs.promises.writeFile(outPath, pngBuf);
}


// Compute features for an RGBA buffer (pixel-only; defaults to 100×100).
function rgbaFeatures(rawBuf, width = 100, height = 100) {
  const W = width|0, H = height|0, N = W*H;
  const u8 = rawBuf; // Buffer / Uint8Array
  let a0 = 0, a0GrayNon0 = 0, a0AnyNon0 = 0, fg = 0;
  const hist = new Uint32Array(256); // histogram of A==0 gray (R==G==B), value 0 ignored

  for (let i=0;i<N;i++){
    const off = i*4;
    const r=u8[off], g=u8[off+1], b=u8[off+2], a=u8[off+3];
    if (a === 0) {
      a0++;
      if ((r|g|b) !== 0) a0AnyNon0++;
      if (r===g && g===b && r!==0) { a0GrayNon0++; hist[r]++; }
    } else {
      fg++;
    }
  }

  // Background nonzero-gray peak.
  let uniq = 0, top1 = 0, top2 = 0;
  for (let v=1; v<256; v++){
    const c = hist[v];
    if (!c) continue;
    uniq++;
    if (c > top1) { top2 = top1; top1 = c; }
    else if (c > top2) { top2 = c; }
  }
  const a0_ratio              = a0 / (N || 1);
  const a0_nonzero_ratio_any  = a0 ? a0AnyNon0 / a0 : 0;
  const a0_gray_nonzero_ratio = a0 ? a0GrayNon0 / a0 : 0;
  const top2_ratio            = a0 ? top2 / a0 : 0;
  const uniq_nonzero_lvls     = uniq;

  // Background entropy (optional).
  let a0_entropy = 0;
  if (a0) {
    for (let v=1; v<256; v++){
      const p = hist[v] / a0;
      if (p > 0) a0_entropy += -p * Math.log2(p);
    }
  }

  // Foreground sharpness (A>0): basic Laplacian variance + Tenengrad (advisory, not enforced).
  // Note: simple implementation, performance is fine; swap for a faster convolution if needed.
  const gray = new Float32Array(N);
  const mask = new Uint8Array(N);
  for (let i=0;i<N;i++){
    const off=i*4; const r=u8[off], g=u8[off+1], b=u8[off+2], a=u8[off+3];
    if (a>0){ gray[i]=0.299*r+0.587*g+0.114*b; mask[i]=1; }
  }
  function conv3x3(K){
    const out = new Float32Array(N);
    for (let y=1;y<H-1;y++){
      for (let x=1;x<W-1;x++){
        let acc=0, k=0;
        for (let dy=-1;dy<=1;dy++){
          for (let dx=-1;dx<=1;dx++){
            const idx=(y+dy)*W+(x+dx);
            acc += (mask[idx]?gray[idx]:0) * K[k++];
          }
        }
        out[y*W+x]=acc;
      }
    }
    return out;
  }
  const Kx=[-1,0,1,-2,0,2,-1,0,1], Ky=[1,2,1,0,0,0,-1,-2,-1], Lp=[0,1,0,1,-4,1,0,1,0];
  const gx=conv3x3(Kx), gy=conv3x3(Ky), lap=conv3x3(Lp);
  let sumMag2=0, cnt=0, s=0, s2=0;
  for (let i=0;i<N;i++){
    if (!mask[i]) continue;
    const mag2=gx[i]*gx[i]+gy[i]*gy[i]; sumMag2+=mag2;
    s+=lap[i]; s2+=lap[i]*lap[i]; cnt++;
  }
  const fg_tenengrad     = cnt ? (sumMag2/cnt) : 0;
  const fg_laplacian_var = cnt ? (s2/cnt - (s/cnt)*(s/cnt)) : 0;

  return {
    a0_ratio,
    a0_nonzero_ratio_any,
    a0_gray_nonzero_ratio,
    uniq_nonzero_lvls,
    top2_ratio,
    a0_entropy,
    fg_ratio: (fg/(N||1)),
    fg_tenengrad,
    fg_laplacian_var
  };
}

function classifyByPngFeatures(feat, thr=PNG_JUDGE){
  return (
    feat.a0_gray_nonzero_ratio >= thr.a0GrayNonzeroMin ||
    feat.uniq_nonzero_lvls     >= thr.uniqNonzeroMin   ||
    feat.top2_ratio            >= thr.top2RatioMin
  );
}



/* ---------------- RNG / PRF ---------------- */
const RNG_SECRET = process.env.RNG_SECRET ? Buffer.from(process.env.RNG_SECRET, 'base64') : crypto.randomBytes(32);
function prfBlock(cid, ns, ctr, rngNonce) {
  const msg = Buffer.from(`${cid}|${ns}|${ctr}|${rngNonce || ''}`);
  return crypto.createHmac('sha256', RNG_SECRET).update(msg).digest();
}
function b64url(buf) { return buf.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''); }
function newCid() { return b64url(crypto.randomBytes(16)); }

/* -----------------------------------------------------------------------------
 * Plan generator (generate ONE session)
 * --------------------------------------------------------------------------- */
function generateOneSession(rngCtx) {
  const { r53, f, i, pick } = rngCtx;
  const W = 100, H = 100;

  // emoji
  const GRID = 6, PER = GRID * GRID, cell = W / GRID;
  function pickN(arr, n) {
    const a = arr.slice();
    for (let k = a.length - 1; k > 0; k--) { const j = i(0, k); [a[k], a[j]] = [a[j], a[k]]; }
    return a.slice(0, Math.min(n, a.length));
  }
  function emojiPlan(list) {
    const items = pickN(list || [], PER).map((s) => {
      const ch = (s != null) ? String(s) : '\u25A1';
      const cps = [];
      for (const cch of ch) { cps.push(cch.codePointAt(0)); if (cps.length === 2) break; }
      return {
        ch, cp: cps[0] || 0x25A1, cp2: cps[1] || null,
        fontSize: +(cell * f(0, 0.3)).toFixed(3),
        dx: +(f(-cell * 0.15, cell * 0.15)).toFixed(3),
        dy: +(f(-cell * 0.15, cell * 0.15)).toFixed(3),
        rot: +(f(-0.25, 0.25)).toFixed(3),
      };
    });
    let bs = i(1, 254);
    return { items, brightnessShift: bs };
  }

  const emoji = {
    faces: emojiPlan(CATEGORIES.faces),
    persons: emojiPlan(CATEGORIES.persons),
    travel: emojiPlan(CATEGORIES.travel),
    hands: emojiPlan(CATEGORIES.hands)
  };

  const SYM = ['∀','∂','∑','ß','Ω','ψ','λ','∞','≠','≈','↔','→','⇑','⇓','$','€','£','¥','₿','≡','±','∏','∆','∫','√','∇','∝','∴','⋯','⋮','⋱','∘','⊕','⊗','⊥','∞'];
  const F_ASCII = ['Arial','Georgia','Menlo','Tahoma','Verdana'];
  const P_BLOBS = [[20,35,30],[50,35,30],[80,35,30],[20,65,30],[50,65,30],[80,65,30],[35,50,30],[65,50,30]];
  const randChar = () => String.fromCharCode(i(0x21, 0xFFFF));
  const rgba = (amin=0,amax=1) => [i(0,255), i(0,255), i(0,255), +f(amin,amax).toFixed(3)];

  function jf(x, amp){ return +(x + f(-amp, amp)).toFixed(3); }
  const RF_BASE = {
    fill: 'rgba(20,20,20,0.92)',
    lines: [
      { font:'Menlo', size:11, x:6, y:16, text:'∀ ∑ λ Ω € ₿' },
      { font:'Georgia', size:12, x:7, y:32, text:'ß ψ ≠ ≈ ± ∞' },
      { font:'Arial', size:11, x:6, y:48, text:'$ £ ¢ ₽ 1 2 3 4 5' },
      { font:'Helvetica', size:10, x:6, y:64, text:'→ ⇑ ⇓ ◎ ○ ◉ ▲ ▼' },
      { font:'Courier New', size:11, x:6, y:80, text:'{ } [ ] ( ) < >' },
      { font:'Times New Roman', size:12, x:6, y:96, text:'é ñ ü å ø æ' }
    ]
  };
  const randomFont = (function(){
    const A = 2.0;
    return {
      fill: RF_BASE.fill,
      items: RF_BASE.lines.map(L => ({ font:L.font, size:L.size, x:jf(L.x,A), y:jf(L.y,A), text:L.text })),
      brightnessShift: i(1,254),              
    maskSalt: i(0, 0x7fffffff)          
    };
  })();

  const moire = {
    spacing:+f(3.5,6.5).toFixed(3),
    angle1:+f(0.10,0.25).toFixed(4),
    angle2:null, lw1:+f(0.7,1.4).toFixed(3), lw2:+f(0.7,1.4).toFixed(3),
    alpha1:+f(0.25,0.45).toFixed(3), alpha2:+f(0.25,0.45).toFixed(3),
    phase:+f(0, Math.PI*2).toFixed(6), freq:+f(1.2,2.0).toFixed(6)
  };
  moire.angle2 = +(parseFloat(moire.angle1) + f(0.02,0.05)).toFixed(4);

  const a_randomString = {
    color:[i(0,255),i(0,255),i(0,255),0.65],
    lines:Array.from({length:5}, () => Array.from({length:11}, () => randChar()).join('')),
    fonts:['serif','sans-serif','monospace','cursive','fantasy']
  };

  function j(x, amp){ return +(x + f(-amp, amp)).toFixed(3); }
  const RL_BASE = {
    strokeWidth: 3, strokeStyle: 'black',
    lineCap: 'round', lineJoin: 'miter', miterLimit: 7.0,
    dash: [4, 3], dashOffsetBase: 0.0,
    bezier: { p0:[15,78], c1:[10,48], c2:[55,26], p3:[95,94] },
    quad:   { p0:[ 8,22], c:[  6,88], p2:[82,22] }
  };

  // gradQuantSteps plan
  const grad_quant_steps_plan = {
    angle: +rngCtx.f(0, Math.PI*2).toFixed(6),
    stops: (() => {
      const n = rngCtx.i(2,5);
      const inner = Array.from({length: Math.max(0,n-2)}, () => +rngCtx.r53().toFixed(6)).sort((a,b)=>a-b);
      return [0, ...inner, 1];
    })(),
    noiseMod: rngCtx.i(7,29),
    noiseAmp: rngCtx.i(0,2)
  };

  // shadowBlurProbe plan
  const shadow_blur_probe_plan = {
    count: rngCtx.i(2,6),
    gco: rngCtx.pick(["source-over","multiply","screen","overlay","darken","lighten","hard-light","soft-light","difference","exclusion"]),
    shapePool: ["circle","rect","diamond"],
    blurMax: +rngCtx.f(8,24).toFixed(3),
    colors: ["rgba(0,0,0,0.6)","rgba(0,0,0,0.4)","rgba(20,0,60,0.4)","rgba(0,40,80,0.4)"]
  };


  return {
    faces: emoji.faces, persons: emoji.persons,
    travel: emoji.travel, 
    hands: emoji.hands,

    randomFont, moire,
    'a-randomString': a_randomString,
    gradQuantSteps: grad_quant_steps_plan,
    shadowBlurProbe: shadow_blur_probe_plan
  };
}

  function makePlanOnServer({ deviceId, visitNonce, numSubSessions = 1 } = {}) {
    // Number of sub-sessions to generate.
    const NUM_SUB = Number(numSubSessions || process.env.NUM_SUBSESSIONS || 10);

    const DEFAULT_COUNT = 1024;
    const COUNT = Number(process.env.RNG_COUNT || DEFAULT_COUNT);

    // Build an rngCtx that draws from a raw byte pool.
    function makeRngCtxFromPool(pool) {
      let poolIdx = 0;
      function r53() {
        if (poolIdx + 8 > pool.length) poolIdx = 0;
        const a = pool.readUInt32BE(poolIdx);
        const b = pool.readUInt32BE(poolIdx + 4) >>> 11;
        poolIdx += 8;
        return (a * 2097152 + b) / 9007199254740992;
      }
      const f = (lo, hi) => lo + (hi - lo) * r53();
      function int(lo, hi) { return (lo + Math.floor((hi - lo + 1) * r53())) | 0; }
      return { r53, f, i: int, pick: (arr) => arr[Math.floor(r53() * arr.length)] };
    }

    // Generate a fully independent random stream per sub-session
    // (independent across visits and across sub-sessions).
    const sessions = [];
    const seeds = []; // optional: remember seeds for reproducibility; safe to drop

    for (let k = 0; k < NUM_SUB; k++) {
      const cidK = newCid();                           // per-sub-session cid
      const rngNonceK = b64url(crypto.randomBytes(16)); // per-sub-session rngNonce
      const NSk = `frozen:${visitNonce || 'v'}:${k}`;

      const blocks = [];
      for (let ctr = 0; ctr < COUNT; ctr++) {
        blocks.push(prfBlock(cidK, NSk, ctr, rngNonceK));
      }
      const poolK = Buffer.concat(blocks);
      const rngCtxK = makeRngCtxFromPool(poolK);

      sessions.push(generateOneSession(rngCtxK));
      seeds.push({ k, cid: cidK, rngNonce: rngNonceK, ns: NSk, count: COUNT }); // optional
    }

      return {
        sessions,
        numSubSessions: NUM_SUB,
        rngSeeds: seeds,  // surfaced so clients/tools can reproduce a sub-session
        meta: {
          deviceId,
          visitNonce: visitNonce || ''
        }
    };
  }



/* -----------------------------------------------------------------------------
 * /save (accepts text/plain chunked JSON; also accepts application/json in one shot)
 * --------------------------------------------------------------------------- */
app.post('/save',
  bodyParser.text({ type: ['application/json', 'text/plain'], limit: '50mb' }),
  async (req, res) => {
    try {
      if (req.socket && req.socket.setNoDelay) req.socket.setNoDelay(true);
      const deviceId = issueDeviceIdIfNeeded(req, res);
      const meta = {};
      const src = sanitizeName(req.query.src || 'unspecified');
      const part = String(req.query.part || '');
      const total = String(req.query.total || '');
      const uaDir = makeParticipantDir(saveDir, deviceId, meta, src);
      await fs.promises.mkdir(uaDir, { recursive: true });

      // bodyParser.text leaves the body as a string when the request is
      // application/json. Try JSON.parse here; on failure keep an empty {}.
      let allData = {};
      if (typeof req.body === 'string') {
        try { allData = JSON.parse(req.body || '{}'); } catch { allData = {}; }
      } else if (req.body && typeof req.body === 'object') {
        allData = req.body;
      }

      const entries = Object.entries(allData).filter(([k]) => k !== 'hash');
      const bytes = req.get('content-length') || (typeof req.body === 'string' ? Buffer.byteLength(req.body) : 0);
      console.log(`[save] device=${deviceId} src=${src} part=${part||'1'}/${total||'?'} keys=${entries.length} bytes=${bytes}`);

      for (const [rawKey, dataURL] of entries) {
        const key = sanitizeName(rawKey);
        try {
          const pngBuf = dataURLToBuffer(dataURL);
          const { data, info } = await sharp(pngBuf).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
          const hash = crypto.createHash('sha256').update(data).digest('hex');

          const pixels = [];
          for (let i = 0; i < data.length; i += info.channels) {
            const r = data[i], g = data[i + 1], b = data[i + 2];
            const a = info.channels === 4 ? data[i + 3] : 255;
            pixels.push(r === 0 && g === 0 && b === 0 && a === 0 ? '0' : `${r},${g},${b},${a}`);
          }
          const filename = `${key}_${getLocalTimestamp()}.json`;
          await fs.promises.writeFile(path.join(uaDir, filename), JSON.stringify(pixels), 'utf8');
        } catch (e) {
          console.warn(`[save] decode failed for key=${key}: ${e.message}`);
        }
      }

      if (allData.hash && typeof allData.hash === 'object') {
        const hashPath = path.join(uaDir, 'hash_log.json');
        let existing = {};
        if (fs.existsSync(hashPath)) {
          try { existing = JSON.parse(await fs.promises.readFile(hashPath, 'utf8')); } catch {}
        }
        existing[new Date().toISOString()] = allData.hash;
        await fs.promises.writeFile(hashPath, JSON.stringify(existing, null, 2), 'utf8');
      }

      res.status(200).send({ ok: true, message: 'Saved', deviceId, ...meta, src, count: entries.length, part, total });
    } catch (err) {
      console.error('Error saving pixel data:', err);
      res.status(500).send({ ok: false, error: 'Failed to save fingerprint data' });
    }
  }
);

function resolveSrc(req, def='visit1') {
  const c = parseCookie(req.headers.cookie || '');
  return sanitizeName(req.query.src || c.PP_LAST_SRC || def);
}

  // ===== randomFont =====
  function _rfCanon(raw){
    const items = (raw?.items||[]).map(it=>({
      font: String(it?.font||''),
      size: +(+(it?.size||0)).toFixed(3),
      x:    +(+(it?.x||0)).toFixed(3),
      y:    +(+(it?.y||0)).toFixed(3),
      text: String(it?.text||'')
    }));
    return JSON.stringify({
      fill: String(raw?.fill||''),
      items,
      brightnessShift: (raw?.brightnessShift|0)
    });
  }


// ===== emoji: canonicalize randomness-related parameters (reproducibility fields) =====
function _emojiCanon(raw){
  const items = (raw?.items||[]).slice(0,36).map(it=>({
    cp:  (Number.isInteger(it?.cp)  ? (it.cp|0)  : 0),
    cp2: (Number.isInteger(it?.cp2) ? (it.cp2|0) : null),
    fontSize: +(+(it?.fontSize||0)).toFixed(3),
    dx: +(+(it?.dx||0)).toFixed(3),
    dy: +(+(it?.dy||0)).toFixed(3),
    rot:+(+(it?.rot||0)).toFixed(3)
  }));
  return JSON.stringify({ items, brightnessShift: (raw?.brightnessShift|0) });
}

function _deriveMaskSalt(serverSecretBuf, canonStr, sIdxNum, serverNonceBuf){
  const h = crypto.createHmac('sha256', serverSecretBuf)
    .update(canonStr).update('|')
    .update(String(sIdxNum)).update('|')
    .update(serverNonceBuf)
    .digest();
  return (h.readUInt32BE(0) & 0x7fffffff) >>> 0; // 31-bit
}


function verifyMaskRGBA(rawBuf, width, height, rec, sessName, cat, opts = {}) {
  if (!rawBuf || rawBuf.length < width*height*4) {
    return { ok:false, reason:'bad-buffer', expected:0, matched:0, ratio:0 };
  }
  const catStr = String(cat);
  const allowed = new Set(['randomFont','faces','persons','travel','hands']);
    if (!allowed.has(catStr)) {
      return { ok:true, reason:'skip-cat', expected:0, matched:0, ratio:1 };
    }

  let planRaw = {};
  try { if (rec?.planSnapshot) planRaw = JSON.parse(rec.planSnapshot); } catch {}
  const bsRaw = ((planRaw.brightnessShift ?? planRaw.bs ?? rec?.bs ?? 0) | 0);
  const maskSalt = ((planRaw.maskSalt ?? rec?.maskSalt ?? 0) | 0) >>> 0;
  const D = Number.isFinite(planRaw.D)
    ? (planRaw.D|0)
    : (Number.isFinite(rec?.density) ? (rec.density|0) : WATERMARK_DENSITY);
  let sIdx = 0; { const m = String(sessName||'').match(/^S(\d+)$/i); if (m) sIdx = (m[1]|0); }
  let code = (bsRaw | 0);
  if (code < 1) code = 1;
  else if (code > 254) code = 254;
  function xorshift32(u){ u>>>0; u^=(u<<13)>>>0; u^=(u>>>17)>>>0; u^=(u<<5)>>>0; return u>>>0; }

  let expected = 0, matched = 0;
  const buf = rawBuf;
  for (let idx=0; idx<width*height; idx++){
    const y = (idx/width)|0, x = idx - y*width;
    let seed = (((maskSalt + bsRaw + (sIdx*1315423911))|0) ^ ((y<<16)|x))|0;
    const h = xorshift32(seed);
    if ((h % 100) < D) {
      const i = idx*4;
      const r = buf[i], g = buf[i+1], b = buf[i+2], a = buf[i+3];
      if (a === 0) {
        expected++;
        if (r===code && g===code && b===code /* && a===0 */) {
          matched++;
        }
      }
    }
  }
  const ratio = expected ? matched/expected : 1;
  const passRatio = opts.passRatio ?? 0.85;
  return { ok: ratio >= passRatio, reason: ratio >= passRatio ? 'ok' : 'low-match', expected, matched, ratio };
}

/* ----------------------------- /save-batch: chunked raw-RGBA upload -----------------------------
 *
 * The frozen template also POSTs ?kind=log per-session timing rows via
 * __sendLog. Those are acknowledged with 200 and not persisted; the iframe
 * shows the per-run render/upload/blocking totals.
 */
app.post('/save-batch',
    (req, res, next) => {
    const kind = String(req.query.kind || '').toLowerCase();
    if (kind !== 'log') return next();
    return express.json({ limit: '100kb' })(req, res, () => res.status(200).json({ ok: true }));
  },
  upload.fields([
    { name: 'img', maxCount: 64 },   // PNG batch (the four PNG-uploaded keys)
    { name: 'bin', maxCount: 1024 }  // raw RGBA
  ]),
  cleanupUploadedTmpFiles,
  async (req, res) => {
  try {
    if (req.socket && req.socket.setNoDelay) req.socket.setNoDelay(true);
    const PIXEL_STORE = (process.env.PIXEL_STORE || 'rgba').toLowerCase(); // rgba | json | json.gz | both
    const zlib = require('zlib');

    const deviceId = issueDeviceIdIfNeeded(req, res);
    const meta = {};
    const src = resolveSrc(req, 'visit1');
    const session = String(req.query.session || '');
    const kind = String(req.query.kind || '').toLowerCase(); // optional: kind=ori
    const uaDir = makeParticipantDir(saveDir, deviceId, meta, src);
    await fs.promises.mkdir(uaDir, { recursive: true });

    const totalBytes = Number(req.get('content-length') || 0);
    const filesPNG = (req.files && req.files['img']) || [];
    const filesORI = (req.files && req.files['bin']) || [];
    const filesCount = filesPNG.length + filesORI.length;
    console.log(`[save-batch] device=${deviceId} src=${src} session=${session} files=${filesCount} bytes=${totalBytes}`);

    if (!filesCount) {
      res.status(400).json({ ok: false, error: 'no files in multipart' });
      return;
    }

    const items = [];

    for (const f of filesORI) {
      // With diskStorage, multer gives us f.path (no f.buffer). Read on-demand
      // so peak memory is ~one file at a time rather than the whole batch.
      if (!f.buffer) f.buffer = await fs.promises.readFile(f.path);
      const rawName = String((f.originalname || '').replace(/\.rgba$/i, ''));
      const key = sanitizeName(rawName, `unnamed_${Date.now()}`);
      const ts  = getLocalTimestamp();

      // Parse S# and category from the key.
      const { session: sessName, cat } = parseSessionCatFromKey(key);

      // Filename = <key>_<timestamp>; bs / maskSalt live in the in-memory
      // bsStore, keyed by deviceId/src/session/cat.
      const baseWithParams = `${key}_${ts}`;
      try {
        if (['randomFont','faces','persons','travel','hands'].includes(cat) && shouldValidateForSrc(src)) {
          const rec = findBsRecord(deviceId, src, sessName, cat);
          if (!rec) return res.status(403).json({ ok:false, error:'record not found' });

          // === a0 guard: cap current a0 at 120% of the visit1 max ===
        let a0Guard = null;
        try {
          const visit1Dir = makeParticipantDir(saveDir, deviceId, meta, 'visit1');
          const canonical = canonicalKeyPrefix(baseWithParams); // current name with _bs/_ms/timestamp stripped
          const curA0     = countA0(f.buffer, 100, 100);
          const baseMax   = await getVisit1MaxA0(visit1Dir, canonical);
          const limit     = baseMax ? Math.ceil(baseMax * 1.20) : null; // no baseline => no check
          const pass      = (limit == null) ? true : (curA0 <= limit);
          if (!pass) console.warn('[a0Guard] A0 over limit', { key, sessName, curA0, baseMax, limit });
          a0Guard = { baselineMax: baseMax, current: curA0, limit, pass };
        } catch (e) {
          console.warn('[a0Guard] error', e?.message || e);
        }

          const sNum = Number(String(sessName).replace(/^S/i,'')) || 1;
          const serverNonceBuf = Buffer.from(String(rec.serverNonce||''), 'base64');
          const planRaw = JSON.parse(rec.planSnapshot || '{}') || {};
          const canonStr = (cat === 'randomFont')
            ? (rec.rfCanon    || _rfCanon(planRaw))
            : (rec.emojiCanon || _emojiCanon(planRaw));
          const expectedMaskSalt = _deriveMaskSalt(RNG_SECRET, canonStr, sNum, serverNonceBuf) >>> 0;

          if ((expectedMaskSalt>>>0) !== (rec.maskSalt>>>0)) {
            console.warn('[save-batch] maskSalt mismatch', {
              key, sessName, expectedMaskSalt, storedMaskSalt: rec.maskSalt, bs: rec.bs
            });
              // Log a maskSalt mismatch.
          try {
            await appendSessionValidation(uaDir, sessName || (session ? `S${session}` : null), {
              at: new Date().toISOString(),
              key,
              cat,
              base: baseWithParams,
              salt: { expected: expectedMaskSalt, actual: rec.maskSalt, match: false },
              marker: null,
              a0Guard: a0Guard,
              pngJudge: null,
              outputs: { rgba: null }
            });
          } catch {}

          } else {
            console.log('[save-batch] rf maskSalt OK', {
              key, sessName, expectedMaskSalt, storedMaskSalt: rec.maskSalt, bs: rec.bs
            });
          }
          const passRatio = 0.9;
          const verify = verifyMaskRGBA(f.buffer, 100, 100, rec, sessName, cat, { passRatio });
          const pct = (verify.ratio * 100).toFixed(2) + '%';
          if (!verify.ok) {
            console.warn('[save-batch] pixel watermark LOW-MATCH', {
              key, sessName, expected: verify.expected, matched: verify.matched, ratio: pct, passRatio
            });
            // Log a pixel-watermark mismatch.
            try {
              await appendSessionValidation(uaDir, sessName || (session ? `S${session}` : null), {
                at: new Date().toISOString(),
                key,
                cat,
                base: baseWithParams,
                salt: { expected: expectedMaskSalt, actual: rec.maskSalt, match: true },
                marker: {
                  match: false,
                  passRatio,
                  expected: verify.expected,
                  matched:  verify.matched,
                  ratio:    verify.ratio
                },
                a0Guard: a0Guard,
                pngJudge: null,
                outputs: { rgba: null }
              });
            } catch {}

          } else {
            console.log('[save-batch] pixel watermark OK', {
              key, sessName, expected: verify.expected, matched: verify.matched, ratio: pct, passRatio
            });
          }
          // === PNG-JUDGE: tampered/not based on pixel features (bs-independent) ===
      if (shouldValidateForSrc(src) && PNG_JUDGE.enable) {
        try {
          const feat = rgbaFeatures(f.buffer, 100, 100);
          const tampered = classifyByPngFeatures(feat, PNG_JUDGE);

          console.log('[save-batch] png-judge', {
            key, sessName, cat,
            a0_gray_nonzero_ratio: +feat.a0_gray_nonzero_ratio.toFixed(4),
            uniq_nonzero_lvls: feat.uniq_nonzero_lvls,
            top2_ratio: +feat.top2_ratio.toFixed(4),
            fg_tenengrad: +feat.fg_tenengrad.toFixed(2),
            fg_laplacian_var: +feat.fg_laplacian_var.toFixed(2),
            tampered
          });

          if (PNG_JUDGE_SAVE) {
            const pngPath = path.join(uaDir, `${baseWithParams}.png`);
            await savePngFromRGBA(f.buffer, 100, 100, pngPath);
            console.log('[save-batch] png-judge saved png:', { pngPath });
          }
          // (feat/tampered/logging/PNG_JUDGE_SAVE handling above already happened)
          // Log the PNG-JUDGE outcome regardless of whether it blocks the upload.
          try {
            await appendSessionValidation(uaDir, sessName || (session ? `S${session}` : null), {
              at: new Date().toISOString(),
              key,
              cat,
              base: baseWithParams,
              salt: { expected: expectedMaskSalt, actual: rec.maskSalt, match: true },
              marker: {
                match: true,
                passRatio,
                expected: verify.expected,
                matched:  verify.matched,
                ratio:    verify.ratio
              },
              a0Guard: a0Guard,
              pngJudge: {
                tampered,
                thresholds: {
                  a0GrayNonzeroMin: PNG_JUDGE.a0GrayNonzeroMin,
                  uniqNonzeroMin:   PNG_JUDGE.uniqNonzeroMin,
                  top2RatioMin:     PNG_JUDGE.top2RatioMin
                },
                features: {
                  a0_ratio: +(feat.a0_ratio ?? 0).toFixed(6),
                  a0_nonzero_ratio_any: +(feat.a0_nonzero_ratio_any ?? 0).toFixed(6),
                  a0_gray_nonzero_ratio: +(feat.a0_gray_nonzero_ratio ?? 0).toFixed(6),
                  uniq_nonzero_lvls: feat.uniq_nonzero_lvls,
                  top2_ratio: +(feat.top2_ratio ?? 0).toFixed(6),
                  a0_entropy: +(feat.a0_entropy ?? 0).toFixed(6),
                  fg_ratio: +(feat.fg_ratio ?? 0).toFixed(6),
                  fg_tenengrad: +(feat.fg_tenengrad ?? 0).toFixed(2),
                  fg_laplacian_var: +(feat.fg_laplacian_var ?? 0).toFixed(2)
                }
              }
            });
          } catch {}


        } catch (e) {
          console.warn('[save-batch] png-judge error', e?.message || e);
          // Do not block the upload on PNG-JUDGE errors.
        }
      }
  }
  } catch (e) {
  console.warn('[save-batch verify error]', e?.message || e);
  return res.status(500).json({ ok:false, error:'mask verification error' });
}
      
      const full = path.join(uaDir, `${baseWithParams}.rgba`);
      await fs.promises.writeFile(full, f.buffer);

      // Surface a pixelSample preview in samples_preview.ndjson so all 9 keys are listed
      // (the PNG path below also writes here, this covers the raw RGBA keys).
      const previewBytes = Math.min(32, f.buffer.length);
      const pixelSample = Array.from(f.buffer.slice(0, previewBytes));
      try {
        await upsertPixelsIndex(uaDir, baseWithParams, {
          key, ts, w: 100, h: 100, channels: 4, pixelSample,
        });
      } catch (e) {
        console.warn('[save-batch] samples_preview.ndjson update failed (raw):', e.message);
      }
      items.push({
        key, bytes: f.size, w: null, h: null,
        kind: 'raw', appendedBs: baseWithParams !== `${key}_${ts}`
      });
  
    }

    // 2.2 Process the PNG batch.
    for (const f of filesPNG) {
      if (!f.buffer) f.buffer = await fs.promises.readFile(f.path);
      const rawName = String((f.originalname || '').replace(/\.png$/i, ''));
      const key = sanitizeName(rawName, `unnamed_${Date.now()}`);
      const ts  = getLocalTimestamp();
      const base = path.join(uaDir, `${key}_${ts}`);

      // Decode PNG -> raw RGBA.
      const { data: rawRGBA, info } = await sharp(f.buffer)
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

      // Preview the first 8 RGBA pixels (32 channel values) so a reader can
      // eyeball the saved buffer. Order is [R0,G0,B0,A0, R1,G1,B1,A1, …].
      const previewBytes = Math.min(32, rawRGBA.length);
      const pixelSample = Array.from(rawRGBA.slice(0, previewBytes));
      const metaSidecar = {
        key, ts,
        w: info.width,
        h: info.height,
        channels: info.channels || 4,
        pixelSample,
      };

      // Persist pixels to disk.
      if (PIXEL_STORE === 'rgba' || PIXEL_STORE === 'both') {
        await fs.promises.writeFile(`${base}.rgba`, rawRGBA);
      }
      if (PIXEL_STORE === 'json' || PIXEL_STORE === 'json.gz' || PIXEL_STORE === 'both') {
        const jsonStr = pixelsToJsonLines(rawRGBA, info.channels || 4);
        if (PIXEL_STORE === 'json') {
          await fs.promises.writeFile(`${base}.json`, jsonStr, 'utf8');
        } else { // json.gz or both
          const gz = zlib.gzipSync(Buffer.from(jsonStr, 'utf8'));
          await fs.promises.writeFile(`${base}.json.gz`, gz);
        }
      }

      // Persist the PNG to disk only when requested.
      if (needPNG(req)) {
        const pngName = `${key}_${ts}.png`;
        await fs.promises.writeFile(path.join(uaDir, pngName), f.buffer);
      }

      // Update the per-directory index (supersedes per-file *.meta.json).
      await upsertPixelsIndex(uaDir, path.basename(base), metaSidecar);

      items.push({ key, bytes: f.size, w: info.width, h: info.height, kind: 'png' });
    }

    res.set('X-Received-Bytes', String(totalBytes));
    res.status(200).json({
      ok: true, session, count: items.length, items,
      saved: {
        rgba:    (PIXEL_STORE === 'rgba' || PIXEL_STORE === 'both'),
        json:    (PIXEL_STORE === 'json' || PIXEL_STORE === 'both'),
        json_gz: (PIXEL_STORE === 'json.gz' || PIXEL_STORE === 'both'),
        png:     needPNG(req)
      },
      deviceId, ...meta
    });
  } catch (e) {
    console.error('[save-batch] error:', e);
    res.status(400).json({ ok: false, error: String(e) });
  }
});


/* -----------------------------------------------------------------------------
 * /pp/frozen.js — return the generated runtime (NUM_SUB sub-sessions)
 * --------------------------------------------------------------------------- */
app.get('/pp/frozen.js', async (req, res) => {
  try {
    const deviceId = issueDeviceIdIfNeeded(req, res);
    const visitNonce = crypto.randomBytes(16).toString('base64url');

    const nsRaw = Number(req.query.ns || NaN);
    const nsClamped = Number.isFinite(nsRaw) ? Math.max(1, Math.min(nsRaw, NUM_SESSIONS_LOCAL)) : NUM_SESSIONS_LOCAL;

    const plan = makePlanOnServer({
      deviceId,
      visitNonce,
      numSubSessions: nsClamped
    });

        // Resolve src (the client passes ?src=visit1).
    const src = sanitizeName(req.query.src || 'visit1');
    res.cookie('PP_LAST_SRC', src, {
        path: '/', httpOnly: false, sameSite: 'Lax', secure: isSecureReq(req), maxAge: ONE_YEAR_MS
      });

    // Record bs; derive maskSalt with the server secret and surface it back.
        // Record bs; for randomFont, the server derives maskSalt and surfaces it back.
      const catsToRecord = ['faces','persons','travel','hands','randomFont']; 
      const emojiCats = ['faces','persons','travel','hands'];
      for (let sIdx = 0; sIdx < plan.sessions.length; sIdx++) {
        const sessionName = `S${sIdx + 1}`;
        const sess = plan.sessions[sIdx] || {};

        for (const cat of catsToRecord) {
          const raw = sess[cat] || {};
          const v = Number.isFinite(raw.bs) ? (raw.bs|0)
                : Number.isFinite(raw.bsCode) ? (raw.bsCode|0)
                : Number.isFinite(raw.brightnessShift) ? (raw.brightnessShift|0)
                : null;
          if (v == null) continue;

          if (cat === 'randomFont') {
            // Same derivation, using the randomFont canonicalization.
            const canon = _rfCanon(raw);
            const serverNonce = crypto.randomBytes(16);
            const derived = _deriveMaskSalt(RNG_SECRET, canon, sIdx+1, serverNonce);

            raw.maskSalt = derived >>> 0;
            raw.D = WATERMARK_DENSITY;
            raw.sIdx = sIdx + 1;
            putBsRecord(deviceId, src, sessionName, cat, {
              bs: (raw?.brightnessShift|0),
              maskSalt: derived >>> 0,
              density: WATERMARK_DENSITY,
              ts: Date.now(),
              planSnapshot: JSON.stringify(raw || {}),
              rfCanon: canon,
              serverNonce: serverNonce.toString('base64'),
            });
            } else if (emojiCats.includes(cat)) {
              const canon = _emojiCanon(raw);
              const serverNonce = crypto.randomBytes(16);
              const derived = _deriveMaskSalt(RNG_SECRET, canon, sIdx+1, serverNonce);

              // Surface to the client so it can write the watermark on A==0 pixels.
              raw.maskSalt = derived >>> 0;
              raw.D = WATERMARK_DENSITY;
              raw.sIdx = sIdx + 1;

              putBsRecord(deviceId, src, sessionName, cat, {
                bs: v,
                maskSalt: derived >>> 0,
                density: WATERMARK_DENSITY,
                ts: Date.now(),
                planSnapshot: JSON.stringify(raw || {}),
                emojiCanon: canon,
                serverNonce: serverNonce.toString('base64')
              });
            }
           else {
            // Other categories: record as-is (use the client's maskSalt if it provided one, otherwise null).
            const m = Number.isFinite(raw.maskSalt) ? (raw.maskSalt|0) : null;
            putBsRecord(deviceId, src, sessionName, cat, {
              bs: v, maskSalt: m, density: (bsLatest.get(`${deviceId}::${sessionName}::${cat}`)?.density) ?? WATERMARK_DENSITY, ts: Date.now()
            });
          }
        }
      }

    const js = buildFrozenBundle({
      sessions: plan.sessions,
      numSubSessions: plan.numSubSessions,
      width: 100,
      height: 100
    });

    res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
    res.setHeader('Cache-Control', 'no-store');
    res.status(200).send(js);
  } catch (e) { console.error(e); res.status(500).send('// frozen error'); }
});

app.get('/export-json', async (req, res) => {
  try {
    // base: stem of the file (no extension)
    // gz:   1 = return gzip-encoded
    // save: 1 = also write the result to disk (.json or .json.gz)
    const dir  = String(req.query.dir || '').replace(/[^a-zA-Z0-9._\-\/]/g, '');
    const base = String(req.query.base || '').replace(/[^a-zA-Z0-9._\-]/g, '');
    const gz   = String(req.query.gz || '0') === '1';
    const save = String(req.query.save || '0') === '1';

    if (!dir || !base) return res.status(400).json({ ok:false, error:'need dir & base' });
    // Reject path traversal: the sanitization regex above keeps "." and "/", so
    // a payload like dir=../../etc would survive. Resolve and confirm the result
    // is still inside saveDir before touching the filesystem.
    if (dir.includes('..') || base.startsWith('.')) {
      return res.status(400).json({ ok:false, error:'bad path' });
    }
    const uaDir = path.resolve(saveDir, dir);
    if (uaDir !== saveDir && !uaDir.startsWith(saveDir + path.sep)) {
      return res.status(400).json({ ok:false, error:'bad path' });
    }
    const rgbaPath = path.join(uaDir, `${base}.rgba`);
    if (!fs.existsSync(rgbaPath)) {
      return res.status(404).json({ ok:false, error:'rgba not found' });
    }

    const channels = await readChannelsFromIndex(uaDir, base); // fall back to 4 when no index entry exists
    const rgba = await fs.promises.readFile(rgbaPath);
    const jsonStr = pixelsToJsonLines(rgba, channels);

    if (save) {
      if (gz) {
        const zlib = require('zlib');
        const out = zlib.gzipSync(Buffer.from(jsonStr,'utf8'));
        await fs.promises.writeFile(path.join(uaDir, `${base}.json.gz`), out);
      } else {
        await fs.promises.writeFile(path.join(uaDir, `${base}.json`), jsonStr, 'utf8');
      }
    }

    if (gz) {
      const zlib = require('zlib');
      const out = zlib.gzipSync(Buffer.from(jsonStr, 'utf8'));
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Content-Encoding', 'gzip');
      return res.status(200).send(out);
    }
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.status(200).send(jsonStr);
  } catch (e) {
    console.error('[export-json] error:', e);
    res.status(500).json({ ok:false, error:String(e) });
  }
});


/* -----------------------------------------------------------------------------
 * POC: list and serve the current device's saved .rgba canvas traces
 * --------------------------------------------------------------------------- */
// Per-key sample cap for the showcase viewer; matches the enrollment budget.
const TRACES_PER_KEY_CAP = 100;

app.get('/images', (req, res) => {
  const deviceId = issueDeviceIdIfNeeded(req, res);
  const devDir = path.join(saveDir, deviceId);
  if (!fs.existsSync(devDir)) {
    return res.json({ ok: true, deviceId, byKey: {}, total: 0, totalOnDisk: 0 });
  }
  // collect (relPath, mtime) per key
  const byKey = {};
  let totalOnDisk = 0;
  for (const visit of fs.readdirSync(devDir)) {
    const vdir = path.join(devDir, visit);
    if (!fs.statSync(vdir).isDirectory()) continue;
    for (const f of fs.readdirSync(vdir)) {
      if (!f.endsWith('.rgba')) continue;
      const m = f.match(/^S\d+_(?:raw_)?([A-Za-z0-9-]+)_\d+_/);
      const key = m ? m[1] : 'other';
      const full = path.join(vdir, f);
      let mt = 0;
      try { mt = fs.statSync(full).mtimeMs; } catch {}
      (byKey[key] = byKey[key] || []).push({ rel: `${visit}/${f}`, mt });
      totalOnDisk++;
    }
  }
  // Sort by mtime then name, take first TRACES_PER_KEY_CAP per key.
  const trimmed = {};
  for (const k of Object.keys(byKey)) {
    const sorted = byKey[k].sort((a, b) => (a.mt - b.mt) || a.rel.localeCompare(b.rel));
    trimmed[k] = sorted.slice(0, TRACES_PER_KEY_CAP).map(x => x.rel);
  }
  const total = Object.values(trimmed).reduce((a, b) => a + b.length, 0);
  return res.json({ ok: true, deviceId, byKey: trimmed, total, totalOnDisk, perKeyCap: TRACES_PER_KEY_CAP });
});

app.get('/image', (req, res) => {
  const deviceId = issueDeviceIdIfNeeded(req, res);
  const rel = String(req.query.path || '');
  // strict path sanitization: only <visitN>/<filename>, no ".." no absolute
  if (!/^visit\d+\/[A-Za-z0-9._-]+\.rgba$/.test(rel)) {
    return res.status(400).send('bad path');
  }
  const full = path.join(saveDir, deviceId, rel);
  if (!full.startsWith(path.join(saveDir, deviceId) + path.sep)) {
    return res.status(400).send('bad path');
  }
  if (!fs.existsSync(full)) return res.status(404).send('not found');
  res.setHeader('Content-Type', 'application/octet-stream');
  res.setHeader('Cache-Control', 'no-store');
  fs.createReadStream(full).pipe(res);
});

// Inspector for the "Show saved data" panel: dumps every *.json the canvas
// pipeline writes for the current device's visit so a user can verify
// the validation / index / stats files without touching the filesystem.
app.get('/saved-data', async (req, res) => {
  const deviceId = issueDeviceIdIfNeeded(req, res);
  const src = sanitizeName(req.query.src || 'visit1');
  const uaDir = path.join(saveDir, deviceId, src);
  if (!fs.existsSync(uaDir)) {
    return res.json({ ok: true, deviceId, src, exists: false, files: {} });
  }
  const out = { ok: true, deviceId, src, exists: true, files: {}, rgbaCount: 0 };
  const entries = await fs.promises.readdir(uaDir);
  for (const name of entries) {
    if (name.endsWith('.rgba')) { out.rgbaCount += 1; continue; }
    try {
      const raw = await fs.promises.readFile(path.join(uaDir, name), 'utf8');
      if (name.endsWith('.ndjson')) {
        // Append-only NDJSON (validation.ndjson): parse line-by-line and
        // surface as { entries: [...] } so the panel renders it the same
        // as a regular { entries } JSON file.
        const parsed = [];
        for (const line of raw.split('\n')) {
          if (!line.trim()) continue;
          try { parsed.push(JSON.parse(line)); } catch {}
        }
        out.files[name] = { entries: parsed };
      } else if (name.endsWith('.json')) {
        out.files[name] = JSON.parse(raw);
      }
    } catch (e) {
      out.files[name] = { _error: String(e && e.message) };
    }
  }
  res.json(out);
});

// Lightweight health endpoint for Render's healthCheckPath.
app.get('/health', (req, res) => {
  res.status(200).json({ ok: true, node: 'ok' });
});

// Clear the device cookies so a reload yields a fresh deviceId.
app.post('/clear-device', (req, res) => {
  res.clearCookie(COOKIE_NAME_HTTPONLY, { path: '/' });
  res.clearCookie(COOKIE_NAME_READABLE, { path: '/' });
  res.json({ ok: true });
});

/* ---- Web Audio collection routes (mounted under /audio/*); samples land at
 * <persist>/audio/save/<deviceId>/<src>/ ---- */
const { mountAudioRoutes } = require(path.join(__dirname, 'audio'));
const AUDIO_SAVE_ROOT = path.join(PERSIST_ROOT, 'audio', 'save');
mountAudioRoutes(app, {
  audioSaveDir: AUDIO_SAVE_ROOT,
  issueDeviceIdIfNeeded,
  sanitizeName,
  getLocalTimestamp,
});

/* -----------------------------------------------------------------------------
 * Start server
 * --------------------------------------------------------------------------- */
app.listen(argv.port, '0.0.0.0', () => {
  console.log(`Server running on http://localhost:${argv.port}/`);
  if (argv.open) open(`http://localhost:${argv.port}/`);
});