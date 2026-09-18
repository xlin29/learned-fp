'use strict';

/**
 * Web Audio collection module for the LearnedFP framework.
 *
 * Mirrors the canvas pipeline: the server seeds 12 audio probes per session,
 * the client renders each in an OfflineAudioContext, and the raw Float32
 * buffers upload as binary. A per-key marker derived from a server secret is
 * embedded in near-silent samples and verified on receipt (the audio
 * counterpart of the canvas marker of paper §4.2; the audio check itself is
 * not described in the paper).
 *
 * Mount under any base path via `mountAudioRoutes(app, opts)`. Routes:
 *   GET  <base>/bootstrap         — issue/return the device cookie + audio config
 *   GET  <base>/pp/frozen.js      — per-session probe bundle (ESM)
 *   POST <base>/save-batch        — upload one session's 12 .f32 buffers
 *
 * Identity is shared with the canvas server via the same deviceId cookie:
 * the host passes its `issueDeviceIdIfNeeded` function in `opts`.
 */

const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const multer = require('multer');
const { buildAudioFrozenBundle } = require(path.join(__dirname, 'templates', 'audio_frozen_template'));

/* ----------------------------- Audio constants ----------------------------- */
const NUM_SESSIONS = 60;
const SAMPLE_RATE = 44100;
const DURATION = 0.5;
const BUFFER_LENGTH = Math.ceil(SAMPLE_RATE * DURATION); // 22050
const EXPECTED_BYTES = BUFFER_LENGTH * 4;                // 88200 bytes (Float32)
const MARKER_DENSITY = Number(process.env.MARKER_DENSITY ?? 60); // % of near-silent samples to mark

const AUDIO_KEYS = [
  'oscillatorMix', 'biquadChain', 'compressorProbe', 'waveShaperProbe',
  'convolverProbe', 'periodicWaveProbe', 'analyserProbe',
  'channelMixProbe', 'stereoPannerProbe',
  'iirFilterProbe', 'hrtfPannerProbe', 'delayProbe',
];

/* ----------------------------- RNG / PRF ----------------------------- */
const RNG_SECRET = process.env.RNG_SECRET
  ? Buffer.from(process.env.RNG_SECRET, 'base64')
  : crypto.randomBytes(32);

function prfBlock(cid, ns, ctr, rngNonce) {
  const msg = Buffer.from(`${cid}|${ns}|${ctr}|${rngNonce || ''}`);
  return crypto.createHmac('sha256', RNG_SECRET).update(msg).digest();
}
function b64url(buf) {
  return buf.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
function newCid() { return b64url(crypto.randomBytes(16)); }

/* ----------------------------- Probe BASE configs -----------------------------
 *
 * Each probe = canonical BASE + small per-session jitter. Frequency, phase
 * and timing are fixed across sessions (a 1-D CNN on raw audio has no
 * frequency-shift invariance); only amplitude / shape parameters jitter.
 * Base frequencies are non-round (813.7, 247.3, ...) to exercise more
 * floating-point rounding in each engine's sin/cos. The jitter is derived
 * from a server PRF, so buffers cannot be pre-computed from the BASE values.
 */
const PROBE_BASE_CONFIGS = {
  compressorProbe: {
    base: { inputType: 'sawtooth', inputFreq: 813.7, inputGain: 1.0,
            threshold: -30, knee: 6, ratio: 8, attack: 0.005, release: 0.1 },
    jitter: { inputFreq: 0, inputGain: 0.01,
              threshold: 0.5, knee: 0.5, ratio: 0.3,
              attack: 0.0002, release: 0.005 },
  },
  analyserProbe: {
    base: { inputType: 'square', inputFreq: 1547.9,
            fftSize: 512, smoothingTimeConstant: 0.8 },
    jitter: { inputFreq: 0, smoothingTimeConstant: 0.02 },
  },
  periodicWaveProbe: {
    base: {
      fundamentalFreq: 247.3, numHarmonics: 14,
      realCoeffs: [0.5, -0.4, 0.35, -0.3, 0.25, -0.2, 0.18,
                  -0.15, 0.12, -0.1, 0.08, -0.06, 0.04, -0.025],
      imagCoeffs: [-0.45, 0.4, -0.32, 0.28, -0.22, 0.2, -0.16,
                    0.14, -0.11, 0.09, -0.07, 0.05, -0.035, 0.02],
    },
    jitter: { fundamentalFreq: 0, coeff: 0.005 },
  },
  channelMixProbe: {
    base: {
      oscillators: [
        { type: 'triangle', frequency:  247.3, gain: 0.25 },
        { type: 'sine',     frequency: 1373.1, gain: 0.25 },
        { type: 'square',   frequency:  691.3, gain: 0.25 },
        { type: 'sawtooth', frequency: 4423.7, gain: 0.25 },
      ],
    },
    jitter: { frequency: 0, gain: 0.01 },
  },
  oscillatorMix: {
    base: {
      oscillators: [
        { type: 'sine',     frequency:  443.7, detune:  12, gain: 0.20 },
        { type: 'triangle', frequency:  887.3, detune:  -7, gain: 0.20 },
        { type: 'square',   frequency:  221.9, detune:   3, gain: 0.15 },
        { type: 'sawtooth', frequency: 1763.1, detune: -15, gain: 0.15 },
      ],
    },
    jitter: { frequency: 0, detune: 0, gain: 0.01 },
  },
  biquadChain: {
    base: {
      inputType: 'sawtooth', inputFreq: 1057.7,
      filters: [
        { type: 'lowpass',  frequency: 1547.9, Q: 1.0,   gain: 0 },
        { type: 'highpass', frequency:  211.3, Q: 0.707, gain: 0 },
        { type: 'peaking',  frequency: 2547.9, Q: 2.0,   gain: 6 },
      ],
    },
    jitter: { inputFreq: 0, filterFreq: 0.5, Q: 0.05, gain: 0.5 },
  },
  waveShaperProbe: {
    base: {
      inputType: 'sine', inputFreq: 1057.7,
      curveSegments: 256,
      curveCoeffs: [0.0, 1.0, -0.3, 0.0, 0.5],
      oversample: '4x',
      preGain: 1.5,
    },
    jitter: { inputFreq: 0, coeff: 0.005, preGain: 0.02 },
  },
  convolverProbe: {
    base: { inputType: 'sine', inputFreq: 691.3,
            irLength: 2048, irDecay: 3.0, irSeed: 0x123456 },
    jitter: { inputFreq: 0, irDecay: 0.02 },
  },
  stereoPannerProbe: {
    base: { inputType: 'sawtooth', inputFreq: 1547.9, pan: 0.3, gain: 0.5 },
    jitter: { inputFreq: 0, pan: 0, gain: 0.01 },
  },
  iirFilterProbe: {
    base: {
      inputType: 'sawtooth', inputFreq: 1057.7,
      feedforward: [0.0156, 0.0312, 0.0156],
      feedback:    [1.0,    -1.601, 0.6634],
      preGain: 0.5,
    },
    jitter: { inputFreq: 0, coeff: 1e-5, preGain: 0.01 },
  },
  hrtfPannerProbe: {
    base: {
      inputType: 'sine', inputFreq: 1103.7,
      positionX: 1.0, positionY: 0.5, positionZ: 0.5,
      gain: 0.5,
    },
    jitter: { inputFreq: 0, position: 0, gain: 0.01 },
  },
  delayProbe: {
    base: {
      inputType: 'square', inputFreq: 211.3,
      delayTime: 0.030, feedbackGain: 0.5,
      dryGain: 0.5, wetGain: 0.5,
    },
    jitter: { inputFreq: 0, delayTime: 0,
              feedbackGain: 0.01, dryGain: 0.01, wetGain: 0.01 },
  },
};

function generateOneAudioSession(rngCtx) {
  const { f } = rngCtx;
  const jf = (x, amp) => +(x + f(-amp, amp)).toFixed(6);
  // jArr preserves exact zeros so a BASE coefficient pattern doesn't drift into noise.
  const jArr = (arr, amp) => arr.map(x => x === 0 ? 0 : jf(x, amp));
  const B = PROBE_BASE_CONFIGS;

  return {
    sampleRate: SAMPLE_RATE,
    duration: DURATION,
    bufferLength: BUFFER_LENGTH,

    compressorProbe: {
      inputType: B.compressorProbe.base.inputType,
      inputFreq: jf(B.compressorProbe.base.inputFreq, B.compressorProbe.jitter.inputFreq),
      inputGain: jf(B.compressorProbe.base.inputGain, B.compressorProbe.jitter.inputGain),
      threshold: jf(B.compressorProbe.base.threshold, B.compressorProbe.jitter.threshold),
      knee:      jf(B.compressorProbe.base.knee,      B.compressorProbe.jitter.knee),
      ratio:     jf(B.compressorProbe.base.ratio,     B.compressorProbe.jitter.ratio),
      attack:    jf(B.compressorProbe.base.attack,    B.compressorProbe.jitter.attack),
      release:   jf(B.compressorProbe.base.release,   B.compressorProbe.jitter.release),
    },
    analyserProbe: {
      inputType: B.analyserProbe.base.inputType,
      inputFreq: jf(B.analyserProbe.base.inputFreq, B.analyserProbe.jitter.inputFreq),
      fftSize:   B.analyserProbe.base.fftSize,
      smoothingTimeConstant: jf(B.analyserProbe.base.smoothingTimeConstant,
                                B.analyserProbe.jitter.smoothingTimeConstant),
    },
    periodicWaveProbe: {
      fundamentalFreq: jf(B.periodicWaveProbe.base.fundamentalFreq,
                          B.periodicWaveProbe.jitter.fundamentalFreq),
      numHarmonics: B.periodicWaveProbe.base.numHarmonics,
      realCoeffs:   jArr(B.periodicWaveProbe.base.realCoeffs, B.periodicWaveProbe.jitter.coeff),
      imagCoeffs:   jArr(B.periodicWaveProbe.base.imagCoeffs, B.periodicWaveProbe.jitter.coeff),
    },
    channelMixProbe: {
      oscillators: B.channelMixProbe.base.oscillators.map(o => ({
        type:      o.type,
        frequency: jf(o.frequency, B.channelMixProbe.jitter.frequency),
        gain:      jf(o.gain,      B.channelMixProbe.jitter.gain),
      })),
    },
    oscillatorMix: {
      oscillators: B.oscillatorMix.base.oscillators.map(o => ({
        type:      o.type,
        frequency: jf(o.frequency, B.oscillatorMix.jitter.frequency),
        detune:    jf(o.detune,    B.oscillatorMix.jitter.detune),
        gain:      jf(o.gain,      B.oscillatorMix.jitter.gain),
      })),
    },
    biquadChain: {
      inputType: B.biquadChain.base.inputType,
      inputFreq: jf(B.biquadChain.base.inputFreq, B.biquadChain.jitter.inputFreq),
      filters: B.biquadChain.base.filters.map(fc => ({
        type:      fc.type,
        frequency: jf(fc.frequency, B.biquadChain.jitter.filterFreq),
        Q:         jf(fc.Q,         B.biquadChain.jitter.Q),
        gain:      jf(fc.gain,      B.biquadChain.jitter.gain),
      })),
    },
    waveShaperProbe: {
      inputType:     B.waveShaperProbe.base.inputType,
      inputFreq:     jf(B.waveShaperProbe.base.inputFreq, B.waveShaperProbe.jitter.inputFreq),
      curveSegments: B.waveShaperProbe.base.curveSegments,
      curveCoeffs:   jArr(B.waveShaperProbe.base.curveCoeffs, B.waveShaperProbe.jitter.coeff),
      oversample:    B.waveShaperProbe.base.oversample,
      preGain:       jf(B.waveShaperProbe.base.preGain, B.waveShaperProbe.jitter.preGain),
    },
    convolverProbe: {
      inputType: B.convolverProbe.base.inputType,
      inputFreq: jf(B.convolverProbe.base.inputFreq, B.convolverProbe.jitter.inputFreq),
      irLength:  B.convolverProbe.base.irLength,
      irDecay:   jf(B.convolverProbe.base.irDecay, B.convolverProbe.jitter.irDecay),
      irSeed:    B.convolverProbe.base.irSeed,
    },
    stereoPannerProbe: {
      inputType: B.stereoPannerProbe.base.inputType,
      inputFreq: jf(B.stereoPannerProbe.base.inputFreq, B.stereoPannerProbe.jitter.inputFreq),
      pan:       jf(B.stereoPannerProbe.base.pan,       B.stereoPannerProbe.jitter.pan),
      gain:      jf(B.stereoPannerProbe.base.gain,      B.stereoPannerProbe.jitter.gain),
    },
    iirFilterProbe: {
      inputType:   B.iirFilterProbe.base.inputType,
      inputFreq:   jf(B.iirFilterProbe.base.inputFreq, B.iirFilterProbe.jitter.inputFreq),
      feedforward: jArr(B.iirFilterProbe.base.feedforward, B.iirFilterProbe.jitter.coeff),
      // feedback[0] must remain exactly 1.0 (denominator normalization).
      feedback:    [1.0, ...jArr(B.iirFilterProbe.base.feedback.slice(1),
                                 B.iirFilterProbe.jitter.coeff)],
      preGain:     jf(B.iirFilterProbe.base.preGain, B.iirFilterProbe.jitter.preGain),
    },
    hrtfPannerProbe: {
      inputType: B.hrtfPannerProbe.base.inputType,
      inputFreq: jf(B.hrtfPannerProbe.base.inputFreq, B.hrtfPannerProbe.jitter.inputFreq),
      positionX: jf(B.hrtfPannerProbe.base.positionX, B.hrtfPannerProbe.jitter.position),
      positionY: jf(B.hrtfPannerProbe.base.positionY, B.hrtfPannerProbe.jitter.position),
      positionZ: jf(B.hrtfPannerProbe.base.positionZ, B.hrtfPannerProbe.jitter.position),
      gain:      jf(B.hrtfPannerProbe.base.gain,      B.hrtfPannerProbe.jitter.gain),
    },
    delayProbe: {
      inputType:    B.delayProbe.base.inputType,
      inputFreq:    jf(B.delayProbe.base.inputFreq,    B.delayProbe.jitter.inputFreq),
      delayTime:    jf(B.delayProbe.base.delayTime,    B.delayProbe.jitter.delayTime),
      feedbackGain: jf(B.delayProbe.base.feedbackGain, B.delayProbe.jitter.feedbackGain),
      dryGain:      jf(B.delayProbe.base.dryGain,      B.delayProbe.jitter.dryGain),
      wetGain:      jf(B.delayProbe.base.wetGain,      B.delayProbe.jitter.wetGain),
    },
  };
}

function makePlanOnServer({ deviceId, visitNonce, numSubSessions = NUM_SESSIONS } = {}) {
  const NUM_SUB = Math.max(1, Number(numSubSessions) || NUM_SESSIONS);
  const COUNT = 1024;

  function makeRngCtxFromPool(pool) {
    let poolIdx = 0;
    function r53() {
      if (poolIdx + 8 > pool.length) poolIdx = 0;
      const a = pool.readUInt32BE(poolIdx);
      const b = pool.readUInt32BE(poolIdx + 4) >>> 11;
      poolIdx += 8;
      return (a * 2097152 + b) / 9007199254740992;
    }
    const fFn = (lo, hi) => lo + (hi - lo) * r53();
    function int(lo, hi) { return (lo + Math.floor((hi - lo + 1) * r53())) | 0; }
    return { r53, f: fFn, i: int, pick: arr => arr[Math.floor(r53() * arr.length)] };
  }

  const sessions = [];
  const seeds = [];
  for (let k = 0; k < NUM_SUB; k++) {
    const cidK = newCid();
    const rngNonceK = b64url(crypto.randomBytes(16));
    const NSk = `audio:${visitNonce || 'v'}:${k}`;
    const blocks = [];
    for (let ctr = 0; ctr < COUNT; ctr++) {
      blocks.push(prfBlock(cidK, NSk, ctr, rngNonceK));
    }
    const poolK = Buffer.concat(blocks);
    sessions.push(generateOneAudioSession(makeRngCtxFromPool(poolK)));
    seeds.push({ k, cid: cidK, rngNonce: rngNonceK, ns: NSk, count: COUNT });
  }
  return { sessions, numSubSessions: NUM_SUB, rngSeeds: seeds, meta: { deviceId, visitNonce: visitNonce || '' } };
}

/* ----------------------------- Marker helpers ----------------------------- */
function _deriveMarkerSalt(secretBuf, keyName, sIdx, serverNonceBuf) {
  const h = crypto.createHmac('sha256', secretBuf)
    .update(keyName).update('|')
    .update(String(sIdx)).update('|')
    .update(serverNonceBuf)
    .digest();
  return (h.readUInt32BE(0) & 0x7fffffff) >>> 0;
}

function xorshift32(u) {
  u = u | 0;
  u ^= u << 13; u ^= u >>> 17; u ^= u << 5;
  return u >>> 0;
}

function verifyAudioMarker(f32Buf, keyName, rec, opts = {}) {
  if (!f32Buf || (f32Buf.length !== BUFFER_LENGTH && f32Buf.length !== BUFFER_LENGTH * 2)) {
    return { ok: false, reason: 'bad-buffer', expected: 0, matched: 0, ratio: 0 };
  }
  // Stereo probes: only verify the first channel.
  if (f32Buf.length === BUFFER_LENGTH * 2) f32Buf = f32Buf.slice(0, BUFFER_LENGTH);

  const markerSalt = (rec.markerSalt | 0) >>> 0;
  const markerValue = rec.markerValue;
  const D = rec.density || MARKER_DENSITY;
  const sIdx = rec.sIdx || 0;
  const tolerance = 1e-6;

  let expected = 0, matched = 0;
  for (let i = 0; i < BUFFER_LENGTH; i++) {
    let seed = (((markerSalt + sIdx * 1315423911) | 0) ^ (i * 2654435761)) | 0;
    const h = xorshift32(seed);
    if ((h % 100) < D) {
      // Client embeds marker only on near-silent samples (|x| < 0.01).
      if (Math.abs(f32Buf[i]) < 0.01 || Math.abs(f32Buf[i] - markerValue) < tolerance) {
        if (Math.abs(f32Buf[i] - markerValue) < tolerance) {
          expected++;
          matched++;
        } else if (Math.abs(f32Buf[i]) < 0.01) {
          // Eligible slot but the marker wasn't applied (could be defense interference).
          expected++;
        }
      }
    }
  }
  const ratio = expected > 0 ? matched / expected : 1;
  const passRatio = opts.passRatio ?? 0.80;
  return { ok: ratio >= passRatio, reason: ratio >= passRatio ? 'ok' : 'low-match', expected, matched, ratio };
}

/* ----------------------------- In-memory stores with TTL ----------------------------- */
const planStore = new Map();   // deviceId::src -> { sessions, _ts }
const markerStore = new Map(); // deviceId::src::session::keyName -> rec
const STORE_TTL_MS = 30 * 60 * 1000;
const STORE_CAP = 5000; // safety cap so a runaway client can't OOM us

setInterval(() => {
  const cutoff = Date.now() - STORE_TTL_MS;
  for (const [k, v] of planStore)   if (v._ts && v._ts < cutoff) planStore.delete(k);
  for (const [k, v] of markerStore) if (v._ts && v._ts < cutoff) markerStore.delete(k);
}, 5 * 60 * 1000).unref();

function evictOldestIfFull(m) {
  if (m.size < STORE_CAP) return;
  const k = m.keys().next().value;
  if (k !== undefined) m.delete(k);
}
function markerKey(deviceId, src, session, keyName) { return `${deviceId}::${src}::${session}::${keyName}`; }
function putMarkerRecord(deviceId, src, session, keyName, rec) {
  evictOldestIfFull(markerStore);
  markerStore.set(markerKey(deviceId, src, session, keyName), { ...rec, _ts: Date.now() });
}
function findMarkerRecord(deviceId, src, session, keyName) {
  return markerStore.get(markerKey(deviceId, src, session, keyName)) || null;
}

/* ----------------------------- Validation log (single NDJSON) -----------------------------
 * One append-only validation.ndjson per device-visit; race-free under
 * concurrent /save-batch because fs.appendFile preserves O_APPEND ordering.
 */
async function appendSessionValidation(uaDir, sessName, record) {
  const s = String(sessName || '').trim();
  const valPath = path.join(uaDir, 'validation.ndjson');
  const line = JSON.stringify({ sample: s || null, ...record }) + '\n';
  await fs.promises.appendFile(valPath, line, 'utf8');
}

/* ----------------------------- Multer (in-memory; .f32 uploads are small) ----------------------------- */
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { files: 1024, fileSize: 50 * 1024 * 1024 },
});

/* ----------------------------- Mount ----------------------------- */
function mountAudioRoutes(app, opts = {}) {
  const {
    audioSaveDir,                  // /<persist>/audio/save
    issueDeviceIdIfNeeded,
    sanitizeName,
    getLocalTimestamp,
    base = '/audio',               // route prefix
  } = opts;

  if (!audioSaveDir || !issueDeviceIdIfNeeded || !sanitizeName ||
      !getLocalTimestamp) {
    throw new Error('mountAudioRoutes: missing required opts');
  }
  fs.mkdirSync(audioSaveDir, { recursive: true });

  function makeAudioParticipantDir(deviceId, src) {
    const safeSrc = sanitizeName(src);
    return safeSrc ? path.join(audioSaveDir, deviceId, safeSrc) : path.join(audioSaveDir, deviceId);
  }

  app.get(`${base}/bootstrap`, (req, res) => {
    const deviceId = issueDeviceIdIfNeeded(req, res);
    res.setHeader('Cache-Control', 'no-store');
    res.json({
      ok: true,
      deviceId,
      audio: {
        keys: AUDIO_KEYS,
        sampleRate: SAMPLE_RATE,
        bufferLength: BUFFER_LENGTH,
        duration: DURATION,
        numSessions: NUM_SESSIONS,
      },
    });
  });

  app.get(`${base}/pp/frozen.js`, async (req, res) => {
    try {
      const deviceId = issueDeviceIdIfNeeded(req, res);
      const src = sanitizeName(req.query.src || 'visit1');
      const visitNonce = crypto.randomBytes(16).toString('base64url');

      const nsRaw = Number(req.query.ns || NaN);
      const nsClamped = Number.isFinite(nsRaw) ? Math.max(1, Math.min(nsRaw, NUM_SESSIONS)) : NUM_SESSIONS;

      const plan = makePlanOnServer({ deviceId, visitNonce, numSubSessions: nsClamped });
      planStore.set(`${deviceId}::${src}`, { sessions: plan.sessions, _ts: Date.now() });

      // Persist a clean copy of the plan (without server-only _marker fields)
      // so the per-session probe parameters can be read offline.
      const planDir = makeAudioParticipantDir(deviceId, src);
      await fs.promises.mkdir(planDir, { recursive: true });
      const cleanSessions = plan.sessions.map(s => {
        const c = {};
        for (const [k, v] of Object.entries(s)) {
          if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
            const { _marker, ...rest } = v;
            c[k] = rest;
          } else c[k] = v;
        }
        return c;
      });
      fs.promises.writeFile(
        path.join(planDir, 'sample_params.json'),
        JSON.stringify(cleanSessions, null, 2), 'utf8',
      ).catch(e => console.warn('[audio plan save]', e.message));

      // Embed a per-key freshness marker in each session.
      for (let sIdx = 0; sIdx < plan.sessions.length; sIdx++) {
        const sessionName = `S${sIdx + 1}`;
        const sess = plan.sessions[sIdx];
        for (const keyName of AUDIO_KEYS) {
          const serverNonce = crypto.randomBytes(16);
          const derived = _deriveMarkerSalt(RNG_SECRET, keyName, sIdx + 1, serverNonce);
          // Marker value: small float in [-0.001, 0.001] derived from the salt.
          const markerFloat = ((derived % 2000) - 1000) / 1000000;

          sess[keyName] = sess[keyName] || {};
          sess[keyName]._marker = {
            salt: derived >>> 0,
            value: +markerFloat.toFixed(7),
            density: MARKER_DENSITY,
            sIdx: sIdx + 1,
          };

          putMarkerRecord(deviceId, src, sessionName, keyName, {
            markerSalt: derived >>> 0,
            markerValue: markerFloat,
            density: MARKER_DENSITY,
            sIdx: sIdx + 1,
            ts: Date.now(),
            serverNonce: serverNonce.toString('base64'),
          });
        }
      }

      const js = buildAudioFrozenBundle({
        sessions: plan.sessions,
        numSubSessions: plan.numSubSessions,
        sampleRate: SAMPLE_RATE,
        duration: DURATION,
        bufferLength: BUFFER_LENGTH,
      });
      res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
      res.setHeader('Cache-Control', 'no-store');
      res.status(200).send(js);
    } catch (e) {
      console.error('[audio frozen.js]', e);
      res.status(500).send('// audio frozen error: ' + String(e && e.message));
    }
  });

  app.post(`${base}/save-batch`,
    upload.fields([{ name: 'bin', maxCount: 1024 }]),
    async (req, res) => {
      try {
        if (req.socket && req.socket.setNoDelay) req.socket.setNoDelay(true);

        const deviceId = issueDeviceIdIfNeeded(req, res);
        const session = String(req.query.session || '');
        const src = sanitizeName(req.query.src || 'visit1');
        const uaDir = makeAudioParticipantDir(deviceId, src);
        await fs.promises.mkdir(uaDir, { recursive: true });

        const filesBin = (req.files && req.files['bin']) || [];
        const totalBytes = Number(req.get('content-length') || 0);
        console.log(`[audio save-batch] device=${deviceId} src=${src} session=${session} files=${filesBin.length} bytes=${totalBytes}`);
        if (!filesBin.length) {
          return res.status(400).json({ ok: false, error: 'no files in multipart' });
        }

        const results = [];
        for (const f of filesBin) {
          const rawName = String((f.originalname || '').replace(/\.f32$/i, ''));
          const key = sanitizeName(rawName, `unnamed_${Date.now()}`);
          const ts = getLocalTimestamp();

          // Filename is S1_audio_oscillatorMix → session=S1, audioKey=oscillatorMix.
          const m = key.match(/^S(\d+)_audio_([A-Za-z0-9_-]+)/);
          const sessName = m ? `S${m[1]}` : '';
          const audioKey = m ? m[2] : '';

          const isStereo = f.buffer.length === EXPECTED_BYTES * 2;
          if (f.buffer.length !== EXPECTED_BYTES && !isStereo) {
            console.warn(`[audio save-batch] wrong size for ${key}: ${f.buffer.length} != ${EXPECTED_BYTES}`);
            results.push({ key, ok: false, reason: 'wrong-size' });
            continue;
          }

          const f32Len = isStereo ? BUFFER_LENGTH * 2 : BUFFER_LENGTH;
          const f32 = new Float32Array(f.buffer.buffer, f.buffer.byteOffset, f32Len);
          const hash = crypto.createHash('sha256').update(f.buffer).digest('hex');

          let markerResult = null;
          if (sessName && audioKey) {
            const rec = findMarkerRecord(deviceId, src, sessName, audioKey);
            if (rec) markerResult = verifyAudioMarker(f32, audioKey, rec);
          }

          const baseName = `${key}_${ts}`;
          const outPath = path.join(uaDir, `${baseName}.f32`);
          await fs.promises.writeFile(outPath, f.buffer);

          // First 32 float samples — preview snippet that mirrors canvas's
          // pixelSample (also 32 entries). Lets anyone eyeball each
          // recording without loading the full .f32 buffer.
          const audioSample = Array.from(f32.slice(0, Math.min(32, f32.length)));

          // Append-only per-sample preview. Up to 8 concurrent /save-batch
          // handlers race on this file; appendFile preserves O_APPEND
          // ordering so each line lands intact (preview only,
          // the .f32 filenames on disk are the source of truth).
          const idxPath = path.join(uaDir, 'samples_preview.ndjson');
          const line = JSON.stringify({
            baseName,
            key: audioKey || key,
            sample: sessName,
            ts,
            sampleRate: SAMPLE_RATE,
            bufferLength: BUFFER_LENGTH,
            stereo: isStereo,
            audioSample,
          }) + '\n';
          await fs.promises.appendFile(idxPath, line, 'utf8');

          if (markerResult && sessName) {
            await appendSessionValidation(uaDir, sessName, {
              at: new Date().toISOString(),
              key,
              audioKey,
              marker: markerResult,
            });
          }
          results.push({ key, ok: true, hash: hash.slice(0, 16), marker: markerResult });
        }

        return res.status(200).json({
          ok: true, deviceId, src, session, count: results.length, results,
        });
      } catch (e) {
        console.error('[audio save-batch] error:', e);
        if (e && (e.code === 'ENOSPC' || /ENOSPC|no space left/i.test(String(e)))) {
          return res.status(507).json({ ok: false, error: 'storage full', code: 'ENOSPC' });
        }
        return res.status(400).json({ ok: false, error: String(e && e.message) });
      }
    }
  );

  // Lightweight inspector for the "Show saved data" UI.
  // Lists files written for the current device under the audio store.
  app.get(`${base}/list`, async (req, res) => {
    const deviceId = issueDeviceIdIfNeeded(req, res);
    const src = sanitizeName(req.query.src || 'visit1');
    const uaDir = makeAudioParticipantDir(deviceId, src);
    if (!fs.existsSync(uaDir)) {
      return res.json({ ok: true, deviceId, src, files: [], totalBytes: 0 });
    }
    const entries = await fs.promises.readdir(uaDir);
    const files = [];
    let totalBytes = 0;
    for (const name of entries) {
      try {
        const st = await fs.promises.stat(path.join(uaDir, name));
        if (!st.isFile()) continue;
        files.push({ name, bytes: st.size, mtimeMs: st.mtimeMs });
        totalBytes += st.size;
      } catch {}
    }
    files.sort((a, b) => a.mtimeMs - b.mtimeMs || a.name.localeCompare(b.name));
    res.json({ ok: true, deviceId, src, files, totalBytes });
  });

  // Inspector for the "Show saved data" panel: dumps every *.json the audio
  // pipeline writes (sample_params, samples, validation_S*,
  // meta) plus a count of .f32 buffers, so the freshness-marker
  // results can be read without filesystem access.
  app.get(`${base}/saved-data`, async (req, res) => {
    const deviceId = issueDeviceIdIfNeeded(req, res);
    const src = sanitizeName(req.query.src || 'visit1');
    const uaDir = makeAudioParticipantDir(deviceId, src);
    if (!fs.existsSync(uaDir)) {
      return res.json({ ok: true, deviceId, src, exists: false, files: {}, f32Count: 0 });
    }
    const out = { ok: true, deviceId, src, exists: true, files: {}, f32Count: 0 };
    const entries = await fs.promises.readdir(uaDir);
    for (const name of entries) {
      if (name.endsWith('.f32')) { out.f32Count += 1; continue; }
      // sample_params.json is internal book-keeping (per-session probe
      // jitter plan); not surfaced in the saved-data panel.
      if (name === 'sample_params.json') continue;
      try {
        const raw = await fs.promises.readFile(path.join(uaDir, name), 'utf8');
        if (name.endsWith('.ndjson')) {
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
}

module.exports = {
  mountAudioRoutes,
  AUDIO_KEYS,
  SAMPLE_RATE,
  BUFFER_LENGTH,
  DURATION,
  NUM_SESSIONS,
};
