(() => {
  'use strict';

  // === Native references captured at document_start =====================
  const Canvas = HTMLCanvasElement;
  const Ctx2D = CanvasRenderingContext2D;
  const nativeGetImageData = Ctx2D.prototype.getImageData;
  const nativePutImageData = Ctx2D.prototype.putImageData;
  const nativeToDataURL = Canvas.prototype.toDataURL;
  const nativeToBlob = Canvas.prototype.toBlob;
  const nativeGetContext = Canvas.prototype.getContext;
  const nativeCreateElement = Document.prototype.createElement;
  const nativeImageData = ImageData;
  const nativeFunctionToString = Function.prototype.toString;

  // === Per-canvas state (keyed by HTMLCanvasElement OR OffscreenCanvas) =
  const canvasState = new WeakMap();

  function getState(canvas) {
    let s = canvasState.get(canvas);
    if (!s) {
      s = {
        generation: 0,
        cachedSnapshot: null,
        cachedRaw: null,
        cachedGen: -1,
        cachedW: 0,
        cachedH: 0,
      };
      canvasState.set(canvas, s);
    }
    return s;
  }

  function bumpGeneration(canvas) {
    const s = getState(canvas);
    s.generation++;
    s.cachedSnapshot = null;
    s.cachedRaw = null;
  }

  const internalCanvases = new WeakSet();

  function makeInternalCanvas(w, h) {
    const c = nativeCreateElement.call(document, 'canvas');
    c.width = w;
    c.height = h;
    internalCanvases.add(c);
    return c;
  }

  // === Display-restore state ============================================
  // Keep the sanitized bytes issued by getImageData and the original raw
  // pixels; when putImageData receives bytes identical to what was issued,
  // draw the originals instead, so the display shows the real picture while
  // scripts only ever saw sanitized bytes. Mutated bytes are drawn as given.
  // Not covered: canvas.captureStream / MediaRecorder.
  const sanitizedOrigin = new WeakMap();

  function bytesEqual(a, b) {
    const n = a.length;
    if (n !== b.length) return false;
    for (let i = 0; i < n; i++) if (a[i] !== b[i]) return false;
    return true;
  }

  function recordIssuedAndOriginal(outData, outRaw, width, height) {
    sanitizedOrigin.set(outData, {
      issuedCopy: new Uint8ClampedArray(outData),
      original: outRaw,
      width,
      height,
    });
  }

  function tryRestoreImgData(imgData) {
    if (!imgData || !imgData.data) return null;
    const orig = sanitizedOrigin.get(imgData.data);
    if (!orig) return null;
    if (imgData.width !== orig.width || imgData.height !== orig.height) return null;
    if (!bytesEqual(imgData.data, orig.issuedCopy)) return null;
    return new nativeImageData(
      new Uint8ClampedArray(orig.original),
      orig.width,
      orig.height
    );
  }

  // === Budget + policy (per-frame) ======================================
  const budget = { readbacks: 0 };

  const config = {
    coarseThreshold: 3,
    blockThreshold: 20,
    blockMode: 'uniform',  // 'uniform' (zero-entropy gray) | 'canonical' (median + 3-bit, content-preserving)
  };

  window.addEventListener('message', (e) => {
    if (e.source !== window) return;
    const d = e.data;
    if (!d) return;
    if (d.__LFPMitigationCfg === true && d.config) {
      if (typeof d.config.coarseThreshold === 'number') {
        config.coarseThreshold = d.config.coarseThreshold;
      }
      if (typeof d.config.blockThreshold === 'number') {
        config.blockThreshold = d.config.blockThreshold;
      }
      if (d.config.blockMode === 'uniform' || d.config.blockMode === 'canonical') {
        config.blockMode = d.config.blockMode;
      }
    }
    if (d.__LFPMitigationReady === true) {
      window.__LFPMitigationReady = true;
      try { document.dispatchEvent(new Event('learnedfp-mitigation-ready')); } catch (_) {}
    }
  });

  function currentMode() {
    if (budget.readbacks <= config.coarseThreshold) return 'normal';
    if (budget.readbacks <= config.blockThreshold) return 'coarse';
    return 'block';
  }

  // === Sanitization =====================================================

  function canonicalizeTransparentPixels(data) {
    for (let i = 3; i < data.length; i += 4) {
      if (data[i] === 0) {
        data[i - 3] = 0;
        data[i - 2] = 0;
        data[i - 1] = 0;
      }
    }
    return data;
  }

  function sanitize(data) {
    return canonicalizeTransparentPixels(data);
  }

  function quantizeHighNibble(data) {
    const out = new Uint8ClampedArray(data.length);
    for (let i = 0; i < data.length; i += 4) {
      out[i]     = data[i]     & 0xF0;
      out[i + 1] = data[i + 1] & 0xF0;
      out[i + 2] = data[i + 2] & 0xF0;
      out[i + 3] = data[i + 3];
    }
    return out;
  }

  // === Canonical (block-mode) normalization =============================
  // Content-preserving alternative to a uniform-constant block:
  //   1. transparent-pixel canonicalization
  //   2. 3x3 median filter (suppresses 1-pixel AA / subpixel jitter)
  //   3. 3-bit-per-channel quantization
  // Deterministic in the content, so a probe cannot detect the defense by
  // hashing for one fixed output.

  function applyMedianFilter3x3(data, w, h) {
    const out = new Uint8ClampedArray(data.length);
    const rs = new Array(9), gs = new Array(9), bs = new Array(9);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let k = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const ny = y + dy < 0 ? 0 : (y + dy >= h ? h - 1 : y + dy);
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx < 0 ? 0 : (x + dx >= w ? w - 1 : x + dx);
            const j = (ny * w + nx) * 4;
            rs[k] = data[j];
            gs[k] = data[j + 1];
            bs[k] = data[j + 2];
            k++;
          }
        }
        rs.sort((a, b) => a - b);
        gs.sort((a, b) => a - b);
        bs.sort((a, b) => a - b);
        const i = (y * w + x) * 4;
        out[i]     = rs[4];
        out[i + 1] = gs[4];
        out[i + 2] = bs[4];
        out[i + 3] = data[i + 3];
      }
    }
    return out;
  }

  function quantize3bit(data) {
    const out = new Uint8ClampedArray(data.length);
    for (let i = 0; i < data.length; i += 4) {
      out[i]     = data[i]     & 0xE0;
      out[i + 1] = data[i + 1] & 0xE0;
      out[i + 2] = data[i + 2] & 0xE0;
      out[i + 3] = data[i + 3];
    }
    return out;
  }

  function makeCanonicalSnapshot(data, w, h) {
    let out = canonicalizeTransparentPixels(new Uint8ClampedArray(data));
    out = applyMedianFilter3x3(out, w, h);
    out = quantize3bit(out);
    return out;
  }

  function makeConstantBlock(w, h) {
    // Zero-entropy uniform output. Every user in block mode gets the same
    // bytes; canvas contributes 0 bits to the fingerprint. Trivially
    // detectable as a defense via fixed-output hash, but information-
    // theoretically the strongest possible reduction.
    const out = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < out.length; i += 4) {
      out[i]     = 127;
      out[i + 1] = 127;
      out[i + 2] = 127;
      out[i + 3] = 255;
    }
    return out;
  }

  // === Counter bridge ===================================================

  function emit(event) {
    try {
      window.postMessage({ __LFPMitigation: true, event }, '*');
    } catch (_) {}
  }

  function registerReadback(api) {
    budget.readbacks++;
    const mode = currentMode();
    emit({ kind: 'readback', api, origin: location.origin, mode, ts: Date.now() });
    return mode;
  }

  function reportRestore(api) {
    try {
      emit({ kind: 'restore', api, origin: location.origin, ts: Date.now() });
    } catch (_) {}
  }

  // === toString stealth =================================================

  const nativeFor = new WeakMap();
  function spoof(patched, native) {
    try { Object.defineProperty(patched, 'name', { value: native.name, configurable: true }); } catch (_) {}
    try { Object.defineProperty(patched, 'length', { value: native.length, configurable: true }); } catch (_) {}
    nativeFor.set(patched, native);
  }

  const patchedFnToString = function toString() {
    const n = nativeFor.get(this);
    if (n) return nativeFunctionToString.call(n);
    return nativeFunctionToString.call(this);
  };
  spoof(patchedFnToString, nativeFunctionToString);
  Function.prototype.toString = patchedFnToString;

  // === Canvas 2D: snapshot + delivery ===================================
  // The snapshot caches BOTH the sanitized output and the raw pixels (so
  // display-restore can reach the originals). Block mode also performs the
  // native read — a small perf hit that keeps display-restore working.

  function getOrCreateSnapshot(ctx, proposedMode) {
    const canvas = ctx.canvas;
    const s = getState(canvas);
    const w = canvas.width;
    const h = canvas.height;

    if (s.cachedSnapshot &&
        s.cachedGen === s.generation &&
        s.cachedW === w &&
        s.cachedH === h) {
      return {
        data: s.cachedSnapshot,
        raw: s.cachedRaw,
        width: w,
        height: h,
        mode: s.cachedMode,
      };
    }

    const raw = nativeGetImageData.call(ctx, 0, 0, w, h);
    const rawCopy = new Uint8ClampedArray(raw.data);
    const sanitized = sanitize(new Uint8ClampedArray(rawCopy));
    // Premultiplied-alpha fixed point: the canvas stores premultiplied
    // bytes, so putImageData -> getImageData on a hidden helper is one
    // premult/unpremult round-trip. Iterate (up to 5 times) until two rounds
    // agree byte for byte, so getImageData, toDataURL/toBlob and a PNG
    // decode + drawImage round-trip all read the same bytes.
    let cycled = sanitized;
    try {
      const cycCanvas = makeInternalCanvas(w, h);
      const cycCtx = nativeGetContext.call(cycCanvas, '2d');
      for (let iter = 0; iter < 5; iter++) {
        nativePutImageData.call(cycCtx, new nativeImageData(cycled, w, h), 0, 0);
        const next = new Uint8ClampedArray(nativeGetImageData.call(cycCtx, 0, 0, w, h).data);
        let same = next.length === cycled.length;
        if (same) {
          for (let i = 0; i < next.length; i++) {
            if (cycled[i] !== next[i]) { same = false; break; }
          }
        }
        cycled = next;
        if (same) break;
      }
    } catch (_) { /* fall back to sanitized on any failure */ }
    s.cachedSnapshot = cycled;
    s.cachedRaw = rawCopy;
    s.cachedGen = s.generation;
    // Lock the chosen delivery mode to this snapshot generation. Every
    // subsequent read of the same generation sees byte-identical output
    // (sub-region vs full read, raw pixels vs PNG re-encode, etc.).
    // A new paint bumps the generation -> mode is re-evaluated.
    s.cachedMode = proposedMode;
    s.cachedW = w;
    s.cachedH = h;
    return {
      data: cycled,
      raw: rawCopy,
      width: w,
      height: h,
      mode: proposedMode,
    };
  }

  function deliver(ctx, mode) {
    const snap = getOrCreateSnapshot(ctx, mode);
    // Use the snapshot's locked mode so all reads of one generation agree.
    const m = snap.mode || mode;
    if (m === 'block') {
      // Default: uniform constant -> zero canvas entropy contribution.
      // Optional 'canonical' mode preserves content shape with reduced
      // (but non-zero) device-specific signal; trades entropy for
      // non-detectability.
      const blockData = config.blockMode === 'canonical'
        ? makeCanonicalSnapshot(snap.data, snap.width, snap.height)
        : makeConstantBlock(snap.width, snap.height);
      return {
        data: blockData,
        raw: snap.raw,
        width: snap.width,
        height: snap.height,
      };
    }
    if (m === 'coarse') {
      return {
        data: quantizeHighNibble(snap.data),
        raw: snap.raw,
        width: snap.width,
        height: snap.height,
      };
    }
    return snap;
  }

  // === Canvas 2D: readback / export patches =============================

  function patchedGetImageData(sx, sy, sw, sh, settings) {
    if (internalCanvases.has(this.canvas)) {
      return nativeGetImageData.call(this, sx, sy, sw, sh, settings);
    }
    try {
      const mode = registerReadback('getImageData');
      if (sw < 0) { sx += sw; sw = -sw; }
      if (sh < 0) { sy += sh; sh = -sh; }
      sx |= 0; sy |= 0; sw |= 0; sh |= 0;

      const d = deliver(this, mode);
      const out = new nativeImageData(sw, sh);
      const outRaw = new Uint8ClampedArray(sw * sh * 4);
      const srcW = d.width, srcH = d.height;
      const src = d.data, srcRaw = d.raw, dst = out.data;
      for (let y = 0; y < sh; y++) {
        const srcY = sy + y;
        if (srcY < 0 || srcY >= srcH) continue;
        for (let x = 0; x < sw; x++) {
          const srcX = sx + x;
          if (srcX < 0 || srcX >= srcW) continue;
          const si = (srcY * srcW + srcX) * 4;
          const di = (y * sw + x) * 4;
          dst[di]     = src[si];
          dst[di + 1] = src[si + 1];
          dst[di + 2] = src[si + 2];
          dst[di + 3] = src[si + 3];
          if (srcRaw) {
            outRaw[di]     = srcRaw[si];
            outRaw[di + 1] = srcRaw[si + 1];
            outRaw[di + 2] = srcRaw[si + 2];
            outRaw[di + 3] = srcRaw[si + 3];
          }
        }
      }
      if (srcRaw) recordIssuedAndOriginal(out.data, outRaw, sw, sh);
      return out;
    } catch (err) {
      console.warn('[LearnedFP mitigation] getImageData fallback:', err);
      return nativeGetImageData.call(this, sx, sy, sw, sh, settings);
    }
  }

  function patchedToDataURL(type, quality) {
    if (internalCanvases.has(this)) {
      return nativeToDataURL.call(this, type, quality);
    }
    try {
      const mode = registerReadback('toDataURL');
      const ctx = nativeGetContext.call(this, '2d');
      if (!ctx) return nativeToDataURL.call(this, type, quality);

      const d = deliver(ctx, mode);
      const helper = makeInternalCanvas(d.width, d.height);
      const hctx = nativeGetContext.call(helper, '2d');
      const imgData = new nativeImageData(d.data, d.width, d.height);
      nativePutImageData.call(hctx, imgData, 0, 0);
      return nativeToDataURL.call(helper, type, quality);
    } catch (err) {
      console.warn('[LearnedFP mitigation] toDataURL fallback:', err);
      return nativeToDataURL.call(this, type, quality);
    }
  }

  function patchedToBlob(callback, type, quality) {
    if (internalCanvases.has(this)) {
      return nativeToBlob.call(this, callback, type, quality);
    }
    try {
      const mode = registerReadback('toBlob');
      const ctx = nativeGetContext.call(this, '2d');
      if (!ctx) return nativeToBlob.call(this, callback, type, quality);

      const d = deliver(ctx, mode);
      const helper = makeInternalCanvas(d.width, d.height);
      const hctx = nativeGetContext.call(helper, '2d');
      const imgData = new nativeImageData(d.data, d.width, d.height);
      nativePutImageData.call(hctx, imgData, 0, 0);
      return nativeToBlob.call(helper, callback, type, quality);
    } catch (err) {
      console.warn('[LearnedFP mitigation] toBlob fallback:', err);
      return nativeToBlob.call(this, callback, type, quality);
    }
  }

  // === Canvas 2D: putImageData with display-restore =====================

  function patchedPutImageData() {
    if (internalCanvases.has(this.canvas)) {
      return nativePutImageData.apply(this, arguments);
    }
    let restored = null;
    try {
      restored = tryRestoreImgData(arguments[0]);
    } catch (err) {
      console.warn('[LearnedFP mitigation] putImageData restore error:', err);
    }
    let result;
    if (restored) {
      const args = Array.from(arguments);
      args[0] = restored;
      result = nativePutImageData.apply(this, args);
      reportRestore('putImageData');
    } else {
      result = nativePutImageData.apply(this, arguments);
    }
    const c = this.canvas;
    if (c instanceof Canvas && !internalCanvases.has(c)) {
      bumpGeneration(c);
    }
    return result;
  }

  // === Canvas 2D: generation invalidation on other draw ops =============

  function wrapCanvasDrawOp(name) {
    const native = Ctx2D.prototype[name];
    if (typeof native !== 'function') return;
    const wrapped = function (...args) {
      const result = native.apply(this, args);
      const c = this.canvas;
      if (c instanceof Canvas && !internalCanvases.has(c)) {
        bumpGeneration(c);
      }
      return result;
    };
    spoof(wrapped, native);
    Ctx2D.prototype[name] = wrapped;
  }

  // === Install Canvas 2D patches ========================================

  spoof(patchedGetImageData, nativeGetImageData);
  Ctx2D.prototype.getImageData = patchedGetImageData;

  spoof(patchedToDataURL, nativeToDataURL);
  Canvas.prototype.toDataURL = patchedToDataURL;

  spoof(patchedToBlob, nativeToBlob);
  Canvas.prototype.toBlob = patchedToBlob;

  spoof(patchedPutImageData, nativePutImageData);
  Ctx2D.prototype.putImageData = patchedPutImageData;

  // putImageData handled separately above for display-restore.
  const drawOps2D = [
    'fillRect', 'strokeRect', 'clearRect',
    'fill', 'stroke',
    'fillText', 'strokeText',
    'drawImage',
  ];
  for (const name of drawOps2D) wrapCanvasDrawOp(name);
  if (typeof Ctx2D.prototype.reset === 'function') wrapCanvasDrawOp('reset');

  for (const prop of ['width', 'height']) {
    const desc = Object.getOwnPropertyDescriptor(Canvas.prototype, prop);
    if (!desc || !desc.set) continue;
    const nativeSet = desc.set;
    const wrappedSet = function (v) {
      nativeSet.call(this, v);
      if (!internalCanvases.has(this)) bumpGeneration(this);
    };
    spoof(wrappedSet, nativeSet);
    Object.defineProperty(Canvas.prototype, prop, { ...desc, set: wrappedSet });
  }

  // === OffscreenCanvas patches (main thread) ============================

  if (typeof OffscreenCanvas !== 'undefined' &&
      typeof OffscreenCanvasRenderingContext2D !== 'undefined') {
    const OC = OffscreenCanvas;
    const OCCtx = OffscreenCanvasRenderingContext2D;
    const nativeOCGetImageData = OCCtx.prototype.getImageData;
    const nativeOCPutImageData = OCCtx.prototype.putImageData;
    const nativeOCConvertToBlob = OC.prototype.convertToBlob;
    const nativeOCGetContext = OC.prototype.getContext;

    const internalOC = new WeakSet();

    const makeInternalOC = (w, h) => {
      const o = new OC(w, h);
      internalOC.add(o);
      return o;
    };

    const getOCSnapshot = (ctx, proposedMode) => {
      const oc = ctx.canvas;
      const s = getState(oc);
      const w = oc.width, h = oc.height;
      if (s.cachedSnapshot &&
          s.cachedGen === s.generation &&
          s.cachedW === w && s.cachedH === h) {
        return {
          data: s.cachedSnapshot,
          raw: s.cachedRaw,
          width: w,
          height: h,
          mode: s.cachedMode,
        };
      }
      const raw = nativeOCGetImageData.call(ctx, 0, 0, w, h);
      const rawCopy = new Uint8ClampedArray(raw.data);
      const sanitized = sanitize(new Uint8ClampedArray(rawCopy));
      // Same premult/unpremult cycle as the 2D path: align getImageData
      // bytes with the bytes that come back through PNG encode + decode.
      let cycled = sanitized;
      try {
        const cycOC = makeInternalOC(w, h);
        const cycCtx = cycOC.getContext('2d');
        nativePutImageData.call(cycCtx, new nativeImageData(sanitized, w, h), 0, 0);
        cycled = new Uint8ClampedArray(nativeOCGetImageData.call(cycCtx, 0, 0, w, h).data);
      } catch (_) { /* fall back to sanitized on any failure */ }
      s.cachedSnapshot = cycled;
      s.cachedRaw = rawCopy;
      s.cachedGen = s.generation;
      // Lock mode to this snapshot so every readback of one generation
      // returns byte-identical output (sub-region vs full, getImageData
      // vs PNG re-encode). Same rationale as the 2D context above.
      s.cachedMode = proposedMode;
      s.cachedW = w; s.cachedH = h;
      return {
        data: cycled,
        raw: rawCopy,
        width: w,
        height: h,
        mode: proposedMode,
      };
    };

    const deliverOC = (ctx, mode) => {
      const snap = getOCSnapshot(ctx, mode);
      const m = snap.mode || mode;
      if (m === 'block') {
        const blockData = config.blockMode === 'canonical'
          ? makeCanonicalSnapshot(snap.data, snap.width, snap.height)
          : makeConstantBlock(snap.width, snap.height);
        return {
          data: blockData,
          raw: snap.raw,
          width: snap.width,
          height: snap.height,
        };
      }
      if (m === 'coarse') {
        return {
          data: quantizeHighNibble(snap.data),
          raw: snap.raw,
          width: snap.width,
          height: snap.height,
        };
      }
      return snap;
    };

    function patchedOCGetImageData(sx, sy, sw, sh, settings) {
      if (internalOC.has(this.canvas)) {
        return nativeOCGetImageData.call(this, sx, sy, sw, sh, settings);
      }
      try {
        const mode = registerReadback('getImageData');
        if (sw < 0) { sx += sw; sw = -sw; }
        if (sh < 0) { sy += sh; sh = -sh; }
        sx |= 0; sy |= 0; sw |= 0; sh |= 0;

        const d = deliverOC(this, mode);
        const out = new nativeImageData(sw, sh);
        const outRaw = new Uint8ClampedArray(sw * sh * 4);
        const srcW = d.width, srcH = d.height;
        const src = d.data, srcRaw = d.raw, dst = out.data;
        for (let y = 0; y < sh; y++) {
          const srcY = sy + y;
          if (srcY < 0 || srcY >= srcH) continue;
          for (let x = 0; x < sw; x++) {
            const srcX = sx + x;
            if (srcX < 0 || srcX >= srcW) continue;
            const si = (srcY * srcW + srcX) * 4;
            const di = (y * sw + x) * 4;
            dst[di]     = src[si];
            dst[di + 1] = src[si + 1];
            dst[di + 2] = src[si + 2];
            dst[di + 3] = src[si + 3];
            if (srcRaw) {
              outRaw[di]     = srcRaw[si];
              outRaw[di + 1] = srcRaw[si + 1];
              outRaw[di + 2] = srcRaw[si + 2];
              outRaw[di + 3] = srcRaw[si + 3];
            }
          }
        }
        if (srcRaw) recordIssuedAndOriginal(out.data, outRaw, sw, sh);
        return out;
      } catch (err) {
        console.warn('[LearnedFP mitigation] OC getImageData fallback:', err);
        return nativeOCGetImageData.call(this, sx, sy, sw, sh, settings);
      }
    }

    function patchedOCConvertToBlob(options) {
      if (internalOC.has(this)) {
        return nativeOCConvertToBlob.call(this, options);
      }
      try {
        const mode = registerReadback('convertToBlob');
        const ctx = nativeOCGetContext.call(this, '2d');
        if (!ctx) return nativeOCConvertToBlob.call(this, options);
        const d = deliverOC(ctx, mode);
        const helper = makeInternalOC(d.width, d.height);
        const hctx = nativeOCGetContext.call(helper, '2d');
        const imgData = new nativeImageData(d.data, d.width, d.height);
        nativeOCPutImageData.call(hctx, imgData, 0, 0);
        return nativeOCConvertToBlob.call(helper, options);
      } catch (err) {
        console.warn('[LearnedFP mitigation] OC convertToBlob fallback:', err);
        return nativeOCConvertToBlob.call(this, options);
      }
    }

    function patchedOCPutImageData() {
      if (internalOC.has(this.canvas)) {
        return nativeOCPutImageData.apply(this, arguments);
      }
      let restored = null;
      try {
        restored = tryRestoreImgData(arguments[0]);
      } catch (err) {
        console.warn('[LearnedFP mitigation] OC putImageData restore error:', err);
      }
      let result;
      if (restored) {
        const args = Array.from(arguments);
        args[0] = restored;
        result = nativeOCPutImageData.apply(this, args);
        reportRestore('putImageData');
      } else {
        result = nativeOCPutImageData.apply(this, arguments);
      }
      const c = this.canvas;
      if (c instanceof OC && !internalOC.has(c)) {
        bumpGeneration(c);
      }
      return result;
    }

    spoof(patchedOCGetImageData, nativeOCGetImageData);
    OCCtx.prototype.getImageData = patchedOCGetImageData;

    spoof(patchedOCConvertToBlob, nativeOCConvertToBlob);
    OC.prototype.convertToBlob = patchedOCConvertToBlob;

    spoof(patchedOCPutImageData, nativeOCPutImageData);
    OCCtx.prototype.putImageData = patchedOCPutImageData;

    const wrapOCDrawOp = (name) => {
      const native = OCCtx.prototype[name];
      if (typeof native !== 'function') return;
      const wrapped = function (...args) {
        const result = native.apply(this, args);
        const c = this.canvas;
        if (c instanceof OC && !internalOC.has(c)) {
          bumpGeneration(c);
        }
        return result;
      };
      spoof(wrapped, native);
      OCCtx.prototype[name] = wrapped;
    };

    // putImageData handled separately above for display-restore.
    const ocDrawOps = [
      'fillRect', 'strokeRect', 'clearRect',
      'fill', 'stroke',
      'fillText', 'strokeText',
      'drawImage',
    ];
    for (const name of ocDrawOps) wrapOCDrawOp(name);
    if (typeof OCCtx.prototype.reset === 'function') wrapOCDrawOp('reset');

    for (const prop of ['width', 'height']) {
      const desc = Object.getOwnPropertyDescriptor(OC.prototype, prop);
      if (!desc || !desc.set) continue;
      const nativeSet = desc.set;
      const wrappedSet = function (v) {
        nativeSet.call(this, v);
        if (!internalOC.has(this)) bumpGeneration(this);
      };
      spoof(wrappedSet, nativeSet);
      Object.defineProperty(OC.prototype, prop, { ...desc, set: wrappedSet });
    }
  }
})();
