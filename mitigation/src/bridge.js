(() => {
  'use strict';

  // === MAIN -> SW: readback events ======================================
  const queue = [];
  let flushTimer = null;

  window.addEventListener('message', (e) => {
    if (e.source !== window) return;
    const d = e.data;
    if (!d || d.__LFPMitigation !== true || !d.event) return;
    queue.push(d.event);
    if (!flushTimer) flushTimer = setTimeout(flush, 50);
  });

  function flush() {
    flushTimer = null;
    if (queue.length === 0) return;
    const batch = queue.splice(0);
    try {
      chrome.runtime.sendMessage({ type: 'events', events: batch });
    } catch (_) {
      // Extension context invalidated (reload/update). Drop batch.
    }
  }

  // === Storage -> MAIN: config push =====================================
  function pushConfig(cfg) {
    try {
      window.postMessage({ __LFPMitigationCfg: true, config: cfg }, '*');
    } catch (_) {}
  }

  chrome.storage.local.get({ coarseThreshold: 3, blockThreshold: 20 }, (r) => {
    pushConfig(r);
    // Signal MAIN world that initial config is applied. Test pages can
    // wait for this before issuing readbacks to avoid the race between
    // page-script execution and the async storage read.
    try { window.postMessage({ __LFPMitigationReady: true }, '*'); } catch (_) {}
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== 'local') return;
    const cfg = {};
    for (const k of Object.keys(changes)) cfg[k] = changes[k].newValue;
    if (Object.keys(cfg).length) pushConfig(cfg);
  });

  // === Test-mode runtime config setter (research-prototype only) ========
  // Allows Playwright tests / pages to update thresholds without going
  // through the popup UI. Intentionally permissive for research; remove
  // or gate behind an authentication token for production deployment.
  window.addEventListener('message', (e) => {
    if (e.source !== window) return;
    const d = e.data;
    if (!d || d.__LFPMitigationSet !== true) return;
    const update = {};
    if (typeof d.coarseThreshold === 'number') {
      update.coarseThreshold = d.coarseThreshold;
    }
    if (typeof d.blockThreshold === 'number') {
      update.blockThreshold = d.blockThreshold;
    }
    if (Object.keys(update).length) {
      try { chrome.storage.local.set(update); } catch (_) {}
    }
  });
})();
