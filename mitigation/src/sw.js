'use strict';

const KEY = (tabId) => `tab:${tabId}`;
const EVENT_BUFFER_SIZE = 200;

function emptyCounters() {
  return {
    readback: 0,
    getImageData: 0,
    toDataURL: 0,
    toBlob: 0,
    convertToBlob: 0,
    normal: 0,
    coarse: 0,
    block: 0,
    downgrade: 0,
    restore: 0,
    perOrigin: {},
    recentEvents: [],
  };
}

function emptyPerOrigin() {
  return { count: 0, normal: 0, coarse: 0, block: 0 };
}

async function getCounters(tabId) {
  const k = KEY(tabId);
  const r = await chrome.storage.session.get(k);
  return r[k] || emptyCounters();
}

async function setCounters(tabId, c) {
  await chrome.storage.session.set({ [KEY(tabId)]: c });
}

async function clearTab(tabId) {
  await chrome.storage.session.remove(KEY(tabId));
}

async function ingest(tabId, events) {
  const c = await getCounters(tabId);
  if (!Array.isArray(c.recentEvents)) c.recentEvents = [];
  for (const e of events) {
    if (e.kind === 'readback') {
      c.readback++;
      if (e.api && typeof c[e.api] === 'number') c[e.api]++;

      const o = e.origin || '(unknown)';
      if (!c.perOrigin[o] || typeof c.perOrigin[o] !== 'object') {
        c.perOrigin[o] = emptyPerOrigin();
      }
      c.perOrigin[o].count++;

      const mode = e.mode || 'normal';
      if (typeof c[mode] === 'number') c[mode]++;
      if (typeof c.perOrigin[o][mode] === 'number') c.perOrigin[o][mode]++;
      if (mode !== 'normal') c.downgrade++;

      c.recentEvents.push({
        ts: typeof e.ts === 'number' ? e.ts : Date.now(),
        api: e.api,
        mode,
        origin: o,
      });
    } else if (e.kind === 'restore') {
      c.restore = (c.restore || 0) + 1;
      c.recentEvents.push({
        ts: typeof e.ts === 'number' ? e.ts : Date.now(),
        api: e.api || 'putImageData',
        mode: 'restored',
        origin: e.origin || '(unknown)',
      });
    }
  }
  if (c.recentEvents.length > EVENT_BUFFER_SIZE) {
    c.recentEvents.splice(0, c.recentEvents.length - EVENT_BUFFER_SIZE);
  }
  await setCounters(tabId, c);
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === 'events' && sender.tab) {
    ingest(sender.tab.id, msg.events || []).then(() => sendResponse({ ok: true }));
    return true;
  }
  if (msg && msg.type === 'getCounters' && typeof msg.tabId === 'number') {
    getCounters(msg.tabId).then(sendResponse);
    return true;
  }
  if (msg && msg.type === 'resetCounters' && typeof msg.tabId === 'number') {
    clearTab(msg.tabId).then(() => sendResponse({ ok: true }));
    return true;
  }
  return false;
});

chrome.tabs.onRemoved.addListener((tabId) => clearTab(tabId));

chrome.webNavigation.onCommitted.addListener((d) => {
  if (d.frameId === 0) clearTab(d.tabId);
});
