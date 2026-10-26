'use strict';

let currentTabId = null;

async function init() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  currentTabId = tab ? tab.id : null;

  document.getElementById('reset').addEventListener('click', async () => {
    if (currentTabId == null) return;
    await chrome.runtime.sendMessage({ type: 'resetCounters', tabId: currentTabId });
    refresh();
  });

  document.getElementById('downloadEvents').addEventListener('click', async () => {
    if (currentTabId == null) return;
    const c = await chrome.runtime.sendMessage({ type: 'getCounters', tabId: currentTabId });
    if (!c) return;
    const blob = new Blob([JSON.stringify(c, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'learnedfp-mitigation-tab' + currentTabId + '-' + Date.now() + '.json';
    a.click();
    URL.revokeObjectURL(url);
  });

  await loadThresholds();
  bindThresholdInputs();

  if (currentTabId != null) {
    refresh();
    setInterval(refresh, 500);
  }
}

async function loadThresholds() {
  const r = await chrome.storage.local.get({ coarseThreshold: 3, blockThreshold: 20 });
  document.getElementById('coarseThreshold').value = r.coarseThreshold;
  document.getElementById('blockThreshold').value = r.blockThreshold;
}

function bindThresholdInputs() {
  let saveTimer = null;
  const schedule = () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      const coarseThreshold = Math.max(0, parseInt(document.getElementById('coarseThreshold').value, 10) || 0);
      const blockThreshold = Math.max(0, parseInt(document.getElementById('blockThreshold').value, 10) || 0);
      await chrome.storage.local.set({ coarseThreshold, blockThreshold });
    }, 300);
  };
  document.getElementById('coarseThreshold').addEventListener('input', schedule);
  document.getElementById('blockThreshold').addEventListener('input', schedule);
}

async function refresh() {
  if (currentTabId == null) return;
  let c;
  try {
    c = await chrome.runtime.sendMessage({ type: 'getCounters', tabId: currentTabId });
  } catch (_) { return; }
  if (!c) return;

  setText('readback', c.readback);
  setText('getImageData', c.getImageData);
  setText('toDataURL', c.toDataURL);
  setText('toBlob', c.toBlob);
  setText('convertToBlob', c.convertToBlob);
  setText('cnt-normal', c.normal);
  setText('cnt-coarse', c.coarse);
  setText('cnt-block', c.block);
  setText('restored', c.restore);

  renderOrigins(c.perOrigin || {});
  renderEvents(c.recentEvents || []);
}

function renderOrigins(perOrigin) {
  const container = document.getElementById('origins');
  const entries = Object.entries(perOrigin)
    .filter(([, v]) => v && typeof v === 'object')
    .sort((a, b) => b[1].count - a[1].count);
  if (entries.length === 0) {
    container.innerHTML = '<div class="hint">(none yet)</div>';
    return;
  }
  container.innerHTML = '';
  for (const [origin, pd] of entries) {
    const row = document.createElement('div');
    row.className = 'origin-row';

    const o = document.createElement('span');
    o.className = 'o';
    o.title = origin;
    o.textContent = origin;

    const v = document.createElement('span');
    v.className = 'v';
    const count = document.createElement('b');
    count.textContent = String(pd.count);
    v.appendChild(count);

    if (pd.coarse) {
      v.appendChild(document.createTextNode(' \u00B7 '));
      v.appendChild(makeSpan('m-coarse', 'c:' + pd.coarse));
    }
    if (pd.block) {
      v.appendChild(document.createTextNode(' \u00B7 '));
      v.appendChild(makeSpan('m-block', 'b:' + pd.block));
    }

    const chip = document.createElement('span');
    const highest = pd.block ? 'block' : pd.coarse ? 'coarse' : 'normal';
    chip.className = 'chip ' + highest;
    chip.textContent = highest.toUpperCase();
    v.appendChild(chip);

    row.appendChild(o);
    row.appendChild(v);
    container.appendChild(row);
  }
}

function renderEvents(events) {
  const container = document.getElementById('events');
  if (!Array.isArray(events) || events.length === 0) {
    container.innerHTML = '<div class="hint">(none yet)</div>';
    return;
  }
  const recent = events.slice().reverse(); // newest first
  container.innerHTML = '';
  for (const e of recent) {
    const row = document.createElement('div');
    row.className = 'event-row';
    row.appendChild(makeSpan('ts', formatTime(e.ts)));
    row.appendChild(makeSpan('mode ' + (e.mode || 'normal'), e.mode || 'normal'));
    row.appendChild(makeSpan('api', e.api || ''));
    const o = makeSpan('origin', e.origin || '');
    o.title = e.origin || '';
    row.appendChild(o);
    container.appendChild(row);
  }
}

function formatTime(ts) {
  if (typeof ts !== 'number') return '--:--:--';
  const d = new Date(ts);
  return String(d.getHours()).padStart(2, '0') + ':'
       + String(d.getMinutes()).padStart(2, '0') + ':'
       + String(d.getSeconds()).padStart(2, '0');
}

function makeSpan(cls, text) {
  const el = document.createElement('span');
  el.className = cls;
  el.textContent = text;
  return el;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = String(value || 0);
}

init();
