// ============================================================
// UniMemo — Background Service Worker
// Brain: routes messages, manages storage, TSGC compression
// (Compression inlined — no ES module import needed)
// ============================================================

const MAX_MESSAGES    = 50;
const STORAGE_KEY     = 'unimemo_context';
const STORAGE_KEY_META= 'unimemo_compression_meta';

// ── Message Router ───────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    try {
      switch (message.type) {

        case 'SAVE_CONTEXT': {
          const { messages, platform } = message.payload;
          await saveContext(messages, platform);
          sendResponse({ ok: true });
          break;
        }

        case 'GET_CONTEXT': {
          const ctx = await getContext();
          sendResponse({ ok: true, context: ctx });
          break;
        }

        case 'GET_COMPRESSION_META': {
          const result = await chrome.storage.local.get(STORAGE_KEY_META);
          sendResponse({ ok: true, meta: result[STORAGE_KEY_META] ?? null });
          break;
        }

        case 'CLEAR_CONTEXT': {
          await clearContext();
          sendResponse({ ok: true });
          break;
        }

        case 'TRIGGER_CAPTURE': {
          const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
          if (!tab?.id) {
            sendResponse({ ok: false, error: 'No active tab found.' });
            break;
          }

          const platformInfo = detectPlatform(tab.url);
          if (!platformInfo) {
            sendResponse({
              ok: false,
              error: 'Open a conversation on Claude, ChatGPT, or Gemini first.'
            });
            break;
          }

          // Content scripts are injected by manifest — send CAPTURE_NOW directly.
          // Fall back to manual injection if content script isn't ready yet.
          try {
            await chrome.tabs.sendMessage(tab.id, { type: 'CAPTURE_NOW' });
          } catch (_) {
            // Page loaded before extension was installed/reloaded — inject now
            await chrome.scripting.executeScript({
              target: { tabId: tab.id },
              files: [`content/${platformInfo.key}-scraper.js`]
            });
            await new Promise(r => setTimeout(r, 500));
            await chrome.tabs.sendMessage(tab.id, { type: 'CAPTURE_NOW' });
          }

          sendResponse({ ok: true, platform: platformInfo.display });
          break;
        }

        default:
          sendResponse({ ok: false, error: `Unknown message: ${message.type}` });
      }
    } catch (err) {
      console.error('[UniMemo SW] Error:', err.message, err.stack);
      sendResponse({ ok: false, error: err.message });
    }
  })();
  return true; // keep async channel open
});

// ── Storage ───────────────────────────────────────────────────

async function saveContext(messages, platform) {
  const capped = messages.slice(-MAX_MESSAGES);

  try {
    const { compressed, meta } = await tsgcCompress(capped);
    await chrome.storage.local.set({ [STORAGE_KEY_META]: meta });
    await chrome.storage.local.set({
      [STORAGE_KEY]: {
        compressed,
        platform,
        capturedAt:   Date.now(),
        messageCount: capped.length,
        isCompressed: true,
      }
    });
    const pct = (meta.overallRatio * 100).toFixed(1);
    console.log(`[UniMemo] Saved ${capped.length} msgs from ${platform} | ${pct}% saved`);
  } catch (err) {
    console.warn('[UniMemo] Compression failed, storing raw:', err.message);
    await chrome.storage.local.set({
      [STORAGE_KEY]: {
        messages:     capped,
        platform,
        capturedAt:   Date.now(),
        messageCount: capped.length,
        isCompressed: false,
      }
    });
  }
}

async function getContext() {
  const result = await chrome.storage.local.get(STORAGE_KEY);
  const ctx = result[STORAGE_KEY];
  if (!ctx) return null;

  if (ctx.isCompressed && ctx.compressed) {
    try {
      const messages = await tsgcDecompress(ctx.compressed);
      return { messages, platform: ctx.platform, capturedAt: ctx.capturedAt, messageCount: ctx.messageCount };
    } catch (err) {
      console.error('[UniMemo] Decompress failed:', err.message);
      return null;
    }
  }
  return ctx; // raw fallback
}

async function clearContext() {
  await chrome.storage.local.remove([STORAGE_KEY, STORAGE_KEY_META]);
}

// ── TSGC Compression (inlined) ────────────────────────────────
//
// Three-layer pipeline:
//   1. Structural deduplication (cross-turn repeated sentences)
//   2. Temporal gradient pruning (Zone1=verbatim, Zone2=40%, Zone3=15%)
//   3. DEFLATE byte compression (native CompressionStream)

const ZONE_VERBATIM = 5;
const ZONE_PRUNED   = 15;

const SIGNAL_WORDS = new Set([
  'because','therefore','however','but','although','instead','error',
  'result','conclusion','solution','problem','issue','answer','key',
  'important','note','warning','step','must','should','never','always',
  'define','means','returns','function','class','approach','algorithm',
  'difference','summary','finally','specifically','example',
]);

const FILLER_RE = [
  /^(sure|of course|certainly|absolutely|great|okay)[!.,]?\s*/i,
  /^(as i (mentioned|said|noted) (earlier|above|before))[,.]?\s*/i,
  /^(to (summarize|recap|reiterate))[,:]?\s*/i,
  /hope (this|that) (helps|clarifies)/i,
  /feel free to (ask|let me know)/i,
];

async function tsgcCompress(messages) {
  const originalJson  = JSON.stringify(messages);
  const originalBytes = new TextEncoder().encode(originalJson).length;

  // Layer 1: dedup
  const deduped = deduplicateMsgs(messages);

  // Layer 2: temporal gradient
  const pruned  = temporalPrune(deduped);

  // Layer 3: deflate
  const prunedJson      = JSON.stringify(pruned.map(m => ({ role: m.role, content: m.content })));
  const compressed      = await deflate(prunedJson);
  const compressedBytes = Math.ceil(compressed.length * 0.75);

  const zones = { 1: 0, 2: 0, 3: 0 };
  pruned.forEach(m => { zones[m._z || 1]++; });

  const meta = {
    originalBytes,
    compressedBytes,
    overallRatio: Math.max(0, 1 - compressedBytes / originalBytes),
    messagesBefore: messages.length,
    messagesAfter:  pruned.length,
    zones: { verbatim: zones[1], pruned: zones[2], summary: zones[3] },
    algorithm: 'TSGC-1.0',
  };

  return { compressed, meta };
}

async function tsgcDecompress(compressed) {
  const json = await inflate(compressed);
  return JSON.parse(json);
}

function deduplicateMsgs(messages) {
  const seen = new Set();
  return messages.map(msg => {
    const sents  = splitSents(msg.content);
    const unique = sents.filter(s => {
      const k = s.trim().toLowerCase().replace(/\s+/g,' ');
      if (k.length < 20) return true;
      if (seen.has(k))   return false;
      seen.add(k); return true;
    });
    return { ...msg, content: unique.join(' ').trim() };
  }).filter(m => m.content.length > 0);
}

function temporalPrune(messages) {
  const total = messages.length;
  return messages.map((msg, idx) => {
    const ri = total - 1 - idx; // 0 = most recent
    if (ri < ZONE_VERBATIM) return { ...msg, _z: 1 };
    const ratio = ri < ZONE_PRUNED ? 0.40 : 0.15;
    if (msg.content.length < 80) return { ...msg, _z: ri < ZONE_PRUNED ? 2 : 3 };
    return { ...msg, content: extractTop(msg.content, ratio, msg.role), _z: ri < ZONE_PRUNED ? 2 : 3 };
  }).filter(m => m.content.trim().length > 0);
}

function extractTop(text, ratio, role) {
  const sents = splitSents(text).filter(s => s.trim().length > 10);
  if (sents.length <= 1) return text;
  const keep = Math.max(1, Math.ceil(sents.length * ratio));
  const cleaned = role === 'assistant'
    ? sents.filter(s => !FILLER_RE.some(r => r.test(s.trim())))
    : sents;
  const scored = cleaned.map((s, i) => ({ s, i, sc: sentScore(s, i, cleaned.length) }));
  return scored.sort((a,b) => b.sc - a.sc).slice(0, keep)
    .sort((a,b) => a.i - b.i).map(x => x.s).join(' ').trim();
}

function sentScore(s, i, total) {
  const relPos  = i / Math.max(total - 1, 1);
  const posS    = 1 - Math.abs(relPos - 0.5) * 0.6;
  const words   = s.toLowerCase().match(/\b[a-z]{3,}\b/g) ?? [];
  const densS   = words.length > 0 ? new Set(words).size / words.length : 0;
  const kwS     = Math.min(1, words.filter(w => SIGNAL_WORDS.has(w)).length / 3);
  return 0.3 * posS + 0.4 * densS + 0.3 * kwS;
}

function splitSents(text) {
  return text.replace(/([.!?])\s+/g,'$1\n').split('\n')
    .map(s => s.trim()).filter(s => s.length > 0);
}

// Native DEFLATE via CompressionStream (Chrome 80+)
async function deflate(text) {
  const bytes  = new TextEncoder().encode(text);
  const cs     = new CompressionStream('deflate-raw');
  const writer = cs.writable.getWriter();
  writer.write(bytes); writer.close();
  const chunks = []; const reader = cs.readable.getReader();
  while (true) { const {done,value} = await reader.read(); if(done) break; chunks.push(value); }
  const out = new Uint8Array(chunks.reduce((n,c)=>n+c.length,0));
  let off = 0; for(const c of chunks){out.set(c,off);off+=c.length;}
  return uint8ToB64(out);
}

async function inflate(b64) {
  const bytes  = b64ToUint8(b64);
  const ds     = new DecompressionStream('deflate-raw');
  const writer = ds.writable.getWriter();
  writer.write(bytes); writer.close();
  const chunks = []; const reader = ds.readable.getReader();
  while (true) { const {done,value} = await reader.read(); if(done) break; chunks.push(value); }
  const out = new Uint8Array(chunks.reduce((n,c)=>n+c.length,0));
  let off = 0; for(const c of chunks){out.set(c,off);off+=c.length;}
  return new TextDecoder().decode(out);
}

function uint8ToB64(bytes) {
  const CHUNK = 0x8000; let bin = '';
  for(let i=0;i<bytes.length;i+=CHUNK)
    bin += String.fromCharCode(...bytes.subarray(i,i+CHUNK));
  return btoa(bin);
}
function b64ToUint8(b64) {
  const bin = atob(b64); const out = new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) out[i]=bin.charCodeAt(i);
  return out;
}

// ── Utility ───────────────────────────────────────────────────

function detectPlatform(url = '') {
  if (url.includes('claude.ai'))         return { key: 'claude',  display: 'Claude' };
  if (url.includes('chatgpt.com'))       return { key: 'chatgpt', display: 'ChatGPT' };
  if (url.includes('gemini.google.com')) return { key: 'gemini',  display: 'Gemini' };
  return null;
}
