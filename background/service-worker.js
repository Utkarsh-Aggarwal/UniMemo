// ============================================================
// UniMemo — Background Service Worker
// Orchestrates context capture, CSGC compression, and storage.
//
// Algorithm: CSGC (Context-Salience Greedy Compression)
// Reference: Aggarwal 2026 — DOI: 10.5281/zenodo.21227878
// ============================================================

import { csgcCompress, csgcDecompress } from '../utils/compression.js';

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
          // Fall back to manual injection if the content script isn't ready yet.
          try {
            await chrome.tabs.sendMessage(tab.id, { type: 'CAPTURE_NOW' });
          } catch (_) {
            // Page was open before extension loaded — inject now
            await chrome.scripting.executeScript({
              target: { tabId: tab.id },
              files:  [`content/${platformInfo.key}-scraper.js`]
            });
            await new Promise(r => setTimeout(r, 500));
            await chrome.tabs.sendMessage(tab.id, { type: 'CAPTURE_NOW' });
          }

          sendResponse({ ok: true, platform: platformInfo.display });
          break;
        }

        default:
          sendResponse({ ok: false, error: `Unknown message type: ${message.type}` });
      }
    } catch (err) {
      console.error('[UniMemo SW] Error:', err.message, err.stack);
      sendResponse({ ok: false, error: err.message });
    }
  })();
  return true; // keep async message channel open
});

// ── Storage ───────────────────────────────────────────────────

async function saveContext(messages, platform) {
  // Cap at MAX_MESSAGES most recent turns
  const capped = messages.slice(-MAX_MESSAGES);

  try {
    const { compressed, meta } = await csgcCompress(capped);

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
    const sup = meta.obsoleteCount + meta.correctionCount;
    console.log(
      `[UniMemo CSGC] Saved ${capped.length} msgs from ${platform} | ` +
      `${pct}% saved | ${meta.obsoleteCount} obsolete, ${meta.correctionCount} corrections detected`
    );
    if (sup === 0) {
      console.log('[UniMemo CSGC] No supersession pivots found in this conversation.');
    }
  } catch (err) {
    // CSGC failed — store raw as safety fallback
    console.warn('[UniMemo CSGC] Compression failed, storing raw:', err.message);
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
  const ctx    = result[STORAGE_KEY];
  if (!ctx) return null;

  if (ctx.isCompressed && ctx.compressed) {
    try {
      const messages = await csgcDecompress(ctx.compressed);
      return {
        messages,
        platform:     ctx.platform,
        capturedAt:   ctx.capturedAt,
        messageCount: ctx.messageCount,
      };
    } catch (err) {
      console.error('[UniMemo CSGC] Decompress failed:', err.message);
      return null;
    }
  }

  // Raw fallback (only if CSGC failed during save)
  return ctx;
}

async function clearContext() {
  await chrome.storage.local.remove([STORAGE_KEY, STORAGE_KEY_META]);
}

// ── Utility ───────────────────────────────────────────────────

function detectPlatform(url = '') {
  if (url.includes('claude.ai'))         return { key: 'claude',  display: 'Claude'  };
  if (url.includes('chatgpt.com'))       return { key: 'chatgpt', display: 'ChatGPT' };
  if (url.includes('gemini.google.com')) return { key: 'gemini',  display: 'Gemini'  };
  return null;
}
