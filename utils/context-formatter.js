// ============================================================
// UniMemo — Context Formatter & Injector
// Section 4: formatContext() + clipboard injection
// ============================================================

/**
 * Platform display config
 */
const PLATFORM_META = {
  Claude:  { emoji: '🟠', color: '#D97757' },
  ChatGPT: { emoji: '🟢', color: '#10A37F' },
  Gemini:  { emoji: '🔵', color: '#4285F4' },
  default: { emoji: '🤖', color: '#888888' }
};

/**
 * Formats a stored context object into a rich prompt string
 * ready to paste into any AI chatbot.
 *
 * @param {Object} ctx - Context object from storage
 * @param {Array}  ctx.messages    - [{role, content}]
 * @param {string} ctx.platform    - 'Claude' | 'ChatGPT' | 'Gemini'
 * @param {number} ctx.capturedAt  - Unix timestamp (ms)
 * @returns {string} Formatted prompt string
 */
export function formatContext(ctx) {
  if (!ctx || !ctx.messages?.length) return '';

  const meta      = PLATFORM_META[ctx.platform] ?? PLATFORM_META.default;
  const date      = new Date(ctx.capturedAt).toLocaleString();
  const msgCount  = ctx.messages.length;

  const lines = [];

  // ── Header ────────────────────────────────────────────────
  lines.push(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  lines.push(`${meta.emoji} CONTEXT FROM ${ctx.platform.toUpperCase()}`);
  lines.push(`Captured: ${date}  |  ${msgCount} message${msgCount !== 1 ? 's' : ''}`);
  lines.push(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  lines.push('');

  // ── Messages ─────────────────────────────────────────────
  for (const msg of ctx.messages) {
    const label = msg.role === 'user' ? '👤 USER' : `🤖 ${ctx.platform.toUpperCase()}`;
    lines.push(`[${label}]`);
    lines.push(msg.content);
    lines.push('');
  }

  // ── Injection prompt ──────────────────────────────────────
  lines.push(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  lines.push('📌 CONTINUE FROM HERE:');
  lines.push(`The above is my conversation history from ${ctx.platform}. Please read it and continue helping me from where we left off.`);
  lines.push(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);

  return lines.join('\n');
}

/**
 * Copies the formatted context to clipboard (MVP approach)
 * Returns { ok: boolean, error?: string }
 */
export async function copyContextToClipboard(ctx) {
  const text = formatContext(ctx);
  if (!text) return { ok: false, error: 'No context to copy' };

  try {
    await navigator.clipboard.writeText(text);
    return { ok: true, charCount: text.length };
  } catch (err) {
    // Fallback: execCommand (older Chrome)
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.cssText = 'position:fixed;top:-9999px;left:-9999px';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      return { ok: true, charCount: text.length };
    } catch (fallbackErr) {
      return { ok: false, error: fallbackErr.message };
    }
  }
}

/**
 * V2: DOM injection — finds the textarea on the target platform
 * and fills it with the formatted context string.
 * Called from a content script, not the popup.
 */
export function injectIntoPage(ctx) {
  const text = formatContext(ctx);
  if (!text) return false;

  // Common textarea/contenteditable selectors across platforms
  const SELECTORS = [
    'textarea[placeholder*="message" i]',
    'textarea[data-id]',
    '[contenteditable="true"][role="textbox"]',
    'div[contenteditable="true"]',
    '#prompt-textarea',
    '[placeholder*="Ask" i]',
  ];

  let el = null;
  for (const sel of SELECTORS) {
    el = document.querySelector(sel);
    if (el) break;
  }

  if (!el) return false;

  // Focus first
  el.focus();

  if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
    el.value = text;
    el.dispatchEvent(new Event('input', { bubbles: true }));
  } else {
    // contenteditable
    el.innerText = text;
    el.dispatchEvent(new InputEvent('input', { bubbles: true }));
  }

  return true;
}

/**
 * Returns a short preview of the last N messages
 */
export function getPreviewMessages(ctx, count = 3) {
  if (!ctx?.messages?.length) return [];
  return ctx.messages.slice(-count);
}

/**
 * Relative time string ("2 minutes ago", "just now")
 */
export function relativeTime(timestamp) {
  if (!timestamp) return '';
  const delta = Date.now() - timestamp;
  const mins  = Math.floor(delta / 60000);
  const hours = Math.floor(delta / 3600000);
  const days  = Math.floor(delta / 86400000);

  if (delta < 60000)  return 'just now';
  if (mins  < 60)     return `${mins}m ago`;
  if (hours < 24)     return `${hours}h ago`;
  return `${days}d ago`;
}
