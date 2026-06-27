// ============================================================
// UniMemo — Popup Script
// Wires up UI to background service worker via messaging
// ============================================================

// ── DOM refs ─────────────────────────────────────────
// Workflow bar
const step1El      = document.getElementById('step1');
const step2El      = document.getElementById('step2');
const statusDot    = document.getElementById('status-dot');
const statusText   = document.getElementById('status-text');
// Content
const emptyState    = document.getElementById('empty-state');
const contextCard   = document.getElementById('context-card');
const platformBadge = document.getElementById('platform-badge');
const platformEmoji = document.getElementById('platform-emoji');
const platformName  = document.getElementById('platform-name');
const msgCount      = document.getElementById('msg-count');
const capturedTime  = document.getElementById('captured-time');
const messagePreview= document.getElementById('message-preview');
// Footer panels
const panelStep1    = document.getElementById('panel-step1');
const panelStep2    = document.getElementById('panel-step2');
// Compression stats
const compressionBar   = document.getElementById('compression-bar');
const compressSavings  = document.getElementById('compress-savings');
const compressZones    = document.getElementById('compress-zones');
// Buttons
const btnCapture    = document.getElementById('btn-capture');
const btnCopy       = document.getElementById('btn-copy');
const btnClear      = document.getElementById('btn-clear');
const toast         = document.getElementById('toast');

// ── Platform metadata ─────────────────────────────────────────
const PLATFORM_META = {
  Claude:  { emoji: '🟠' },
  ChatGPT: { emoji: '🟢' },
  Gemini:  { emoji: '🔵' },
};

// ── Initialise ────────────────────────────────────────────────
(async function init() {
  setStatus('loading', 'Loading context…');
  const ctx = await bgMessage({ type: 'GET_CONTEXT' });
  renderContext(ctx?.context ?? null);
})();

// ── Button handlers ───────────────────────────────────────────

btnCapture.addEventListener('click', async () => {
  btnCapture.classList.add('loading');
  btnCapture.querySelector('.btn-icon').textContent = '⏳';
  btnCapture.disabled = true;

  const res = await bgMessage({ type: 'TRIGGER_CAPTURE' });

  btnCapture.classList.remove('loading');
  btnCapture.querySelector('.btn-icon').textContent = '⚡';
  btnCapture.disabled = false;

  if (res?.ok) {
    setStatus('loading', `Capturing from ${res.platform}…`);
    setTimeout(async () => {
      const ctx = await bgMessage({ type: 'GET_CONTEXT' });
      renderContext(ctx?.context ?? null);
      if (ctx?.context) {
        showToast(`✅ Captured! Now switch to another AI and click Copy.`, 'success');
      }
    }, 1200);
  } else {
    showToast(res?.error ?? 'Capture failed — make sure you\'re on Claude, ChatGPT, or Gemini', 'error');
    setStatus('error', res?.error ?? 'Capture failed');
  }
});

btnCopy.addEventListener('click', async () => {
  const ctx = await bgMessage({ type: 'GET_CONTEXT' });
  if (!ctx?.context) { showToast('Nothing to copy', 'error'); return; }

  const text = formatContext(ctx.context);
  try {
    await navigator.clipboard.writeText(text);
    showToast('📋 Context copied! Paste it in your new AI chat.', 'success');
  } catch (_) {
    // fallback
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.cssText = 'position:fixed;opacity:0';
    document.body.appendChild(ta); ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    showToast('📋 Context copied!', 'success');
  }
});

btnClear.addEventListener('click', async () => {
  const res = await bgMessage({ type: 'CLEAR_CONTEXT' });
  if (res?.ok) {
    renderContext(null);
    showToast('🗑 Context cleared — ready to capture a new one', '');
  }
});

// ── Render ─────────────────────────────────────────────────────

async function renderContext(ctx) {
  if (!ctx || !ctx.messages?.length) {
    // ─ No context: show Step 1 panel, hide Step 2
    emptyState.hidden  = false;
    contextCard.hidden = true;
    panelStep1.hidden  = false;
    panelStep2.hidden  = true;
    // Workflow bar: Step 1 active, Step 2 dim
    step1El.dataset.active = 'true';
    delete step1El.dataset.done;
    delete step2El.dataset.active;
    delete step2El.dataset.done;
    setStatus('idle', 'Ready to capture');
    return;
  }

  // ─ Has context: show Step 2 panel, hide Step 1
  emptyState.hidden  = true;
  contextCard.hidden = false;
  panelStep1.hidden  = true;
  panelStep2.hidden  = false;
  // Workflow bar: Step 1 done (✓), Step 2 active
  delete step1El.dataset.active;
  step1El.dataset.done   = 'true';
  step2El.dataset.active = 'true';
  // Update step-num to checkmark
  step1El.querySelector('.step-num').textContent = '✓';

  // Platform badge
  const meta = PLATFORM_META[ctx.platform] ?? { emoji: '🤖' };
  platformEmoji.textContent = meta.emoji;
  platformName.textContent  = ctx.platform;
  platformBadge.dataset.platform  = ctx.platform;
  contextCard.dataset.platform    = ctx.platform;

  // Counts
  msgCount.textContent    = `${ctx.messages.length} msg${ctx.messages.length !== 1 ? 's' : ''}`;
  capturedTime.textContent = relativeTime(ctx.capturedAt);

  // Status
  setStatus('active', `Context from ${ctx.platform} ready`);

  // Compression stats
  const metaRes = await bgMessage({ type: 'GET_COMPRESSION_META' });
  if (metaRes?.meta) {
    const m = metaRes.meta;
    const pct = Math.round(m.overallRatio * 100);
    compressSavings.textContent = `${pct}% saved`;
    compressZones.textContent   =
      `Z1:${m.zones.verbatim} Z2:${m.zones.pruned} Z3:${m.zones.summary}`;
    compressionBar.hidden = false;
  } else {
    compressionBar.hidden = true;
  }

  // Message preview — last 3
  const preview = ctx.messages.slice(-3);
  messagePreview.innerHTML = '';
  preview.forEach((msg, i) => {
    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${msg.role}`;
    bubble.style.animationDelay = `${i * 60}ms`;

    const roleLabel = document.createElement('div');
    roleLabel.className = 'bubble-role';
    roleLabel.textContent = msg.role === 'user' ? '👤 You' : `🤖 ${ctx.platform}`;

    const content = document.createElement('div');
    content.className = 'bubble-content';
    content.textContent = msg.content;

    bubble.appendChild(roleLabel);
    bubble.appendChild(content);
    messagePreview.appendChild(bubble);
  });
}

// ── Status bar helper ──────────────────────────────────────────

function setStatus(state, text) {
  statusDot.className = `status-dot ${state === 'idle' ? '' : state}`;
  statusText.textContent = text;
}

// ── Background messaging ───────────────────────────────────────

function bgMessage(payload) {
  return new Promise(resolve => {
    chrome.runtime.sendMessage(payload, response => {
      if (chrome.runtime.lastError) {
        console.warn('[UniMemo popup]', chrome.runtime.lastError.message);
        resolve(null);
      } else {
        resolve(response);
      }
    });
  });
}

// ── Toast ──────────────────────────────────────────────────────

let toastTimer = null;
function showToast(msg, type = '') {
  clearTimeout(toastTimer);
  toast.textContent = msg;
  toast.className   = `toast ${type} show`;
  toastTimer = setTimeout(() => { toast.className = 'toast'; }, 2800);
}

// ── Context formatter (inline, no ES module in popup) ──────────

const PLATFORM_LABELS = {
  Claude: 'CLAUDE', ChatGPT: 'CHATGPT', Gemini: 'GEMINI'
};

function formatContext(ctx) {
  if (!ctx?.messages?.length) return '';
  const date = new Date(ctx.capturedAt).toLocaleString();
  const lines = [
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `${(PLATFORM_META[ctx.platform] ?? { emoji: '🤖' }).emoji} CONTEXT FROM ${ctx.platform.toUpperCase()}`,
    `Captured: ${date}  |  ${ctx.messages.length} messages`,
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    '',
  ];
  for (const msg of ctx.messages) {
    const label = msg.role === 'user' ? '👤 USER' : `🤖 ${ctx.platform.toUpperCase()}`;
    lines.push(`[${label}]`, msg.content, '');
  }
  lines.push(
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    '📌 CONTINUE FROM HERE:',
    `The above is my conversation history from ${ctx.platform}. Please read it and continue helping me from where we left off.`,
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
  );
  return lines.join('\n');
}

// ── Relative time ─────────────────────────────────────────────

function relativeTime(ts) {
  if (!ts) return '';
  const d = Date.now() - ts;
  const m = Math.floor(d / 60000);
  const h = Math.floor(d / 3600000);
  if (d < 60000) return 'just now';
  if (m < 60)    return `${m}m ago`;
  if (h < 24)    return `${h}h ago`;
  return `${Math.floor(d / 86400000)}d ago`;
}
