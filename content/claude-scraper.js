// ============================================================
// UniMemo — Claude.ai Conversation Scraper (v2 — robust)
// Multiple selector strategies + nuclear fallback
// ============================================================

(function () {
  'use strict';

  if (window.__unimemoClaude) return;
  window.__unimemoClaude = true;

  // ── Text cleaner ─────────────────────────────────────────────

  function cleanText(node) {
    if (!node) return '';
    const clone = node.cloneNode(true);
    clone.querySelectorAll(
      'button, svg, [aria-hidden="true"], script, style, .sr-only, ' +
      '[data-testid*="action"], [class*="action-bar"]'
    ).forEach(el => el.remove());
    return (clone.innerText ?? clone.textContent ?? '')
      .replace(/\n{3,}/g, '\n\n').replace(/\t/g, ' ').trim();
  }

  // ── Strategy 1: data-testid human/assistant turn ─────────────

  function strategyTestId() {
    const turns = document.querySelectorAll(
      '[data-testid="human-turn"], [data-testid="assistant-turn"]'
    );
    if (!turns.length) return [];
    return Array.from(turns).map(turn => {
      const isHuman = turn.dataset.testid === 'human-turn';
      return { role: isHuman ? 'user' : 'assistant', content: cleanText(turn) };
    }).filter(m => m.content);
  }

  // ── Strategy 2: class name patterns Claude uses ───────────────

  function strategyClassName() {
    // Claude often uses identifiable class fragments
    const pairs = [
      { sel: '[class*="human"], [class*="HumanTurn"]',    role: 'user' },
      { sel: '[class*="assistant"], [class*="AssistantTurn"]', role: 'assistant' },
    ];
    const allEls = document.querySelectorAll(
      '[class*="human"], [class*="HumanTurn"], [class*="assistant"], [class*="AssistantTurn"]'
    );
    if (!allEls.length) return [];
    const msgs = [];
    for (const el of allEls) {
      // skip if this is a child of another matched element
      if (el.closest('[class*="human"] [class*="human"]') ||
          el.closest('[class*="assistant"] [class*="assistant"]')) continue;
      const cls = (el.className ?? '').toLowerCase();
      const isUser = cls.includes('human');
      const content = cleanText(el);
      if (content.length > 10) msgs.push({ role: isUser ? 'user' : 'assistant', content });
    }
    return msgs;
  }

  // ── Strategy 3: font-claude-message prose blocks ─────────────

  function strategyProse() {
    // Claude renders AI responses with specific typography classes
    const proseBlocks = document.querySelectorAll(
      '.font-claude-message, [class*="prose"], [class*="message-content"]'
    );
    if (!proseBlocks.length) return [];
    const msgs = [];
    for (const block of proseBlocks) {
      const content = cleanText(block);
      if (content.length > 10) msgs.push({ role: 'assistant', content });
    }
    return msgs;
  }

  // ── Strategy 4: nuclear DOM scan ─────────────────────────────

  function strategyNuclear() {
    const container = document.querySelector(
      'main, [class*="conversation"], [class*="chat-window"], [class*="thread"]'
    ) ?? document.body;
    const candidates = Array.from(container.querySelectorAll('div, p'))
      .filter(el => {
        const text = el.innerText?.trim() ?? '';
        return text.length > 30 && el.children.length < 8;
      })
      .filter((el, _, arr) => !arr.some(o => o !== el && o.contains(el)));
    if (!candidates.length) return [];
    return candidates.slice(0, 40).map((el, i) => ({
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: cleanText(el),
    })).filter(m => m.content.length > 0);
  }

  // ── Main scraper ──────────────────────────────────────────────

  function scrapeConversation() {
    const strategies = [strategyTestId, strategyClassName, strategyProse, strategyNuclear];
    for (const fn of strategies) {
      try {
        const msgs = fn();
        if (msgs.length > 0) {
          console.log(`[UniMemo Claude] ✅ Strategy "${fn.name}" found ${msgs.length} messages`);
          return msgs;
        } else {
          console.log(`[UniMemo Claude] ⬜ Strategy "${fn.name}" found 0 — trying next`);
        }
      } catch (e) {
        console.warn(`[UniMemo Claude] Strategy "${fn.name}" error:`, e.message);
      }
    }
    console.warn('[UniMemo Claude] ❌ All strategies failed');
    return [];
  }

  // ── Capture & send ────────────────────────────────────────────

  function capture() {
    const messages = scrapeConversation();
    if (!messages.length) return;
    chrome.runtime.sendMessage({
      type: 'SAVE_CONTEXT',
      payload: { messages, platform: 'Claude' }
    }).catch(err => console.error('[UniMemo Claude] sendMessage failed:', err));
  }

  // ── CAPTURE_NOW listener ──────────────────────────────────────

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === 'CAPTURE_NOW') {
      capture();
      sendResponse({ ok: true });
    }
  });

  // ── Auto-capture on DOM changes ───────────────────────────────

  let debounceTimer = null;
  const observer = new MutationObserver(() => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      const hasTurns = document.querySelector(
        '[data-testid="human-turn"], [data-testid="assistant-turn"]'
      );
      if (hasTurns) capture();
    }, 2000);
  });

  observer.observe(document.querySelector('main') ?? document.body, {
    childList: true, subtree: true
  });

  console.log('[UniMemo] Claude scraper v2 ready');
})();
