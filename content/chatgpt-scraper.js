// ============================================================
// UniMemo — ChatGPT Conversation Scraper (v2 — robust)
// Multiple selector strategies + nuclear fallback
// ============================================================

(function () {
  'use strict';

  if (window.__unimemoChatGPT) return;
  window.__unimemoChatGPT = true;

  // ── Text cleaner ─────────────────────────────────────────────

  function cleanText(node) {
    if (!node) return '';
    const clone = node.cloneNode(true);
    clone.querySelectorAll(
      'button, svg, [aria-hidden="true"], [data-testid*="action"], ' +
      'script, style, .sr-only'
    ).forEach(el => el.remove());
    return (clone.innerText ?? clone.textContent ?? '')
      .replace(/\n{3,}/g, '\n\n').replace(/\t/g, ' ').trim();
  }

  // ── Strategy 1: data-message-author-role (current ChatGPT) ───

  function strategyRole() {
    const turns = document.querySelectorAll('[data-message-author-role]');
    if (!turns.length) return [];
    const msgs = [];
    for (const t of turns) {
      const role = t.dataset.messageAuthorRole;
      if (role !== 'user' && role !== 'assistant') continue;
      const el = t.querySelector('.markdown, .prose, [class*="prose"], [class*="whitespace-pre-wrap"]') ?? t;
      const content = cleanText(el);
      if (content) msgs.push({ role, content });
    }
    return msgs;
  }

  // ── Strategy 2: article[data-testid] turn structure ──────────

  function strategyArticle() {
    const articles = document.querySelectorAll('article[data-testid]');
    if (!articles.length) return [];
    const msgs = [];
    for (const a of articles) {
      const roleEl = a.querySelector('[data-message-author-role]');
      if (roleEl) {
        const role = roleEl.dataset.messageAuthorRole;
        if (role !== 'user' && role !== 'assistant') continue;
        const content = cleanText(roleEl);
        if (content) msgs.push({ role, content });
      } else {
        // Guess role from testid index (even = user, odd = assistant)
        const idx = parseInt(a.dataset.testid?.replace(/\D/g, '') ?? '0', 10);
        const content = cleanText(a);
        if (content) msgs.push({ role: idx % 2 === 1 ? 'user' : 'assistant', content });
      }
    }
    return msgs;
  }

  // ── Strategy 3: class-name heuristics ────────────────────────

  function strategyClassName() {
    const userSels = [
      '[class*="user-message"]', '[class*="human-message"]',
      '[class*="user-turn"]', '[class*="human-turn"]',
    ];
    const assistantSels = [
      '[class*="assistant-message"]', '[class*="bot-message"]',
      '[class*="gpt-message"]', '[class*="assistant-turn"]',
    ];
    const msgs = [];
    // Interleave in DOM order
    const allCandidates = document.querySelectorAll(
      [...userSels, ...assistantSels].join(', ')
    );
    for (const el of allCandidates) {
      const cls = el.className ?? '';
      const isUser = userSels.some(s => el.matches(s));
      const content = cleanText(el);
      if (content) msgs.push({ role: isUser ? 'user' : 'assistant', content });
    }
    return msgs;
  }

  // ── Strategy 4: nuclear — scan main for substantial text ─────

  function strategyNuclear() {
    const main = document.querySelector('main') ?? document.body;
    // Find all leaf-ish divs with substantial text content
    const candidates = Array.from(main.querySelectorAll('div, p'))
      .filter(el => {
        const text = el.innerText?.trim() ?? '';
        // Must have >20 chars, no many children (i.e., leaf node)
        return text.length > 20 && el.children.length < 5 &&
               !el.querySelector('[data-message-author-role]'); // avoid containers
      })
      .filter((el, _, arr) => !arr.some(other => other !== el && other.contains(el)));

    if (!candidates.length) return [];

    // Alternate user/assistant — imperfect but better than nothing
    return candidates.slice(0, 40).map((el, i) => ({
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: cleanText(el),
    })).filter(m => m.content.length > 0);
  }

  // ── Main scraper ──────────────────────────────────────────────

  function scrapeConversation() {
    const strategies = [strategyRole, strategyArticle, strategyClassName, strategyNuclear];
    for (const fn of strategies) {
      try {
        const msgs = fn();
        if (msgs.length > 0) {
          console.log(`[UniMemo ChatGPT] ✅ Strategy "${fn.name}" found ${msgs.length} messages`);
          return msgs;
        } else {
          console.log(`[UniMemo ChatGPT] ⬜ Strategy "${fn.name}" found 0 — trying next`);
        }
      } catch (e) {
        console.warn(`[UniMemo ChatGPT] Strategy "${fn.name}" error:`, e.message);
      }
    }
    console.warn('[UniMemo ChatGPT] ❌ All strategies failed — page structure unknown');
    return [];
  }

  // ── Capture & send ────────────────────────────────────────────

  function capture() {
    const messages = scrapeConversation();
    if (!messages.length) return;
    chrome.runtime.sendMessage({
      type: 'SAVE_CONTEXT',
      payload: { messages, platform: 'ChatGPT' }
    }).catch(err => console.error('[UniMemo ChatGPT] sendMessage failed:', err));
  }

  // ── CAPTURE_NOW listener ──────────────────────────────────────

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === 'CAPTURE_NOW') {
      capture();
      sendResponse({ ok: true });
    }
  });

  // ── Auto-capture on conversation changes ──────────────────────

  let debounceTimer = null;
  let lastUrl = location.href;

  const observer = new MutationObserver(() => {
    const currentUrl = location.href;
    if (currentUrl !== lastUrl) {
      lastUrl = currentUrl;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(capture, 3000);
      return;
    }
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      if (document.querySelector('[data-message-author-role]')) capture();
    }, 2500);
  });

  observer.observe(document.querySelector('main') ?? document.body, {
    childList: true, subtree: true
  });

  console.log('[UniMemo] ChatGPT scraper v2 ready');
})();
