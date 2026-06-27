// ============================================================
// UniMemo — Gemini Conversation Scraper
// Extracts human/model turns from gemini.google.com DOM
// ============================================================

(function () {
  'use strict';

  // Guard: prevent duplicate initialization if injected more than once
  if (window.__unimemoGemini) return;
  window.__unimemoGemini = true;

  // ── Helpers ─────────────────────────────────────────────────

  function cleanText(node) {
    if (!node) return '';
    const clone = node.cloneNode(true);

    // Remove action chips, copy buttons, thumbs, avatars
    clone.querySelectorAll(
      'button, svg, mat-icon, [class*="actions"], [class*="toolbar"], ' +
      '[class*="avatar"], [class*="timestamp"], [aria-hidden="true"]'
    ).forEach(el => el.remove());

    const text = clone.innerText ?? clone.textContent ?? '';
    return text
      .replace(/\n{3,}/g, '\n\n')
      .replace(/\t/g, ' ')
      .trim();
  }

  // ── Scraper ──────────────────────────────────────────────────

  function scrapeConversation() {
    const messages = [];

    // Gemini uses Angular Material components
    // User turns: <user-query> or <div class="query-text">
    // Model turns: <model-response> or <message-content>

    // Strategy 1: semantic custom elements
    const userEls  = document.querySelectorAll('user-query, .query-container');
    const modelEls = document.querySelectorAll('model-response, .response-container');

    if (userEls.length > 0 || modelEls.length > 0) {
      // Interleave in DOM order
      const allTurns = Array.from(
        document.querySelectorAll('user-query, model-response, .query-container, .response-container')
      );

      for (const turn of allTurns) {
        const tagName = turn.tagName?.toLowerCase() ?? '';
        const classes = turn.className ?? '';
        const isUser  = tagName === 'user-query' || classes.includes('query');
        const role    = isUser ? 'user' : 'assistant';
        const content = cleanText(turn);
        if (content) messages.push({ role, content });
      }
    }

    // Strategy 2: message-content divs
    if (messages.length === 0) {
      const contentDivs = document.querySelectorAll('[class*="message-content"], [class*="chat-turn"]');
      contentDivs.forEach((div, i) => {
        const content = cleanText(div);
        if (content) messages.push({ role: i % 2 === 0 ? 'user' : 'assistant', content });
      });
    }

    return messages;
  }

  // ── Send to background ────────────────────────────────────────

  function capture() {
    const messages = scrapeConversation();
    if (messages.length === 0) {
      console.warn('[UniMemo Gemini] No messages found — page may not be loaded yet.');
      return;
    }
    chrome.runtime.sendMessage({
      type: 'SAVE_CONTEXT',
      payload: { messages, platform: 'Gemini' }
    }).catch(err => console.error('[UniMemo Gemini] sendMessage failed:', err));
  }

  // ── Listen for manual trigger ─────────────────────────────────
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === 'CAPTURE_NOW') {
      capture();
      sendResponse({ ok: true });
    }
  });

  // Auto-capture — Gemini is an Angular SPA
  let debounceTimer = null;
  const observer = new MutationObserver(() => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      const hasTurns = document.querySelector('user-query, model-response, .query-container');
      if (hasTurns) capture();
    }, 2500);
  });

  const root = document.querySelector('main, chat-window') ?? document.body;
  observer.observe(root, { childList: true, subtree: true });

  console.log('[UniMemo] Gemini scraper ready');
})();
