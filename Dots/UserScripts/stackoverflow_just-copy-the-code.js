// ==UserScript==
// @name         Stack Overflow: copy just the code when you copy code.
// @namespace    https://example.com/
// @version      1.0
// @description  Remove Stack Overflow's injected attribution/header when copying code blocks
// @match        https://stackoverflow.com/*
// @match        https://*.stackexchange.com/*
// @match        https://superuser.com/*
// @match        https://serverfault.com/*
// @match        https://askubuntu.com/*
// @grant        GM_setClipboard
// @run-at       document-start
// ==/UserScript==

(function () {
  'use strict';

  function isCopyButton(el) {
    if (!(el instanceof Element)) return false;

    const button = el.closest('button, [role="button"]');
    if (!button) return false;

    const label = [
      button.getAttribute('aria-label'),
      button.getAttribute('title'),
      button.textContent,
      button.dataset?.controller,
      button.className,
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase();

    return label.includes('copy');
  }

  function findCodeElement(button) {
    // Look upward for a container that holds this copy button and a code block.
    for (let node = button; node; node = node.parentElement) {
      const code = node.querySelector('pre > code, pre, code');
      if (code) return code;
    }

    // Fallback: nearby siblings.
    let node = button.parentElement;
    while (node) {
      const prev = node.previousElementSibling;
      const next = node.nextElementSibling;

      const code =
        prev?.matches?.('pre, code') ? prev :
        prev?.querySelector?.('pre > code, pre, code') ||
        next?.matches?.('pre, code') ? next :
        next?.querySelector?.('pre > code, pre, code');

      if (code) return code;
      node = node.parentElement;
    }

    return null;
  }

  function getCodeText(el) {
    // innerText usually matches what the user sees/copies best on code blocks.
    // Trim one trailing newline only, to avoid adding extra blank lines.
    let text = (el.innerText ?? el.textContent ?? '').replace(/\r\n/g, '\n');
    text = text.replace(/\n$/, '');
    return text;
  }

  async function copyText(text) {
    if (typeof GM_setClipboard === 'function') {
      GM_setClipboard(text, 'text');
      return;
    }

    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }

    // Last-resort fallback
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.top = '-1000px';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
  }

  document.addEventListener(
    'click',
    async (event) => {
      if (!isCopyButton(event.target)) return;

      const button = event.target.closest('button, [role="button"]');
      const codeEl = findCodeElement(button);
      if (!codeEl) return;

      const text = getCodeText(codeEl);
      if (!text) return;

      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();

      try {
        await copyText(text);

        // Optional tiny visual feedback
        const originalAria = button.getAttribute('aria-label');
        const originalTitle = button.getAttribute('title');
        const originalText = button.textContent;

        if (button.tagName === 'BUTTON') {
          button.setAttribute('aria-label', 'Copied');
          button.setAttribute('title', 'Copied');
          if (button.textContent.trim().toLowerCase() === 'copy') {
            button.textContent = 'Copied';
            setTimeout(() => {
              if (originalText != null) button.textContent = originalText;
            }, 1200);
          }
          setTimeout(() => {
            if (originalAria != null) button.setAttribute('aria-label', originalAria);
            if (originalTitle != null) button.setAttribute('title', originalTitle);
          }, 1200);
        }
      } catch (err) {
        console.error('Failed to copy code text:', err);
      }
    },
    true
  );
})();
