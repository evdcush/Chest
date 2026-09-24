// ==UserScript==
// @name         Strip utm_source=chatgpt.com from ChatGPT links
// @namespace    https://github.com/
// @version      1.0.0
// @description  Removes the utm_source=chatgpt.com tracking parameter from all URLs in ChatGPT responses (link targets and visible link text).
// @match        https://chatgpt.com/*
// @match        https://chat.openai.com/*
// @run-at       document-idle
// @grant        none
// ==/UserScript==

(function () {
  'use strict';

  const TRACKING_PARAM = /^utm_source=chatgpt\.com$/i;
  const QUICK_CHECK = /utm_source=chatgpt\.com/i;

  // String-based cleaning (instead of the URL API) so nothing else in the
  // URL gets normalized, e.g. no trailing slash added, no re-encoding.
  function cleanUrl(url) {
    if (!QUICK_CHECK.test(url)) return url;

    const hashIdx = url.indexOf('#');
    const hash = hashIdx >= 0 ? url.slice(hashIdx) : '';
    const beforeHash = hashIdx >= 0 ? url.slice(0, hashIdx) : url;

    const qIdx = beforeHash.indexOf('?');
    if (qIdx < 0) return url;

    const path = beforeHash.slice(0, qIdx);
    const params = beforeHash
      .slice(qIdx + 1)
      .split('&')
      .filter((p) => !TRACKING_PARAM.test(p));

    return path + (params.length ? '?' + params.join('&') : '') + hash;
  }

  function cleanAnchor(a) {
    // 1. The link target
    const href = a.getAttribute('href');
    if (href && QUICK_CHECK.test(href)) {
      const cleaned = cleanUrl(href);
      if (cleaned !== href) a.setAttribute('href', cleaned);
    }

    // 2. The visible link text (ChatGPT often renders the raw URL as text)
    if (QUICK_CHECK.test(a.textContent)) {
      const walker = document.createTreeWalker(a, NodeFilter.SHOW_TEXT);
      let node;
      while ((node = walker.nextNode())) {
        if (!QUICK_CHECK.test(node.nodeValue)) continue;
        // Clean each whitespace-separated token so surrounding text is preserved
        const cleaned = node.nodeValue.replace(/\S+/g, cleanUrl);
        if (cleaned !== node.nodeValue) node.nodeValue = cleaned;
      }
    }
  }

  function cleanAll(root) {
    if (!root || root.nodeType !== Node.ELEMENT_NODE) return;
    if (root.matches && root.matches('a')) cleanAnchor(root);
    root.querySelectorAll('a[href*="utm_source"], a').forEach(cleanAnchor);
  }

  // Batch mutations so streaming responses don't trigger a scan per token.
  let scheduled = false;
  const pending = new Set();

  function flush() {
    scheduled = false;
    for (const node of pending) {
      if (node.isConnected) cleanAll(node);
    }
    pending.clear();
  }

  function schedule(node) {
    pending.add(node);
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(flush);
    }
  }

  const observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.type === 'childList') {
        m.addedNodes.forEach((n) => {
          if (n.nodeType === Node.ELEMENT_NODE) schedule(n);
          else if (n.parentElement) schedule(n.parentElement);
        });
      } else if (m.type === 'attributes') {
        schedule(m.target);
      } else if (m.type === 'characterData' && m.target.parentElement) {
        // Streaming text updates inside an existing link
        schedule(m.target.parentElement.closest('a') || m.target.parentElement);
      }
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['href'],
    characterData: true,
  });

  // Initial pass for anything already on the page
  cleanAll(document.body);
})();
