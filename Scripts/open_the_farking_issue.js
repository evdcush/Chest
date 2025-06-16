// ==UserScript==
// @name         GitHub, issues: restore the expected behavior of clicking a link and going to a page
// @description  Fixes GitHub's latest solution in search of a problem
// @namespace    https://github.com/evdcush
// @downloadURL  https://raw.githubusercontent.com/evdcush/Chest/master/Scripts/open_the_farking_issue.js
// @updateURL    https://raw.githubusercontent.com/evdcush/Chest/master/Scripts/open_the_farking_issue.js
// @version      6.9
// @author       evdcush
// @match        https://github.com/*
// @icon         https://www.google.com/s2/favicons?sz=64&domain=github.com
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    // disable github's "turbo"/overlay system for issue links
    document.addEventListener('click', function(e) {
        if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;

        const link = e.target.closest('a');
        if (!link || !link.href) return;

        const url = new URL(link.href);
        if (url.hostname !== 'github.com') return;

        // check if this is any kind of issue link
        if (url.pathname.includes('/issues/') || url.searchParams.get('issue')) {
            // force full page navigation by setting target="_top"
            link.setAttribute('data-turbo', 'false');
            link.setAttribute('target', '_top');

            // manually navigate to be sure
            e.preventDefault();
            e.stopPropagation();

            if (url.searchParams.get('issue')) {
                // handle &issue= format
                const parts = url.searchParams.get('issue').split('|');
                if (parts.length === 3) {
                    window.location.href = `https://github.com/${parts[0]}/${parts[1]}/issues/${parts[2]}`;
                    return;
                }
            }

            // for all other issue links, just navigate directly
            window.location.href = link.href;
        }
    }, true);
})();
