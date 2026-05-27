// thirdeye saved-views: read/write the active filter to localStorage
// keyed by page (e.g. "sessions"), so navigating to a bare URL
// restores the user's last filter. The "Save current" form captures
// window.location.search at submit time via hx-vals — no JS wiring
// of the form is needed.
(function () {
  const PAGE_KEY = (page) => `thirdeye.${page}.filter`;

  function currentQuery() {
    return window.location.search || "";
  }

  function restoreIfBareAndStored(page, defaultQuery) {
    if (currentQuery()) {
      localStorage.setItem(PAGE_KEY(page), currentQuery());
      return;
    }
    const stored = localStorage.getItem(PAGE_KEY(page));
    if (stored && stored !== "") {
      window.location.replace(window.location.pathname + stored);
    } else if (defaultQuery) {
      window.location.replace(window.location.pathname + defaultQuery);
    }
  }

  function clearStored(page) {
    localStorage.removeItem(PAGE_KEY(page));
  }

  window.thirdeyeViews = { restoreIfBareAndStored, clearStored };
})();

document.addEventListener("DOMContentLoaded", function () {
  if (window.location.pathname === "/") {
    window.thirdeyeViews.restoreIfBareAndStored("sessions", "?since=7d&order=newest");
  }
});
