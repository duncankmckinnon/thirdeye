// thirdeye saved-views: read/write the active filter to localStorage
// keyed by page (e.g. "sessions"), and inject the current query string
// into the "Save current" form before submit.
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

  function wireSaveForm() {
    document.querySelectorAll("[data-current-query]").forEach((el) => {
      el.value = currentQuery();
    });
  }

  function clearStored(page) {
    localStorage.removeItem(PAGE_KEY(page));
  }

  window.thirdeyeViews = { restoreIfBareAndStored, wireSaveForm, clearStored };
})();

document.addEventListener("DOMContentLoaded", function () {
  if (window.location.pathname === "/") {
    window.thirdeyeViews.restoreIfBareAndStored("sessions", "?since=7d&order=newest");
    window.thirdeyeViews.wireSaveForm();
  }
});
