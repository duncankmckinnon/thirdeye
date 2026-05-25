/* tree.js v1 — vendored 2026-05-25 */

function _treeSummaries() {
  const tree = document.getElementById("tree");
  if (!tree) return [];
  return Array.from(tree.querySelectorAll("summary"));
}

function _toggleAll(open) {
  const tree = document.getElementById("tree");
  if (!tree) return;
  tree.querySelectorAll("details").forEach((d) => {
    d.open = open;
  });
}

function _bindControls() {
  const tree = document.getElementById("tree");
  if (!tree) return;
  const expand = tree.querySelector(".tree-expand-all");
  const collapse = tree.querySelector(".tree-collapse-all");
  if (expand) expand.addEventListener("click", () => _toggleAll(true));
  if (collapse) collapse.addEventListener("click", () => _toggleAll(false));
}

function _onKey(e) {
  const tree = document.getElementById("tree");
  if (!tree || !tree.contains(document.activeElement)) return;
  const summaries = _treeSummaries();
  if (summaries.length === 0) return;
  const idx = summaries.indexOf(document.activeElement);
  if (e.key === "ArrowDown") {
    e.preventDefault();
    const next = summaries[Math.min(summaries.length - 1, Math.max(0, idx) + 1)];
    if (next) next.focus();
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    const prev = summaries[Math.max(0, (idx < 0 ? 0 : idx) - 1)];
    if (prev) prev.focus();
  } else if (e.key === "Enter") {
    if (idx >= 0) {
      e.preventDefault();
      summaries[idx].click();
    }
  }
}

document.addEventListener("htmx:afterSwap", (e) => {
  if (e.target && e.target.id === "tree") {
    _bindControls();
  }
});

document.addEventListener("keydown", _onKey);
