/* sse.js v1 — vendored 2026-05-25 */
(function () {
  "use strict";

  function init() {
    var tree = document.getElementById("tree");
    if (!tree) return;
    if (tree.dataset.status !== "open") return;

    var sid = tree.dataset.sid;
    var lastSeq = parseInt(tree.dataset.lastSeq, 10);
    if (isNaN(lastSeq)) lastSeq = -1;
    if (!sid) return;

    // The tree contents are rendered via htmx on `load`. Wait for the
    // swap to complete before opening the SSE connection, otherwise
    // events that arrive before the <ol> exists are dropped AND any
    // <li> we appended early gets wiped out by htmx's innerHTML swap.
    if (!tree.querySelector("ol.tree")) {
      tree.addEventListener("htmx:afterSwap", function onSwap() {
        tree.removeEventListener("htmx:afterSwap", onSwap);
        startStream(tree, sid, lastSeq);
      });
      return;
    }
    startStream(tree, sid, lastSeq);
  }

  function startStream(tree, sid, lastSeq) {
    var url = "/sessions/" + encodeURIComponent(sid) + "/stream?last_seq=" + lastSeq;
    var es = new EventSource(url);

    es.onmessage = function (msg) {
      try {
        var ev = JSON.parse(msg.data);
        if (!appendEvent(tree, ev)) {
          // Tree not ready — don't advance lastSeq so we don't lose
          // events; browser will reconnect with the unchanged offset.
          return;
        }
        if (typeof ev.seq === "number" && ev.seq > lastSeq) {
          lastSeq = ev.seq;
          tree.dataset.lastSeq = String(lastSeq);
        }
      } catch (err) {
        console.error("sse: bad event payload", err);
      }
    };

    es.addEventListener("closed", function () {
      es.close();
      tree.dataset.status = "closed";
    });

    es.addEventListener("degraded", function (msg) {
      console.warn("sse: degraded", msg.data);
    });

    es.onerror = function () {
      // EventSource auto-reconnects on transient errors; just log.
      console.warn("sse: connection error, browser will retry");
    };
  }

  function appendEvent(tree, ev) {
    var list = tree.querySelector("ol.tree");
    if (!list) return false;
    var sid = tree.dataset.sid;
    var t = ev.t || "unknown";

    var li = document.createElement("li");
    li.className = "evt evt-" + t;
    li.dataset.seq = String(ev.seq);

    var details = document.createElement("details");
    details.open = true;

    var summary = document.createElement("summary");
    summary.tabIndex = 0;
    summary.setAttribute(
      "hx-get",
      "/sessions/" + encodeURIComponent(sid) + "/events/" + encodeURIComponent(ev.seq)
    );
    summary.setAttribute("hx-target", "#detail-pane");
    summary.setAttribute("hx-swap", "innerHTML");

    var seqSpan = document.createElement("span");
    seqSpan.className = "seq";
    seqSpan.textContent = String(ev.seq);

    var typeSpan = document.createElement("span");
    typeSpan.className = "type";
    typeSpan.textContent = t;

    var hintSpan = document.createElement("span");
    hintSpan.className = "hint";
    var hint = "";
    try {
      hint = JSON.stringify(ev.data == null ? null : ev.data);
    } catch (e) {
      hint = String(ev.data);
    }
    if (hint && hint.length > 80) hint = hint.slice(0, 77) + "...";
    hintSpan.textContent = hint;

    summary.appendChild(seqSpan);
    summary.appendChild(document.createTextNode(" "));
    summary.appendChild(typeSpan);
    summary.appendChild(document.createTextNode(" "));
    summary.appendChild(hintSpan);

    details.appendChild(summary);
    li.appendChild(details);
    list.appendChild(li);

    if (typeof window !== "undefined" && window.htmx && typeof window.htmx.process === "function") {
      window.htmx.process(li);
    }
    return true;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
