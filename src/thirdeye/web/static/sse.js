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

    var url = "/sessions/" + encodeURIComponent(sid) + "/stream?last_seq=" + lastSeq;
    var es = new EventSource(url);

    es.onmessage = function (msg) {
      try {
        var ev = JSON.parse(msg.data);
        appendEvent(tree, ev);
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
    var list = tree.querySelector("ul.event-tree") || tree.querySelector("ul");
    if (!list) return;
    var li = document.createElement("li");
    li.className = "event event-" + (ev.t || "unknown");
    li.dataset.seq = String(ev.seq);
    li.dataset.type = ev.t || "unknown";

    var seqSpan = document.createElement("span");
    seqSpan.className = "seq";
    seqSpan.textContent = "#" + ev.seq;

    var typeSpan = document.createElement("span");
    typeSpan.className = "type";
    typeSpan.textContent = ev.t || "unknown";

    var tsSpan = document.createElement("span");
    tsSpan.className = "ts";
    tsSpan.textContent = ev.ts || "";

    li.appendChild(seqSpan);
    li.appendChild(document.createTextNode(" "));
    li.appendChild(typeSpan);
    li.appendChild(document.createTextNode(" "));
    li.appendChild(tsSpan);

    list.appendChild(li);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
