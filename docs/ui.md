# Browser UI

`thirdeye ui` launches a local, browser-based alternative to the CLI for
exploring sessions, visualizing traces, tagging events, and authoring evals.
The server is opt-in and binds loopback only.

## Install

The UI ships as an optional extra so the default `pip install thrdi` stays
lean.

```bash
pip install 'thrdi[ui]'
```

Using pipx? Inject the deps into the existing thrdi venv:

```bash
pipx inject thrdi starlette uvicorn jinja2
```

## Launch

```bash
thirdeye ui                        # http://127.0.0.1:8765 by default
thirdeye ui --port 9000            # pick a port
thirdeye ui --host 127.0.0.1       # interface to bind (loopback only)
thirdeye ui --no-browser           # don't auto-open the browser
```

The CLI opens your default browser unless `--no-browser` is passed. Ctrl-C
stops the server.

If the extra isn't installed, `thirdeye ui` prints a hint and exits with
status 1:

```
The 'ui' extra is required. Install it with: pip install 'thrdi[ui]'
```

## What you can do

- **Browse sessions.** The list at `/` shows every recorded session
  across every platform (claude, codex, cursor), with filters
  for platform, cwd, status (open / closed / stale), date range, and a
  **tag multi-select** dropdown
  populated from every tag in your history. Defaults to the last 7
  days, newest first. The active filter persists in `localStorage` so
  navigating away and back keeps your view.
- **Ask panel (agentic filters).** Both `/` and `/search` carry an
  "Ask" textarea. Type a natural-language description ("find sessions
  about the workbench plan", "long-lasting claude runs this week"),
  pick a CLI agent (claude / codex), and submit. The agent
  receives a grounded vocabulary block (your platforms, cwds, tags)
  plus the bundled `thirdeye-filter` skill and returns a structured
  JSON envelope. The UI **auto-fills the existing filter form** with
  the proposed values — review or tweak any field, then hit the
  Search / Filter button to run. No auto-execute, no separate Run
  button to learn.
- **Saved filter views.** From `/`, name a filter combination and pin
  it to the sidebar (`/views/sessions` under the hood). Saved views
  persist on disk at `<thirdeye_home>/views/sessions.json` and are
  manageable from the CLI too: `thirdeye views list / save / delete
  --page sessions`.
- **Open a session.** Each session view renders the event stream as a
  collapsible tree, color-coded by event type (user, assistant, tool
  call, tool result, etc.). Click any event to expand its full JSON
  payload in the side pane.
- **Tag events.** Add or remove tags from any event inline. Tags are
  stored in the same per-session tag store the CLI uses, so
  `thirdeye search --tag <name>` picks them up immediately.
- **Search across sessions.** Substring search with the same filters
  as `thirdeye search`. Each hit links straight to the event in its
  session.
- **Usage charts.** `/usage` renders daily tokens-over-time
  (stacked input / output) and sessions-per-day charts via vendored
  Chart.js, plus totals cards. Filter by platform / since / until.
  Per-session usage at `/sessions/<id>/usage` keeps the existing
  per-turn detail view.
- **Author and run evals.** Browse, edit, and create eval definitions
  (YAML rubrics) in the browser. Dispatch a run against any single
  session, or select multiple sessions on `/` and dispatch a **batch**
  of runs in one click. Each run gets a unique `run_id`; the same
  definition can be re-run on the same session.
- **Eval-keyed cross-cut.** `/evals/defs/<name>/results` shows every
  run of a given definition across every session it's been applied
  to — sortable, with verdict / agent / started-at / duration and a
  link to each run's findings.
- **Per-(session, definition) panel.** From a session's eval list,
  click a definition chip to open
  `/sessions/<id>/evals/<def_name>`. The panel renders the
  definition's full directive text plus a table of every run of that
  definition on that session, with parsed columns: run id, verdict,
  one column per score key seen across runs (with `—` for absent),
  findings count, duration, started, agent. Useful for comparing
  iterations of the same rubric on one session.
- **Live-tail open sessions.** Sessions that are still receiving
  events stream new entries into the tree in real time via
  Server-Sent Events. No refresh needed.

## Logfire span tree

With Logfire export enabled (see the project README), each session is also a
Logfire trace whose spans nest the way the session view nests events: an
`agent-turn` contains a `chat` span, which contains `tool` spans.

For local Cursor IDE and CLI sessions, a dispatched subagent is exported as its
own `agent-turn` under the `Task` tool span that launched it, and the subagent's
own tool calls nest under that child turn:

```text
agent-turn
└── chat
    └── tool: Task
        └── agent-turn (Cursor subagent)
            └── chat
                └── tool: read_file
```

The `Task` span may be exported before or after its child, but stable span IDs
preserve the same tree regardless of the order spans arrive.

## Keyboard shortcuts

Inside the session tree:

- `↑` / `↓` — move between sibling events.
- `←` — collapse the focused node (or jump to its parent if already
  collapsed).
- `→` — expand the focused node.
- `Enter` — open the focused event in the side pane.

## Security model

- The server **binds `127.0.0.1` by default** and refuses to listen on
  other interfaces unless `--host` is overridden explicitly.
- **No authentication.** The UI is intended for local use on a trusted
  workstation. Don't expose it over a network or tunnel it without adding
  your own auth layer in front.
- All reads and writes operate on the same `<thirdeye_home>/` tree the
  CLI uses (`THIRDEYE_HOME` env var, default `~/.thirdeye/`).

## Troubleshooting

**`ModuleNotFoundError: starlette`** — the `ui` extra isn't installed. Run
`pip install 'thrdi[ui]'` (or `pipx inject thrdi starlette uvicorn jinja2`).

**`OSError: [Errno 48] Address already in use`** — another process holds the
port. Pass `--port` with a free port, or stop the other process.

**Browser didn't open** — pass `--no-browser` to suppress the auto-open and
copy the printed URL manually. On headless machines, always use
`--no-browser`.

**Live tail isn't updating** — confirm the session is still `open` (the
session header shows status). Closed sessions render the full timeline but
don't stream. Some restrictive corporate proxies break SSE; running the
browser on the same host as the server avoids the proxy entirely.

**404 on a session ID** — IDs accept any unique prefix. If you typed an
ambiguous prefix the 404 page lists the closest matches; click one to
disambiguate.
