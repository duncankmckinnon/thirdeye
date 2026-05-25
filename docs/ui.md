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

- **Browse sessions.** The session list shows every recorded session across
  every platform, with filters for platform, cwd, status (open / closed /
  stale), tags, and date range.
- **Open a session.** Each session view renders the event stream as a
  collapsible tree, color-coded by event type (user, assistant, tool call,
  tool result, etc.). Click any event to expand its full JSON payload in
  the side pane.
- **Tag events.** Add or remove tags from any event inline. Tags are stored
  in the same per-session tag store the CLI uses, so `thirdeye search
  --tag <name>` picks them up immediately.
- **Search across sessions.** Substring search with the same filters as
  `thirdeye search`. Each hit links straight to the event in its session.
- **Inspect token usage.** Per-session and global usage views surface the
  same data as `thirdeye usage`: model, input / output / total tokens, and
  per-turn rollups.
- **Author and run evals.** Browse, edit, and create eval definitions
  (YAML rubrics) in the browser. Dispatch a run against any session and
  watch its status poll until the job completes; results render inline
  with the per-turn findings anchored to event `seq`.
- **Live-tail open sessions.** Sessions that are still receiving events
  stream new entries into the tree in real time via Server-Sent Events. No
  refresh needed.

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
