# Setup and tracing

Install thirdeye and register hooks so events are captured automatically.

## Install

```bash
brew install duncankmckinnon/tap/thirdeye   # macOS / Linux
pipx install thrdi                          # isolated tool install
uv tool install thrdi                       # via uv
```

The PyPI package is `thrdi`; the installed commands are `thirdeye` and `thrdi`
(aliases of each other).

## Enable tracing for a platform

```bash
thirdeye add --claude       # Claude Code
thirdeye add --codex        # OpenAI Codex CLI
thirdeye add --cursor       # Cursor
thirdeye add --copilot      # GitHub Copilot CLI
```

`thirdeye add` is idempotent — running it twice for the same platform leaves
the existing hook entries in place rather than duplicating them.

Hook entries are written into each platform's own config file: Claude Code
uses `~/.claude/settings.json`, Codex uses `~/.codex/config.toml`, and
Cursor uses `~/.cursor/hooks.json`
(covering both the IDE chat and the `cursor-agent` CLI).

## Detach

```bash
thirdeye remove --claude    # remove only Claude hooks
thirdeye remove --codex     # etc.
```

## Verify tracing is live

After the next agent run, a new session should appear:

```bash
thirdeye list                      # JSON-per-line, newest first
thirdeye list --tree               # human-readable
thirdeye events <sid>              # events for one session
```

`<sid>` accepts any unique prefix — usually 4-8 characters is enough.

## Data layout

All captured data lives under `<thirdeye_home>/traces/<platform>/<sid>/`:

| File           | Purpose                                                          |
| -------------- | ---------------------------------------------------------------- |
| `events.alog`  | Append-only event log (msgpack frames). Never mutated.           |
| `events.idx`   | Index of frame offsets for fast seq lookup.                      |
| `tags.jsonl`   | Sidecar with tag add/remove operations (append-only, replayable).|
| `meta.yaml`    | Session metadata (`platform`, `cwd`, timestamps, status).        |

`<thirdeye_home>` defaults to `~/.thirdeye`. Override with the `THIRDEYE_HOME`
environment variable if needed.
