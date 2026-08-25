<p align="center">
  <img src="docs/img/logo_1.jpeg" alt="thirdeye" width="160" />
</p>

# thirdeye

[![PyPI](https://img.shields.io/pypi/v/thrdi.svg)](https://pypi.org/project/thrdi/)
[![Homebrew](https://img.shields.io/badge/homebrew-duncankmckinnon%2Ftap-orange)](https://github.com/duncankmckinnon/homebrew-tap)
[![CI](https://github.com/duncankmckinnon/thirdeye/actions/workflows/test.yml/badge.svg)](https://github.com/duncankmckinnon/thirdeye/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/duncankmckinnon/thirdeye/branch/main/graph/badge.svg)](https://codecov.io/gh/duncankmckinnon/thirdeye)
[![Python](https://img.shields.io/pypi/pyversions/thrdi.svg)](https://pypi.org/project/thrdi/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Trace every agent session on your machine — Claude Code, Codex, Gemini, Cursor — into one history you and your agents can manage, search, and evaluate.

## Install

```bash
brew install duncankmckinnon/tap/thirdeye    # macOS / Linux
pipx install thrdi                           # or: uv tool install thrdi
```

## Install agent skills locally

Install thirdeye's bundled skills for agents working in the current project:

```bash
thirdeye skills list                                 # show bundled skill names
thirdeye skills add                                  # all skills → .agents/skills/
thirdeye skills add --claude --codex                 # both agent-specific folders
thirdeye skills add -p path/to/folder                # custom parent folder
thirdeye skills add --only thirdeye-review           # just one
thirdeye skills add --force                          # replace existing entries
```

Skills install as symlinks, so upgrading thirdeye (`brew upgrade thirdeye` or
`pipx upgrade thrdi`) automatically refreshes them in every repository where
they are installed.

## Enable tracing

```bash
thirdeye add --claude        # also: --cursor, --codex
```

To detach: `thirdeye remove --claude`.

## Read your history

```bash
thirdeye list                          # every session, every platform
thirdeye events <id>                   # one session, terse
thirdeye tail <id> -n 5                # last few events
thirdeye event <id> <seq>              # one event, fully expanded
thirdeye search "migration"            # substring across all sessions
thirdeye stats                         # totals
```

## Tag and filter

```bash
thirdeye tag <id> <seq> --add bug,review     # tag an event
thirdeye tag <id> --list                     # list tagged events in a session
thirdeye tag <id> <seq> --remove bug         # untag
thirdeye tags                                # global tag inventory
thirdeye search "migration" --tag review --platform claude --since 2026-05-01
```

Add `--json` for parseable JSONL, `--tree` for human-readable, `--platform` / `--cwd` / `--tag` / `--since` / `--until` to filter. Session IDs accept any unique prefix. Run `thirdeye --help` for the full reference.

## Per-turn usage

thirdeye captures model name and token counts per turn into an append-only
sidecar (`usage.jsonl`) and a global SQLite index (`usage.db`). Capture starts
automatically on the next agent run after `thirdeye add`.

```bash
thirdeye usage                          # global rollup, sessions by token spend
thirdeye usage <id>                     # per-turn detail for one session
thirdeye usage --top 5 --since 2026-05-01
thirdeye usage <id> --json              # parseable JSONL rows
thirdeye usage reindex                  # rebuild SQLite from sidecars
thirdeye usage errors                   # tail the capture audit log
```

Filters: `--platform` / `--harness`, `--model SUBSTR`, `--since` / `--until`,
`--top N`, `--sort total|input|output|ts`.

## Export to Pydantic Logfire

Mirror every captured session into [Logfire](https://pydantic.dev/logfire) live, as traces — no separate export step. Once enabled, each thirdeye session becomes one Logfire trace: tool calls appear as spans with real durations (paired from `PreToolUse`/`PostToolUse`, or Codex's `call_id`), everything else (messages, notifications, compaction, ...) as timeline markers, all searchable by `gen_ai.conversation.id`.

On Claude Code, each individual model call within a turn gets its own `chat <model>` span. Codex uses a turn-level `chat <model>` span reconstructed from the completed turn in its rollout JSONL, with `execute_tool <name>` children for every paired tool call. Both use OpenTelemetry GenAI attributes for model, messages, and token usage; Codex repeated token reports are deduplicated by their cumulative watermark before turn totals are calculated.

```bash
pip install 'thrdi[logfire]'
thirdeye logfire enable                                           # securely prompts for gateway key
thirdeye logfire status
thirdeye logfire disable                                          # keeps the saved key
```

Or from `thirdeye ui`, under **settings**: paste the gateway key and project name and hit Enable — persisted the same way, in `~/.thirdeye/config.yaml`.

Export is dispatched from the same Claude Code / Codex hooks that already capture events, but the actual Logfire call (including a flush, a real network round trip) runs in a detached background process — the hook itself never waits on the network, so enabling this adds no latency to your tool calls.

## Browse in a browser

For a richer experience than the CLI, install the UI extra and launch:

```bash
pip install 'thrdi[ui]'
thirdeye ui
```

The local browser UI covers:

- **Sessions list** with platform / cwd / status / date filters and a
  tag multi-select drawn from every tag in your history, defaulting to
  the last 7 days, newest first.
- **Ask panel** — type "find sessions about the workbench plan" or
  "long-lasting claude runs this week" and a CLI agent of your choice
  (claude / codex / gemini) auto-fills the filter form. Review the
  populated fields and hit Search / Filter to run.
- **Saved filter views** — name a filter combination and pin it to the
  sidebar; restored across browser sessions via local storage.
- **Session view** — collapsible event tree color-coded by event type,
  inline tag editing, live-tail via Server-Sent Events for open sessions.
- **Evals** — author and edit YAML rubrics; dispatch a run on one
  session or a batch on a selection. Two complementary tables: per-
  definition cross-cut (`/evals/defs/<name>/results`) for comparing a
  rubric across sessions, and a per-(session, definition) panel showing
  the directive text plus every run on that session with parsed verdict
  and score columns.
- **Usage charts** — daily tokens-over-time and sessions-per-day with a
  platform filter and totals cards.

The server binds loopback only.

See [docs/ui.md](docs/ui.md) for full reference.

## Evaluations

Grade a recorded session by dispatching one of your installed CLI agents
(claude / codex / gemini) as an LLM-as-judge. Eval definitions are named
rubrics — directive text shipped with sensible defaults and editable per-user.

```bash
thirdeye eval def list                                          # available rubrics
thirdeye eval def show default                                  # see the directive
thirdeye eval def create my-rubric --directive "<text>"         # custom rubric

thirdeye eval run <id> --agent claude                           # foreground
thirdeye eval run <id> --agent gemini --using token-efficiency --background

thirdeye eval show <id>                                         # latest result
thirdeye eval list --since 2026-05-01 --verdict warn            # history
thirdeye eval status                                            # background jobs
```

Per-turn findings are stored with the event `seq` they anchor to, and
`thirdeye events <id>` annotates the timeline inline by default (suppress with
`--no-findings`, filter with `--eval NAME`). The eval invocation itself is a
thirdeye-traced session, so every grading run has its own audit trail.

Dispatched agents run in read-only mode (Claude `--allowedTools` allowlist,
Codex `--sandbox read-only`, Gemini `--approval-mode plan`). No new Python
deps — thirdeye shells out to the agent binaries you already have installed.

## Agent

Dispatch an AI agent (Claude Code, Codex, or Gemini) against your thirdeye
history directly from the CLI. The agent is pre-loaded with its analysis and
evaluation skills and runs in read-only mode by default.

```bash
thirdeye agent "review my sessions from the last week"
thirdeye agent "find sessions where token usage spiked" --stream
thirdeye agent "fix inefficient tool use in session abc123" --fix
thirdeye agent "summarize eval findings" --agent gemini
```

Flags:

| Flag | Description |
|------|-------------|
| `--stream` | Print tool calls and results in real time as the agent explores |
| `--fix` | Unlock full tool access so the agent can edit files (default: read-only) |
| `--agent NAME` | Agent to dispatch: `claude` (default), `codex`, or `gemini` |
| `--skill PATH` | Inject an additional skill from a local file (repeatable) |
| `--skills` | List the built-in skills and exit |
| `--cwd PATH` | Working directory context injected into the prompt |

New sessions opened by the agent are automatically tagged `thirdeye-agent`
so you can filter them with `thirdeye list --tag thirdeye-agent`.

### Skills used by `thirdeye agent`

Four bundled skills are injected into every `thirdeye agent` run by default:

- **`use-thirdeye`** — basic CLI fluency: enable tracing, search sessions,
  debug tool calls, analyze token usage.
- **`thirdeye-evals`** — eval workflow: create rubrics, dispatch
  evaluators, view per-turn findings.
- **`thirdeye-review`** — audit other agents' traces to find
  inefficiencies and propose invocation, skill, or convention changes
  (cohort scoping, tool-mix patterns, token spikes, recommendation
  templates).
- **`thirdeye-filter`** — directive used by the browser UI's Ask panel
  to translate natural-language queries into filter JSON. Installed
  alongside the others; not invoked directly by agents.

Pass `--skill path/to/skill.md` to inject additional skills from local files
alongside the defaults. Run `thirdeye agent --skills` to see the default list.

## License

MIT.
