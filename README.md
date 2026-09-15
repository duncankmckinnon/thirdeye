<p align="center">
  <a href="https://third3y3.com">
    <img src="docs/img/logo_transparent.png" alt="thirdeye" width="160" />
  </a>
</p>

[![PyPI](https://img.shields.io/pypi/v/thrdi.svg)](https://pypi.org/project/thrdi/)
[![Homebrew](https://img.shields.io/badge/homebrew-duncankmckinnon%2Ftap-orange)](https://github.com/duncankmckinnon/homebrew-tap)
[![CI](https://github.com/duncankmckinnon/thirdeye/actions/workflows/test.yml/badge.svg)](https://github.com/duncankmckinnon/thirdeye/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/duncankmckinnon/thirdeye/branch/main/graph/badge.svg)](https://codecov.io/gh/duncankmckinnon/thirdeye)
[![Python](https://img.shields.io/pypi/pyversions/thrdi.svg)](https://pypi.org/project/thrdi/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Trace every agent session on your machine — Claude Code, Codex, Cursor, Copilot —
into one history you and your agents can manage, search, and evaluate.

Thirdeye persists each session as a durable, append-only local event history.
Prompts, responses, tool calls, subagents, usage, and session metadata remain
available after the original agent process exits, giving both people and agents
a consistent record to inspect. Remote export is optional; the local history
remains the source of truth.

Visit [third3y3.com](https://third3y3.com) for guides and the full command reference.

## Installation

Homebrew is the simplest option on macOS and Linux. On Windows, use `pipx`
or `uv`.

```bash
brew install duncankmckinnon/tap/thirdeye
```

Alternatively, install with `pipx`:

```bash
pipx install thrdi
```

Or use `uv` or `pip`:

```bash
uv tool install thrdi
pip install thrdi
```

## Setup

Run the guided setup after installation:

```bash
thirdeye setup
```

The setup flow:

1. Selects the agent integrations you want to trace.
2. Installs the required hooks or agent configuration.
3. Offers to install thirdeye's bundled skills in the matching agent directories.
4. Optionally connects Pydantic Logfire and saves the write token used for export.

You can rerun setup whenever you add an agent or change your Logfire configuration.

## Logfire

When enabled during setup, thirdeye mirrors captured sessions to
[Pydantic Logfire](https://logfire.pydantic.dev) as OpenTelemetry traces. A
session becomes a trace containing agent turns, model calls, tool executions,
subagents, and available token usage. Export happens outside the agent hook so
network latency does not interrupt the agent session.

See [third3y3.com](https://third3y3.com) for Logfire configuration and export details.

## Skills

Thirdeye includes focused skills that let coding agents work directly with the
history it captures:

- **`use-thirdeye`** — inspect sessions, search events, debug tool calls, and
  analyze captured usage.
- **`thirdeye-evals`** — create evaluation rubrics, run evaluators, and interpret
  per-turn findings.
- **`thirdeye-review`** — audit agent behavior for inefficiencies and recommend
  concrete invocation or skill improvements.
- **`thirdeye-filter`** — translate natural-language questions from the UI into
  structured session and turn filters.
- **`thirdeye-logfire`** — configure, verify, and troubleshoot thirdeye's
  Logfire integration.

The setup flow can install these for you. To add them later, run
`thirdeye skills add`; use `thirdeye skills list` to see what is available.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for local
development, validation, and pull-request guidance.
