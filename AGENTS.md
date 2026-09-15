# Working in thirdeye

This guide is for coding agents and contributors changing the repository. Keep
changes focused, preserve unrelated working-tree edits, and treat local agent
data and credentials as private.

## Environment and portability

- Use Python 3.11 or newer and run project commands through `uv`.
- Install development dependencies with `uv sync --extra dev`.
- CI runs Ubuntu on Python 3.11, 3.12, 3.13, and 3.14, plus Windows on Python
  3.12. Do not assume POSIX-only paths, process behavior, symlinks, or shell
  utilities in production code.
- Put platform-neutral filesystem and process behavior in `src/thirdeye/_compat/`.
  Prefer those helpers when process detachment, locking, or filesystem details
  differ across operating systems.
- Always run pre-commit before handing off changes or opening a pull request.
  It checks structured files, merge markers, whitespace, line endings, Ruff
  linting, and Ruff formatting.

```bash
uv run pre-commit run --all-files
uv run ruff check .
```

## Repository map

- `src/thirdeye/commands/` contains Click command implementations; `cli.py`
  is the command entry point.
- `config.py`, `paths.py`, `store.py`, `writer.py`, and `reader.py` define the
  durable local session store. Preserve the append-only event model.
- `src/thirdeye/platforms/` contains harness-specific adapters:
  `claude/`, `codex/`, `cursor/`, and `copilot/`. Adapters translate their
  hook, transcript, or local-state inputs into the shared event/turn model;
  do not leak platform-specific parsing into core modules.
- `src/thirdeye/tracing/model.py` is the generic, JSON-serializable turn
  envelope. Platform adapters build it; core export consumes it.
- `otel_export.py` and `otel_worker.py` turn completed turns into OpenTelemetry
  spans. `usage/` owns local token accounting and its indexes.
- `web/` is the local Starlette UI: routes, templates, and static assets.
- `tests/platforms/`, `tests/shared/`, `tests/web/`, and `tests/parity/` mirror
  those boundaries. Add coverage beside the behavior you change.

## Platform capture and Logfire

The local event store is authoritative. Platform adapters capture raw events,
reconstruct completed turns, and then hand the generic turn envelope to the
export layer. Keep platform parsing, correlation, and recovery conservative:
supplement missing data rather than overriding data directly supplied by a
harness.

Logfire export is optional and must never block an agent hook. `otel_export`
writes a job and starts a detached `otel_worker`; the worker performs the
network request and flush. Export failures must be fail-open, avoid stdout in
hook processes, and be recorded through the capture error path instead of
affecting the agent run. Preserve the OpenTelemetry GenAI message, usage, and
tool-span conventions when changing turn or span data.

## Bundled skills

`src/thirdeye/skills/` contains skills packaged with thirdeye and installed by
`thirdeye skills add`:

- `use-thirdeye` — inspect and search captured sessions.
- `thirdeye-evals` — define and run evaluations over recorded sessions.
- `thirdeye-review` — find agent inefficiencies and suggest improvements.
- `thirdeye-filter` — translate UI Ask requests into structured filters.
- `thirdeye-logfire` — configure and troubleshoot thirdeye's Logfire export.

Keep each skill narrowly scoped, make its trigger and safety boundaries clear,
and update `pyproject.toml` package-data entries when adding packaged skill
resources.

## Development and testing

Start with the narrowest relevant test, then expand before handoff:

```bash
uv run pytest tests/platforms/cursor -q
uv run pytest tests/shared -q
uv run pytest -q
uv run pre-commit run --all-files
```

Always run `uv run pre-commit run --all-files` before a pull request. Also run
`uv run pytest -q` for changes that cross shared storage, tracing, CLI, or
platform boundaries. For platform work, test the matching adapter and add
fixtures that represent only synthetic input.

Tests must be hermetic. Use `tmp_path`, `monkeypatch`, and the protections in
`tests/conftest.py`; never read or modify real agent configuration, credentials,
session transcripts, local databases, or user home-directory data. When
changing an event shape, turn reconstruction, or exported span attribute,
assert both the local model and its user-visible/exported behavior where
appropriate.

Before creating a pull request, check the diff for unrelated changes, describe
the user-visible impact, and state the validation you ran.
