# Contributing to thirdeye

Thanks for contributing.

## Development setup

Thirdeye requires Python 3.11 or newer and uses [uv](https://docs.astral.sh/uv/)
for local development.

```bash
git clone https://github.com/duncankmckinnon/thirdeye.git
cd thirdeye
uv sync --extra dev
```

Run the CLI from the checkout with:

```bash
uv run thirdeye --help
```

## Validate changes

Run the relevant tests while developing, then run the full suite before opening
a pull request:

```bash
uv run pytest tests/platforms/cursor -q
uv run pytest -q
uv run ruff check .
```

Add or update tests for behavior changes. Tests must be hermetic: do not read
or modify real agent configuration, credentials, session history, or other
developer-local data.

## Pull requests

- Keep each pull request focused and describe the user-visible change.
- Include validation you ran in the PR description.
- Preserve user-owned working-tree changes; avoid broad formatting or unrelated
  refactors.
- Update documentation when a command or public behavior changes.

For larger changes, open an issue or discuss the design before implementation.
