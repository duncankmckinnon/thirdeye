# Windows support

Windows support is **experimental**. The test suite runs on `windows-latest` in
CI across Python 3.11 and 3.13, and Claude Code tracing is the verified
integration. The Codex CLI and Cursor installers are implemented but have not
been exercised against those tools on Windows. Please file Windows issues at the
[issue tracker](https://github.com/duncankmckinnon/thirdeye/issues).

Install with `pipx` or `uv` — Homebrew stays macOS/Linux only.

```bash
pipx install 'thrdi[ui,logfire]'
# or
uv tool install 'thrdi[ui,logfire]'
```

## Deliberate platform differences

These are design decisions, not defects.

### 1. Exclusive-only locking

thirdeye uses `msvcrt` file locking on Windows, which has no shared mode. Where
POSIX lets concurrent readers hold a shared `flock`, on Windows every lock is
exclusive, so readers serialize instead of sharing. This is correct but slower.
For a local, single-user tool the extra contention is acceptable.

### 2. Copied skills

`thirdeye skills add` symlinks the bundled skill directory into your project.
Creating a symlink on Windows requires Developer Mode or administrator rights;
without them, thirdeye falls back to copying the directory instead.

A copied skill does not track a thirdeye upgrade the way a symlink does. After
`pipx upgrade thrdi` (or `uv tool upgrade`), rerun `thirdeye skills add --force`
to refresh the copied skills.

### 3. Unverified Codex and Cursor hooks

The Codex CLI and Cursor installers are correct by construction but have not been
run against the real tools on Windows. How each tool invokes a hook command
there — `cmd.exe`, PowerShell, or a direct `CreateProcess` — is unconfirmed, and
therefore so is whether a path containing spaces needs quoting.

To sidestep the question, on Windows only, when the resolved hook binary path
contains a space thirdeye writes the bare binary name into the tool's config
instead of the absolute path. `shutil.which()` found the binary on `PATH`, and
the agent almost certainly inherits that `PATH`, so the bare name resolves
without any shell quoting.

## Deferred

- **Native Windows packaging** (winget, scoop). `pipx` and `uv` already work.
- **A `pywin32`-backed `LockFileEx` backend** for true shared locks, if reader
  contention ever proves to matter in practice. thirdeye adds no runtime
  dependency beyond the stdlib (`msvcrt`, `ctypes`) for Windows support today.
