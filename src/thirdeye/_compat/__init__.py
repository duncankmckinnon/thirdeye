"""Platform-specific primitives for thirdeye.

This package is the only place in the codebase permitted to branch on the
operating system. Nothing outside ``thirdeye._compat`` may import ``fcntl``,
``msvcrt``, or inspect ``sys.platform`` directly.

The package ``__init__`` intentionally exports only :data:`IS_WINDOWS`; submodule
imports stay explicit (``from thirdeye._compat.locking import LockMode, locked``).
"""

import sys

IS_WINDOWS: bool = sys.platform.startswith("win")
