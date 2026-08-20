"""The single canonical read path over usage sidecars.

The sidecar (``usage.jsonl``) is a faithful raw mirror: writers append one row
per source frame and never read-modify-write, so the same logical LLM call can
appear many times. Claude repeats the identical ``message.usage`` across every
content-block frame of one API call; Codex emits byte-identical repeat reports.
``iter_calls`` collapses those duplicates into one row per distinct
``(session_id, call_id)``.

This is why capture needs no locking anywhere: two triggers racing on the same
transcript offset append the same rows, and their identical ``call_id``s
collapse here on read. Deduplication happens only in this module — every
consumer that wants logical calls (as opposed to the raw mirror) goes through
``iter_calls``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from thirdeye.usage.store import UsageStore
from thirdeye.usage.types import UsageRow


def _collapse_by_call(rows: Iterable[UsageRow]) -> list[UsageRow]:
    """Collapse to one row per distinct (session_id, call_id); last occurrence
    wins. Order follows first appearance of each call_id, so output is stable
    and roughly chronological even though the surviving value is the last one
    seen. Last-wins is safe because duplicate rows for one call_id carry
    identical token values, so which copy survives cannot change any result.
    """
    latest: dict[tuple[str, str], UsageRow] = {}
    for row in rows:
        latest[(row.session_id, row.call_id)] = row
    return list(latest.values())


def iter_calls(session_dir_: Path) -> Iterator[UsageRow]:
    """Yield one row per distinct (session_id, call_id); last occurrence wins.

    Reads the raw sidecar via UsageStore.iter_rows() and collapses duplicates.
    """
    yield from _collapse_by_call(UsageStore(session_dir_).iter_rows())


def call_totals(rows: Iterable[UsageRow]) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) summed over rows."""
    input_total = 0
    output_total = 0
    for row in rows:
        input_total += row.input_tokens
        output_total += row.output_tokens
    return input_total, output_total
