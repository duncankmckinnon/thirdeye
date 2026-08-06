from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from thirdeye.paths import sessions_root, usage_db_path, usage_jsonl_path
from thirdeye.usage.errlog import log_capture_error
from thirdeye.usage.read import iter_calls

# Bumped to 2 for the OTel GenAI schema. usage.db is purely derived from the
# sidecars, so a user_version below SCHEMA_VERSION drops and rebuilds rather
# than migrating.
SCHEMA_VERSION = 2

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS usage (
    session_id   TEXT NOT NULL,
    call_id      TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    ts           TEXT NOT NULL,
    platform     TEXT NOT NULL,
    gen_ai_provider_name  TEXT NOT NULL,
    gen_ai_operation_name TEXT NOT NULL,
    gen_ai_response_model TEXT NOT NULL,
    gen_ai_usage_input_tokens  INTEGER NOT NULL,
    gen_ai_usage_output_tokens INTEGER NOT NULL,
    gen_ai_usage_cache_read_input_tokens     INTEGER,
    gen_ai_usage_cache_creation_input_tokens INTEGER,
    gen_ai_usage_reasoning_output_tokens     INTEGER,
    PRIMARY KEY (session_id, call_id)
);
CREATE INDEX IF NOT EXISTS idx_usage_model    ON usage (gen_ai_response_model);
CREATE INDEX IF NOT EXISTS idx_usage_ts       ON usage (ts);
CREATE INDEX IF NOT EXISTS idx_usage_platform ON usage (platform);
CREATE INDEX IF NOT EXISTS idx_usage_seq      ON usage (seq);
CREATE TABLE IF NOT EXISTS usage_sync (
    session_id      TEXT PRIMARY KEY,
    last_jsonl_size INTEGER NOT NULL
);
"""

# Column order for the usage table, mirrored by the upsert below.
_INSERT_SQL = """
INSERT INTO usage (
    session_id, call_id, seq, ts, platform,
    gen_ai_provider_name, gen_ai_operation_name, gen_ai_response_model,
    gen_ai_usage_input_tokens, gen_ai_usage_output_tokens,
    gen_ai_usage_cache_read_input_tokens, gen_ai_usage_cache_creation_input_tokens,
    gen_ai_usage_reasoning_output_tokens
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (session_id, call_id) DO UPDATE SET
    seq = excluded.seq,
    ts = excluded.ts,
    platform = excluded.platform,
    gen_ai_provider_name = excluded.gen_ai_provider_name,
    gen_ai_operation_name = excluded.gen_ai_operation_name,
    gen_ai_response_model = excluded.gen_ai_response_model,
    gen_ai_usage_input_tokens = excluded.gen_ai_usage_input_tokens,
    gen_ai_usage_output_tokens = excluded.gen_ai_usage_output_tokens,
    gen_ai_usage_cache_read_input_tokens = excluded.gen_ai_usage_cache_read_input_tokens,
    gen_ai_usage_cache_creation_input_tokens = excluded.gen_ai_usage_cache_creation_input_tokens,
    gen_ai_usage_reasoning_output_tokens = excluded.gen_ai_usage_reasoning_output_tokens
"""


class UsageIndex:
    def __init__(self, thirdeye_home: Path) -> None:
        self.thirdeye_home = thirdeye_home
        self.db_path = usage_db_path(thirdeye_home)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        if current < SCHEMA_VERSION:
            # usage.db is purely derived from the sidecars, so an out-of-date
            # schema is dropped and rebuilt rather than migrated.
            conn.executescript("DROP TABLE IF EXISTS usage; DROP TABLE IF EXISTS usage_sync;")
            conn.executescript(SCHEMA_SQL)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
        else:
            conn.executescript(SCHEMA_SQL)
        return conn

    def refresh(self, conn: sqlite3.Connection) -> int:
        """Pull new rows from every session's usage.jsonl into the DB.

        Returns the total number of rows written (inserted or updated) across
        all sessions. Anomalies (shrunk sidecars, malformed lines) are logged
        but do not raise.
        """
        written = 0
        root = sessions_root(self.thirdeye_home)
        if not root.exists():
            return 0
        for platform_dir_ in sorted(root.iterdir()):
            if not platform_dir_.is_dir():
                continue
            for session_dir_ in sorted(platform_dir_.iterdir()):
                if not session_dir_.is_dir():
                    continue
                written += self._refresh_one(conn, session_dir_.name, session_dir_)
        conn.commit()
        return written

    def refresh_session(self, conn: sqlite3.Connection, session_id: str, session_dir_: Path) -> int:
        n = self._refresh_one(conn, session_id, session_dir_)
        conn.commit()
        return n

    def _refresh_one(self, conn: sqlite3.Connection, sid: str, session_dir_: Path) -> int:
        jsonl = usage_jsonl_path(session_dir_)
        if not jsonl.exists():
            return 0
        try:
            current_size = jsonl.stat().st_size
        except FileNotFoundError:
            return 0

        cur = conn.execute(
            "SELECT last_jsonl_size FROM usage_sync WHERE session_id = ?", (sid,)
        ).fetchone()
        last_size = cur[0] if cur else 0

        if current_size == last_size:
            return 0
        if current_size < last_size:
            log_capture_error(
                thirdeye_home=self.thirdeye_home,
                phase="index_sync",
                message=f"sidecar shrank from {last_size} to {current_size}",
                session_id=sid,
                source_path=str(jsonl),
            )
            conn.execute("DELETE FROM usage WHERE session_id = ?", (sid,))
            last_size = 0

        # The sidecar has grown. iter_calls is last-wins over the whole file, so
        # re-read it entirely and upsert every logical call; a later append that
        # corrects a call overwrites the earlier value. Malformed lines are
        # silently dropped by iter_calls, so scan the grown region separately to
        # log them.
        self._log_malformed(sid, jsonl, last_size)

        written = 0
        for row in iter_calls(session_dir_):
            try:
                cursor = conn.execute(
                    _INSERT_SQL,
                    (
                        row.session_id,
                        row.call_id,
                        row.seq,
                        row.ts,
                        row.platform,
                        row.provider_name,
                        row.operation_name,
                        row.response_model,
                        row.input_tokens,
                        row.output_tokens,
                        row.cache_read_input_tokens,
                        row.cache_creation_input_tokens,
                        row.reasoning_output_tokens,
                    ),
                )
                written += cursor.rowcount
            except sqlite3.Error as e:
                log_capture_error(
                    thirdeye_home=self.thirdeye_home,
                    phase="index_sync",
                    error=e,
                    session_id=sid,
                    source_path=str(jsonl),
                )

        conn.execute(
            "INSERT INTO usage_sync (session_id, last_jsonl_size) "
            "VALUES (?, ?) "
            "ON CONFLICT (session_id) DO UPDATE SET last_jsonl_size = excluded.last_jsonl_size",
            (sid, current_size),
        )
        return written

    def _log_malformed(self, sid: str, jsonl: Path, from_offset: int) -> None:
        """Log any lines that are not valid JSON in the grown region.

        Purely observability: the actual upsert reads through iter_calls, which
        drops malformed lines. Scans only from ``from_offset`` to avoid
        re-logging lines seen in a prior refresh.
        """
        try:
            with jsonl.open("rb") as f:
                f.seek(from_offset)
                for raw in f:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        log_capture_error(
                            thirdeye_home=self.thirdeye_home,
                            phase="index_sync",
                            message="malformed jsonl line",
                            session_id=sid,
                            source_path=str(jsonl),
                        )
        except OSError:
            return
