"""SQLite persistence for AuditEnvelope. One row per audit emission.

Designed for write-light, read-heavy: audit calls insert; analysis queries
during paper-figure generation read.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from aria_audit.core import AuditEnvelope

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class EnvelopeLogger:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))

    def log(self, env: AuditEnvelope, suite_name: str = "", suite_item_id: str = "") -> int:
        def _maybe_json(x: object) -> str | None:
            if x is None:
                return None
            return json.dumps(asdict(x) if hasattr(x, "__dataclass_fields__") else x, default=str)

        cur = self.conn.execute(
            """INSERT INTO audit_envelopes
               (schema_version, request_id, model_name, prompt, response,
                retrieved_chunk_ids, calibration_json, faithfulness_json,
                consistency_json, equity_json, attribution_json, drift_json,
                composite_score, latency_ms_generation, latency_ms_audit,
                peak_vram_gb, timestamp, suite_name, suite_item_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                env.SCHEMA_VERSION,
                env.request_id,
                env.model_name,
                env.prompt,
                env.response,
                json.dumps(env.retrieved_chunk_ids),
                _maybe_json(env.calibration),
                _maybe_json(env.faithfulness),
                _maybe_json(env.consistency),
                _maybe_json(env.equity),
                _maybe_json(env.attribution),
                json.dumps([asdict(d) for d in env.drift]),
                env.composite_score,
                env.latency_ms_generation,
                env.latency_ms_audit,
                env.peak_vram_gb,
                env.timestamp,
                suite_name,
                suite_item_id,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def iter_envelopes(self, suite_name: str | None = None) -> Iterator[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        q = "SELECT * FROM audit_envelopes"
        params: tuple = ()
        if suite_name is not None:
            q += " WHERE suite_name = ?"
            params = (suite_name,)
        return iter(self.conn.execute(q, params))

    def close(self) -> None:
        self.conn.close()
