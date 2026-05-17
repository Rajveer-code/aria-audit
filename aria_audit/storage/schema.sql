-- ARIA-Audit SQLite schema v0.1.0
-- One row per AuditEnvelope emission.

CREATE TABLE IF NOT EXISTS audit_envelopes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_version TEXT NOT NULL,
    request_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    retrieved_chunk_ids TEXT,            -- JSON array
    calibration_json TEXT,
    faithfulness_json TEXT,
    consistency_json TEXT,
    equity_json TEXT,
    attribution_json TEXT,
    drift_json TEXT,
    composite_score REAL,
    latency_ms_generation REAL,
    latency_ms_audit REAL,
    peak_vram_gb REAL,
    timestamp REAL NOT NULL,
    suite_name TEXT,
    suite_item_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_envelope_timestamp ON audit_envelopes(timestamp);
CREATE INDEX IF NOT EXISTS idx_envelope_model ON audit_envelopes(model_name);
CREATE INDEX IF NOT EXISTS idx_envelope_suite ON audit_envelopes(suite_name);
