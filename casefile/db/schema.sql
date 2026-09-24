CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    raw_input JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS claim_state_snapshots (
    id SERIAL PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    step_number INT NOT NULL,
    node_name TEXT NOT NULL,
    state_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS trace_events (
    id SERIAL PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    step_number INT NOT NULL,
    node_name TEXT NOT NULL,
    tokens_in INT NOT NULL DEFAULT 0,
    tokens_out INT NOT NULL DEFAULT 0,
    dollars NUMERIC NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pending_approvals (
    claim_id TEXT PRIMARY KEY REFERENCES claims(claim_id),
    recommendation_json JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    decided_by TEXT,
    decided_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS claims_system_writeback (
    claim_id TEXT PRIMARY KEY REFERENCES claims(claim_id),
    payout_amount NUMERIC NOT NULL,
    written_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
