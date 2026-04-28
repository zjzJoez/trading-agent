-- Phase 2.6.5 follow-up — enforce param_archive_cells.cell_key shape
-- Run via: ta-pg-migrate
-- Idempotent: NOT VALID first to admit existing rows, then VALIDATE.
--
-- Codex review (af480bd) noted that cell_key was a free-form TEXT column,
-- so a typo in `archive.py:CellKey.to_text` could land a malformed row.
-- The contract is "<regime>|<strategy>|<dte_bucket>|<iv_bucket>" — exactly
-- 3 pipe separators, no empty segment.

BEGIN;

-- 1. Drop any leftover constraint from a prior partial migration attempt
--    (defensive — pg_constraint won't error on missing names but ALTER will).
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'param_archive_cells_cell_key_shape'
    ) THEN
        ALTER TABLE param_archive_cells
            DROP CONSTRAINT param_archive_cells_cell_key_shape;
    END IF;
END
$$;

-- 2. Add NOT VALID so existing rows are not re-checked (archive table is
--    new in 003 so this should be a no-op in practice, but keep
--    re-runnable on environments where 003 was filled by hand).
ALTER TABLE param_archive_cells
    ADD CONSTRAINT param_archive_cells_cell_key_shape
    CHECK (cell_key ~ '^[^|]+\|[^|]+\|[^|]+\|[^|]+$')
    NOT VALID;

-- 3. Validate against current rows.  Will raise if anything malformed
--    sneaked in pre-migration; that's the desired behaviour.
ALTER TABLE param_archive_cells
    VALIDATE CONSTRAINT param_archive_cells_cell_key_shape;

INSERT INTO schema_migrations (version, notes)
VALUES ('005_cell_key_shape',
        'Phase 2.6.5 follow-up — CHECK constraint on param_archive_cells.cell_key shape (4 pipe-delimited non-empty segments)')
ON CONFLICT (version) DO NOTHING;

COMMIT;
