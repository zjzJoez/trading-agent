"""Headless cron jobs — run on Dublin EC2 (no OpenD required), sync back to Mac.

Each job has a main() entry point, emits structured JSON to stdout, and writes
its DB rows into trader.ec2.db (not trader.db). `db_sync.py` merges the EC2
DB back to Mac via rsync + `INSERT OR IGNORE` on composite keys.
"""
