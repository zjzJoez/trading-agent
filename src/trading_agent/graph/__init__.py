"""LangGraph orchestrator — typed state, fixed subgraph topology, Postgres checkpointer.

§5 of the Phase 2 v3 plan. Subgraphs:
    premarket_scan_graph        08:30 (digest) + 10:15 + 13:30 ET (execute)
    candidate_entry_graph       per Scout candidate
    intraday_monitor_graph      every 15 min during US hours
    eod_review_graph            ~21:30 UTC
    weekly_learning_graph       Friday post-close
    healthcheck_graph           hourly
"""
