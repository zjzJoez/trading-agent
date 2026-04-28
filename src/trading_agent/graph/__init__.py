"""LangGraph orchestrator — typed state, fixed subgraph topology, Postgres checkpointer.

§5 of the Phase 2 v3 plan. Subgraphs:
    premarket_scan_graph        12:30 UTC
    candidate_entry_graph       per Scout candidate
    intraday_monitor_graph      every 15 min during US hours
    eod_review_graph            ~21:30 UTC
    weekly_learning_graph       Friday post-close
    healthcheck_graph           hourly
"""
