"""Stub subgraph nodes for Phase 2.1.

Each node is a small Python function that takes the state dict, does
some work, and returns a partial state update. LangGraph merges the
update into the running state.

Phase 2.1 implements only stubs that emit agent_events + ntfy heartbeats.
The real logic for regime / scan / research / risk / executor / learning
lands in Phase 2.2 onwards.
"""
