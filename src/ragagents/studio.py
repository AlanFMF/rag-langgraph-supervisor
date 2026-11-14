"""Ponto de entrada para o LangGraph Studio (`langgraph dev`), usado no langgraph.json."""

from __future__ import annotations

from .config import Settings
from .graph import build_graph_from_settings

# O servidor do Studio cuida da persistência, então não passamos checkpointer.
graph = build_graph_from_settings(Settings.from_env())
