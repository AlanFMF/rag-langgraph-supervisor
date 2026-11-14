"""Integração com o LangSmith.

O rastreamento em si é automático: com LANGSMITH_TRACING=true e LANGSMITH_API_KEY
definidos, toda execução de chain, agente, ferramenta e LLM vira um trace no
projeto LANGSMITH_PROJECT. Aqui só padronizamos nomes, tags e metadados para os
traces ficarem fáceis de filtrar e comparar.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.runnables import RunnableConfig

from .config import Settings


def tracing_enabled() -> bool:
    flag = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    return flag.strip().lower() == "true" and bool(key)


def tracing_status() -> str:
    if tracing_enabled():
        project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "default"
        return f"LangSmith: rastreamento ativo no projeto '{project}'."
    return "LangSmith: rastreamento desligado (defina LANGSMITH_TRACING=true e LANGSMITH_API_KEY)."


def run_config(settings: Settings, kind: str, **metadata: Any) -> RunnableConfig:
    return {
        "run_name": f"ragagents:{kind}",
        "tags": ["ragagents", kind],
        "metadata": {
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "top_k": settings.top_k,
            "chunk_size": settings.chunk_size,
            **metadata,
        },
        "recursion_limit": 50,
    }
