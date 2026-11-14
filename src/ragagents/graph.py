"""Montagem do grafo: supervisor coordenando pesquisador e redator.

    START -> supervisor -> researcher -> supervisor -> writer -> supervisor -> END

O supervisor decide (com saída estruturada) se falta pesquisa ou se já dá para
responder. Pesquisador e redator sempre devolvem o controle ao supervisor.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .agents import (
    RESEARCHER,
    SUPERVISOR,
    WRITER,
    TeamState,
    make_researcher,
    make_supervisor,
    make_writer,
)
from .config import Settings
from .models import get_chat_model
from .vectorstore import get_retriever, unique


def build_graph(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    *,
    max_steps: int = 6,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    builder = StateGraph(TeamState)
    builder.add_node(
        SUPERVISOR, make_supervisor(llm, max_steps), destinations=(RESEARCHER, WRITER, END)
    )
    builder.add_node(RESEARCHER, make_researcher(llm, retriever))
    builder.add_node(WRITER, make_writer(llm))

    builder.add_edge(START, SUPERVISOR)
    builder.add_edge(RESEARCHER, SUPERVISOR)
    builder.add_edge(WRITER, SUPERVISOR)
    return builder.compile(checkpointer=checkpointer, name="rag_team")


def build_graph_from_settings(
    settings: Settings, *, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    return build_graph(
        get_chat_model(settings),
        get_retriever(settings),
        max_steps=settings.max_steps,
        checkpointer=checkpointer,
    )


def initial_input(question: str) -> dict:
    return {"messages": [HumanMessage(content=question)], "steps": 0}


def final_answer(state: dict) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.name == WRITER:
            return msg.text
    return ""


def final_sources(state: dict) -> list[str]:
    return unique(state.get("sources", []))
