"""Os três nós do time: supervisor, pesquisador (researcher) e redator (writer)."""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, MessagesState
from langgraph.types import Command
from pydantic import BaseModel, Field

from .prompts import (
    RESEARCH_NOTES_PREFIX,
    RESEARCHER_SYSTEM,
    SUPERVISOR_ROUTE,
    SUPERVISOR_SYSTEM,
    WRITER_SYSTEM,
)
from .vectorstore import format_docs, source_label, unique

RESEARCHER = "researcher"
WRITER = "writer"
SUPERVISOR = "supervisor"


class TeamState(MessagesState):
    """Estado compartilhado do grafo.

    messages: conversa (pergunta do usuário + contribuições dos agentes).
    sources: fontes encontradas pelo pesquisador, acumuladas.
    steps: decisões do supervisor na pergunta atual (guarda contra laços).
    next / reason: última decisão do supervisor, útil para depurar.
    """

    sources: Annotated[list[str], operator.add]
    steps: int
    next: str
    reason: str


class Route(BaseModel):
    """Decisão do supervisor sobre quem age a seguir."""

    next: Literal["researcher", "writer"] = Field(description="Quem deve agir agora.")
    reason: str = Field(description="Justificativa curta da escolha.")


def to_llm_messages(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Converte o histórico do time para o formato que os modelos esperam.

    As notas do pesquisador entram como mensagem de usuário (contexto para o
    próximo agente) e não como resposta do assistente. Isso mantém a conversa
    terminando em mensagem de usuário, o que alguns provedores exigem.
    """
    converted: list[AnyMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.name == RESEARCHER:
            converted.append(HumanMessage(content=f"{RESEARCH_NOTES_PREFIX}\n{msg.text}"))
        elif isinstance(msg, AIMessage) and msg.name == WRITER:
            converted.append(AIMessage(content=msg.text))
        else:
            converted.append(msg)
    return converted


def make_search_tool(retriever: BaseRetriever) -> BaseTool:
    @tool("search_documents", response_format="content_and_artifact")
    def search_documents(query: str) -> tuple[str, list[Document]]:
        """Busca trechos relevantes nos documentos internos da empresa.

        Use consultas curtas e específicas. Para perguntas com várias partes,
        chame a ferramenta uma vez para cada parte.
        """
        docs = retriever.invoke(query)
        # O texto vai para o modelo; os Documents ficam no artifact para extrair as fontes.
        return format_docs(docs), docs

    return search_documents


def make_supervisor(llm: BaseChatModel, max_steps: int) -> Callable[..., Command]:
    router = llm.with_structured_output(Route)

    def supervisor(state: TeamState, config: RunnableConfig) -> Command:
        steps = state.get("steps", 0)
        last = state["messages"][-1]

        # Decisões determinísticas primeiro: não gastam chamada ao modelo.
        if isinstance(last, AIMessage) and last.name == WRITER:
            return Command(
                goto=END, update={"steps": 0, "next": "FINISH", "reason": "resposta pronta"}
            )
        if steps >= max_steps:
            return Command(
                goto=WRITER,
                update={"steps": steps + 1, "next": WRITER, "reason": "limite de passos atingido"},
            )

        route: Route = router.invoke(
            [
                SystemMessage(content=SUPERVISOR_SYSTEM),
                *to_llm_messages(state["messages"]),
                HumanMessage(content=SUPERVISOR_ROUTE),
            ],
            config,
        )
        return Command(
            goto=route.next,
            update={"steps": steps + 1, "next": route.next, "reason": route.reason},
        )

    return supervisor


def make_researcher(llm: BaseChatModel, retriever: BaseRetriever) -> Callable[..., dict[str, Any]]:
    agent = create_agent(
        llm,
        tools=[make_search_tool(retriever)],
        system_prompt=RESEARCHER_SYSTEM,
        name="researcher_agent",
    )

    def researcher(state: TeamState, config: RunnableConfig) -> dict[str, Any]:
        result = agent.invoke({"messages": to_llm_messages(state["messages"])}, config)
        found = [
            source_label(doc)
            for msg in result["messages"]
            if isinstance(msg, ToolMessage) and msg.artifact
            for doc in msg.artifact
        ]
        notes = result["messages"][-1].text
        return {"messages": [AIMessage(content=notes, name=RESEARCHER)], "sources": unique(found)}

    return researcher


def make_writer(llm: BaseChatModel) -> Callable[..., dict[str, Any]]:
    def writer(state: TeamState, config: RunnableConfig) -> dict[str, Any]:
        response = llm.invoke(
            [SystemMessage(content=WRITER_SYSTEM), *to_llm_messages(state["messages"])], config
        )
        return {"messages": [AIMessage(content=response.text, name=WRITER)]}

    return writer
