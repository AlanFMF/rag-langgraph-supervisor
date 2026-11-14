from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ragagents.agents import Route, to_llm_messages
from ragagents.graph import build_graph, final_answer, final_sources, initial_input
from ragagents.rag_chain import build_rag_chain

from .fakes import ScriptedChatModel


def _search_call(query: str, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "search_documents", "args": {"query": query}, "id": call_id}],
    )


def test_rag_chain_puts_retrieved_context_in_prompt(retriever) -> None:
    llm = ScriptedChatModel([AIMessage(content="R$ 320,00 por noite [1].")])
    out = build_rag_chain(llm, retriever).invoke("limite de hospedagem em capitais")

    assert out["answer"] == "R$ 320,00 por noite [1]."
    assert out["docs"][0].metadata["source"] == "politica-reembolso.md"
    prompt = llm.calls[0][-1].content
    assert "(fonte: politica-reembolso.md)" in prompt
    assert "R$ 320,00 por noite" in prompt


def test_team_researches_then_writes(retriever) -> None:
    llm = ScriptedChatModel(
        [
            Route(next="researcher", reason="precisa dos documentos"),
            _search_call("limite hospedagem capitais"),
            AIMessage(
                content="Hospedagem em capitais: R$ 320,00/noite (fonte: politica-reembolso.md)"
            ),
            Route(next="writer", reason="notas suficientes"),
            AIMessage(content="O limite é R$ 320,00 por noite (fonte: politica-reembolso.md)."),
        ]
    )
    graph = build_graph(llm, retriever)
    state = graph.invoke(initial_input("Qual o limite de hotel em capitais?"))

    assert final_answer(state) == "O limite é R$ 320,00 por noite (fonte: politica-reembolso.md)."
    assert final_sources(state)[0] == "politica-reembolso.md"
    assert [m.name for m in state["messages"] if isinstance(m, AIMessage)] == [
        "researcher",
        "writer",
    ]
    # Só as notas do pesquisador entram no estado do time, não as mensagens internas do agente.
    assert not any(isinstance(m, ToolMessage) for m in state["messages"])
    assert llm.remaining == 0
    assert state["steps"] == 0  # zerado ao terminar, pronto para a próxima pergunta

    writer_prompt = llm.calls[-1]
    assert isinstance(writer_prompt[-1], HumanMessage)
    assert "R$ 320,00/noite" in writer_prompt[-1].content


def test_supervisor_can_send_straight_to_writer(retriever) -> None:
    llm = ScriptedChatModel(
        [
            Route(next="writer", reason="cumprimento, não precisa de pesquisa"),
            AIMessage(content="Olá! Posso ajudar com dúvidas sobre as políticas internas."),
        ]
    )
    state = build_graph(llm, retriever).invoke(initial_input("oi, tudo bem?"))
    assert final_answer(state).startswith("Olá!")
    assert final_sources(state) == []


def test_step_limit_forces_an_answer(retriever) -> None:
    # Um supervisor "teimoso" que sempre pede mais pesquisa não trava o grafo.
    max_steps = 2
    script: list = []
    for i in range(max_steps):
        script += [
            Route(next="researcher", reason="mais pesquisa"),
            _search_call("reembolso", f"call_{i}"),
            AIMessage(content="Notas parciais (fonte: politica-reembolso.md)"),
        ]
    script.append(AIMessage(content="Resposta com o que foi encontrado."))

    llm = ScriptedChatModel(script)
    state = build_graph(llm, retriever, max_steps=max_steps).invoke(initial_input("reembolso?"))
    assert final_answer(state) == "Resposta com o que foi encontrado."
    assert state["reason"] == "resposta pronta"
    assert llm.remaining == 0


def test_research_notes_are_passed_as_user_context() -> None:
    converted = to_llm_messages(
        [
            HumanMessage(content="pergunta"),
            AIMessage(content="fato X", name="researcher"),
            AIMessage(content="resposta", name="writer"),
        ]
    )
    assert isinstance(converted[1], HumanMessage)
    assert converted[1].content.endswith("fato X")
    assert isinstance(converted[2], AIMessage) and converted[2].name is None
