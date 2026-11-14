"""Linha de comando: ingest, ask, team, graph e eval."""

from __future__ import annotations

import argparse
import sys

from .config import Settings
from .tracing import run_config, tracing_status


def _cmd_ingest(settings: Settings, args: argparse.Namespace) -> None:
    from .ingest import ingest

    report = ingest(settings, reset=args.reset)
    print(
        f"{report.files} arquivo(s), {report.chunks} trecho(s) na coleção "
        f"(+{report.added} novos, -{report.removed} removidos)."
    )


def _cmd_ask(settings: Settings, args: argparse.Namespace) -> None:
    from .models import get_chat_model
    from .rag_chain import build_rag_chain
    from .vectorstore import cited_by_number, get_retriever, source_label, unique

    chain = build_rag_chain(get_chat_model(settings), get_retriever(settings))
    out = chain.invoke(args.question, run_config(settings, "ask"))
    print(out["answer"])
    _print_sources(
        cited_by_number(out["answer"], out["docs"]),
        unique(source_label(d) for d in out["docs"]),
    )


def _cmd_team(settings: Settings, args: argparse.Namespace) -> None:
    from .graph import build_graph_from_settings, final_answer, final_sources, initial_input

    graph = build_graph_from_settings(settings)
    state: dict = {}
    for mode, chunk in graph.stream(
        initial_input(args.question),
        run_config(settings, "team"),
        stream_mode=["updates", "values"],
    ):
        if mode == "values":
            state = chunk
        elif args.verbose:
            for node, update in chunk.items():
                if node == "supervisor" and update:
                    print(f"[supervisor] -> {update.get('next')} ({update.get('reason')})")
                elif node != "supervisor":
                    print(f"[{node}] concluiu")
    from .vectorstore import cited_by_name

    answer = final_answer(state)
    print()
    print(answer)
    _print_sources(cited_by_name(answer, final_sources(state)), final_sources(state))


def _cmd_graph(settings: Settings, args: argparse.Namespace) -> None:
    from langchain_core.documents import Document
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.retrievers import BaseRetriever

    from .graph import build_graph

    # Só a estrutura importa aqui: modelo e retriever de mentira, sem chave de API.
    class _NoopModel(FakeListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    class _EmptyRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager) -> list[Document]:
            return []

    graph = build_graph(_NoopModel(responses=["-"]), _EmptyRetriever())
    print(graph.get_graph().draw_mermaid())


def _cmd_eval(settings: Settings, args: argparse.Namespace) -> None:
    from .evaluation import run_evaluation
    from .tracing import tracing_enabled

    if not tracing_enabled():
        sys.exit(
            "A avaliação grava os resultados no LangSmith: configure LANGSMITH_API_KEY "
            "e LANGSMITH_TRACING=true no .env."
        )
    name = run_evaluation(settings, args.target)
    print(f"Experimento '{name}' registrado no LangSmith (aba Datasets & Experiments).")


def _print_sources(cited: list[str], consulted: list[str]) -> None:
    """Mostra as fontes citadas na resposta; sem citação, mostra o que foi consultado."""
    title, sources = ("Fontes citadas", cited) if cited else ("Trechos consultados", consulted)
    if sources:
        print(f"\n{title}:")
        for s in sources:
            print(f"  - {s}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragagents",
        description="RAG sobre documentos (LangChain + Chroma) e time de agentes no LangGraph.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="indexa a pasta de documentos no Chroma")
    p.add_argument("--reset", action="store_true", help="apaga a coleção e reindexa do zero")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("ask", help="pergunta usando o RAG simples (uma busca + uma resposta)")
    p.add_argument("question")
    p.set_defaults(func=_cmd_ask)

    p = sub.add_parser("team", help="pergunta usando o time supervisor + pesquisador + redator")
    p.add_argument("question")
    p.add_argument("-v", "--verbose", action="store_true", help="mostra as decisões do supervisor")
    p.set_defaults(func=_cmd_team)

    p = sub.add_parser("graph", help="imprime o diagrama do grafo em Mermaid")
    p.set_defaults(func=_cmd_graph)

    p = sub.add_parser("eval", help="roda a avaliação no LangSmith")
    p.add_argument("--target", choices=["chain", "team"], default="team")
    p.set_defaults(func=_cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> None:
    from .models import MissingCredentialsError
    from .vectorstore import EmbeddingMismatchError

    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.command in {"ask", "team", "ingest"}:
            print(tracing_status(), file=sys.stderr)
        args.func(settings, args)
    except (MissingCredentialsError, EmbeddingMismatchError, FileNotFoundError, ValueError) as exc:
        sys.exit(f"Erro: {exc}")


if __name__ == "__main__":
    main()
