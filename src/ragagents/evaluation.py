"""Avaliação offline no LangSmith: compara o RAG simples com o time de agentes.

Cria (uma vez) um dataset de perguntas com resposta e fonte esperadas, roda o
alvo escolhido sobre ele e registra um experimento com duas métricas:

- source_hit: o documento esperado apareceu entre as fontes recuperadas?
- correctness: um LLM-juiz compara a resposta com a resposta de referência.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langsmith import Client
from pydantic import BaseModel, Field

from .config import Settings
from .graph import build_graph, final_answer, final_sources, initial_input
from .models import get_chat_model
from .rag_chain import build_rag_chain
from .tracing import run_config
from .vectorstore import get_retriever, source_label, unique

DATASET_NAME = "ragagents-aurora-qa"
DATASET_PATH = Path("data/eval/qa.json")

JUDGE_PROMPT = """Você avalia respostas de um assistente sobre documentos internos.
Compare a RESPOSTA com a REFERÊNCIA. A resposta está correta se contém os mesmos
fatos essenciais da referência (valores, prazos, canais), sem contradizê-la.
Detalhes extras corretos não tornam a resposta errada. Se a referência diz que a
informação não consta nos documentos, a resposta só está correta se também disser
que não encontrou a informação.

PERGUNTA: {question}
REFERÊNCIA: {reference}
RESPOSTA: {answer}"""


class Grade(BaseModel):
    reasoning: str = Field(description="Explicação curta do veredito.")
    correct: bool = Field(description="True se a resposta está correta.")


def load_examples(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "inputs": {"question": row["question"]},
            "outputs": {"answer": row["answer"], "source": row.get("source")},
        }
        for row in rows
    ]


def ensure_dataset(client: Client, name: str = DATASET_NAME, path: Path = DATASET_PATH) -> str:
    if client.has_dataset(dataset_name=name):
        return name
    dataset = client.create_dataset(
        dataset_name=name, description="Perguntas sobre os documentos fictícios da Aurora Entregas."
    )
    client.create_examples(dataset_id=dataset.id, examples=load_examples(path))
    return name


def source_hit(outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs.get("source")
    if not expected:
        return {"key": "source_hit", "score": None, "comment": "pergunta sem fonte esperada"}
    found = outputs.get("sources") or []
    return {"key": "source_hit", "score": any(s.startswith(expected) for s in found)}


def make_correctness(judge: BaseChatModel) -> Callable[..., dict]:
    grader = judge.with_structured_output(Grade)

    def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        grade: Grade = grader.invoke(
            JUDGE_PROMPT.format(
                question=inputs["question"],
                reference=reference_outputs["answer"],
                answer=outputs.get("answer", ""),
            )
        )
        return {"key": "correctness", "score": grade.correct, "comment": grade.reasoning}

    return correctness


def make_target(settings: Settings, kind: str) -> Callable[[dict], dict]:
    llm = get_chat_model(settings)
    retriever = get_retriever(settings)

    if kind == "chain":
        chain = build_rag_chain(llm, retriever)

        def run_chain(inputs: dict) -> dict:
            out = chain.invoke(inputs["question"], run_config(settings, "eval-chain"))
            return {
                "answer": out["answer"],
                "sources": unique(source_label(d) for d in out["docs"]),
            }

        return run_chain

    if kind == "team":
        graph = build_graph(llm, retriever, max_steps=settings.max_steps)

        def run_team(inputs: dict) -> dict:
            state = graph.invoke(
                initial_input(inputs["question"]), run_config(settings, "eval-team")
            )
            return {"answer": final_answer(state), "sources": final_sources(state)}

        return run_team

    raise ValueError(f"Alvo desconhecido: {kind!r} (use 'chain' ou 'team').")


def run_evaluation(settings: Settings, kind: str, *, client: Client | None = None) -> str:
    client = client or Client()
    dataset = ensure_dataset(client)
    results = client.evaluate(
        make_target(settings, kind),
        data=dataset,
        evaluators=[source_hit, make_correctness(get_chat_model(settings))],
        experiment_prefix=f"{kind}-{settings.chat_model.replace(':', '-')}",
        metadata={"target": kind, "chat_model": settings.chat_model, "top_k": settings.top_k},
        max_concurrency=2,
    )
    return results.experiment_name
