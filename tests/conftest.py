from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ragagents.config import Settings
from ragagents.ingest import ingest
from ragagents.offline import HashingEmbeddings
from ragagents.vectorstore import get_retriever

SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Testes nunca enviam traces, mesmo que o .env local tenha LangSmith ligado.
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    docs = tmp_path / "docs"
    shutil.copytree(SAMPLE_DOCS, docs)
    return Settings(
        embedding_model="offline:hashing",
        docs_dir=docs,
        chroma_dir=tmp_path / "chroma",
        collection="test",
        top_k=3,
    )


@pytest.fixture
def embeddings() -> HashingEmbeddings:
    return HashingEmbeddings()


@pytest.fixture
def retriever(settings: Settings, embeddings: HashingEmbeddings):
    ingest(settings, embeddings=embeddings)
    return get_retriever(settings, embeddings)
