"""Acesso ao Chroma persistido em disco e formatação dos trechos com citação."""

from __future__ import annotations

import re
from collections.abc import Iterable

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from .config import Settings
from .models import get_embeddings


class EmbeddingMismatchError(RuntimeError):
    """A coleção foi criada com outro modelo de embeddings (dimensões incompatíveis)."""


def get_vectorstore(
    settings: Settings, embeddings: Embeddings | None = None, *, reset: bool = False
) -> Chroma:
    """Abre (ou cria) a coleção. Com reset=True, apaga e recria antes de checar o modelo."""
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    store = Chroma(
        client=client,
        collection_name=settings.collection,
        embedding_function=embeddings or get_embeddings(settings),
        # Guardamos o modelo usado para detectar troca de embeddings sem reindexar.
        collection_metadata={"embedding_model": settings.embedding_model},
    )
    if reset:
        store.reset_collection()
    stored = (store._collection.metadata or {}).get("embedding_model")
    if stored and stored != settings.embedding_model:
        raise EmbeddingMismatchError(
            f"A coleção '{settings.collection}' foi indexada com '{stored}', mas o "
            f"EMBEDDING_MODEL atual é '{settings.embedding_model}'. "
            "Rode `ragagents ingest --reset` para reindexar."
        )
    return store


def get_retriever(settings: Settings, embeddings: Embeddings | None = None) -> VectorStoreRetriever:
    return get_vectorstore(settings, embeddings).as_retriever(search_kwargs={"k": settings.top_k})


def source_label(doc: Document) -> str:
    """Rótulo legível da origem de um trecho, ex.: 'beneficios.pdf, p. 2'."""
    source = doc.metadata.get("source", "desconhecido")
    page = doc.metadata.get("page")
    return f"{source}, p. {int(page) + 1}" if page is not None else str(source)


def format_docs(docs: Iterable[Document]) -> str:
    blocks = [
        f"[{i}] (fonte: {source_label(doc)})\n{doc.page_content.strip()}"
        for i, doc in enumerate(docs, start=1)
    ]
    return "\n\n".join(blocks) if blocks else "Nenhum trecho relevante encontrado."


def unique(items: Iterable[str]) -> list[str]:
    """Remove duplicatas mantendo a ordem."""
    return list(dict.fromkeys(items))


def cited_by_number(answer: str, docs: list[Document]) -> list[str]:
    """Fontes citadas como [n] na resposta do RAG simples (n = posição do trecho)."""
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return unique(source_label(d) for i, d in enumerate(docs, start=1) if i in numbers)


def cited_by_name(answer: str, sources: Iterable[str]) -> list[str]:
    """Fontes cujo nome de arquivo aparece no texto da resposta do time."""
    return [s for s in unique(sources) if s.split(",")[0] in answer]
