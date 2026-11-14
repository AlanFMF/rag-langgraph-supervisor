from __future__ import annotations

import pytest

from ragagents.config import Settings
from ragagents.ingest import ingest, load_documents
from ragagents.vectorstore import EmbeddingMismatchError, get_vectorstore, source_label


def test_loads_markdown_and_pdf_with_relative_sources(settings: Settings) -> None:
    docs = load_documents(settings.docs_dir)
    sources = {d.metadata["source"] for d in docs}
    assert sources == {
        "guia-de-beneficios.pdf",
        "politica-home-office.md",
        "politica-reembolso.md",
        "seguranca-da-informacao.md",
    }
    pdf_pages = [d for d in docs if d.metadata["source"].endswith(".pdf")]
    assert len(pdf_pages) == 2
    assert source_label(pdf_pages[1]) == "guia-de-beneficios.pdf, p. 2"


def test_ingest_is_idempotent(settings: Settings, embeddings) -> None:
    first = ingest(settings, embeddings=embeddings)
    assert first.files == 4
    assert first.added == first.chunks > 4
    second = ingest(settings, embeddings=embeddings)
    assert (second.added, second.removed, second.chunks) == (0, 0, first.chunks)


def test_ingest_syncs_changed_and_deleted_files(settings: Settings, embeddings) -> None:
    first = ingest(settings, embeddings=embeddings)

    path = settings.docs_dir / "politica-home-office.md"
    path.write_text(
        path.read_text(encoding="utf-8") + "\n\nNova regra: sextas são livres.\n", encoding="utf-8"
    )
    (settings.docs_dir / "seguranca-da-informacao.md").unlink()

    report = ingest(settings, embeddings=embeddings)
    assert report.files == 3
    assert report.added >= 1 and report.removed >= 1

    store = get_vectorstore(settings, embeddings)
    stored_sources = {m["source"] for m in store.get(include=["metadatas"])["metadatas"]}
    assert "seguranca-da-informacao.md" not in stored_sources
    assert len(store.get(include=[])["ids"]) == report.chunks
    assert report.chunks != first.chunks


def test_empty_folder_does_not_wipe_collection(settings: Settings, embeddings, tmp_path) -> None:
    ingest(settings, embeddings=embeddings)
    empty = tmp_path / "vazia"
    empty.mkdir()
    with pytest.raises(ValueError, match="Nenhum arquivo"):
        ingest(Settings(**{**settings.__dict__, "docs_dir": empty}), embeddings=embeddings)
    assert get_vectorstore(settings, embeddings).get(include=[])["ids"]


def test_embedding_model_change_is_detected(settings: Settings, embeddings) -> None:
    ingest(settings, embeddings=embeddings)
    other = Settings(**{**settings.__dict__, "embedding_model": "openai:text-embedding-3-small"})
    with pytest.raises(EmbeddingMismatchError, match="--reset"):
        get_vectorstore(other, embeddings)


def test_reset_reindexes_with_new_embedding_model(settings: Settings, embeddings) -> None:
    ingest(settings, embeddings=embeddings)
    other = Settings(**{**settings.__dict__, "embedding_model": "outro:modelo"})
    report = ingest(other, embeddings=embeddings, reset=True)
    assert report.added == report.chunks
    store = get_vectorstore(other, embeddings)
    assert store._collection.metadata["embedding_model"] == "outro:modelo"


def test_missing_api_key_has_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragagents.models import MissingCredentialsError, get_chat_model

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingCredentialsError, match="OPENAI_API_KEY"):
        get_chat_model(Settings(chat_model="openai:gpt-4.1-mini"))


def test_retrieval_finds_the_right_document(retriever) -> None:
    docs = retriever.invoke("limite de hospedagem em capitais por noite")
    assert docs[0].metadata["source"] == "politica-reembolso.md"
    assert "320" in docs[0].page_content

    docs = retriever.invoke("auxílio academia")
    assert docs[0].metadata["source"] == "guia-de-beneficios.pdf"
