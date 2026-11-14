"""Ingestão incremental: carrega, divide em trechos e sincroniza com o Chroma.

Cada trecho recebe um id determinístico (hash do conteúdo + posição). Assim a
coleção espelha a pasta de documentos: rodar de novo só gera embeddings para o
que mudou, e trechos de arquivos alterados ou removidos são apagados.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from pypdf import PdfReader

from .config import Settings
from .vectorstore import get_vectorstore

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
_BATCH_SIZE = 256


@dataclass(frozen=True)
class IngestReport:
    files: int
    chunks: int
    added: int
    removed: int


def _load_file(path: Path, source: str) -> list[Document]:
    if path.suffix.lower() == ".pdf":
        # Um Document por página, para a citação apontar a página certa.
        return [
            Document(page_content=text, metadata={"source": source, "page": i})
            for i, page in enumerate(PdfReader(path).pages)
            if (text := (page.extract_text() or "").strip())
        ]
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": source})] if text.strip() else []


def load_documents(docs_dir: Path) -> list[Document]:
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Pasta de documentos não encontrada: {docs_dir}")

    paths = sorted(
        p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    documents: list[Document] = []
    for path in paths:
        # Caminho relativo e com "/" para a citação ser igual em Windows e Linux.
        documents.extend(_load_file(path, path.relative_to(docs_dir).as_posix()))
    return documents


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def chunk_id(doc: Document) -> str:
    key = "|".join(
        [
            str(doc.metadata.get("source")),
            str(doc.metadata.get("page", "")),
            str(doc.metadata.get("start_index", "")),
            doc.page_content,
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


@traceable(name="ingest", run_type="chain")
def ingest(
    settings: Settings, *, embeddings: Embeddings | None = None, reset: bool = False
) -> IngestReport:
    documents = load_documents(settings.docs_dir)
    if not documents:
        # Sem isso, uma pasta vazia apagaria a coleção inteira na sincronização.
        raise ValueError(
            f"Nenhum arquivo {sorted(SUPPORTED_SUFFIXES)} encontrado em {settings.docs_dir}."
        )

    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    wanted = {chunk_id(c): c for c in chunks}

    store = get_vectorstore(settings, embeddings, reset=reset)
    existing = set(store.get(include=[])["ids"])
    to_remove = sorted(existing - wanted.keys())
    to_add = [i for i in wanted if i not in existing]

    if to_remove:
        store.delete(ids=to_remove)
    for start in range(0, len(to_add), _BATCH_SIZE):
        batch = to_add[start : start + _BATCH_SIZE]
        store.add_documents([wanted[i] for i in batch], ids=batch)

    return IngestReport(
        files=len({d.metadata["source"] for d in documents}),
        chunks=len(wanted),
        added=len(to_add),
        removed=len(to_remove),
    )
