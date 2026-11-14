"""Configuração lida de variáveis de ambiente (e de um arquivo .env, se existir)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} precisa ser um número inteiro (recebi {raw!r}).") from exc
    if value <= 0:
        raise ValueError(f"{name} precisa ser maior que zero (recebi {value}).")
    return value


@dataclass(frozen=True)
class Settings:
    # Modelos no formato "provedor:modelo", resolvidos por init_chat_model / init_embeddings.
    chat_model: str = "openai:gpt-4.1-mini"
    embedding_model: str = "openai:text-embedding-3-small"

    docs_dir: Path = Path("data/docs")
    chroma_dir: Path = Path(".chroma")
    collection: str = "docs"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    # Teto de decisões do supervisor por pergunta (evita laço infinito entre agentes).
    max_steps: int = 6

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP precisa ser menor que CHUNK_SIZE.")
        if ":" not in self.chat_model or ":" not in self.embedding_model:
            raise ValueError(
                "CHAT_MODEL e EMBEDDING_MODEL usam o formato 'provedor:modelo', "
                "por exemplo 'openai:gpt-4.1-mini'."
            )

    @classmethod
    def from_env(cls, *, dotenv: bool = True) -> Settings:
        if dotenv:
            load_dotenv(override=False)
        return cls(
            chat_model=os.getenv("CHAT_MODEL", cls.chat_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            docs_dir=Path(os.getenv("DOCS_DIR", str(cls.docs_dir))),
            chroma_dir=Path(os.getenv("CHROMA_DIR", str(cls.chroma_dir))),
            collection=os.getenv("CHROMA_COLLECTION", cls.collection),
            chunk_size=_positive_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_positive_int("CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_positive_int("TOP_K", cls.top_k),
            max_steps=_positive_int("MAX_STEPS", cls.max_steps),
        )
