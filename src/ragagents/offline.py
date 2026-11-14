"""Embeddings determinísticos que não precisam de rede nem de chave de API.

Servem para testes, CI e para experimentar a ingestão sem gastar nada. Não são
semânticos: cada palavra vira uma posição do vetor (bag-of-words com hashing),
então a busca funciona por sobreposição de palavras, não por significado.
Use EMBEDDING_MODEL=offline:hashing para ativá-los.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata

from langchain_core.embeddings import Embeddings

OFFLINE_EMBEDDING_MODEL = "offline:hashing"

_STOPWORDS = {
    "que",
    "para",
    "com",
    "uma",
    "por",
    "dos",
    "das",
    "nos",
    "nas",
    "não",
    "nao",
    "mais",
    "como",
    "qual",
    "quais",
    "ser",
    "são",
    "sao",
    "tem",
    "the",
    "and",
    "for",
}


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = normalized.encode("ascii", "ignore").decode()
    return [t for t in re.findall(r"[a-z0-9]+", ascii_text) if len(t) > 2 and t not in _STOPWORDS]


class HashingEmbeddings(Embeddings):
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vec[int.from_bytes(digest, "big") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
