"""Criação dos modelos de chat e de embeddings a partir da configuração.

init_chat_model / init_embeddings aceitam "provedor:modelo", então trocar de
OpenAI para Gemini, Anthropic ou Ollama é só mudar o .env (e instalar o extra).
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings
from .offline import OFFLINE_EMBEDDING_MODEL, HashingEmbeddings

# Variável de chave esperada por provedor (Ollama roda local e não precisa).
_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class MissingCredentialsError(RuntimeError):
    pass


def _require_key(model: str) -> None:
    provider = model.split(":", 1)[0]
    env = _API_KEY_ENV.get(provider)
    if env and not os.getenv(env):
        raise MissingCredentialsError(
            f"O modelo '{model}' precisa da variável {env}. Preencha no .env "
            "(veja o .env.example) ou troque de provedor."
        )


def get_chat_model(settings: Settings) -> BaseChatModel:
    _require_key(settings.chat_model)
    return init_chat_model(settings.chat_model, temperature=0)


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_model == OFFLINE_EMBEDDING_MODEL:
        return HashingEmbeddings()
    _require_key(settings.embedding_model)
    return init_embeddings(settings.embedding_model)
