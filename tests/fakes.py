"""Modelo de chat roteirizado para testar o grafo sem chamar nenhuma API.

Cada chamada consome a próxima resposta da fila: AIMessage para chamadas normais
(inclusive com tool_calls) e instâncias Pydantic para with_structured_output.
Também guarda as mensagens recebidas, para os testes inspecionarem os prompts.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, PrivateAttr


class ScriptedChatModel(BaseChatModel):
    _script: deque = PrivateAttr(default_factory=deque)
    _calls: list = PrivateAttr(default_factory=list)

    def __init__(self, script: list[Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._script = deque(script)
        self._calls = []

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    @property
    def remaining(self) -> int:
        return len(self._script)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, expected: type) -> Any:
        if not self._script:
            raise AssertionError(
                "O roteiro acabou: o grafo chamou o modelo mais vezes que o esperado."
            )
        item = self._script.popleft()
        if not isinstance(item, expected):
            raise AssertionError(f"Esperava {expected.__name__} no roteiro, veio {item!r}")
        return item

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any):
        self._calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self._next(AIMessage))])

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any):
        def respond(messages: Any) -> BaseModel:
            self._calls.append(messages if isinstance(messages, str) else list(messages))
            return self._next(schema)

        return RunnableLambda(respond)
