"""RAG "clássico" em LCEL: recupera trechos, monta o prompt e gera a resposta."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from .prompts import RAG_HUMAN, RAG_SYSTEM
from .vectorstore import format_docs

RAG_PROMPT = ChatPromptTemplate.from_messages([("system", RAG_SYSTEM), ("human", RAG_HUMAN)])


def _prompt_inputs(x: dict[str, Any]) -> dict[str, str]:
    return {"context": format_docs(x["docs"]), "question": x["question"]}


def build_rag_chain(llm: BaseChatModel, retriever: BaseRetriever) -> Runnable:
    """Recebe a pergunta (str) e devolve {"question", "docs", "answer"}."""
    answer = RunnableLambda(_prompt_inputs) | RAG_PROMPT | llm | StrOutputParser()
    chain = RunnableParallel(docs=retriever, question=RunnablePassthrough()) | (
        RunnablePassthrough.assign(answer=answer)
    )
    return chain.with_config(run_name="rag_chain")
