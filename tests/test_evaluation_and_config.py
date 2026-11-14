from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ragagents.config import Settings
from ragagents.evaluation import Grade, ensure_dataset, load_examples, make_correctness, source_hit

from .fakes import ScriptedChatModel

DATA = Path(__file__).resolve().parents[1] / "data"
QA_PATH = DATA / "eval" / "qa.json"


def test_eval_examples_reference_existing_documents() -> None:
    docs = {p.name for p in (DATA / "docs").iterdir()}
    examples = load_examples(QA_PATH)
    assert len(examples) >= 8
    for ex in examples:
        assert ex["inputs"]["question"] and ex["outputs"]["answer"]
        source = ex["outputs"]["source"]
        assert source is None or source in docs


def test_source_hit() -> None:
    ref = {"source": "guia-de-beneficios.pdf"}
    assert source_hit({"sources": ["guia-de-beneficios.pdf, p. 2"]}, ref)["score"] is True
    assert source_hit({"sources": ["politica-reembolso.md"]}, ref)["score"] is False
    assert source_hit({"sources": []}, {"source": None})["score"] is None


def test_correctness_uses_judge_verdict() -> None:
    judge = ScriptedChatModel([Grade(reasoning="mesmo valor", correct=True)])
    result = make_correctness(judge)(
        {"question": "auxílio academia?"}, {"answer": "R$ 80"}, {"answer": "R$ 80,00 por mês"}
    )
    assert result == {"key": "correctness", "score": True, "comment": "mesmo valor"}
    assert "R$ 80,00 por mês" in judge.calls[0]


def test_ensure_dataset_creates_only_once() -> None:
    client = MagicMock()
    client.has_dataset.return_value = False
    client.create_dataset.return_value = SimpleNamespace(id="ds-1")
    ensure_dataset(client, "x", QA_PATH)
    client.create_examples.assert_called_once()
    assert client.create_examples.call_args.kwargs["dataset_id"] == "ds-1"

    client.reset_mock()
    client.has_dataset.return_value = True
    ensure_dataset(client, "x", QA_PATH)
    client.create_dataset.assert_not_called()


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_MODEL", "google_genai:gemini-2.5-flash")
    monkeypatch.setenv("TOP_K", "6")
    s = Settings.from_env(dotenv=False)
    assert s.chat_model == "google_genai:gemini-2.5-flash" and s.top_k == 6

    monkeypatch.setenv("TOP_K", "zero")
    with pytest.raises(ValueError, match="TOP_K"):
        Settings.from_env(dotenv=False)


def test_settings_validation() -> None:
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError, match="provedor:modelo"):
        Settings(chat_model="gpt-4.1-mini")


def test_rate_limiter_is_shared_between_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragagents.models import get_chat_model

    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")
    limited = Settings(chat_model="openai:gpt-4.1-mini", requests_per_minute=12)
    a, b = get_chat_model(limited), get_chat_model(limited)
    assert a.rate_limiter is not None and a.rate_limiter is b.rate_limiter
    assert get_chat_model(Settings(chat_model="openai:gpt-4.1-mini")).rate_limiter is None


def test_requests_per_minute_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_REQUESTS_PER_MINUTE", "12")
    assert Settings.from_env(dotenv=False).requests_per_minute == 12
    monkeypatch.setenv("LLM_REQUESTS_PER_MINUTE", "0")
    assert Settings.from_env(dotenv=False).requests_per_minute == 0
