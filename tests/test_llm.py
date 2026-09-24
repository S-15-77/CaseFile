import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from casefile.llm import call_structured, LLMCallError


class Dummy(BaseModel):
    value: str


def _fake_response(content: str, tokens_in=10, tokens_out=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out),
    )


def test_call_structured_success_first_try():
    good = _fake_response(json.dumps({"value": "hello"}))
    with patch("casefile.llm.get_client") as get_client:
        get_client.return_value.chat.completions.create.return_value = good
        parsed, tin, tout = call_structured("sys", "user", Dummy)
    assert parsed.value == "hello"
    assert tin == 10 and tout == 5


def test_call_structured_retries_once_then_succeeds():
    bad = _fake_response("not json")
    good = _fake_response(json.dumps({"value": "recovered"}))
    with patch("casefile.llm.get_client") as get_client:
        get_client.return_value.chat.completions.create.side_effect = [bad, good]
        parsed, _, _ = call_structured("sys", "user", Dummy)
    assert parsed.value == "recovered"


def test_call_structured_raises_after_two_failures():
    bad = _fake_response("still not json")
    with patch("casefile.llm.get_client") as get_client:
        get_client.return_value.chat.completions.create.side_effect = [bad, bad]
        with pytest.raises(LLMCallError):
            call_structured("sys", "user", Dummy)
