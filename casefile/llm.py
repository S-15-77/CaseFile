import json
import os

from groq import Groq
from pydantic import BaseModel, ValidationError

MODEL = "llama-3.3-70b-versatile"

_client = None


class LLMCallError(Exception):
    pass


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def call_structured(
    system_prompt: str, user_prompt: str, schema: type[BaseModel]
) -> tuple[BaseModel, int, int]:
    client = get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None
    for attempt in range(2):
        if attempt == 1:
            messages.append({
                "role": "user",
                "content": (
                    "Your last response was not valid JSON matching this "
                    f"schema: {schema.model_json_schema()}. "
                    "Return ONLY valid JSON, nothing else."
                ),
            })
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
            parsed = schema.model_validate(data)
            return parsed, response.usage.prompt_tokens, response.usage.completion_tokens
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": content})
    raise LLMCallError(f"failed to get valid {schema.__name__} after 2 attempts: {last_error}")
