"""LLM client for any OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter, Ollama, ...)."""
from __future__ import annotations

from collections.abc import Iterator

PRESETS = {
    "Groq (free tier)": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "OpenRouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct"),
    "Ollama (local)": ("http://localhost:11434/v1", "llama3.2"),
}


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.1):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key or "none", base_url=base_url)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, stream=True
        )
        for event in resp:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content

    def complete(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages))
