"""Groq provider — fast inference for Llama, Mixtral, Gemma via Groq Cloud."""

from __future__ import annotations

import os
from typing import AsyncIterator

from agentic_graphs.llm.base import LLM, Message, Chunk

try:
    from groq import AsyncGroq
except ImportError as exc:
    raise ImportError("Install Groq SDK: pip install groq") from exc


class GroqLLM(LLM):
    """Groq Cloud provider (llama-3.3-70b-versatile, mixtral-8x7b-32768, etc.)."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.model = model
        self._client = AsyncGroq(
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            timeout=timeout,
        )

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        kwargs: dict = dict(model=self.model, messages=messages)  # type: ignore
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        result: Message = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        # Capture usage if available
        if response.usage:
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return result

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[Chunk]:
        kwargs: dict = dict(model=self.model, messages=messages, stream=True)  # type: ignore
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield {"content": delta.content, "done": False}
        yield {"content": "", "done": True}
