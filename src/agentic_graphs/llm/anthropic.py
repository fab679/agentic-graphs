"""Anthropic provider — Claude models via the Anthropic SDK."""

from __future__ import annotations

import os
from typing import AsyncIterator

from agentic_graphs.llm.base import LLM, Message, Chunk

try:
    import anthropic as _anthropic
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Install the Anthropic SDK: pip install anthropic"
    ) from exc


def _openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-style tool schemas to Anthropic format."""
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def _openai_messages_to_anthropic(
    messages: list[Message],
) -> tuple[str, list[dict]]:
    """Split system prompt out and convert message list to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    """
    system = ""
    converted: list[dict] = []
    pending_tool_results: list[dict] = []

    for m in messages:
        role = m.get("role", "user")

        if role == "system":
            system = m.get("content") or ""
            continue

        if role == "tool":
            # Accumulate tool results to bundle with the next user turn
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": m.get("content") or "",
            })
            continue

        if pending_tool_results:
            converted.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

        if role == "assistant":
            content: list = []
            text = m.get("content") or ""
            if text:
                content.append({"type": "text", "text": text})
            for tc in m.get("tool_calls") or []:
                import json
                try:
                    inp = json.loads(tc["function"]["arguments"])
                except Exception:
                    inp = {}
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": inp,
                })
            converted.append({"role": "assistant", "content": content or text})
        else:
            converted.append({"role": "user", "content": m.get("content") or ""})

    if pending_tool_results:
        converted.append({"role": "user", "content": pending_tool_results})

    return system, converted


class AnthropicLLM(LLM):
    """Claude provider (claude-sonnet-4-20250514, claude-opus-4-20250514, etc.)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = _anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=timeout,
        )

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        import json
        system, converted = _openai_messages_to_anthropic(messages)
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=converted,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _openai_tools_to_anthropic(tools)

        response = await self._client.messages.create(**kwargs)

        content_text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        result: Message = {"role": "assistant", "content": content_text}
        if tool_calls:
            result["tool_calls"] = tool_calls

        # Capture usage if available
        if hasattr(response, 'usage') and response.usage:
            result["usage"] = {
                "prompt_tokens": getattr(response.usage, 'input_tokens', 0),
                "completion_tokens": getattr(response.usage, 'output_tokens', 0),
                "total_tokens": getattr(response.usage, 'input_tokens', 0) + getattr(response.usage, 'output_tokens', 0),
            }

        return result

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[Chunk]:
        import json
        system, converted = _openai_messages_to_anthropic(messages)
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=converted,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _openai_tools_to_anthropic(tools)

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield {"content": text, "done": False}

        final = await stream.get_final_message()
        tool_calls = []
        for block in final.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })
        yield {
            "content": "",
            "tool_call": tool_calls[0] if tool_calls else None,
            "done": True,
        }
