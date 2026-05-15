"""LiteRT-LM provider — on-device inference (Gemma 4 via LiteRT).

Wraps the existing litert_agent.py pattern into the standard LLM interface.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

from agentic_graphs.llm.base import LLM, Message, Chunk

try:
    import litert_lm
    from litert_lm import LlmInference, LlmInferenceOptions
except ImportError as exc:
    raise ImportError(
        "Install LiteRT-LM: pip install 'litert-lm>=0.11.0'"
    ) from exc


class LiteRTLLM(LLM):
    """On-device LiteRT-LM provider (Gemma 4 and compatible models).

    Usage::
        llm = LiteRTLLM(model_path="/path/to/model.litertlm")
    """

    def __init__(
        self,
        model_path: str | None = None,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ):
        self.model = "litert-lm"
        self._timeout = timeout  # local inference; unused but keeps interface consistent
        path = model_path or os.environ.get("MODEL_PATH", "")
        if not path:
            raise ValueError(
                "Provide model_path= or set MODEL_PATH env var "
                "(e.g. /path/to/gemma-4-E2B-it.litertlm/model.litertlm)"
            )
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        opts = LlmInferenceOptions(max_tokens=max_tokens)
        self._inference = LlmInference(model_path=path, options=opts)

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        import asyncio

        # Build a simple text prompt (LiteRT tool calling is handled externally)
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content") or ""
            parts.append(f"<{role}>\n{content}\n</{role}>")
        prompt = "\n".join(parts) + "\n<assistant>"

        def _run():
            return self._inference.generate_response(prompt)

        output = await asyncio.to_thread(_run)
        return {"role": "assistant", "content": output}

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[Chunk]:
        msg = await self.generate(messages, tools)
        yield {"content": msg.get("content") or "", "done": True}
