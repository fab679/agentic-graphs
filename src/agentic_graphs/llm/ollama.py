"""Ollama provider — local models (Llama 3, Mistral, Phi-3, Qwen, etc.)."""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

from agentic_graphs.llm.base import LLM, Message, Chunk

try:
    import httpx
except ImportError as exc:
    raise ImportError("Install httpx: pip install httpx") from exc


class OllamaLLM(LLM):
    """Ollama local inference provider.

    Talks to the Ollama REST API (default: http://localhost:11434).
    Supports tool calling for models that have it (llama3.2, qwen2.5, etc.).

    Usage::
        llm = OllamaLLM(model="llama3.2")
        # Make sure Ollama is running: ollama serve
        # and the model is pulled: ollama pull llama3.2
    """

    def __init__(
        self,
        model: str = "llama3.2",
        embed_model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        self.model = model
        self.embed_model = embed_model or model
        self._timeout = timeout
        self._base = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(f"{self._base}/api/embed", json={
                "model": self.embed_model,
                "input": text,
            })
            r.raise_for_status()
            data = r.json()
        return data["embeddings"][0]

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(f"{self._base}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()

        msg = data.get("message", {})
        result: Message = {"role": "assistant", "content": msg.get("content") or ""}

        tool_calls_raw = msg.get("tool_calls") or []
        if tool_calls_raw:
            result["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": json.dumps(tc["function"].get("arguments", {})),
                    },
                }
                for i, tc in enumerate(tool_calls_raw)
            ]

        # Capture usage if available
        if "prompt_eval_count" in data or "eval_count" in data:
            result["usage"] = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }

        return result

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[Chunk]:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", f"{self._base}/api/chat", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = chunk.get("message", {}).get("content") or ""
                    done = chunk.get("done", False)
                    if content:
                        yield {"content": content, "done": False}
                    if done:
                        yield {"content": "", "done": True}
                        return
        yield {"content": "", "done": True}
