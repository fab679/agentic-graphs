"""Azure OpenAI provider — same interface as OpenAILLM but via Azure endpoints."""

from __future__ import annotations

import os
from typing import AsyncIterator

from agentic_graphs.llm.base import LLM, Message, Chunk

try:
    from openai import AsyncAzureOpenAI
except ImportError as exc:
    raise ImportError("Install OpenAI SDK: pip install openai") from exc


class AzureOpenAILLM(LLM):
    """Azure OpenAI provider.

    Required env vars (or pass as constructor args):
        AZURE_OPENAI_API_KEY
        AZURE_OPENAI_ENDPOINT      (e.g. https://my-resource.openai.azure.com/)
        AZURE_OPENAI_API_VERSION   (e.g. 2024-12-01-preview)
        AZURE_OPENAI_DEPLOYMENT    (your deployment name)
    """

    def __init__(
        self,
        deployment: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
        timeout: float = 120.0,
    ):
        self.model = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self._client = AsyncAzureOpenAI(
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_version=api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            timeout=timeout,
        )

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            tools=tools or None,
        )
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
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            tools=tools or None,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield {"content": delta.content, "done": False}
        yield {"content": "", "done": True}
