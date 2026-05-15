"""OpenAI provider — implements the LLM interface with streaming."""

import json, os
from typing import AsyncIterator

from openai import AsyncOpenAI

from agentic_graphs.llm.base import LLM, Message, Chunk


def _load_env():
    for candidate in (".env", os.path.expanduser("~/.env")):
        path = os.path.abspath(candidate)
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            return


_load_env()


class OpenAILLM(LLM):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        timeout: float = 120.0,
    ):
        self.model = model
        self.embed_model = embed_model
        self._timeout = timeout
        self._client = client or AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=timeout,
        )

    async def embed(self, text: str) -> list[float]:
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=self._timeout)
        resp = await client.embeddings.create(input=text, model=self.embed_model)
        return resp.data[0].embedding

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        api_key = os.environ.get("OPENAI_API_KEY")
        client = AsyncOpenAI(api_key=api_key, timeout=self._timeout)
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            tools=tools or None,
        )
        msg = response.choices[0].message

        result: Message = {"role": "assistant", "content": msg.content or ""}

        if response.usage:
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

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

        return result

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[Chunk]:
        api_key = os.environ.get("OPENAI_API_KEY")
        client = AsyncOpenAI(api_key=api_key, timeout=self._timeout)
        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            tools=tools or None,
            stream=True,
        )

        tool_call_index: dict[int, dict] = {}
        finish_reason = None
        last_event = None

        async for event in stream:
            last_event = event
            delta = event.choices[0].delta if event.choices else None
            finish = event.choices[0].finish_reason if event.choices else None
            if finish:
                finish_reason = finish

            if delta is None:
                continue

            if delta.content:
                yield {"content": delta.content, "done": False}

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_index:
                        tool_call_index[idx] = {
                            "id": tc_delta.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.id:
                        tool_call_index[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_call_index[idx]["function"]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_call_index[idx]["function"]["arguments"] += tc_delta.function.arguments

        content = ""
        tcs = list(tool_call_index.values()) if tool_call_index else None

        final_chunk: Chunk = {
            "content": content,
            "tool_call": tcs[0] if tcs else None,
            "tool_calls": tcs if tcs else None,
            "done": True,
        }

        # Capture usage from the final event
        if last_event is not None and hasattr(last_event, 'usage') and last_event.usage:
            final_chunk["usage"] = {
                "prompt_tokens": last_event.usage.prompt_tokens,
                "completion_tokens": last_event.usage.completion_tokens,
                "total_tokens": last_event.usage.total_tokens,
            }

        yield final_chunk
