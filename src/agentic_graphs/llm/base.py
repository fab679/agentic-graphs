"""Abstract LLM interface — all providers implement this."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, TypedDict


class ToolCall(TypedDict):
    id: str
    type: str
    function: dict


class Usage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Message(TypedDict, total=False):
    role: str
    content: str | None
    tool_calls: list[ToolCall]
    tool_call_id: str
    usage: Usage | None


class Chunk(TypedDict, total=False):
    """A streamed token fragment.

    During streaming, multiple chunks are yielded with ``content`` fragments.
    The final chunk has ``done=True`` and may include ``tool_calls`` (list of
    complete ToolCall dicts) if the LLM requested tools.
    """
    content: str
    tool_call: ToolCall | None   # deprecated — use tool_calls
    tool_calls: list[ToolCall]   # all tool calls (final chunk)
    usage: Usage | None
    done: bool


class LLM(ABC):
    """Unified interface for any LLM provider.

    All providers accept a ``timeout`` parameter (default 120s) which is
    passed to the underlying HTTP client (httpx, aiohttp, etc.) to set
    the maximum wait time for an API response.  The retry logic in
    ``Agent.process_node`` will retry on timeout errors with backoff.

    Usage:
        llm = OpenAILLM(model="gpt-4o-mini")
        reply = await llm.generate(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[add.schema],
        )
    """

    model: str

    async def embed(self, text: str) -> list[float]:
        """Return a vector embedding for *text*.

        Default implementation raises NotImplementedError.
        Override in provider subclasses that support embeddings.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support embeddings")

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        """Send messages + optional tool schemas → get a response message.

        If the response contains tool_calls, the caller is responsible for
        executing them and feeding the results back in a follow-up call.
        """
        ...

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[Chunk]:
        """Streamed version of ``generate``.

        Yields Chunk dicts as tokens arrive.  The final chunk has ``done=True``
        and the complete ``Message`` under ``message``.

        Default implementation calls ``generate`` and yields one chunk.
        Override for true streaming.
        """
        msg = await self.generate(messages, tools)
        yield {
            "content": msg.get("content") or "",
            "tool_call": msg.get("tool_calls", [None])[0] if msg.get("tool_calls") else None,
            "done": True,
        }
