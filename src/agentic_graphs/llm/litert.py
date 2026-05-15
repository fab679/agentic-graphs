"""LiteRT-LM provider — on-device inference (Gemma 4 via LiteRT).

Uses ``litert_lm.Engine`` with ``create_conversation(tools=...)`` for
native tool-calling support.  Tool implementations are looked up from a
registry passed at construction time, or set per-call via
``set_tool_fns()`` (used by ``LiteRTAgent``).

Note: LiteRT SDK expects plain Python functions (not Tool wrappers) for
proper signature introspection.  ``Tool.fn`` is used when available.

Concurrency: the LiteRT Engine only supports one conversation session at
a time.  An ``asyncio.Lock`` serializes all ``generate()`` calls.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from typing import AsyncIterator, Callable

from agentic_graphs.llm.base import LLM, Message, Chunk

try:
    import litert_lm
except ImportError as exc:
    raise ImportError(
        "Install LiteRT-LM: pip install 'litert-lm>=0.11.0'"
    ) from exc

log = logging.getLogger(__name__)


def _snap(s: str, n: int = 120) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "…"


_MAX_TOOL_CALLS = 20

# Shared counter across all tools in a single generate() call
_tool_call_counter: dict[str, int] = {}
# Recorded tool calls for FalkorDB message subgraph
_tool_call_history: list[dict] = []


def _unwrap(fn: Callable) -> Callable:
    """Unwrap Tool objects → raw function for LiteRT signature introspection."""
    import functools as _ft

    raw = fn.fn if hasattr(fn, "fn") and inspect.isfunction(fn.fn) else fn
    tool_name = raw.__name__

    @_ft.wraps(raw)
    def _logged(*args, **kwargs):
        _tool_call_counter["total"] = _tool_call_counter.get("total", 0) + 1
        if _tool_call_counter["total"] > _MAX_TOOL_CALLS:
            msg = f"Conversation hit max tool calls ({_MAX_TOOL_CALLS})"
            log.warning("  \u2717 %s", msg)
            raise RuntimeError(msg)
        args_repr = ", ".join(
            [repr(a) for a in args] +
            [f"{k}={v!r}" for k, v in kwargs.items()]
        )

        call_args = kwargs if kwargs else {
            p.name: a for p, a in zip(
                [p for p in inspect.signature(raw).parameters.values()
                 if p.name not in ("self", "cls")],
                args
            )
        }
        entry = {
            "name": tool_name,
            "arguments": json.dumps(call_args),
            "result": None,
        }
        _tool_call_history.append(entry)

        log.info("  \u2699 %s(%s)", tool_name, _snap(args_repr))
        try:
            result = raw(*args, **kwargs)
            entry["result"] = str(result) if result is not None else ""
            log.info("  \u2190 %s", _snap(str(result)))
            return result
        except Exception as e:
            entry["error"] = str(e)
            log.error("  \u2717 %s error: %s", tool_name, e)
            raise

    return _logged


class LiteRTLLM(LLM):
    """On-device LiteRT-LM provider (Gemma 4 and compatible models).

    Supports tool calling via the LiteRT SDK's native function-calling
    API (``Engine.create_conversation``).  Pass tool implementations
    through the *tool_registry* or via ``set_tool_fns()`` (used by
    ``LiteRTAgent``).

    Usage::

        llm = LiteRTLLM(model_path="/path/to/model.litertlm",
                        tool_registry={"add": add, "multiply": multiply})
    """

    def __init__(
        self,
        model_path: str | None = None,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        tool_registry: dict[str, Callable] | None = None,
    ):
        self.model = "litert-lm"
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._tool_registry = tool_registry or {}
        self._current_tool_fns: dict[str, Callable] | None = None

        path = model_path or os.environ.get("MODEL_PATH", "")
        if not path:
            raise ValueError(
                "Provide model_path= or set MODEL_PATH env var "
                "(e.g. /path/to/gemma-4-E2B-it.litertlm/model.litertlm)"
            )
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        log.info("Loading LiteRT model: %s", path)
        self._engine = litert_lm.Engine(path)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Tool-fn injection (called by LiteRTAgent before generate)
    # ------------------------------------------------------------------

    def set_tool_fns(self, fns: dict[str, Callable]) -> None:
        """Set the current node's tool implementations for the next ``generate()`` call."""
        self._current_tool_fns = fns

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(messages: list[Message]) -> str:
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content") or ""
            parts.append(f"<{role}>\n{content}\n</{role}>")
        return "\n".join(parts) + "\n<assistant>"

    def _resolve_tools(
        self,
        tool_schemas: list[dict] | None,
    ) -> list[Callable]:
        """Convert OpenAI-style tool schemas to Python functions for LiteRT SDK.

        Lookup order:
          1. ``_current_tool_fns`` (set by ``LiteRTAgent`` per node)
          2. ``_tool_registry`` (user-provided at construction)
        """
        fns: list[Callable] = []
        if not tool_schemas:
            return fns

        current = self._current_tool_fns or {}

        for t in tool_schemas:
            fn_info = t.get("function", t)
            name = fn_info["name"]
            fn = current.get(name) or self._tool_registry.get(name)
            if fn is not None:
                fns.append(_unwrap(fn))

        return fns

    # ------------------------------------------------------------------
    # LLM interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        prompt = self._build_prompt(messages)

        tool_fns = self._resolve_tools(tools)

        async with self._lock:
            _tool_call_counter.clear()
            _tool_call_history.clear()
            def _run() -> dict:
                with self._engine.create_conversation(
                    tools=tool_fns or None,
                ) as conv:
                    return conv.send_message(prompt)

            response = await asyncio.to_thread(_run)

        text = ""
        for block in response.get("content", []):
            if isinstance(block, dict) and "text" in block:
                text += block["text"]

        log.info("Gemma response: %s", text[:200])

        result: Message = {"role": "assistant", "content": text}

        prompt_tokens = len(prompt.split())
        completion_tokens = len(text.split()) if text else 0
        result["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        return result

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[Chunk]:
        msg = await self.generate(messages, tools)
        content = msg.get("content") or ""

        chunk_size = 50
        for i in range(0, len(content), chunk_size):
            yield {"content": content[i : i + chunk_size], "done": False}

        yield {"content": "", "done": True, "usage": msg.get("usage", {})}
