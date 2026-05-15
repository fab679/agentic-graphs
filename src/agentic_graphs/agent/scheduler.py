"""Standalone scheduler utilities for running Graph-based agents.

These functions can be used without subclassing Agent — useful for
procedurally-built graphs or multi-agent orchestration.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

from agentic_graphs.core.graph import Graph, Node, S, NT
from agentic_graphs.llm.base import LLM, Message

log = logging.getLogger(__name__)

# Optional hook called after every mutation: (graph) -> None
_SYNC_HOOK: Callable[[Graph], None] | None = None


def set_sync_hook(fn: Callable[[Graph], None] | None) -> None:
    """Register a callback invoked after every node state change.

    Use this to auto-persist graph mutations to FalkorDB::

        backend = FalkorDBBackend()
        set_sync_hook(lambda g: backend.sync(g))
    """
    global _SYNC_HOOK
    _SYNC_HOOK = fn


def _fire_hook(graph: Graph) -> None:
    if _SYNC_HOOK is not None:
        try:
            _SYNC_HOOK(graph)
        except Exception as exc:
            log.warning("sync hook error: %s", exc)


def _snap(s: str, n: int = 60) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "..."


async def process_node(
    node: Node,
    graph: Graph,
    llm: LLM,
    system_prompt: str,
    tool_schemas: list[dict] | None = None,
    tool_fns: dict[str, Any] | None = None,
    retries: int = 2,
) -> str:
    """Process a single READY node with an LLM + optional tools.

    Marks the node ACTIVE \u2192 RESOLVED (or FAILED) and fires the sync hook.
    Transient failures (timeouts, rate limits) are retried up to *retries* times.
    """
    log.info("Node %s [%s] %r \u2192 ACTIVE", node.id[:8], node.type.value, _snap(node.label))
    graph.set_state(node.id, S.ACTIVE)
    _fire_hook(graph)

    messages: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": node.label},
    ]

    tool_fns = tool_fns or {}
    iterations = 0
    last_exc: Exception | None = None

    for attempt in range(1 + retries):
        if attempt > 0:
            log.warning("Retry attempt %d/%d for node %s",
                        attempt + 1, retries + 1, node.id[:8])

        try:
            while iterations < 10:
                iterations += 1
                response = await llm.generate(messages, tools=tool_schemas or None)
                messages.append(response)

                tool_calls = response.get("tool_calls") or []
                if not tool_calls:
                    output = response.get("content") or ""
                    graph.resolve(node.id, output)
                    _fire_hook(graph)
                    log.info("Node %s [%s] \u2192 RESOLVED  (%d chars)",
                             node.id[:8], node.type.value, len(output))
                    return output

                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    log.info("  \u2699 %s(%s)", fn_name, _snap(json.dumps(args), 120))
                    fn = tool_fns.get(fn_name)
                    out = fn(**args) if fn else f"[tool '{fn_name}' not found]"
                    result = str(out) if out is not None else ""
                    log.info("  \u2190 %s", _snap(result, 120))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

            log.warning("Node %s hit max tool iterations", node.id[:8])
            graph.resolve(node.id, "[max tool iterations]")
            _fire_hook(graph)
            return "[max tool iterations]"

        except Exception as exc:
            last_exc = exc
            is_transient = any(
                s in str(exc).lower()
                for s in ("timed out", "timeout", "rate limit", "429", "503", "502")
            )
            if is_transient and attempt < retries:
                wait = 2.0 * (attempt + 1)
                log.warning("Node %s transient error (attempt %d/%d): %s. "
                            "Retrying in %.0fs...",
                            node.id[:8], attempt + 1, retries + 1, exc, wait)
                await asyncio.sleep(wait)
                # Reset iterator counter for retry (messages preserved)
                iterations = 0
                continue
            break

    log.error("Node %s error: %s", node.id, last_exc)
    graph.set_state(node.id, S.FAILED)
    _fire_hook(graph)
    raise last_exc  # type: ignore[misc]


async def run_scheduler(
    graph: Graph,
    llm: LLM,
    system_prompt: str,
    tool_schemas: list[dict] | None = None,
    tool_fns: dict[str, Any] | None = None,
    max_iterations: int = 50,
) -> str:
    """Drive the graph to completion using concurrent node execution.

    Returns the output of the last resolved node.
    """
    last_output = ""
    passes = 0
    for iteration in range(max_iterations):
        ready = graph.ready()
        if not ready:
            break
        passes = iteration + 1

        log.info("Schedule pass %d: %d node(s) ready", passes, len(ready))
        results = await asyncio.gather(
            *(
                process_node(n, graph, llm, system_prompt, tool_schemas, tool_fns)
                for n in ready
            ),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, str):
                last_output = r

    resolved = sum(1 for n in graph.nodes.values() if n.state == S.RESOLVED)
    failed = sum(1 for n in graph.nodes.values() if n.state == S.FAILED)
    total = len(graph.nodes)
    log.info("Scheduler done  passes=%d  %d/%d resolved  %d failed",
             passes, resolved, total, failed)

    # Deterministic output: SYNTHESIS > GOAL > last loop output
    for n in graph.nodes.values():
        if n.state == S.RESOLVED and n.type == NT.SYNTHESIS:
            return n.output
    for n in graph.nodes.values():
        if n.state == S.RESOLVED and n.type == NT.GOAL:
            return n.output
    return last_output


def collect_answer(graph: Graph) -> str:
    """Return the concatenated output of all RESOLVED nodes."""
    return "\n\n".join(
        f"[{n.type.upper()} — {n.label}]\n{n.output}"
        for n in graph.nodes.values()
        if n.state == S.RESOLVED and n.output
    )
