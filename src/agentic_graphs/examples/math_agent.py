#!/usr/bin/env python3
"""Math agent — solves arithmetic problems via the mutation tool pipeline.

The mutation pipeline is built-in (Agent provides create_task, create_action,
resolve_current_node, etc. automatically).  We only register the arithmetic
tools as extra_action_tools — they become available on ACTION nodes.
"""

import asyncio
import logging
import sys

from agentic_graphs import Agent, OpenAILLM, tool
from agentic_graphs.core.falkordb_backend import FalkorDBBackend
from agentic_graphs.llm.base import Message


# -- colored logging ---------------------------------------------------------

class _Color:
    cyan = "\033[36m"
    yellow = "\033[33m"
    green = "\033[32m"
    magenta = "\033[35m"
    red = "\033[91m"
    bold = "\033[1m"
    reset = "\033[0m"


class _LogFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        tag = f"{_Color.cyan}{_Color.bold}[agt]{_Color.reset}"
        msg = msg.replace("\u2192 ACTIVE", f"{_Color.yellow}\u2192 ACTIVE{_Color.reset}")
        msg = msg.replace("\u2192 RESOLVED", f"{_Color.green}\u2192 RESOLVED{_Color.reset}")
        msg = msg.replace("\u2192 FAILED", f"{_Color.red}\u2192 FAILED{_Color.reset}")
        msg = msg.replace("\u2699 ", f"{_Color.magenta}\u2699 {_Color.reset}")
        msg = msg.replace("\u2190 ", f"{_Color.green}\u2190 {_Color.reset}")
        return f"{tag} {msg}"


def _setup_logging():
    log = logging.getLogger("agentic_graphs.agent")
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_LogFormatter("%(message)s"))
    for existing in list(log.handlers):
        log.removeHandler(existing)
    log.addHandler(h)
    log.propagate = False


# -- arithmetic tools (registered as extra_action_tools) --------------------

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b (b must not be zero)."""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b


_MATH_TOOLS = {t.name: t for t in [add, subtract, multiply, divide]}
_MATH_SCHEMAS = [t.schema for t in [add, subtract, multiply, divide]]


# -- runner ------------------------------------------------------------------

_PROJECT_GRAPH = "math_project"

async def _run_once(goal: str, backend=None, history_messages=None):
    llm = OpenAILLM(model="gpt-4o-mini")

    agent = Agent(
        llm, goal,
        graph_name=_PROJECT_GRAPH,
        on_token=lambda text: print(text, end="", flush=True),
        extra_action_tools=_MATH_TOOLS,
        extra_action_schemas=_MATH_SCHEMAS,
        history_messages=history_messages or [],
    )
    if backend:
        agent.attach_backend(backend)
    result = await agent.run()
    # Update the GOAL node output with the actual final answer so
    # cross-turn semantic memory surfaces real answers, not decomposition text.
    if backend and agent._root_id:
        backend.resolve_node(agent._root_id, result, "resolved", _PROJECT_GRAPH)
    print(f"\n\u2713 {result}\n")
    return result


async def _chat(backend=None):
    print("Math agent \u2014 type a math problem or 'quit'")
    history: list[Message] = []
    while True:
        try:
            goal = input("question> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if goal.lower() in ("quit", "exit", "q"):
            break
        if not goal:
            continue
        history.append({"role": "user", "content": goal})
        reply = await _run_once(goal, backend, history[:-1])
        history.append({"role": "assistant", "content": reply})
        # Keep only last 6 turns to avoid blowing the context window
        if len(history) > 12:
            history = history[-12:]


if __name__ == "__main__":
    _setup_logging()
    backend = None
    try:
        backend = FalkorDBBackend(graph_name="math_agent")
    except Exception:
        pass

    if "--chat" in sys.argv:
        asyncio.run(_chat(backend))
    else:
        goal = " ".join(sys.argv[1:]) or "What is (17 * 3 + 42) / 9?"
        asyncio.run(_run_once(goal, backend))
