#!/usr/bin/env python3
"""Math agent using LiteRT-LM (Gemma 4 on-device) with native tool calling.

Uses MODEL_PATH from environment or --model argument.

LiteRT's Engine.create_conversation(tools=...) handles the tool-call loop
internally — the SDK calls your Python functions, feeds results back to
the model, and returns the final text.  Tool calls are logged to stdout
so you can see them happening.
"""

import asyncio
import logging
import os
import sys

from agentic_graphs import LiteRTAgent, LiteRTLLM, tool
from agentic_graphs.core.falkordb_backend import FalkorDBBackend


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
        tag = f"{_Color.cyan}{_Color.bold}[litert]{_Color.reset}"
        msg = msg.replace("→ ACTIVE", f"{_Color.yellow}→ ACTIVE{_Color.reset}")
        msg = msg.replace("→ RESOLVED", f"{_Color.green}→ RESOLVED{_Color.reset}")
        msg = msg.replace("→ FAILED", f"{_Color.red}→ FAILED{_Color.reset}")
        msg = msg.replace("⚙ ", f"{_Color.magenta}⚙ {_Color.reset}")
        msg = msg.replace("← ", f"{_Color.green}← {_Color.reset}")
        return f"{tag} {msg}"


def _setup_logging():
    for name in ("agentic_graphs.agent", "agentic_graphs.llm.litert"):
        log = logging.getLogger(name)
        log.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(_LogFormatter("%(message)s"))
        for existing in list(log.handlers):
            log.removeHandler(existing)
        log.addHandler(h)
        log.propagate = False


# -- arithmetic tools -------------------------------------------------------

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


# -- runner -----------------------------------------------------------------

async def _run_once(goal: str, model_path: str, backend=None):
    print(f"{_Color.bold}Model:{_Color.reset} {model_path}")
    print(f"{_Color.bold}Goal:{_Color.reset} {goal}")
    print()

    llm = LiteRTLLM(
        model_path=model_path,
        tool_registry=_MATH_TOOLS,
    )

    agent = LiteRTAgent(
        llm, goal,
        graph_name="litert_math",
        extra_action_tools=_MATH_TOOLS,
        extra_action_schemas=_MATH_SCHEMAS,
    )
    if backend:
        agent.attach_backend(backend)
    result = await agent.run()
    if backend and agent._root_id:
        backend.resolve_node(agent._root_id, result, "resolved", "litert_math")
    print(f"\n{_Color.green}✓ {result}{_Color.reset}")
    return result


async def _chat(model_path: str, backend=None):
    print("LiteRT math agent — type a math problem or 'quit'")
    print(f"Model: {model_path}\n")
    while True:
        try:
            goal = input("question> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if goal.lower() in ("quit", "exit", "q"):
            break
        if not goal:
            continue
        await _run_once(goal, model_path, backend)


if __name__ == "__main__":
    _setup_logging()

    model_path = os.environ.get("MODEL_PATH", "")
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--model" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 2]
        elif arg.startswith("--model="):
            model_path = arg.split("=", 1)[1]

    if not model_path:
        print("Error: MODEL_PATH not set. Provide it via:")
        print("  export MODEL_PATH=/path/to/model.litertlm")
        print("  or: --model /path/to/model.litertlm")
        sys.exit(1)

    backend = None
    try:
        backend = FalkorDBBackend(graph_name="litert_math")
        print(f"{_Color.green}✓ FalkorDB connected{_Color.reset} — open http://localhost:3000\n")
    except Exception as e:
        print(f"{_Color.yellow}⚠ FalkorDB not available: {e}{_Color.reset}")
        print("  Run: docker run -d -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest\n")

    if "--chat" in sys.argv:
        asyncio.run(_chat(model_path, backend))
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        goal = " ".join(args) or "What is (17 * 3 + 42) / 9?"
        asyncio.run(_run_once(goal, model_path, backend))
