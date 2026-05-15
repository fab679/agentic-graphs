#!/usr/bin/env python3
"""Skill Learning demo — agent creates reusable skills from experience.

The agent discovers procedures during problem-solving and saves them
as skills via ``create_skill``.  Later queries trigger ``search_skills``
and ``activate_learned_skill`` to reuse the learned procedure.

Usage:
    uv run python -m agentic_graphs.examples.skill_learning_demo
"""

import asyncio, logging, sys

from agentic_graphs import Agent, OpenAILLM, tool
from agentic_graphs.core.falkordb_backend import FalkorDBBackend
from agentic_graphs.llm.base import Message

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="[skill] %(message)s")
log = logging.getLogger("agentic_graphs.agent")
log.setLevel(logging.INFO)


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


_TOOLS = {"multiply": multiply, "add": add}
_SCHEMAS = [multiply.schema, add.schema]


async def main():
    backend = None
    try:
        backend = FalkorDBBackend()
    except Exception:
        pass

    user_id = "skill_demo"
    history: list[Message] = []

    turns = [
        "A widget costs 17 credits. Sales tax is 8.25%. What is the total price?",
        "Another widget costs 34 credits. What is the total with same tax rate?",
        "If a gadget costs 50 credits with 8.25% tax, what is the total?",
    ]

    for msg in turns:
        print(f"\n  >>> {msg}")
        history.append({"role": "user", "content": msg})

        agent = Agent(
            OpenAILLM(),
            msg,
            graph_name="skill_demo",
            user_id=user_id,
            history_messages=history[:-1],
            extra_action_tools=_TOOLS,
            extra_action_schemas=_SCHEMAS,
        )
        if backend:
            agent.attach_backend(backend)

        reply = await agent.run()
        history.append({"role": "assistant", "content": reply})
        print(f"  <<< {reply}")

    print(f"\n  Done. Memory graph: memory:{user_id}")


if __name__ == "__main__":
    asyncio.run(main())
