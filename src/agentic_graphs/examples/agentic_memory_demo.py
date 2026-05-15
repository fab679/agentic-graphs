#!/usr/bin/env python3
"""Agentic Memory demo — agents learn user preferences and recall them across turns.

The agent has built-in tools: ``remember``, ``recall``, ``search_memory``,
and ``list_memory_schemas``.  Facts are stored in ``memory:<user_id>`` graphs,
isolated per user.  Before each GOAL, relevant facts are automatically
injected into the prompt so the agent never forgets.

Usage:
    uv run python -m agentic_graphs.examples.agentic_memory_demo
"""

import asyncio, logging, sys

from agentic_graphs import Agent, OpenAILLM, tool
from agentic_graphs.core.falkordb_backend import FalkorDBBackend
from agentic_graphs.llm.base import Message

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="[mem] %(message)s")
log = logging.getLogger("agentic_graphs.agent")
log.setLevel(logging.INFO)


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


_TOOLS = {"add": add}
_SCHEMAS = [add.schema]


async def main():
    backend = None
    try:
        backend = FalkorDBBackend()
    except Exception:
        pass

    user_id = "demo_user"
    history: list[Message] = []

    turns = [
        "My name is Alice and I prefer metric units",
        "What is 17 * 3?",
        "What is 12 inches in cm?",
        "What is my name and what units do I prefer?",
    ]

    for msg in turns:
        print(f"\n  >>> {msg}")
        history.append({"role": "user", "content": msg})

        agent = Agent(
            OpenAILLM(),
            msg,
            graph_name="memory_demo",
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

    print(f"\n  Done.  User memory in FalkorDB: memory:{user_id}")

    # Cross-session test: fresh agent with no history, must recall from memory
    print("\n  === CROSS-SESSION TEST (no history) ===")
    fresh = Agent(
        OpenAILLM(),
        "What do you know about me?",
        graph_name="memory_demo",
        user_id=user_id,
    )
    if backend:
        fresh.attach_backend(backend)
    reply = await fresh.run()
    print(f"  <<< {reply}")


if __name__ == "__main__":
    asyncio.run(main())
