"""Chat session demo — multi-turn conversation with graph persistence.

The mutation pipeline is built-in (Agent provides create_task, create_action,
resolve_current_node, etc. automatically).  We register ``search_knowledge``
as an extra_action_tool so ACTION nodes can research topics.

Each turn builds a fresh Graph, runs the agent, and persists everything
to FalkorDB.  History from prior turns is injected as LLM context.

Usage:
    uv run python -m agentic_graphs.examples.chat_session
    uv run python -m agentic_graphs.examples.chat_session --thread <id>
"""

import asyncio
import logging
import sys

from agentic_graphs import Agent, OpenAILLM, tool, NT
from agentic_graphs.core.falkordb_backend import FalkorDBBackend
from agentic_graphs.session import Session


# -- colored logging (logs go to stderr, streamed output goes to stdout) ------

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
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(_LogFormatter("%(message)s"))
    for existing in list(log.handlers):
        log.removeHandler(existing)
    log.addHandler(h)
    log.propagate = False


# -- tools (registered as extra_action_tools on Agent) ---------------------

@tool
def search_knowledge(query: str) -> str:
    """Search a simulated knowledge base.

    Args:
        query: The topic to look up.
    """
    db = {
        "python": "Python 3.13 introduced the JIT compiler and free-threaded mode.",
        "graphs": "Graphs consist of nodes (vertices) connected by edges (relationships).",
        "falkordb": "FalkorDB is a graph database built on Redis with Cypher query support.",
    }
    for key, val in db.items():
        if key in query.lower():
            return val
    return f"No results for {query!r}."


# -- session demo -------------------------------------------------------------

async def main():
    backend = FalkorDBBackend()

    thread_id = None
    if "--thread" in sys.argv:
        idx = sys.argv.index("--thread")
        thread_id = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None

    session = await Session.create(
        llm=OpenAILLM(model="gpt-4o-mini"),
        agent_class=Agent,
        backend=backend,
        thread_name="Chat session demo",
        user_id="demo_user",
        thread_id=thread_id,
        agent_kwargs={
            "on_token": lambda text: print(text, end="", flush=True),
            "extra_action_tools": {"search_knowledge": search_knowledge},
            "extra_action_schemas": [search_knowledge.schema],
        },
    )

    print(f"Thread: {session.thread.id}", file=sys.stderr)
    print(f"  meta:thread:{session.thread.id}  (conversation history)", file=sys.stderr)
    print(f"  project:{session.thread.id}         (execution graph)", file=sys.stderr)
    print(f"  memory:{session.thread.user_id}     (agentic memory)", file=sys.stderr)
    print("Type your messages, or 'quit' to exit.\n", file=sys.stderr)

    while True:
        try:
            message = input("question> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if message.lower() in ("quit", "exit", "q"):
            break
        if not message:
            continue

        reply = await session.chat(message)
        print(f"\n  {reply}\n")

    print(f"\nThread saved. Resume with:  --thread {session.thread.id}", file=sys.stderr)


if __name__ == "__main__":
    _setup_logging()
    asyncio.run(main())
