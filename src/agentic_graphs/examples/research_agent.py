"""Research agent — one agent tackling a complex question.

The mutation pipeline is built-in (Agent provides create_task, create_action,
resolve_current_node, etc. automatically).  This example only customises
the ACTION guide and registers ``search_topic`` as an extra action tool.

Usage:
    uv run python -m agentic_graphs.examples.research_agent
    uv run python -m agentic_graphs.examples.research_agent "Your question"
"""

import asyncio
import logging
import sys

from agentic_graphs import Agent, tool, OpenAILLM, NT, FalkorDBBackend
from agentic_graphs.agent.defaults import DEFAULT_GUIDES, default_build_prompt
from agentic_graphs.agent.scheduler import collect_answer


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


# -- tools -------------------------------------------------------------------

@tool
def search_topic(query: str) -> str:
    """Simulate researching a topic.

    Args:
        query: The topic or question to research.
    """
    knowledge = {
        "python": "Python is a dynamically-typed language known for readability.",
        "async": "Async/await enables cooperative multitasking via an event loop.",
        "graph": "A graph is nodes connected by edges. DAGs model dependencies.",
    }
    for key, val in knowledge.items():
        if key in query.lower():
            return val
    return f"Research results for: {query}"


# -- agent (only overrides build_prompt for a custom ACTION guide) ----------

class ResearchAgent(Agent):
    """Research agent with a custom ACTION node guide."""

    def __init__(self, llm, goal, **kwargs):
        self._guides = dict(DEFAULT_GUIDES)
        self._guides[NT.ACTION] = (
            "You are processing an ACTION node.\n"
            "Call search_topic() to research, then resolve_current_node('<result>')."
        )
        super().__init__(
            llm, goal,
            extra_action_tools={"search_topic": search_topic},
            extra_action_schemas=[search_topic.schema],
            **kwargs,
        )

    def build_prompt(self, node, graph):
        return default_build_prompt(node, graph, guides=self._guides)


# -- entry -------------------------------------------------------------------

_PROJECT = "research_project"

async def main():
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Research how Python async/await works and how graphs model dependencies."
    )

    backend = None
    try:
        backend = FalkorDBBackend()
    except Exception:
        pass

    agent = ResearchAgent(
        OpenAILLM(),
        goal,
        graph_name=_PROJECT,
        on_token=lambda text: print(text, end="", flush=True),
    )
    if backend:
        agent.attach_backend(backend)

    await agent.run()

    ans = collect_answer(agent.graph)
    if ans:
        bar = "\u2500" * 50
        print(f"\n  {bar}\n  Final answer:\n  {ans[:500]}")

    if backend:
        print(f"  [OK] Project: {_PROJECT}  http://localhost:3000")


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    _setup_logging()
    asyncio.run(main())
