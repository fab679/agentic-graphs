#!/usr/bin/env python3
"""Multi-agent demo — subagents, skills, and router patterns.

All agents share a single graph. Children inherit the parent's graph
context, providing shared memory and full FalkorDB traceability.

Usage:
    uv run python -m agentic_graphs.examples.multi_agent_demo
    uv run python -m agentic_graphs.examples.multi_agent_demo --chat
"""

import asyncio
import logging
import sys

from agentic_graphs import Agent, OpenAILLM, tool
from agentic_graphs.core.falkordb_backend import FalkorDBBackend
from agentic_graphs.agent.defaults import default_build_tools, default_build_prompt
from agentic_graphs.session import Session


# -- colored logging ---------------------------------------------------------

class _Color:
    cyan = "\033[36m"
    yellow = "\033[33m"
    green = "\033[32m"
    magenta = "\033[35m"
    red = "\033[91m"
    bold = "\033[1m"
    dim = "\033[2m"
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


# -- domain tools ------------------------------------------------------------

@tool
def search_weather(city: str) -> str:
    """Get current weather for a city."""
    data = {"london": "12\u00b0C, cloudy", "paris": "18\u00b0C, sunny",
            "tokyo": "22\u00b0C, humid", "nairobi": "27\u00b0C, clear"}
    return data.get(city.lower(), f"Weather data not available for {city}")

@tool
def search_news(topic: str) -> str:
    """Search recent news on a topic."""
    data = {"ai": "AI agents are transforming enterprise workflows (2026)",
            "python": "Python 3.14 beta released with pattern matching improvements",
            "climate": "Global renewable energy capacity grew 25% in 2025"}
    return data.get(topic.lower(), f"No recent news on {topic}")

@tool
def calculate(expr: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


# -- subagent: research expert -----------------------------------------------

_RESEARCH_SYSTEM = """You are a research specialist.
Decompose the task into at most 2 sub-tasks.

IMPORTANT — extract the city/location from the task label first:
  - If the task asks about weather in London, create action with
    search_weather(city="London") — do NOT guess other cities.
  - If the task asks about news on AI, create action with
    search_news(topic="AI").

Each ACTION node must call search_weather or search_news exactly once
with the correct extracted parameters, then resolve_current_node().
Use exact node IDs from create_task/create_action output."""

class ResearchSubAgent(Agent):
    def build_prompt(self, node, graph):
        base = default_build_prompt(node, graph)
        return base + "\n\n━━ EXTRA RESEARCH INSTRUCTIONS ━━\n" + _RESEARCH_SYSTEM
    def build_tools(self, node):
        return default_build_tools(
            self.graph, node,
            extra_action_tools={"search_weather": search_weather,
                                "search_news": search_news},
            extra_action_schemas=[search_weather.schema, search_news.schema],
        )


# -- subagent: math expert ---------------------------------------------------

_MATH_SYSTEM = """You are a math specialist. Decompose into at most 2 sub-tasks.
Each ACTION node must call calculate() with a valid expression.
Use exact node IDs from create_task/create_action output — never guess."""

class MathSubAgent(Agent):
    def build_prompt(self, node, graph):
        base = default_build_prompt(node, graph)
        return base + "\n\n━━ EXTRA MATH INSTRUCTIONS ━━\n" + _MATH_SYSTEM
    def build_tools(self, node):
        return default_build_tools(
            self.graph, node,
            extra_action_tools={"calculate": calculate},
            extra_action_schemas=[calculate.schema],
        )


# -- main agent (uses subagents + skills + handoff) --------------------------

_MAIN_SYSTEM = """You are a versatile orchestrator agent.
You have access to subagents: subagent_math and subagent_research.

PROTOCOL — follow exactly:
  1. Call ALL relevant subagents in the SAME response — they run in parallel.
  2. Each returns "[goal:<id>] <output>".  The <id> is the subgoal node ID.
  3. Once you have both results, output your final answer as plain text.
     Do NOT make any more tool calls.

Example:
  User: "What is 2+2 and weather in London?"
  You: call subagent_math({"task": "2 + 2"})
       and subagent_research({"task": "weather in London"})
  Then output: "2 + 2 = 4 and the weather in London is 12°C, cloudy."

Rules:
  - Call each subagent AT MOST ONCE
  - After receiving results, STOP making tool calls and output the answer"""

class OrchestratorAgent(Agent):
    def __init__(self, llm, goal, **kwargs):
        super().__init__(llm, goal, **kwargs)
        self.register_subagent(
            "research", ResearchSubAgent,
            description="Research topics using weather and news tools",
        )
        self.register_subagent(
            "math", MathSubAgent,
            description="Perform mathematical calculations",
        )
    def build_prompt(self, node, graph):
        return _MAIN_SYSTEM


# -- runner ------------------------------------------------------------------

async def run_demo():
    backend = None
    try:
        backend = FalkorDBBackend()
    except Exception:
        pass

    def on_token(text):
        print(text, end="", flush=True)

    question = "What is 25 * 4 + 100? And what's the weather in London?"
    print(f"\n  {_Color.bold}{_Color.yellow}Q: {question}{_Color.reset}\n")

    agent = OrchestratorAgent(
        OpenAILLM(model="gpt-4o-mini"), question,
        on_token=on_token,
    )
    if backend:
        agent.attach_backend(backend)
        backend.sync(agent.graph, agent.graph_name)

    result = await agent.run()
    print(f"\n  {_Color.bold}{_Color.green}A: {result}{_Color.reset}\n")

    if backend:
        backend.sync(agent.graph, agent.graph_name)
        print(f"  Graph: {agent.graph_name}  ({len(agent.graph.nodes)} nodes)")
        print("  http://localhost:3000")


async def run_chat():
    backend = FalkorDBBackend()

    session = await Session.create(
        llm=OpenAILLM(model="gpt-4o-mini"),
        agent_class=OrchestratorAgent,
        backend=backend,
        thread_name="Multi-agent chat",
        user_id="demo",
        agent_kwargs={
            "on_token": lambda text: print(text, end="", flush=True),
        },
    )

    print(f"Multi-agent chat  [{session.thread.id}]")
    print("Ask anything (or 'quit'). Subagents handle research & math.\n")

    while True:
        try:
            msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in ("quit", "exit", "q"):
            break
        if not msg:
            continue

        reply = await session.chat(msg)
        print(f"\n  {reply}\n")


if __name__ == "__main__":
    _setup_logging()
    if "--chat" in sys.argv:
        asyncio.run(run_chat())
    else:
        asyncio.run(run_demo())
