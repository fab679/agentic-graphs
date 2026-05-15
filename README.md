# agentic-graphs

A graph-native framework for building AI agents. Persistence and visualisation backed by **FalkorDB** — every agent's state is a first-class graph you can inspect at **http://localhost:3000**.

```
uv add agentic-graphs
```

## Quick start

```python
from agentic_graphs import Agent, OpenAILLM, tool

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

class MathAgent(Agent):
    def build_prompt(self, node, graph):
        return f"Process {node.label}"
    def build_tools(self, node):
        return [add.schema], {"add": add}

agent = MathAgent(OpenAILLM(), "15 + 27")
await agent.run()
```

## Multi-turn chat sessions

Each turn builds a fresh Graph and persists it to FalkorDB. History is injected automatically as LLM context.

```python
from agentic_graphs import OpenAILLM
from agentic_graphs.core.falkordb_backend import FalkorDBBackend
from agentic_graphs.session import Session

session = await Session.create(
    llm=OpenAILLM(model="gpt-4o-mini"),
    agent_class=ChatAgent,
    backend=FalkorDBBackend(),
    thread_name="My chat",
    user_id="alice",
)

reply = await session.chat("What is the capital of Kenya?")
reply = await session.chat("And the population?")

# Resume later by thread ID:
session = await Session.create(
    ..., thread_id=session.thread_id,
)
```

## Run examples

```bash
# Start FalkorDB (one-time)
docker run -d -p 6379:6379 -p 3000:3000 --name falkordb falkordb/falkordb:latest

# Math chat (single-turn)
uv run python -m agentic_graphs.examples.math_agent --chat

# Multi-agent research
uv run python -m agentic_graphs.examples.multi_agent

# Single research agent
uv run python -m agentic_graphs.examples.research_agent "Your topic"

# Multi-turn chat session (new)
uv run python -m agentic_graphs.examples.chat_session

# Multi-turn chat session
uv run python -m agentic_graphs.examples.chat_session

# Multi-agent demo — orchestrator delegates to math + research subagents
uv run python -m agentic_graphs.examples.multi_agent_demo
# Subagents share the parent's graph. Each sub-goal runs its own scoped
# scheduler with the full GOAL→TASK→ACTION→SYNTHESIS pipeline.
# Requires an LLM capable of structured tool use (gpt-4o, claude-sonnet-4, etc.)

# Session debugger — inspect threads, turns, and execution graphs
uv run python -m agentic_graphs.examples.debug_session
uv run python -m agentic_graphs.examples.debug_session --thread <id>
uv run python -m agentic_graphs.examples.debug_session --raw   # list all graphs
```

Every example syncs its graph to FalkorDB automatically. Open **http://localhost:3000** to browse.

**Tip:** If you see "No threads found", check FalkorDB is running:
```bash
docker run -d -p 6379:6379 -p 3000:3000 --name falkordb falkordb/falkordb:latest
```
The debugger now checks connectivity and shows available graphs.

## Architecture

- **Graph** — single source of truth for all agent state (nodes + edges)
- **Nodes** — GOAL, TASK, ACTION, SYNTHESIS with states PENDING -> READY -> ACTIVE -> RESOLVED/FAILED
- **Edges** — REQUIRES (blocking), PART_OF (structural), PRODUCES (informational)
- **Scheduler** — processes READY nodes concurrently; fan-out via `asyncio.gather`, fan-in via REQUIRES edges
- **LLM** — abstract provider interface (OpenAI, LiteRT-LM, etc.)
- **@tool** — auto-generates OpenAI-compatible schema from type hints + docstrings
- **Session** — multi-turn conversation with per-turn graph traceability
- **FalkorDB** — native graph persistence with built-in browser UI

## Using FalkorDB directly

```python
from agentic_graphs import FalkorDBBackend, Graph, Node, NT

backend = FalkorDBBackend(graph_name="my_agent")

# Push
backend.sync(my_graph)

# Pull
restored = backend.load("my_agent")

# Query
result = backend.query("MATCH (n:GOAL) RETURN n.label, n.state")
for row in result.result_set:
    print(row)

# Auto-sync every mutation
from agentic_graphs import set_sync_hook
set_sync_hook(lambda g: backend.sync(g))
```

## Project structure

```
src/agentic_graphs/
├── __init__.py          # Public API
├── log.py               # Coloured logging
├── core/
│   ├── graph.py         # Graph, Node, Edge, algorithms
│   ├── tool.py          # @tool decorator
│   └── falkordb_backend.py  # FalkorDB persistence
├── llm/
│   ├── base.py          # Abstract LLM interface
│   └── openai.py        # OpenAI provider
├── agent/
│   ├── base.py          # Agent base class (history_messages support)
│   ├── defaults.py      # Default prompts, guides, mutation tools
│   └── scheduler.py     # Run loop with retry + concurrency
├── session/
│   ├── models.py        # Thread, Turn, TurnStatus, SessionConfig
│   ├── store.py         # ThreadStore — FalkorDB-backed persistence
│   └── session.py       # Session — multi-turn chat with graph traceability
└── examples/
    ├── math_agent.py       # Math solver with @tool tools
    ├── multi_agent.py      # Coordinator + parallel sub-agents
    ├── research_agent.py   # Single research agent (fixed)
    ├── chat_session.py     # Multi-turn session demo (new)
    └── debug_session.py    # Session debugger (new)
```
