# agentic-graphs

**Graph-native framework for building AI agents with persistent execution state.**

Every agent's execution is a directed graph—GOAL → TASK → ACTION → SYNTHESIS. Persistence is backed by **FalkorDB**, so you can inspect the full state of every agent at [http://localhost:3000](http://localhost:3000).

- 🚀 **Built-in tool pipeline** — agents don't hallucinate; they mutate graphs explicitly
- 🔄 **Multi-turn chat** — sessions preserve full context across turns
- 🤝 **Multi-agent patterns** — subagents, skills, handoffs, routers all share the parent graph
- 💾 **Persistent execution** — every node, edge, and message is queryable in FalkorDB
- 🧠 **Semantic memory** — cross-turn reasoning with embeddings

## Installation

### Via pip (stable)
```bash
pip install agentic-graphs
```

### Via uv (development)
```bash
uvx agentic-graphs  # or
uv add agentic-graphs
```

### Optional dependencies
```bash
# Use other LLM providers
pip install agentic-graphs[anthropic,gemini,groq]  # or [all]

# Development
pip install agentic-graphs[dev]
```

## Quick Start

### 1. Set up FalkorDB (one-time)

```bash
docker run -d -p 6379:6379 -p 3000:3000 --name falkordb falkordb/falkordb:latest
```

Then open [http://localhost:3000](http://localhost:3000) to inspect agent state.

### 2. Create your first agent

```python
import asyncio
from agentic_graphs import Agent, OpenAILLM, tool

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

class MathAgent(Agent):
    """Simple math agent with custom tools."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._extra_action_tools = {"add": add, "multiply": multiply}
        self._extra_action_schemas = [add.schema, multiply.schema]

    def build_tools(self, node):
        schemas, impls = super().build_tools(node)
        schemas += self._extra_action_schemas
        impls.update(self._extra_action_tools)
        return schemas, impls

async def main():
    llm = OpenAILLM(model="gpt-4o-mini")
    agent = MathAgent(llm, "What is (17 * 3 + 42) / 9?")
    result = await agent.run()
    print(f"Answer: {result}")

asyncio.run(main())
```

### 3. Multi-turn chat sessions

```python
from agentic_graphs.session import Session
from agentic_graphs.core.falkordb_backend import FalkorDBBackend

async def chat():
    session = await Session.create(
        llm=OpenAILLM(model="gpt-4o-mini"),
        agent_class=MathAgent,
        backend=FalkorDBBackend(),
        thread_name="math-session",
        user_id="alice",
    )

    # Each turn builds a fresh graph, history is auto-injected
    reply = await session.chat("What is 5 * 8?")
    print(reply)

    reply = await session.chat("Add 10 to that result")
    print(reply)

    # Resume later by thread ID
    session = await Session.create(..., thread_id=session.thread_id)
```

## Run Examples

```bash
# Math agent (single-turn)
uv run python -m agentic_graphs.examples.math_agent "What is 15 + 27?"
uv run python -m agentic_graphs.examples.math_agent --chat  # Interactive

# Multi-turn chat session
uv run python -m agentic_graphs.examples.chat_session

# Multi-agent with subagents (research + math)
uv run python -m agentic_graphs.examples.multi_agent_demo

# Session debugger — inspect execution graphs
uv run python -m agentic_graphs.examples.debug_session
uv run python -m agentic_graphs.examples.debug_session --thread <thread-id>
```

Open [http://localhost:3000](http://localhost:3000) to see the graph state in real-time.

## Core Concepts

### Graph Nodes

Every agent execution builds a directed graph. Node types and states:

```
GOAL → TASK → ACTION → SYNTHESIS → (resolved)
  ↓
 (pending/ready/active/resolved/failed)
```

- **GOAL** — user's request
- **TASK** — sub-goal decomposition
- **ACTION** — tool calls (mutations, external APIs)
- **SYNTHESIS** — final answer generation

### Built-in Tools

Agents automatically have:

- `create_task(label, ...)` — create a TASK node
- `create_action(label, instruction)` — create an ACTION node
- `resolve_current_node(output)` — mark node as RESOLVED
- `add_dependency(waiting, prereq)` — set up blocking edges
- `get_token_usage()` — retrieve accumulated token costs

ACTION nodes also get your custom tools (e.g., `add`, `multiply`).

### Multi-Agent Patterns

**Subagents** — delegate to child agents in sub-GOAL nodes:

```python
agent.register_subagent(
    "researcher",
    agent_class=ResearchAgent,
    description="Research any topic",
)
```

**Skills** — progressive disclosure of domain tools:

```python
agent.register_skill(
    "web_search",
    tools=[search.schema],
    tool_fns={"search": search},
    description="Search the web",
)
```

**Handoffs** — state-driven agent switching:

```python
if "financial" in task:
    await agent.handoff_to("financial_analyst")
```

## Supported LLM Providers

- ✅ **OpenAI** — gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-5
- ✅ **Anthropic** — claude-sonnet-4, claude-opus
- ✅ **Google Gemini** — gemini-2.0-flash, gemini-pro
- ✅ **Groq** — mixtral, llama
- ✅ **Ollama** — local models
- ✅ **Azure OpenAI**

Switch providers with:

```python
from agentic_graphs import AnthropicLLM, GeminiLLM

llm = AnthropicLLM(model="claude-sonnet-4")
llm = GeminiLLM(model="gemini-2.0-flash")
```

## Token Usage Tracking

Each node tracks cumulative token costs:

```python
agent = Agent(llm, goal)
result = await agent.run()

# Check usage on any node
for node_id, node in agent.graph.nodes.items():
    usage = node.props.get("_usage", {})
    print(f"{node.label}: {usage.get('total_tokens')} tokens")
```

## Architecture

**Graph Model:**
- Every agent owns a `Graph` object
- Nodes have states: PENDING → READY → ACTIVE → RESOLVED | FAILED
- Edges define dependencies: REQUIRES (blocking), PART_OF (structural), PRODUCES (data)

**Scheduler:**
- Processes nodes in dependency order
- Auto-retries failed nodes with exponential backoff
- Persists state to FalkorDB on every mutation

**Backend:**
- FalkorDB stores all nodes, edges, messages as queryable graphs
- Each turn is a separate graph; sessions maintain history
- Semantic search finds similar prior problems for transfer learning

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for deep dive.

## Configuration

### Environment variables

```bash
export OPENAI_API_KEY=sk-...          # OpenAI
export ANTHROPIC_API_KEY=sk-ant-...   # Anthropic
export GOOGLE_API_KEY=AIzaSy...       # Gemini
export GROQ_API_KEY=gsk_...           # Groq
export FALKORDB_HOST=localhost        # FalkorDB (default: localhost:6379)
```

### Custom timeout

```python
llm = OpenAILLM(model="gpt-4o-mini", timeout=300.0)
```

## Tests

```bash
uv run pytest tests/ -v
uv run ruff check src/  # Linting
```

## Contributing

Pull requests welcome! Please:

1. Run `uv run ruff check src/ --fix` before committing
2. Add tests for new features
3. Update docs if behavior changes

## License

MIT — see [LICENSE](LICENSE)

## Links

- 📖 [Detailed Usage Guide](USAGE.md)
- 🏗️ [Architecture](docs/ARCHITECTURE.md)
- 🚀 [Publishing to PyPI](PUBLISHING.md)
- 📝 [Research Paper](docs/PAPER.md)
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
