# Usage Guide

Comprehensive guide to building agents with agentic-graphs.

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Your First Agent](#your-first-agent)
3. [Custom Tools](#custom-tools)
4. [Multi-Turn Chat Sessions](#multi-turn-chat-sessions)
5. [Multi-Agent Patterns](#multi-agent-patterns)
6. [Error Handling & Retries](#error-handling--retries)
7. [Performance & Costs](#performance--costs)
8. [Debugging](#debugging)

---

## Installation & Setup

### Install the Package

```bash
# Stable (pip)
pip install agentic-graphs

# Development (uv)
uv add agentic-graphs
```

### Install FalkorDB (Required for state persistence)

FalkorDB is a graph database that backs all agent execution state. It's optional, but **strongly recommended**.

```bash
# Docker (easiest)
docker run -d -p 6379:6379 -p 3000:3000 --name falkordb falkordb/falkordb:latest

# Then open http://localhost:3000 to see your agent graphs in real-time
```

If you don't have Docker, see [FalkorDB docs](https://www.falkordb.com) for installation options.

### Set API Keys

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
export GOOGLE_API_KEY=AIzaSy...

# Groq
export GROQ_API_KEY=gsk_...
```

Or pass them directly:

```python
from agentic_graphs import OpenAILLM

llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-...")
```

---

## Your First Agent

### 1. Define Your Agent

The simplest agent needs:
- An LLM provider (OpenAI, Anthropic, etc.)
- A goal (what you want it to do)
- Optional tools (functions it can call)

```python
import asyncio
from agentic_graphs import Agent, OpenAILLM

class SimpleAgent(Agent):
    """A basic agent that answers questions."""
    pass

async def main():
    llm = OpenAILLM(model="gpt-4o-mini")
    agent = SimpleAgent(llm, goal="What is the capital of France?")
    result = await agent.run()
    print(f"Answer: {result}")

asyncio.run(main())
```

**Output:**
```
Answer: The capital of France is Paris.
```

### 2. Run with Progress Feedback

```python
class SimpleAgent(Agent):
    def init_node_props(self, node, **kwargs):
        props = super().init_node_props(node, **kwargs)
        print(f"→ {node.type}: {node.label}")
        return props

agent = SimpleAgent(llm, "What was the first landing on the moon?")
result = await agent.run()
```

**Output:**
```
→ GOAL: What was the first landing on the moon?
→ TASK: Research the first moon landing
→ ACTION: Use available tools to retrieve information
→ SYNTHESIS: Generate a comprehensive answer
Answer: The first moon landing was Apollo 11 on July 20, 1969...
```

### 3. Understand the Execution Model

Every agent execution follows:

```
GOAL (user request)
  ↓
TASK (decomposition)
  ↓
ACTION (tool calls, mutations)
  ↓
SYNTHESIS (final answer)
```

- **GOAL** — the raw user request
- **TASK** — agent breaks down goal into concrete steps
- **ACTION** — agent calls tools or external APIs
- **SYNTHESIS** — agent summarizes findings into a response

Each node is stored in FalkorDB and queryable at http://localhost:3000.

---

## Custom Tools

### Add Tools to Your Agent

Tools are functions that your agent can call. Define them with the `@tool` decorator:

```python
from agentic_graphs import Agent, OpenAILLM, tool

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

class MathAgent(Agent):
    """Agent with math tools."""
    def build_tools(self, node):
        schemas, impls = super().build_tools(node)
        
        # Add custom tools
        schemas += [add.schema, multiply.schema]
        impls.update({"add": add, "multiply": multiply})
        
        return schemas, impls

async def main():
    llm = OpenAILLM(model="gpt-4o-mini")
    agent = MathAgent(llm, "What is (17 * 3 + 42) / 9?")
    result = await agent.run()
    print(f"Answer: {result}")

asyncio.run(main())
```

**Output:**
```
Answer: The answer is 7. Here's how: 17 * 3 = 51, 51 + 42 = 93, and 93 / 9 = 7.
```

### Type-Safe Tool Definitions

Always include **type hints** and **docstrings**. The agent uses these to understand what your tool does:

```python
@tool
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web for information.
    
    Args:
        query: What to search for
        max_results: How many results to return (max 10)
    
    Returns:
        List of search results with 'title', 'url', 'snippet'
    """
    # Implementation
    return [{"title": "...", "url": "...", "snippet": "..."}]
```

### Built-in Tools

Every agent automatically has these tools:

| Tool | Args | Returns | Purpose |
|------|------|---------|---------|
| `create_task` | `label: str, description: str` | `task_id: str` | Create a TASK node |
| `create_action` | `label: str, instruction: str` | `action_id: str` | Create an ACTION node |
| `resolve_current_node` | `output: str` | `"success"` | Mark current node as RESOLVED |
| `add_dependency` | `waiting_node_id: str, prereq_node_id: str` | `"added"` | Create REQUIRES edge |
| `get_token_usage` | (none) | `{prompt_tokens, completion_tokens, total_tokens}` | Get cumulative token costs |

Example:

```python
class DecomposedAgent(Agent):
    async def on_goal(self, goal_node, graph):
        # Manually create subtasks
        task_id = await self._call_tool(
            "create_task",
            label="Research topic",
            description="Find credible sources",
        )
        return f"Created task {task_id}"
```

---

## Multi-Turn Chat Sessions

### Create a Persistent Session

Chat sessions preserve history across turns. The agent automatically has access to prior context.

```python
from agentic_graphs.session import Session
from agentic_graphs.core.falkordb_backend import FalkorDBBackend
from agentic_graphs import OpenAILLM

class ChatAgent(Agent):
    """Simple question-answering agent."""
    pass

async def main():
    # Create a new session
    session = await Session.create(
        llm=OpenAILLM(model="gpt-4o-mini"),
        agent_class=ChatAgent,
        backend=FalkorDBBackend(host="localhost", port=6379),
        thread_name="math-session",
        user_id="alice",
    )

    # Turn 1
    reply = await session.chat("What is 5 * 8?")
    print(f"Assistant: {reply}")
    
    # Turn 2 (has full context from Turn 1)
    reply = await session.chat("Add 10 to that result")
    print(f"Assistant: {reply}")

asyncio.run(main())
```

**Output:**
```
Assistant: 5 * 8 = 40
Assistant: 40 + 10 = 50
```

### Resume a Session Later

```python
# Save this after first session
thread_id = session.thread_id

# Later, resume the same session
session = await Session.create(
    llm=OpenAILLM(model="gpt-4o-mini"),
    agent_class=ChatAgent,
    backend=FalkorDBBackend(),
    thread_id=thread_id,  # Resume by ID
)

# Continue the conversation
reply = await session.chat("What was my first question?")
# Assistant has full context from earlier turns
```

---

## Multi-Agent Patterns

### Pattern 1: Subagents (Delegation)

Have one agent delegate to specialized subagents:

```python
from agentic_graphs import Agent, OpenAILLM

class MathSubagent(Agent):
    """Specialist in math."""
    def build_tools(self, node):
        schemas, impls = super().build_tools(node)
        schemas += [add.schema, multiply.schema]
        impls.update({"add": add, "multiply": multiply})
        return schemas, impls

class ResearchSubagent(Agent):
    """Specialist in research."""
    def build_tools(self, node):
        schemas, impls = super().build_tools(node)
        schemas += [search_web.schema]
        impls.update({"search_web": search_web})
        return schemas, impls

class OrchestratorAgent(Agent):
    """Top-level agent that delegates."""
    
    def __init__(self, llm, goal, **kwargs):
        super().__init__(llm, goal, **kwargs)
        # Register subagents
        self.register_subagent(
            name="math_expert",
            agent_class=MathSubagent,
            description="Use for math calculations",
        )
        self.register_subagent(
            name="researcher",
            agent_class=ResearchSubagent,
            description="Use for researching facts",
        )

async def main():
    llm = OpenAILLM(model="gpt-4o-mini")
    agent = OrchestratorAgent(llm, "How many people watched the Olympics? Calculate total TV revenue.")
    result = await agent.run()
    print(result)

asyncio.run(main())
```

When the OrchestratorAgent encounters a math question, it automatically delegates to MathSubagent. For research, it delegates to ResearchSubagent.

### Pattern 2: Skills (Progressive Disclosure)

Register domain-specific skill sets that the agent can activate:

```python
class FinancialAgent(Agent):
    def __init__(self, llm, goal, **kwargs):
        super().__init__(llm, goal, **kwargs)
        
        # Stock trading skill
        self.register_skill(
            name="stock_trading",
            description="Access to real-time stock quotes and trading",
            tools=[get_stock_price.schema, place_trade.schema],
            tool_fns={"get_stock_price": get_stock_price, "place_trade": place_trade},
        )
        
        # Portfolio analysis skill
        self.register_skill(
            name="portfolio_analysis",
            description="Analyze portfolio allocation and risk",
            tools=[analyze_portfolio.schema],
            tool_fns={"analyze_portfolio": analyze_portfolio},
        )

async def main():
    llm = OpenAILLM(model="gpt-4o-mini")
    agent = FinancialAgent(llm, "Should I buy Tesla stock? What's my portfolio risk?")
    result = await agent.run()

asyncio.run(main())
```

### Pattern 3: Handoffs (State-Driven Switching)

Switch to a different agent based on the current state:

```python
class SmartRouter(Agent):
    async def on_task(self, task_node, graph):
        task_label = task_node.label.lower()
        
        if "math" in task_label or "calculate" in task_label:
            # Switch to math specialist
            math_agent = MathSubagent(self.llm, task_node.label)
            return await math_agent.run()
        
        elif "research" in task_label or "find" in task_label:
            # Switch to research specialist
            research_agent = ResearchSubagent(self.llm, task_node.label)
            return await research_agent.run()
        
        else:
            # Default processing
            return f"Processing: {task_node.label}"
```

---

## Error Handling & Retries

### Automatic Retries

The agent automatically retries failed nodes with exponential backoff:

```python
class RobustAgent(Agent):
    def __init__(self, llm, goal, **kwargs):
        # Max 3 retries, 2s exponential backoff
        super().__init__(llm, goal, max_tool_retries=3, **kwargs)

async def main():
    agent = RobustAgent(OpenAILLM(model="gpt-4o-mini"), goal)
    result = await agent.run()  # Retries automatically on tool failure
```

### Handle Tool Failures Gracefully

```python
@tool
def risky_operation(data: str) -> str:
    """Operation that might fail."""
    if not data:
        raise ValueError("Empty input!")
    return f"Processed: {data}"

class SafeAgent(Agent):
    def build_tools(self, node):
        schemas, impls = super().build_tools(node)
        schemas.append(risky_operation.schema)
        impls["risky_operation"] = risky_operation
        return schemas, impls
```

The agent will catch the error, note it in the graph, and optionally retry.

### Check Node Status

```python
result = await agent.run()

# Inspect which nodes failed
for node_id, node in agent.graph.nodes.items():
    if node.state == "FAILED":
        print(f"Failed: {node.label}")
        print(f"  Error: {node.props.get('error')}")
```

---

## Performance & Costs

### Monitor Token Usage

Every node tracks its token cost. Check total usage after a run:

```python
result = await agent.run()

total_tokens = 0
for node_id, node in agent.graph.nodes.items():
    usage = node.props.get("_usage", {})
    node_tokens = usage.get("total_tokens", 0)
    total_tokens += node_tokens
    print(f"{node.label}: {node_tokens} tokens")

print(f"\nTotal: {total_tokens} tokens")
```

**Estimate Cost:**

- OpenAI gpt-4o: $0.005 per 1K prompt tokens, $0.015 per 1K output tokens
- Anthropic Claude 3 Opus: $0.015 per 1K prompt tokens, $0.075 per 1K output tokens
- Gemini 2.0 Flash: $0.075 per 1M input tokens, $0.3 per 1M output tokens

### Reduce Token Usage

1. **Use smaller models** (gpt-4o-mini, claude-haiku)
2. **Limit context** — trim session history after N turns
3. **Cache tool schemas** — don't regenerate them per node
4. **Batch requests** — run multiple agents in parallel

```python
# Use a faster, cheaper model
llm = OpenAILLM(model="gpt-4o-mini")

# Or use a local model
from agentic_graphs import OllamaLLM
llm = OllamaLLM(model="mistral")  # Runs on your machine, free
```

### Parallel Execution

```python
import asyncio
from agentic_graphs import Agent, OpenAILLM

llm = OpenAILLM(model="gpt-4o-mini")

agents = [
    Agent(llm, f"What is {i}^2?" for i in range(1, 11))
]

results = await asyncio.gather(*[agent.run() for agent in agents])
```

---

## Debugging

### Enable Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("agentic_graphs")
logger.setLevel(logging.DEBUG)

# Now run your agent
agent = Agent(llm, goal)
result = await agent.run()
```

### Inspect the Graph

Open FalkorDB UI at http://localhost:3000:

1. Select your thread
2. Click a node to see its properties
3. Hover over edges to see dependencies
4. Run Cypher to query the graph

**Example Cypher query:**

```cypher
MATCH (n) WHERE n.type = "TASK" RETURN n.label, n.state
```

### Export Graph State

```python
import json

# After agent.run()
graph_state = {
    "nodes": {
        node_id: {
            "label": node.label,
            "type": node.type,
            "state": node.state,
            "props": node.props,
        }
        for node_id, node in agent.graph.nodes.items()
    },
    "edges": [
        {
            "source": edge.source_id,
            "target": edge.target_id,
            "type": edge.type,
        }
        for edge in agent.graph.edges
    ]
}

with open("graph.json", "w") as f:
    json.dump(graph_state, f, indent=2)
```

### Debug Session Script

Use the debug session script to analyze any recorded thread:

```bash
# List all threads
uv run python -m agentic_graphs.examples.debug_session

# Inspect a specific thread
uv run python -m agentic_graphs.examples.debug_session --thread <thread-id>

# Export raw graph data
uv run python -m agentic_graphs.examples.debug_session --raw
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Agent loops endlessly | Set `max_tool_iters=5` (default: 10) |
| Tools not called | Ensure tool has proper docstring and type hints |
| FalkorDB not found | Run `docker run -d -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest` |
| High token usage | Switch to smaller model (gpt-4o-mini, claude-haiku) or local model (Ollama) |
| Agent gets stuck on ACTION | Check tool implementation for infinite loops |

---

## Next Steps

- **[README.md](README.md)** — Quick start and core concepts
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Deep dive on graph model and scheduler
- **[examples/](src/agentic_graphs/examples/)** — Runnable examples with different patterns
- **[API Docs](https://github.com/fab679/agentic-graphs)** — Full API reference

**Questions?** File an issue on GitHub or check existing discussions.
