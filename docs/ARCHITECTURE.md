# Agentic Graphs — Architecture

## Overview

Agentic Graphs is a graph-native framework for building AI agents. Instead of treating an LLM as a black box that plans in its head and executes tools, the framework makes the *reasoning process itself* a first-class graph structure. Every thought, every tool call, every intermediate result is a node in a directed dependency graph. The LLM mutates this graph explicitly; the scheduler executes it deterministically.

This gives four properties that prompt-based agents lack:

- **Traceability** — every reasoning step is a node with known state and output
- **Determinism** — the scheduler processes nodes in dependency order, not LLM whim
- **Reusability** — past reasoning chains are stored as vector-indexed embeddings
- **Learnability** — the agent creates skills from its own procedures and stores facts as queryable triplets

---

## The Graph

A single directed property graph is the single source of truth. It lives in memory during execution and is persisted to FalkorDB for cross-turn memory.

### Node Types

| Type | Purpose | Created By |
|------|---------|-----------|
| `GOAL` | A user request or top-level objective | Agent bootstrap |
| `TASK` | A unit of decomposable work | `create_task()` |
| `ACTION` | A concrete tool invocation | `create_action()` |
| `SYNTHESIS` | Combines results of tasks | `create_synthesis_node()` |
| `MESSAGE` | An LLM message in a turn transcript | Agent internals |
| `TOOLCALL` | A tool invocation record | Agent internals |
| `EMBEDDING` | A vector embedding of a resolved node | `store_embedding()` |
| `Entity` | A fact subject/object in user memory | `store_triplet()` |
| `Skill` | A reusable procedure stored in memory | `create_skill()` |

### Edge Types

| Type | Semantics | Direction |
|------|-----------|-----------|
| `REQUIRES` | src depends on dst | dependent → prerequisite |
| `PART_OF` | src is a child of dst | child → parent |
| `PRODUCES` | src produced dst (output) | producer → product |
| `HAS_MSG` | node has an LLM message transcript | node → message |
| `NEXT` | message ordering | earlier → later |
| `CALLED` | message invoked a tool | message → toolcall |
| `RETURNED` | tool returned a result | result → toolcall |
| `SEEN_BEFORE` | GOAL linked to a semantically similar past node | current → past |
| `OF` | embedding node points to its original node | embedding → original |
| *dynamic* | user-defined predicates in memory triplets | Entity → Entity |

### Node States

```
PENDING → READY → ACTIVE → RESOLVED
                         ↘ FAILED
```

A node starts PENDING. When all its REQUIRES prerequisites are RESOLVED, it becomes READY. The scheduler picks READY nodes and calls `process_node`, which sets state to ACTIVE, invokes the LLM, and sets RESOLVED or FAILED.

---

## The Mutation Pipeline

The core innovation: instead of asking the LLM to produce a plan as JSON text (which must be parsed and executed), the LLM is given *graph mutation tools* that it calls directly. The LLM builds the graph as it reasons.

### Tools by Node Type

| Node Type | Available Tools |
|-----------|---------------|
| GOAL | `create_task`, `create_synthesis_node`, `add_dependency`, *memory tools*, *skill tools* |
| TASK | `create_action`, `add_dependency`, `resolve_current_node` |
| ACTION | *domain tools* (user-provided), `resolve_current_node` |
| SYNTHESIS | `resolve_current_node` |

### The Scheduler Loop

```
for each pass:
  ready = graph.ready()              # nodes whose deps are all resolved
  if not ready: break
  for each node in ready:            # parallel via asyncio.gather
    process_node(node)               # LLM + tool loop
  # new nodes created by LLM may be ready next pass
```

The scheduler is stateless between passes — it only reads the graph. This makes it deterministic and testable.

---

## Agentic Memory

Three layers of memory, each in its own FalkorDB graph:

### 1. Chat History (message transcript)
- Stored as `MESSAGE` and `TOOLCALL` nodes in the project graph
- Linked via `HAS_MSG`, `NEXT`, `CALLED`, `RETURNED` edges
- Injected as `history_messages` in the LLM context

### 2. Semantic Memory (past reasoning patterns)
- Every resolved node (GOAL, TASK, ACTION, SYNTHESIS) gets an `EMBEDDING` node
- Stored in the project graph: `project:<thread_id>`
- Connected to the original node via `OF` edge
- Before a new GOAL, vector search finds similar past embeddings
- Matches show the node LABEL (what was asked) — NOT the output (which may contain stale tool results)
- `SEEN_BEFORE` edges link current GOALs to past matches, meaning "same reasoning approach"
- **Crucially**: results from tool calls (weather, stock prices, live data) are dynamic and may change between runs. Reusing the OUTPUT is dangerous. Reusing the LOGIC — how the problem was decomposed, what tools were called — is valuable. SEEN_BEFORE represents reasoning pattern similarity, not answer caching.

### 3. Agentic Memory (user facts + skills)
- Per-user graph: `memory:<user_id>`
- **Triplets**: `(:Entity)-[:PREDICATE]->(:Entity)` with vector embeddings
  - `store_triplet(subject, predicate, object_)` — agent learns facts
  - `query_triplets(subject, predicate, object_)` — retrieve with wildcards
  - `search_memory(query)` — semantic search across entities
  - `get_entity_graph(entity_name)` — explore connections
- **Skills**: `(:Skill {name, description, procedure, embedding})`
  - `create_skill(name, description, procedure)` — agent saves procedures
  - `search_skills(query)` — find relevant skills
  - `activate_learned_skill(name)` — load skill context into prompt
- Auto-injected into GOAL prompts via vector search

---

## Multi-Agent Patterns

The framework implements all four patterns from the blog post:

| Pattern | Implementation | When to Use |
|---------|---------------|-------------|
| **Subagents** | `register_subagent()` — parent calls child as a tool; child runs in its own GOAL node in the shared graph | Multiple distinct domains, need parallel execution |
| **Skills** | `register_skill()` + `activate_skill_*` tool — progressive prompt/tool disclosure | Single agent with many specializations |
| **Handoffs** | `handoff_to(target)` tool — state-driven agent switching | Sequential workflows with state transitions |
| **Router** | `RouterAgent` — parallel dispatch + synthesis across specialized agents | Distinct verticals, parallel queries |
| **Learned Skills** | `create_skill()` + `activate_learned_skill()` — agent-created skills | Dynamic procedure learning from experience |

---

## Persistence (FalkorDB)

Three graph families in FalkorDB:

| Graph Pattern | Contents | Lifespan |
|--------------|----------|----------|
| `project:<thread_id>` | All GOAL/TASK/ACTION/SYNTHESIS/MESSAGE/TOOLCALL nodes + EMBEDDING nodes | Per thread |
| `meta:thread:<thread_id>` | THREAD → TURN conversation metadata | Permanent |
| `memory:<user_id>` | Entity triplets + Skill nodes | Permanent per user |

All writes use incremental `MERGE`/`SET` — no full-graph sync (which would DETACH DELETE prior data).

Vector indexes are dropped and recreated before each search because FalkorDB's HNSW index is static (does not auto-index data added after creation).

---

## Embedding Providers

| Provider | Embedding Model | Config |
|----------|----------------|--------|
| `OpenAILLM` | `text-embedding-3-small` (1536d) | `embed_model` param |
| `GeminiLLM` | `models/embedding-001` | `embed_model` param |
| `OllamaLLM` | `/api/embed` on any model | `embed_model` param |
| Others | `NotImplementedError` | Use `embedder=` callable |

The Agent accepts a separate `embedder` callable to decouple chat LLM from embedding model:

```python
agent = Agent(
    OpenAILLM("gpt-4o"),
    goal,
    embedder=OllamaLLM(model="nomic-embed-text").embed,
)
```
