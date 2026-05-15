# Graphs Are All You Need: A Graph-Native Framework for Agentic Reasoning

## Abstract

We present a graph-native architecture for building AI agents where the reasoning process itself is a first-class directed graph. Unlike conventional agent frameworks that treat LLMs as opaque planners which produce opaque text, our approach gives the LLM direct graph mutation tools — `create_task`, `create_action`, `add_dependency`, `resolve_current_node` — that it calls to explicitly construct a dependency graph of its own reasoning. A deterministic scheduler then executes nodes in dependency order. The result is a system with full traceability, deterministic execution, and a persistent memory that the agent reads from and writes to across conversations.

---

## 1. Introduction

### 1.1 The Problem with Prompt-Based Agents

Contemporary AI agent frameworks share a common design: an LLM is given a system prompt describing available tools, asked to produce a plan (often as JSON), and a runtime parses that plan and executes the tools. This approach has three fundamental problems:

1. **Opacity** — The LLM's reasoning is hidden inside a text generation. There is no structured trace of why a particular tool was called, in what order, or what the intermediate results were.

2. **Non-Determinism** — The same prompt can produce completely different plans on different calls. There is no mechanism to enforce consistent execution order.

3. **No Memory** — Each interaction starts from scratch. The agent cannot learn from past reasoning chains or build a persistent knowledge base.

### 1.2 The Graph-Native Alternative

Instead of asking the LLM to *describe* a plan that a runtime *interprets*, we give the LLM tools to *build* a graph. Every reasoning step becomes a node. Every dependency becomes an edge. The LLM does not tell a runtime what to do — the LLM *does it*, mutating the graph directly.

This is not a new idea. It is how humans reason on paper: we write down what we know, draw arrows between related ideas, and the structure of the diagram guides our thinking. The graph is not a description of the reasoning — it *is* the reasoning.

---

## 2. The Graph Model

### 2.1 Nodes as Reasoning Steps

Every node in the graph represents a discrete unit of reasoning:

- A **GOAL** is a user request. It is the root of the reasoning tree.
- A **TASK** is a subproblem that must be solved. It decomposes a goal into pieces.
- An **ACTION** is a concrete operation — a tool call, a calculation, a database query.
- A **SYNTHESIS** node combines the outputs of tasks into a coherent result.

Each node has a **state** that progresses deterministically: PENDING → READY → ACTIVE → RESOLVED (or FAILED).

### 2.2 The Dependency Contract

Edges encode dependencies. A `REQUIRES` edge from node A to node B means "A depends on B". The scheduler enforces this contract strictly: a node becomes READY only when all its REQUIRES prerequisites are RESOLVED.

This is the key insight: **the LLM defines the dependencies, but the scheduler enforces them**. The LLM can create any graph structure it likes, but execution order is determined by the graph, not by the LLM's next token.

### 2.3 Why Not DAGs?

We use a general directed graph (not a DAG) because cycles are informative: they indicate that the LLM is going in circles. The framework detects cycles via DFS and rejects edges that would create them, providing immediate feedback to the LLM. A DAG-based system would silently allow the LLM to create deadlock.

---

## 3. The Mutation Pipeline

### 3.1 Tools as Graph Operations

The LLM is given four primitive graph operations as function calls:

- **`create_task(label)`** — Adds a TASK node and connects it to the current GOAL via a PART_OF edge.
- **`create_action(label, instruction)`** — Adds an ACTION node with an instruction, connects it to the current TASK via REQUIRES, and sets the TASK back to PENDING (it must wait for the action to complete).
- **`add_dependency(waiting, prereq)`** — Creates a REQUIRES edge, with cycle detection.
- **`resolve_current_node(output)`** — Marks the current node as RESOLVED with the given output, blocked if unresolved REQUIRES prerequisites exist.

These four operations are sufficient to represent any chain of reasoning. Additional domain tools (search, calculate, database queries) are added to ACTION nodes only.

### 3.2 The LLM as Graph Builder

When an LLM processes a GOAL node, it receives:

1. The node's type and label
2. The graph neighborhood (neighboring nodes and their states/outputs)
3. Its available tools (mutation operations + domain tools + memory tools)
4. Conversation history

The LLM then enters a tool-calling loop. In each iteration it may call one or more tools, receive results, and decide whether to continue or respond. The loop terminates when the LLM outputs text without tool calls, or when a maximum iteration count is reached.

### 3.3 The Scheduler

The scheduler is a single loop that runs in passes:

1. Find all READY nodes (dependencies resolved, not yet processed)
2. Process them in parallel via `asyncio.gather`
3. Repeat until no READY nodes remain

Processing a node means: set state to ACTIVE, call the LLM with the node's context, wait for the LLM to finish (including any tool calls), set state to RESOLVED with the LLM's output.

The scheduler is stateless between passes. It reads only the graph, which the LLM mutates. This separation of concerns is critical: the scheduler provides deterministic execution guarantees, while the LLM provides flexible reasoning.

---

## 4. Memory Architecture

A framework where every interaction starts with an empty graph is not agentic. We need memory — persistent, queryable, learnable.

### 4.1 Three Layers of Memory

**Chat History.** Each LLM turn produces a transcript of messages and tool calls, stored as MESSAGE and TOOLCALL nodes linked to the GOAL. This transcript is injected as context on subsequent turns, enabling the LLM to reference its own prior reasoning.

**Semantic Memory.** Every resolved node (GOAL, TASK, ACTION, SYNTHESIS) is embedded using text-embedding-3-small and stored as an EMBEDDING node in the project graph. Before processing a new GOAL, the system performs a vector search for semantically similar past nodes. Matches are injected into the prompt as "SIMILAR PROBLEMS SEEN BEFORE", and SEEN_BEFORE edges link the current GOAL to past matches. The revisit count on these edges provides an importance signal.

A critical design decision: SEEN_BEFORE represents *reasoning pattern similarity*, not answer caching. Tool call results (weather data, stock prices, live API responses) are inherently dynamic — the same tool called twice may return different values. Reusing the OUTPUT of a past reasoning chain is dangerous because the result may be stale. Reusing the LOGIC — how the problem was decomposed, which tools were called, in what order — is always safe and valuable. Therefore, semantic memory injection shows only the past node's LABEL (what was asked) and TYPE, never its OUTPUT. The agent sees "this type of problem was solved before with this structure" and re-applies the approach, getting fresh results.

**Agentic Memory.** Each user has a dedicated memory graph where the agent stores:

- **Triplets** — Subject-predicate-object facts (`Alice --[PREFERS]--> metric units`) stored as Entity nodes connected by typed relationships. The agent calls `store_triplet()` when it learns something and `query_triplets()` when it needs to recall.

- **Skills** — Procedural knowledge saved as Skill nodes with name, description, and step-by-step instructions. The agent calls `create_skill()` after solving a multi-step problem and `search_skills()` / `activate_learned_skill()` on subsequent similar problems.

### 4.2 Why Triplets, Not Key-Value Stores

Key-value stores flatten knowledge into isolated pairs. Triplets create a graph — entities are nodes, relationships are edges, and the graph can be traversed. This enables:

- **Transitive reasoning**: If Alice prefers metric units and metric units are used in Europe, the agent can infer Alice might like Europe.
- **Entity-centric retrieval**: `get_entity_graph("Alice")` returns everything connected to Alice, regardless of predicate type.
- **Schema evolution**: The agent can create new relationship types as needed, without schema migration.

### 4.3 Why Vector Indexes

Not all memory retrieval is exact. When the user asks "what do you know about me?", the agent needs a semantic search, not a key lookup. Vector embeddings on Entity and Skill nodes enable fuzzy retrieval: the query "what are my measurements?" can find "8 kg --[IS_EQUAL_TO]--> 8000 grams" even though the words don't match.

---

## 5. Multi-Agent Patterns

### 5.1 Subagents

A parent agent registers specialized children via `register_subagent()`. When the parent calls a subagent tool, the subagent creates its own GOAL node in the shared graph and runs its own scheduler within that scope. The parent's graph and the subagent's graph are the same object — there is no data duplication. Scope isolation is achieved by restricting the scheduler to descendants of the subagent's GOAL via `_descendants()`.

### 5.2 Skills (Static)

Skills are registered at construction time as prompt/tool bundles. The agent knows only their names and descriptions until it calls `activate_skill_<name>`, which swaps the current node's prompt and toolset. This is progressive disclosure in the blog post's sense.

### 5.3 Skills (Learned)

Learned skills extend the static skill pattern: the agent creates them dynamically via `create_skill()` after solving a problem. The skill is persisted to the user's memory graph and found via vector search on subsequent turns. This is the key to continuous learning — the agent does not need a developer to define its skills upfront.

### 5.4 Handoffs

The `handoff_to(target)` tool sets a flag that the Session reads to determine the next agent. This enables state-driven workflows where different agents handle different stages of a process.

### 5.5 Router

The RouterAgent pattern (parallel dispatch + synthesis) is implemented as a built-in Agent subclass. It decomposes the query, dispatches to specialized subagents in parallel, and synthesizes their outputs.

---

## 6. Persistence Model

### 6.1 FalkorDB Integration

The framework uses FalkorDB (a Redis-backed graph database with Cypher query support) for persistence. Three graph families coexist:

- `project:<thread_id>` — Execution graphs with GOAL, TASK, ACTION, SYNTHESIS, EMBEDDING nodes, plus MESSAGE/TOOLCALL transcripts for traceability.
- `meta:thread:<thread_id>` — THREAD and TURN nodes for conversation history, used by the Session abstraction.
- `memory:<user_id>` — Entity triplets and Skill nodes for agentic memory.

### 6.2 Incremental Persistence

The framework does full-graph sync only on the first write. Subsequent writes use incremental `MERGE` and `SET` operations, which update individual nodes and edges without touching unrelated data. This is essential for the single-project-graph model, where a sync would DETACH DELETE prior turns.

### 6.3 Vector Index Management

FalkorDB's HNSW vector index is static — data added after index creation is not indexed. The framework works around this by dropping and recreating the vector index before each search. For small graphs (the common case in agentic applications), this rebuild is effectively instant.

---

## 7. Design Decisions and Tradeoffs

### 7.1 Why a Graph, Not a List

A list of steps implies total ordering. A graph captures partial ordering — some steps can happen in parallel, some depend on others. The LLM expresses parallelism naturally by creating independent TASK nodes, and the scheduler exploits it via asyncio.gather.

### 7.2 Why Not JSON Plans

JSON plans require a parser and executor. The parser adds a failure mode (malformed JSON), and the executor adds indirection (the LLM's intent is separated from execution by a serialization layer). Graph mutation tools eliminate both: the LLM calls Python functions directly, and the framework provides them.

### 7.3 Why Not a Single LLM Call

A single LLM call produces a flat response. Multi-step reasoning requires multiple calls, each building on the previous. The graph captures this naturally — each node is one LLM call, and the edges encode the dependencies between calls.

### 7.4 The Cost of Structure

Graph nodes and edges consume tokens: the graph neighborhood is included in every LLM prompt, and the mutation tool descriptions occupy space in the tool list. For very simple queries (e.g., "what is 2+2?"), this overhead is larger than a free-text response. The framework compensates by allowing the LLM to respond directly (no tool calls) for simple cases, bypassing the graph entirely.

### 7.5 The Learning Overhead

Memory vector searches, embedding computations, and index rebuilds add latency and cost. For a single-turn interaction, these are pure overhead. The benefit accrues over time: each interaction enriches the graph, and future interactions leverage the accumulated knowledge. The framework is designed for persistent agents, not one-shot queries.

---

## 8. Conclusion

We have presented a graph-native architecture for agentic AI that makes reasoning itself a first-class graph structure. The LLM builds the graph explicitly via mutation tools; the scheduler executes it deterministically; the memory layer persists and retrieves knowledge across conversations.

The framework's key insight is that a graph is not a data structure for storing the *results* of reasoning — it is the reasoning itself. Every node, every edge, every state transition is a trace of the agent's thought process. By making this trace explicit, queryable, and persistent, we enable agents that can explain their reasoning, learn from experience, and improve over time.

Unlike conventional agent frameworks that hide reasoning inside opaque text generation, our approach provides full transparency, deterministic execution, and continuous learning — properties that are essential for production AI systems.
