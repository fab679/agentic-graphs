# Agentic Graphs

A graph-native framework for building AI agents with persistent memory, skill learning, and full traceability.

## Quick Start

```bash
# Math agent — single-agent arithmetic with tools
uv run python -m agentic_graphs.examples.math_agent

# Multi-agent demo — orchestrator + math + research subagents
uv run python -m agentic_graphs.examples.multi_agent_demo

# Chat session — multi-turn with FalkorDB persistence
uv run python -m agentic_graphs.examples.chat_session

# Agentic memory — learn user preferences as triplets
uv run python -m agentic_graphs.examples.agentic_memory_demo

# Skill learning — agent creates and reuses procedures
uv run python -m agentic_graphs.examples.skill_learning_demo

# Deep memory test — 8/8 verification of triplet + skill storage
uv run python -m agentic_graphs.examples.deep_memory_test

# LiteRT math agent — on-device Gemma 4 with tool calling
uv run python -m agentic_graphs.examples.litert_math_agent "What is 2 + 3?"
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.
See [PAPER.md](PAPER.md) for a detailed theoretical explanation.

## Core Concepts

**Graph-Native Reasoning:** The LLM is given graph mutation tools (`create_task`, `create_action`, `add_dependency`, `resolve_current_node`) that it calls to explicitly build a directed dependency graph of its own reasoning. A deterministic scheduler executes nodes in dependency order.

**Three-Layer Memory:**

| Layer | Storage | What It Stores |
|-------|---------|---------------|
| Chat History | In-memory + MESSAGE nodes | Prior conversation turns |
| Semantic Memory | EMBEDDING nodes in project graph | Past reasoning chains with vector embeddings |
| Agentic Memory | Entity triplets + Skill nodes in `memory:<user_id>` | User facts and learned procedures |

**Multi-Agent Patterns:** Subagents, Skills, Handoffs, and Router — all four patterns from the architecture blog post, plus Learned Skills (agent-created procedures).

## Persistence

FalkorDB-backed with three graph families:
- `project:<thread_id>` — execution graphs
- `meta:thread:<thread_id>` — conversation history
- `memory:<user_id>` — per-user agentic memory

## Examples

| Example | What It Demonstrates |
|---------|---------------------|
| `math_agent` | Single-agent arithmetic with tool calling |
| `multi_agent_demo` | Orchestrator + math/research subagents |
| `chat_session` | Multi-turn conversation with Session |
| `research_agent` | Custom ACTION guide with research tools |
| `agentic_memory_demo` | Triplet memory + cross-session recall |
| `skill_learning_demo` | Agent creates/reuses learned skills |
| `deep_memory_test` | Full 8/8 verification of memory storage |
| `litert_math_agent` | On-device Gemma 4 math with tool calling |
