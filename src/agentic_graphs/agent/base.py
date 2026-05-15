"""
agent/base.py — Agent base class.

Design:
  An Agent owns a single Graph and drives it to completion via an LLM
  using a built-in mutation tool pipeline (GOAL -> TASK -> ACTION -> SYNTHESIS).

  The mutation pipeline is the framework's core differentiator:
  instead of asking the LLM to plan in its head, it provides *graph mutation
  tools* (create_task, create_action, add_dependency, resolve_current_node)
  that the LLM calls to explicitly build a directed dependency graph.
  The scheduler then executes nodes in dependency order, reducing
  hallucination and providing full traceability.

  Multi-agent patterns (Subagents, Skills, Handoffs, Router):
  All agents share a single graph — children inherit the parent's graph
  context, providing shared memory, full traceability, and no data
  duplication.  Four patterns from the blog post are built in:

    register_subagent(name, agent_class, ...)   — Subagents pattern
      Parent calls child agents as tools. Child runs in its own sub-GOAL
      node within the shared graph, linked via PART_OF.

    register_skill(name, prompt, tools)         — Skills pattern
      Progressive prompt/tool disclosure loaded on demand via a tool call.

    handoff_to(target)                          — Handoffs pattern
      State-driven agent switching across turns.

    RouterAgent subclass                        — Router pattern
      Parallel dispatch + synthesis across specialized agents.

  Key defaults (override via subclass):
    build_prompt(node, graph) -> str   — uses default guides per node type
    build_tools(node)         -> tuple  — mutation tools + user's extra tools

  Subclasses that need custom domain tools just pass them at construction::

      agent = MyAgent(llm, goal, extra_action_tools={"search": search_fn},
                      extra_action_schemas=[search_fn.schema])

  Streaming: set ``on_token`` to receive content tokens as they stream in.

  Node lifecycle: PENDING -> READY -> ACTIVE -> RESOLVED | FAILED
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Callable

from agentic_graphs.core.graph import Graph, Node, Edge, S, NT, ET, _uid
from agentic_graphs.llm.base import LLM, Message
from agentic_graphs.agent.defaults import default_build_prompt, default_build_tools

log = logging.getLogger(__name__)


def _snap(s: str, n: int = 60) -> str:
    """Truncate *s* to *n* chars for logging."""
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "..."


class Agent:
    """Graph-native agent with built-in mutation tool pipeline.

    Constructor args:
        llm:                  Any LLM provider (OpenAI, Anthropic, Gemini, ...).
        goal:                 The user's request — becomes the root GOAL node label.
        graph:                Optional pre-built Graph (skip bootstrapping).
        graph_name:           FalkorDB graph key for this turn's execution graph.
        max_iterations:       Scheduler loop cap.
        history_messages:     Prior conversation turns (injected by Session).
        on_token:             Callable receiving content tokens as they stream in.
        extra_action_tools:   {name: callable} for ACTION nodes.
        extra_action_schemas: OpenAI schema dicts for ACTION nodes.

    Override points:
        build_prompt(node, graph)  — custom per-node prompts/guides
        build_tools(node)          — custom tool pipeline per node type
    """

    def __init__(
        self,
        llm: LLM,
        goal: str,
        graph: Graph | None = None,
        graph_name: str | None = None,
        max_iterations: int = 50,
        history_messages: list[Message] | None = None,
        on_token: Callable[[str], None] | None = None,
        extra_action_tools: dict[str, Callable] | None = None,
        extra_action_schemas: list[dict] | None = None,
        embedder: Callable[[str], list[float]] | None = None,
        user_id: str | None = None,
    ):
        self.llm = llm
        self.embedder = embedder
        self.user_id = user_id
        self.goal_text = goal
        self.graph = graph or Graph()
        self.graph_name = graph_name or f"agent_{uuid.uuid4().hex[:8]}"
        self.max_iterations = max_iterations
        self.history_messages: list[Message] = history_messages or []
        self._backend = None
        self.on_token = on_token
        self._extra_action_tools = extra_action_tools or {}
        self._extra_action_schemas = extra_action_schemas or []
        self._root_id: str | None = None

        # Multi-agent registries
        self._subagents: dict[str, dict] = {}
        self._skills: dict[str, dict] = {}
        self._active_skill: str | None = None
        self._handoff_target: str | None = None

        if not self.graph.nodes:
            self._bootstrap()

    # -- bootstrapping ---------------------------------------------------------

    def _bootstrap(self) -> None:
        """Seed the graph with a root GOAL node from the user's message."""
        goal_node = Node(id=_uid(), type=NT.GOAL, label=self.goal_text)
        self.graph.add_node(goal_node)
        self._root_id = goal_node.id
        log.info("Bootstrapped GOAL node %s: %s", goal_node.id[:8], _snap(self.goal_text))

    # -- persistence hooks -----------------------------------------------------

    def attach_backend(self, backend) -> None:
        """Attach a FalkorDBBackend for incremental persistence."""
        self._backend = backend
        log.info("Backend attached: %s", type(backend).__name__)

    def _persist_node(self, node: Node) -> None:
        if self._backend:
            try:
                self._backend.upsert_node(node, self.graph_name)
            except Exception as exc:
                log.warning("persist_node failed: %s", exc)

    def _persist_edge(self, edge: Edge, src_type: NT, dst_type: NT) -> None:
        if self._backend:
            try:
                self._backend.upsert_edge(edge, src_type, dst_type, self.graph_name)
            except Exception as exc:
                log.warning("persist_edge failed: %s", exc)

    # -- multi-agent: subagents, skills, handoffs ------------------------------

    def register_subagent(
        self,
        name: str,
        agent_class: type | None = None,
        llm: LLM | None = None,
        build_prompt: Callable | None = None,
        extra_action_tools: dict | None = None,
        extra_action_schemas: list | None = None,
        description: str = "",
    ) -> None:
        """Register a subagent that the parent LLM can invoke as a tool.

        When called, the subagent runs in a sub-GOAL node within the
        *shared graph*, linked to the parent via PART_OF.  All nodes
        created by the subagent are visible in the same FalkorDB graph,
        providing unified traceability.

        Patterns: Subagents (centralised orchestration).
        """
        self._subagents[name] = {
            "agent_class": agent_class or Agent,
            "llm": llm or self.llm,
            "build_prompt": build_prompt,
            "extra_action_tools": extra_action_tools or {},
            "extra_action_schemas": extra_action_schemas or [],
            "description": description or f"Delegate work to {name}",
        }

    def register_skill(
        self,
        name: str,
        prompt: str | None = None,
        tools: list | None = None,
        tool_fns: dict | None = None,
        description: str = "",
    ) -> None:
        """Register a skill that can be activated on-demand.

        Activating a skill swaps the node's prompt and tools for the
        duration of the current LLM turn.  Skills provide progressive
        disclosure — the agent knows they exist but only loads context
        when relevant.

        Patterns: Skills (progressive disclosure).
        """
        self._skills[name] = {
            "prompt": prompt or "",
            "tool_schemas": tools or [],
            "tool_fns": tool_fns or {},
            "description": description or f"Use skill {name}",
        }

    async def _run_subagent(
        self, name: str, task: str
    ) -> str:
        """Execute a registered subagent on *task* within the shared graph.

        Creates a sub-GOAL node linked to the parent via PART_OF, then runs
        a full scheduler for the subagent within the shared graph.  Every
        node the subagent creates (TASK, ACTION, SYNTHESIS) is processed
        using the subagent's own ``build_tools`` and ``build_prompt``.

        Returns the subagent's final output (SYNTHESIS > GOAL > last output).
        """
        cfg = self._subagents.get(name)
        if not cfg:
            return f"[unknown subagent: {name}]"

        goal_node = Node(id=_uid(), type=NT.GOAL, label=task)
        self.graph.add_node(goal_node)
        self._persist_node(goal_node)
        pe = Edge(id=_uid(), type=ET.PART_OF, src=goal_node.id, dst=self._root_id)
        self.graph.add_edge(pe)
        self._persist_edge(pe, self.graph.nodes[pe.src].type,
                          self.graph.nodes[pe.dst].type)

        agent = cfg["agent_class"](
            llm=cfg["llm"],
            goal=task,
            graph=self.graph,
            graph_name=self.graph_name,
            history_messages=self.history_messages,
            on_token=self.on_token,
        )
        if self._backend:
            agent.attach_backend(self._backend)
        if cfg["build_prompt"]:
            agent.build_prompt = cfg["build_prompt"]  # type: ignore[method-assign]
        if cfg["extra_action_tools"] or cfg["extra_action_schemas"]:
            agent._extra_action_tools = cfg["extra_action_tools"]
            agent._extra_action_schemas = cfg["extra_action_schemas"]

        try:
            output = await agent.run(scope_root=goal_node.id)
        except Exception as exc:
            return f"[subagent {name} error: {exc}]"

        return f"[goal:{goal_node.id}] {output}"

    # -- prompt & tool builders (defaults use the mutation pipeline) -----------

    def build_prompt(self, node: Node, graph: Graph) -> str:
        """Return the system prompt for processing *node*.

        Default uses DEFAULT_GUIDES per node type.  If a skill is active
        (via activate_skill tool), the skill's prompt replaces the default.
        """
        prompt = default_build_prompt(node, graph)
        if self._active_skill and self._active_skill in self._skills:
            skill = self._skills[self._active_skill]
            prompt = skill["prompt"] or prompt
        # For GOAL nodes, list available domain tools so the LLM only
        # creates tasks that can actually be executed.
        if node.type == NT.GOAL and self._extra_action_tools:
            tool_names = ", ".join(sorted(self._extra_action_tools.keys()))
            prompt += (
                f"\n\n━━ AVAILABLE DOMAIN TOOLS ━━\n"
                f"The only tools available for ACTION nodes are: {tool_names}\n"
                f"Only create tasks that use these tools. "
                f"Do NOT create tasks for tools that do not exist."
            )
        # Override the default GOAL guide when memory & skill tools are available
        if node.type == NT.GOAL and self._backend and self.user_id:
            prompt = (
                "You are processing the top-level GOAL node.\n"
                "Available: create_task, create_synthesis_node, add_dependency,\n"
                "           store_triplet, query_triplets, search_memory,\n"
                "           create_skill, search_skills, activate_learned_skill\n\n"
                "FIRST: search_skills() to see if you have a learned procedure "
                "for this type of problem. If yes, activate_learned_skill().\n\n"
                "For each user request:\n"
                "  1. Store personal info via store_triplet() immediately.\n"
                "  2. If it's a multi-step problem, AFTER solving it, "
                "call create_skill() to save the procedure for future reuse.\n"
                "  3. On subsequent similar requests, search_skills() + "
                "activate_learned_skill() instead of creating tasks from scratch.\n\n"
                "Do NOT decompose into tasks if a learned skill already covers it."
            )
        return prompt

    def build_tools(self, node: Node) -> tuple[list[dict], dict[str, Any]]:
        """Return (tool_schema_list, {name: callable}).

        Default returns mutation tools (create_task, create_action, ...)
        based on node type, plus any extra_action_tools for ACTION nodes
        and registered subagent/skill/handoff tools for non-ACTION nodes.

        Override for a completely custom tool pipeline.
        """
        def _on_node(n: Node) -> None:
            self._persist_node(n)

        def _on_edge(e: Edge) -> None:
            if e.src in self.graph.nodes and e.dst in self.graph.nodes:
                self._persist_edge(
                    e,
                    self.graph.nodes[e.src].type,
                    self.graph.nodes[e.dst].type,
                )

        schemas, impls = default_build_tools(
            self.graph, node,
            extra_action_tools=self._extra_action_tools,
            extra_action_schemas=self._extra_action_schemas,
            on_node_added=_on_node if self._backend else None,
            on_edge_added=_on_edge if self._backend else None,
        )

        # Inject multi-agent tools on non-ACTION nodes
        if node.type != NT.ACTION:
            for sa_name, sa_cfg in self._subagents.items():
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": f"subagent_{sa_name}",
                        "description": sa_cfg["description"],
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": f"Task to delegate to {sa_name}",
                                },
                            },
                            "required": ["task"],
                        },
                    },
                })
                async def _sa_fn(task: str, _n=sa_name) -> str:
                    return await self._run_subagent(_n, task)
                impls[f"subagent_{sa_name}"] = _sa_fn

            for sk_name, sk_cfg in self._skills.items():
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": f"activate_skill_{sk_name}",
                        "description": sk_cfg["description"],
                        "parameters": {"type": "object", "properties": {}},
                    },
                })
                impls[f"activate_skill_{sk_name}"] = (
                    lambda _n=sk_name: self._activate_skill(_n)
                )

        # Agentic memory tools (GOAL / TASK nodes, requires backend + user_id)
        if self._backend and self.user_id and node.type != NT.ACTION:
            _uid = self.user_id
            _embed = self.embedder or self.llm.embed

            async def _store_triplet(subject: str, predicate: str, object_: str) -> str:
                emb = await _embed(f"{subject} {predicate} {object_}")
                return self._backend.store_triplet(_uid, subject, predicate, object_, emb)

            async def _query_triplets(
                subject: str = "", predicate: str = "", object_: str = "",
            ) -> str:
                hits = self._backend.query_triplets(
                    _uid,
                    subject or None,
                    predicate or None,
                    object_ or None,
                )
                if not hits:
                    return "[no matching triplets]"
                return "\n".join(
                    f"  {h['subject']} --[{h['predicate']}]--> {h['object']}"
                    for h in hits
                )

            async def _search_memory(query: str) -> str:
                emb = await _embed(query)
                hits = self._backend.search_memory(_uid, emb)
                if not hits:
                    return "[no relevant memories]"
                return "\n".join(
                    f"  {h['subject']} --[{h['predicate']}]--> {h['object']}"
                    for h in hits
                )

            async def _get_entity_graph(entity_name: str) -> str:
                hits = self._backend.get_entity_graph(_uid, entity_name)
                if not hits:
                    return f"[no triplets for {entity_name}]"
                return "\n".join(
                    f"  {h['subject']} --[{h['predicate']}]--> {h['object']}"
                    for h in hits
                )

            # Learned skills: the agent creates and stores reusable procedures
            async def _create_skill(name: str, description: str, procedure: str) -> str:
                emb = await _embed(f"{name} {description} {procedure}")
                return self._backend.store_skill(_uid, name, description, procedure, emb)

            async def _search_skills(query: str) -> str:
                emb = await _embed(query)
                hits = self._backend.search_skills(_uid, emb)
                if not hits:
                    return "[no relevant skills]"
                lines = []
                for h in hits:
                    lines.append(f"  {h['name']}: {h['description']}")
                    lines.append(f"    Procedure: {h['procedure']}")
                return "\n".join(lines)

            async def _activate_learned_skill(name: str) -> str:
                skill = self._backend.get_skill(_uid, name)
                if not skill:
                    return f"[skill '{name}' not found]"
                safe = name.replace(" ", "_").replace("-", "_").lower()
                self._skills[safe] = {
                    "prompt": f"Using skill: {skill['description']}\n\n{skill['procedure']}",
                    "tool_schemas": [],
                    "tool_fns": {},
                    "description": skill["description"],
                }
                self._active_skill = safe
                return f"Skill '{name}' activated. {skill['description']}"

            for fn_def in [
                ("store_triplet",
                 "Store a fact as subject -[predicate]-> object. Use this when you learn something.",
                 {"subject": "the subject/entity", "predicate": "the relationship", "object_": "the object/entity"}),
                ("query_triplets",
                 "Find triplets. Leave fields empty to use as wildcard.",
                 {"subject": "", "predicate": "", "object_": ""}),
                ("search_memory",
                 "Semantic search across all stored knowledge.",
                 {"query": "natural language query"}),
                ("get_entity_graph",
                 "Get all triplets connected to an entity.",
                 {"entity_name": "name of the entity"}),
                ("create_skill",
                 "Create a reusable skill from a procedure you just performed.",
                 {"name": "skill name", "description": "what it does",
                  "procedure": "step-by-step instructions"}),
                ("search_skills",
                 "Find a skill you created earlier.",
                 {"query": "natural language query"}),
                ("activate_learned_skill",
                 "Load a skill's context into your active prompt.",
                 {"name": "skill name"}),
            ]:
                name, desc, props = fn_def
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": {
                            "type": "object",
                            "properties": {k: {"type": "string"} for k in props},
                            "required": [k for k, v in props.items() if v],
                        },
                    },
                })
            impls["store_triplet"] = _store_triplet
            impls["query_triplets"] = _query_triplets
            impls["search_memory"] = _search_memory
            impls["get_entity_graph"] = _get_entity_graph
            impls["create_skill"] = _create_skill
            impls["search_skills"] = _search_skills
            impls["activate_learned_skill"] = _activate_learned_skill

        # Handoff tool (available on all node types)
        schemas.append({
            "type": "function",
            "function": {
                "name": "handoff_to",
                "description": "Switch to a different agent for the next turn",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Name of the agent to hand off to",
                        },
                    },
                    "required": ["target"],
                },
            },
        })
        impls["handoff_to"] = self._handoff

        # Token usage tool (available on all node types)
        schemas.append({
            "type": "function",
            "function": {
                "name": "get_token_usage",
                "description": "Get the accumulated token usage for this node",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        })
        def _get_token_usage() -> str:
            usage = node.props.get("_usage", {})
            if not usage:
                return "[no token usage recorded yet]"
            return f"Prompt tokens: {usage.get('prompt_tokens', 0)}, Completion tokens: {usage.get('completion_tokens', 0)}, Total tokens: {usage.get('total_tokens', 0)}"
        impls["get_token_usage"] = _get_token_usage

        return schemas, impls

    def _activate_skill(self, name: str) -> str:
        if name in self._skills:
            self._active_skill = name
            return f"Skill '{name}' activated. Using its prompt and tools."
        return f"[unknown skill: {name}]"

    def _handoff(self, target: str) -> str:
        self._handoff_target = target
        return f"Handing off to {target} on next turn."

    # -- LLM tool-call loop ----------------------------------------------------

    async def _generate_streamed(
        self, messages: list[Message], tool_schemas: list[dict] | None
    ) -> Message:
        """Call ``generate_stream``, forward tokens to ``on_token``, return full Message."""
        content_parts: list[str] = []
        all_tool_calls = None

        async for chunk in self.llm.generate_stream(messages, tools=tool_schemas or None):
            text = chunk.get("content", "")
            if text:
                content_parts.append(text)
                if self.on_token:
                    self.on_token(text)
            if chunk.get("done"):
                all_tool_calls = chunk.get("tool_calls")
                break

        result: Message = {"role": "assistant", "content": "".join(content_parts)}
        if all_tool_calls:
            result["tool_calls"] = all_tool_calls
        if chunk.get("usage"):
            result["usage"] = chunk["usage"]
        return result

    async def _call_llm(self, node: Node) -> str:
        """Run the full LLM turn for *node*, executing tools until done.

        Message ordering for a turn:
          [system: build_prompt()]
          [user/assistant: history_messages (injected by Session)]
          [user: node.label]  <- the current node's task
          ... tool results ...
          [assistant: final text]

        The full message transcript (including all tool calls and results)
        is stored on ``node.props["_messages"]`` and persisted to FalkorDB
        via the node's ``props`` JSON.
        """
        tool_schemas, tool_fns = self.build_tools(node)
        system_prompt = self.build_prompt(node, self.graph)

        # Cross-turn semantic memory: for GOAL nodes, find semantically
        # similar past nodes (GOAL / TASK / ACTION / SYNTHESIS) and inject
        # their resolved outputs as context via SEEN_BEFORE edges.
        if node.type == NT.GOAL and self._backend and self.graph_name:
            try:
                _embed = self.embedder or self.llm.embed
                emb = await _embed(node.label)
                past = self._backend.find_similar_nodes(
                    emb, self.graph_name, k=5, threshold=0.78,
                )
                if past:
                    lines = [
                        "\n\n━━ SIMILAR PROBLEMS SEEN BEFORE (same reasoning pattern) ━━",
                        "The following past problems used a similar approach.",
                        "Do NOT reuse their results — they may be stale. "
                        "Instead, reuse the LOGIC: how the problem was decomposed.",
                    ]
                    for p in past:
                        tag = p.get("node_type", "?")
                        lines.append(f"  [{tag}] {p['label']}")
                        # Link current GOAL → past EMBEDDING node via SEEN_BEFORE
                        if p.get("node_id") and self._backend:
                            from agentic_graphs.core.falkordb_backend import _cypher_val
                            self._backend.mutate(
                                f"MATCH (s {{id: {_cypher_val(node.id)}}}) "
                                f"MATCH (e:EMBEDDING {{node_id: {_cypher_val(p['node_id'])}}}) "
                                f"MERGE (s)-[:SEEN_BEFORE]->(e)",
                                self.graph_name,
                            )
                    lines.append("━━ END SIMILAR PROBLEMS ━━\n")
                    system_prompt += "\n".join(lines)
                    log.info("Semantic memory for %r: linked %d past node(s) by reasoning pattern",
                             node.label[:60], len(past))
            except (NotImplementedError, Exception) as exc:
                log.debug("Semantic memory skipped: %s", exc)

        # Agentic memory: inject relevant user facts into the prompt
        if node.type == NT.GOAL and self._backend and self.user_id:
            try:
                _embed = self.embedder or self.llm.embed
                emb = await _embed(node.label)
                mem_hits = self._backend.search_memory(
                    self.user_id, emb, k=5, threshold=0.45,
                )
                if mem_hits:
                    lines = ["\n━━ YOUR MEMORY (relevant triplets) ━━"]
                    for h in mem_hits:
                        lines.append(
                            f"  {h['subject']} --[{h['predicate']}]--> {h['object']}"
                        )
                    lines.append("━━ END MEMORY ━━\n")
                    system_prompt += "\n".join(lines)
                    log.info("Agentic memory for %r: injected %d triplet(s)",
                             node.label[:60], len(mem_hits))
            except (NotImplementedError, Exception) as exc:
                log.debug("Agentic memory skipped: %s", exc)

        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            *self.history_messages,
            {"role": "user",   "content": node.label},
        ]

        max_tool_iters = 12
        final_content = "[max tool iterations reached]"

        _called_subagents: set[str] = set()

        # Accumulate token usage across LLM calls
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for iteration in range(max_tool_iters):
            if self.on_token:
                response = await self._generate_streamed(messages, tool_schemas)
            else:
                response = await self.llm.generate(messages, tools=tool_schemas or None)
            messages.append(response)

            # Accumulate usage if present
            if response.get("usage"):
                usage = response["usage"]
                total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                total_usage["total_tokens"] += usage.get("total_tokens", 0)
            if self.on_token:
                response = await self._generate_streamed(messages, tool_schemas)
            else:
                response = await self.llm.generate(messages, tools=tool_schemas or None)
            messages.append(response)

            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                final_content = response.get("content") or ""
                break

            # Filter out repeated subagent calls — each can be called once
            filtered: list[dict] = []
            for tc in tool_calls:
                fn = tc["function"]["name"]
                if fn.startswith("subagent_") and fn in _called_subagents:
                    tc["function"]["arguments"] = '{"task": "[already called — use previous result]"}'
                    tc["function"]["name"] = "_noop"
                else:
                    if fn.startswith("subagent_"):
                        _called_subagents.add(fn)
                filtered.append(tc)
            tool_calls = filtered
            tool_fns["_noop"] = lambda **kwargs: "[already called this turn — use previous result]"

            # Execute all tool calls in parallel — subagents run concurrently
            tool_results: list[Message] = []
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                log.info("  \u2699 %s(%s)", fn_name, _snap(json.dumps(args), 120))
                fn = tool_fns.get(fn_name)
                if fn is None:
                    result = f"[unknown tool: {fn_name!r}]"
                    log.warning("  \u2190 %s", result)
                    tool_results.append({
                        "role": "tool", "tool_call_id": tc["id"], "content": result,
                    })
                else:
                    try:
                        out = fn(**args)
                        if asyncio.iscoroutine(out):
                            # Subagent calls and other async tools run concurrently
                            tool_results.append((tc["id"], out))
                        else:
                            result = "" if out is None else str(out)
                            log.info("  \u2190 %s", _snap(result, 120))
                            tool_results.append({
                                "role": "tool", "tool_call_id": tc["id"], "content": result,
                            })
                    except Exception as exc:
                        log.warning("  \u2190 tool error: %s", exc)
                        tool_results.append({
                            "role": "tool", "tool_call_id": tc["id"],
                            "content": f"[tool error: {exc}]",
                        })

            # Await any async (subagent) calls in parallel.
            # tool_results is a mixed list: dicts (sync results already formatted)
            # and (tc_id, coroutine) tuples (async subagent calls).  We must
            # filter by type, NOT try to unpack every element — iterating a
            # 3-key dict as (tid, c) raises "too many values to unpack".
            pending = [item for item in tool_results if isinstance(item, tuple)]
            if pending:
                tids = [p[0] for p in pending]
                coros = [p[1] for p in pending]
                results = await asyncio.gather(*coros, return_exceptions=True)
                for tid, res in zip(tids, results):
                    if isinstance(res, Exception):
                        result = f"[tool error: {res}]"
                    else:
                        result = "" if res is None else str(res)
                    log.info("  \u2190 %s", _snap(result, 120))
                    tool_results.append({
                        "role": "tool", "tool_call_id": tid, "content": result,
                    })

            # Append sync results (already resolved) and async results
            for tr in tool_results:
                if isinstance(tr, dict) and "role" in tr:
                    messages.append(tr)

            # If the node was resolved by a tool call, stop the loop
            if self.graph.nodes[node.id].state == S.RESOLVED:
                final_content = response.get("content") or ""
                break

        # When a backend is attached, create message/toolcall subgraph nodes
        # in the in-memory graph so they sync to FalkorDB as queryable nodes.
        if self._backend and len(messages) > 2:
            self._build_message_subgraph(node, messages)

        if final_content == "[max tool iterations reached]":
            log.warning("Max tool iterations (%d) reached for node %s",
                        max_tool_iters, node.id[:8])

        # Store accumulated token usage on the node and persist if backed.
        node.props["_usage"] = total_usage
        if self._backend:
            self._persist_node(node)

        return final_content


    # -- message subgraph (FalkorDB-visible message + toolcall nodes) ---------

    def _build_message_subgraph(
        self, node: Node, messages: list[Message]
    ) -> None:
        """Create MESSAGE and TOOLCALL nodes in the in-memory graph.

        These are persisted to FalkorDB via the standard sync/upsert machinery
        but excluded from ``subgraph_text()`` so LLM prompts stay lean.

        FalkorDB model (fully connected to the root GOAL)::

          (:GOAL)<-[:PART_OF]-(:MESSAGE)  ← every message is part of the goal
                     |-[:NEXT]->(:MESSAGE)
                     |-[:CALLED]->(:TOOLCALL)
                                   ⋮

        Every node and edge is incrementally persisted so the full trace
        is visible in FalkorDB even before the final ``sync()``.
        """
        prev_msg_id: str | None = None
        tc_nodes: dict[str, str] = {}  # tool_call_id -> node id
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            msg_node = Node(
                id=_uid(), type=NT.MESSAGE, label=role,
                state=S.RESOLVED, output=content,
                props={"idx": i},
            )
            self.graph.add_node(msg_node)
            self._persist_node(msg_node)
            e = Edge(id=_uid(), type=ET.HAS_MSG, src=node.id, dst=msg_node.id)
            self.graph.add_edge(e)
            self._persist_edge(e, self.graph.nodes[e.src].type,
                              self.graph.nodes[e.dst].type)
            # Link every message to the root GOAL so the full story is connected
            if self._root_id and self._root_id != node.id:
                pe = Edge(id=_uid(), type=ET.PART_OF, src=msg_node.id, dst=self._root_id)
                self.graph.add_edge(pe)
                self._persist_edge(pe, self.graph.nodes[pe.src].type,
                                  self.graph.nodes[pe.dst].type)
            if prev_msg_id:
                e = Edge(id=_uid(), type=ET.NEXT, src=prev_msg_id, dst=msg_node.id)
                self.graph.add_edge(e)
                self._persist_edge(e, self.graph.nodes[e.src].type,
                                  self.graph.nodes[e.dst].type)
            prev_msg_id = msg_node.id

            # Assistant messages may contain tool calls
            for tc in msg.get("tool_calls", []):
                tc_id = tc["id"]
                fn_name = tc["function"]["name"]
                args_str = tc["function"]["arguments"]
                tc_node = Node(
                    id=_uid(), type=NT.TOOLCALL, label=fn_name,
                    state=S.RESOLVED, output=args_str,
                    props={"tc_id": tc_id},
                )
                self.graph.add_node(tc_node)
                self._persist_node(tc_node)
                e = Edge(id=_uid(), type=ET.CALLED, src=msg_node.id, dst=tc_node.id)
                self.graph.add_edge(e)
                self._persist_edge(e, self.graph.nodes[e.src].type,
                                  self.graph.nodes[e.dst].type)
                if self._root_id and self._root_id != node.id:
                    pe = Edge(id=_uid(), type=ET.PART_OF,
                              src=tc_node.id, dst=self._root_id)
                    self.graph.add_edge(pe)
                    self._persist_edge(pe, self.graph.nodes[pe.src].type,
                                      self.graph.nodes[pe.dst].type)
                tc_nodes[tc_id] = tc_node.id

            # Tool messages contain results; link to the matching TOOLCALL
            tc_id = msg.get("tool_call_id")
            if tc_id and tc_id in tc_nodes:
                tcnid = tc_nodes[tc_id]
                self.graph.nodes[tcnid].props["result"] = content
                self._persist_node(self.graph.nodes[tcnid])
                e = Edge(id=_uid(), type=ET.RETURNED, src=msg_node.id, dst=tcnid)
                self.graph.add_edge(e)
                self._persist_edge(e, self.graph.nodes[e.src].type,
                                  self.graph.nodes[e.dst].type)

    # -- node processing -------------------------------------------------------

    async def process_node(self, node: Node, retries: int = 2) -> str:
        """Mark *node* ACTIVE, call the LLM, mark RESOLVED (or FAILED).

        Transient failures (timeouts, rate limits) are retried up to
        *retries* times before marking the node FAILED.
        """
        log.info("Node %s [%s] %r \u2192 ACTIVE", node.id[:8], node.type.value, _snap(node.label))
        self.graph.set_state(node.id, S.ACTIVE)
        self._persist_node(self.graph.nodes[node.id])

        last_exc: Exception | None = None
        for attempt in range(1 + retries):
            try:
                # Auto-resolve TASK nodes whose child ACTIONs are all resolved.
                # Skip if there are task-to-task REQUIRES edges — those need LLM.
                if node.type == NT.TASK:
                    prereqs = []
                    has_task_dep = False
                    for eid in self.graph._out.get(node.id, []):
                        e = self.graph.edges.get(eid)
                        if e and e.type == ET.REQUIRES:
                            dst = self.graph.nodes.get(e.dst)
                            if dst:
                                prereqs.append(dst)
                                if dst.type != NT.ACTION:
                                    has_task_dep = True
                    if prereqs and not has_task_dep and all(d.state == S.RESOLVED for d in prereqs):
                        # Prefer ACTION outputs (actual computed values)
                        action_outputs = [d.output for d in prereqs
                                          if d.type == NT.ACTION and d.output]
                        outputs = [d.output for d in prereqs if d.output]
                        chosen = action_outputs if action_outputs else outputs
                        combined = "; ".join(chosen) if chosen else "[completed]"
                        self.graph.resolve(node.id, combined)
                        self._persist_node(self.graph.nodes[node.id])
                        log.info("Node %s [%s] \u2192 RESOLVED (auto, %d chars)",
                                 node.id[:8], node.type.value, len(combined))
                        return combined

                output = await self._call_llm(node)
                # If create_action put us back to PENDING, don't resolve yet —
                # wait for child actions to complete first.
                if self.graph.nodes[node.id].state == S.PENDING:
                    log.info("Node %s [%s] \u2192 PENDING (waiting on action)",
                             node.id[:8], node.type.value)
                    self._persist_node(self.graph.nodes[node.id])
                    return output
                # Don't resolve again if already resolved (e.g. by resolve_current_node tool)
                if self.graph.nodes[node.id].state != S.RESOLVED:
                    self.graph.resolve(node.id, output)
                self._persist_node(self.graph.nodes[node.id])
                resolved = self.graph.nodes[node.id].output
                # Store embedding for cross-turn semantic memory (all types)
                if self._backend and self.graph_name:
                    try:
                        _embed = self.embedder or self.llm.embed
                        emb = await _embed(node.label)
                        self._backend.store_embedding(
                            node.id, node.type.value.upper(),
                            node.label, resolved, emb, self.graph_name,
                        )
                    except (NotImplementedError, Exception):
                        pass
                log.info("Node %s [%s] \u2192 RESOLVED  (%d chars)",
                         node.id[:8], node.type.value,
                         len(resolved))
                return output
            except Exception as exc:
                last_exc = exc
                is_transient = any(
                    s in str(exc).lower()
                    for s in ("timed out", "timeout", "rate limit", "429", "503", "502")
                )
                if is_transient and attempt < retries:
                    wait = 2.0 * (attempt + 1)
                    log.warning("Node %s transient error (attempt %d/%d): %s. "
                                "Retrying in %.0fs...",
                                node.id[:8], attempt + 1, retries + 1, exc, wait)
                    await asyncio.sleep(wait)
                else:
                    break

        log.error("Node %s (%r) failed: %s", node.id, node.label[:60], last_exc)
        self.graph.set_state(node.id, S.FAILED)
        self._persist_node(self.graph.nodes[node.id])
        raise last_exc  # type: ignore[misc]

    # -- scheduler loop --------------------------------------------------------

    def _descendants(self, root_id: str) -> set[str]:
        """Return all node IDs reachable from *root_id* via PART_OF edges
        traversed from parent to child.

        PART_OF is stored child→parent, so we follow edges where
        ``e.dst == nid`` to walk the tree downward.  This correctly scopes
        subagents to their own subtree within a shared graph.
        """
        seen: set[str] = set()
        stack = [root_id]
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            for e in self.graph.edges.values():
                if e.type == ET.PART_OF and e.dst == nid and e.src not in seen:
                    stack.append(e.src)
        return seen

    async def run(self, scope_root: str | None = None) -> str:
        """Drive the graph to completion; return the final answer.

        If *scope_root* is set, only nodes reachable from that root via
        forward edges are processed.  This lets subagents run their own
        scheduler within the shared graph without touching unrelated nodes.
        """
        last_output = ""
        passes = 0
        for iteration in range(self.max_iterations):
            ready = self.graph.ready()
            if scope_root:
                descendants = self._descendants(scope_root)
                ready = [n for n in ready if n.id in descendants]
            if not ready:
                break
            passes = iteration + 1

            log.info("Schedule pass %d: %d node(s) ready", passes, len(ready))
            results = await asyncio.gather(
                *(self.process_node(n) for n in ready),
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, str):
                    last_output = r

        # Log scheduler summary
        resolved = sum(1 for n in self.graph.nodes.values() if n.state == S.RESOLVED)
        failed = sum(1 for n in self.graph.nodes.values() if n.state == S.FAILED)
        total = len(self.graph.nodes)
        log.info("Scheduler done  passes=%d  %d/%d resolved  %d failed",
                 passes, resolved, total, failed)

        # Deterministic output:
        #   scope mode  → synthesis > goal within scope
        #   root mode   → root GOAL > synthesis > goal
        if scope_root:
            scope = self._descendants(scope_root)
            for n in self.graph.nodes.values():
                if n.id in scope and n.state == S.RESOLVED and n.type == NT.SYNTHESIS:
                    return n.output
            for n in self.graph.nodes.values():
                if n.id in scope and n.state == S.RESOLVED and n.type == NT.GOAL:
                    return n.output
        else:
            # Prefer SYNTHESIS nodes that are direct children of the root
            # GOAL (via PART_OF).  These are "combine results" nodes that
            # contain the computed answer.  Fall back to root GOAL output
            # (used by orchestrator-style agents that answer inline).
            if self._root_id:
                for n in self.graph.nodes.values():
                    if n.state == S.RESOLVED and n.type == NT.SYNTHESIS:
                        for eid in self.graph._out.get(n.id, []):
                            e = self.graph.edges.get(eid)
                            if e and e.type == ET.PART_OF and e.dst == self._root_id:
                                return n.output
            root = self.graph.nodes.get(self._root_id) if self._root_id else None
            if root and root.state == S.RESOLVED:
                return root.output
            for n in self.graph.nodes.values():
                if n.state == S.RESOLVED and n.type == NT.SYNTHESIS:
                    return n.output
            for n in self.graph.nodes.values():
                if n.state == S.RESOLVED and n.type == NT.GOAL:
                    return n.output
        return last_output