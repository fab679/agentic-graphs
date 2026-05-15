"""Default agent internals — mutation tools, guides, tool schemas.

These are the built-in defaults that let any ``Agent`` subclass work
out of the box without re-implementing the graph-mutation tooling.
"""

from typing import Callable

from agentic_graphs.core.graph import Graph, Node, NT, ET, S, Edge, _uid


# ── which tools each node type may call ─────────────────────────────────────

PERMITTED_TOOLS: dict[NT, set[str]] = {
    NT.GOAL:      {"create_task", "create_synthesis_node", "add_dependency"},
    NT.TASK:      {"create_action", "add_dependency", "resolve_current_node"},
    NT.SYNTHESIS: {"resolve_current_node"},
    NT.ACTION:    set(),
}


# ── prompts per node type ───────────────────────────────────────────────────

DEFAULT_GUIDES: dict[NT, str] = {
    NT.GOAL: (
        "You are processing the top-level GOAL node.\n"
        "Available: create_task, create_synthesis_node, add_dependency\n\n"
        "CHECK THE CONVERSATION HISTORY above. If the user's question was "
        "already asked and answered, simply repeat the previous answer and "
        "do NOT create any tasks.\n\n"
        "If the goal is a simple greeting or chat, respond directly by "
        "outputting text. IMPORTANT: if the user shares personal information, "
        "call store_triplet() to remember it before responding.\n\n"
        "If the available tools cannot help with the goal, use your own "
        "knowledge to answer directly — do not create fake tasks.\n\n"
        "For actual work that requires tools:\n"
        "  1) create_task() for each distinct piece of work.\n"
        "  2) If tasks depend on each other (e.g. multi-step arithmetic),\n"
        "     add_dependency(DEPENDENT_TASK_ID, PREREQ_TASK_ID).\n"
        "  3) create_synthesis_node() once to combine results.\n"
        "  4) add_dependency(YOUR_SYNTHESIS_ID, EACH_TASK_ID) for every task."
    ),
    NT.TASK: (
        "You are processing a TASK node — one unit of work.\n"
        "Available: create_action, add_dependency, resolve_current_node\n\n"
        "Check neighbourhood for existing actions:\n"
        "  • Already have actions → read their outputs, resolve_current_node('Summary: ...').\n"
        "  • No actions yet → create_action() once with a clear instruction.\n"
        "    Do NOT resolve yet — wait for the action to complete.\n"
        "Rules:\n"
        "  - Create at most ONE action per task\n"
        "  - Never create duplicate actions"
    ),
    NT.ACTION: (
        "You are processing an ACTION node — execute one concrete operation.\n"
        "Call exactly ONE tool with the correct parameters, then "
        "resolve_current_node('<the result>').\n"
        "Do NOT call the same tool multiple times with different parameters."
    ),
    NT.SYNTHESIS: (
        "You are processing a SYNTHESIS node.\n"
        "Only tool: resolve_current_node\n"
        "Read all resolved task outputs and combine them into the final answer."
    ),
}


# ── default prompt builder ──────────────────────────────────────────────────

def default_build_prompt(node: Node, graph: Graph, guides: dict | None = None) -> str:
    """Build a prompt for the given node using default guides."""
    ctx = graph.subgraph_text(node.id)
    guide = (guides or DEFAULT_GUIDES).get(node.type, "Use the tools provided.")
    extra = ""
    if node.props.get("instruction"):
        extra = f"Instruction: {node.props['instruction']}\n\n"
    return (
        f"You are the reasoning core of a graph-based agent.\n\n"
        f"━━ CURRENT NODE ━━\nType: {node.type.upper()}\nLabel: {node.label!r}\nID: {node.id}\n\n"
        f"{extra}"
        f"━━ GRAPH NEIGHBOURHOOD ━━\n{ctx}\n\n"
        f"━━ WHAT TO DO ━━\n{guide}\n\n"
        f"Think step by step, then call the appropriate tools."
    )


# ── mutation tool factory ───────────────────────────────────────────────────

def make_mutation_tools(
    graph: Graph,
    focal_id: str,
    on_node_added: Callable[[Node], None] | None = None,
    on_edge_added: Callable[[Edge], None] | None = None,
) -> tuple[list[dict], dict[str, Callable]]:
    """Return (OpenAI tool_defs, tool_implementations) for graph mutation.

    The implementations are closures bound to *graph* and *focal_id* so
    every tool call mutates the correct graph and node.

    If *on_node_added* / *on_edge_added* are provided they are called
    after every mutation so callers (e.g. Agent) can persist incrementally.
    """

    def _node(n: Node) -> Node:
        if on_node_added:
            on_node_added(n)
        return n

    def _edge(e: Edge) -> Edge:
        if on_edge_added:
            on_edge_added(e)
        return e

    def create_task(label: str) -> str:
        n = Node(id=_uid(), type=NT.TASK, label=label)
        graph.add_node(n)
        _node(n)
        e = Edge(id=_uid(), type=ET.PART_OF, src=n.id, dst=focal_id)
        graph.add_edge(e)
        _edge(e)
        return f"Task created \u2014 id={n.id!r}  label={n.label!r}"

    def create_action(label: str, instruction: str) -> str:
        # Inject outputs from resolved prerequisites so the action has
        # the actual numbers it needs (critical for multi-step arithmetic).
        prereq_lines = []
        for eid in graph._out.get(focal_id, []):
            e = graph.edges.get(eid)
            if e and e.type == ET.REQUIRES:
                prereq = graph.nodes.get(e.dst)
                if prereq and prereq.state == S.RESOLVED and prereq.output:
                    prereq_lines.append(f"[{prereq.label}]: {prereq.output}")
        full_instruction = instruction
        if prereq_lines:
            full_instruction += "\n\nResolved prerequisite outputs:\n" + "\n".join(prereq_lines)
        n = Node(id=_uid(), type=NT.ACTION, label=label,
                 props={"instruction": full_instruction})
        graph.add_node(n)
        _node(n)
        e1 = Edge(id=_uid(), type=ET.PART_OF, src=n.id, dst=focal_id)
        graph.add_edge(e1)
        _edge(e1)
        e2 = Edge(id=_uid(), type=ET.REQUIRES, src=focal_id, dst=n.id)
        graph.add_edge(e2)
        _edge(e2)
        # Put focal node back to PENDING so it waits for this action
        # to complete before resolving.
        graph.set_state(focal_id, S.PENDING)
        if on_node_added:
            on_node_added(graph.nodes[focal_id])
        return f"Action created \u2014 id={n.id!r}"

    def create_synthesis_node(label: str) -> str:
        n = Node(id=_uid(), type=NT.SYNTHESIS, label=label)
        graph.add_node(n)
        _node(n)
        n.state = S.PENDING
        e = Edge(id=_uid(), type=ET.PART_OF, src=n.id, dst=focal_id)
        graph.add_edge(e)
        _edge(e)
        return f"Synthesis created \u2014 id={n.id!r}. Call add_dependency for each task."

    def add_dependency(waiting: str, prereq: str) -> str:
        if waiting == prereq:
            return "Error: cannot add self-dependency"
        if waiting not in graph.nodes:
            valid = [n.id[:8] for n in graph.nodes.values()][:6]
            return (f"Error: node {waiting!r} not found. "
                    f"Valid node IDs: {valid}. "
                    f"Use the exact id string from create_task/create_action output.")
        if prereq not in graph.nodes:
            valid = [n.id[:8] for n in graph.nodes.values()][:6]
            return (f"Error: node {prereq!r} not found. "
                    f"Valid node IDs: {valid}. "
                    f"Use the exact id string from create_task/create_action output.")
        # Tentatively add edge, then check for cycles
        e = Edge(id=_uid(), type=ET.REQUIRES, src=waiting, dst=prereq)
        graph.add_edge(e)
        if graph.has_cycle():
            del graph.edges[e.id]
            graph._out.setdefault(waiting, []).remove(e.id)
            graph._in.setdefault(prereq, []).remove(e.id)
            return (f"Error: adding {waiting} \u2192 {prereq} would create a cycle. "
                    f"Rejected.")
        _edge(e)
        return f"Dependency set: {waiting} \u2192 {prereq}"

    def resolve_current_node(output: str) -> str:
        output = str(output) if not isinstance(output, str) else output
        # Block resolve if there are unresolved REQUIRES prereqs.
        pending = []
        for eid in graph._out.get(focal_id, []):
            e = graph.edges.get(eid)
            if e and e.type == ET.REQUIRES:
                dst = graph.nodes.get(e.dst)
                if dst and dst.state not in (S.RESOLVED, S.FAILED):
                    pending.append(dst.id[:8])
        if pending:
            return (f"Cannot resolve yet: still waiting on action(s): "
                    f"{pending}. The scheduler will re-check when they complete.")
        graph.resolve(focal_id, output)
        if on_node_added:
            on_node_added(graph.nodes[focal_id])
        return "Node resolved."

    impls: dict[str, Callable] = {
        "create_task": create_task,
        "create_action": create_action,
        "create_synthesis_node": create_synthesis_node,
        "add_dependency": add_dependency,
        "resolve_current_node": resolve_current_node,
    }

    defs: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": (fn.__doc__ or "").split(".")[0],
                "parameters": {
                    "type": "object",
                    "properties": {
                        p: {"type": "string"}
                        for p in list(fn.__code__.co_varnames[:fn.__code__.co_argcount])
                    },
                    "required": list(fn.__code__.co_varnames[:fn.__code__.co_argcount]),
                },
            },
        }
        for name, fn in impls.items()
    ]
    return defs, impls


# ── default tool-builder ────────────────────────────────────────────────────

def default_build_tools(
    graph: Graph,
    node: Node,
    extra_action_tools: dict[str, Callable] | None = None,
    extra_action_schemas: list[dict] | None = None,
    permitted: dict[NT, set[str]] | None = None,
    on_node_added: Callable[[Node], None] | None = None,
    on_edge_added: Callable[[Edge], None] | None = None,
) -> tuple[list[dict], dict[str, Callable]]:
    """Build tool definitions + implementations for a given node.

    Respects PERMITTED_TOOLS and dynamically removes ``create_action``
    from TASK nodes that already have action children.

    Pass *on_node_added* / *on_edge_added* to receive incremental
    persistence callbacks from the mutation tool implementations.
    """
    mut_defs, mut_impls = make_mutation_tools(
        graph, node.id,
        on_node_added=on_node_added,
        on_edge_added=on_edge_added,
    )
    allowed = set((permitted or PERMITTED_TOOLS).get(node.type, set()))

    # ACTION nodes: mutation tools disabled, external action tools enabled
    if node.type == NT.ACTION:
        tools = (extra_action_schemas or []) + [
            d for d in mut_defs if d["function"]["name"] == "resolve_current_node"
        ]
        impls = {k: v for k, v in mut_impls.items() if k == "resolve_current_node"}
        if extra_action_tools:
            impls.update(extra_action_tools)
        return tools, impls

    # TASK nodes: if actions already exist, forbid creating new ones
    if node.type == NT.TASK:
        has_actions = any(
            nid in graph.nodes
            for eid in graph._out.get(node.id, [])
            if (e := graph.edges.get(eid))
            and e.type == ET.REQUIRES
            and (nid := e.dst)
        )
        if has_actions:
            allowed.discard("create_action")

    tools = [d for d in mut_defs if d["function"]["name"] in allowed]
    impls = {k: v for k, v in mut_impls.items() if k in allowed}
    return tools, impls
