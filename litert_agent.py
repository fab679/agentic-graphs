#!/usr/bin/env python3
"""
Graph Reasoning Agent  ·  LiteRT-LM / Gemma 4 Edition
───────────────────────────────────────────────────────
Runs entirely on-device using Gemma 4 via LiteRT-LM.

The key architectural shift from the Anthropic version:
  Instead of asking the model to return JSON that we parse into mutations,
  we define graph mutation functions AS TOOLS and let Gemma 4 call them
  directly. The model calls Python, Python mutates the graph. No parsing.

Requirements:
    pip install "litert-lm>=0.11.0" "litert-lm-api-nightly>=0.11.0.dev20260422"

Config (via .env or shell):
    MODEL_PATH=/home/picard/.litert-lm/models/gemma-4-E2B-it.litertlm/model.litertlm

Usage:
    python graph_agent_litert.py
    python graph_agent_litert.py "Your goal here"
"""

import asyncio, logging, os, sys, time
import litert_lm

from graph.core import S, NT, ET, Node, Edge, Graph, _uid

log = logging.getLogger("litert_agent")

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/home/picard/.litert-lm/models/gemma-4-E2B-it.litertlm/model.litertlm",
)
litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)


# ══════════════════════════════════════════════════════════════════════════════
#  3.  GRAPH MUTATION TOOLS
#
#  Plain Python closures the model calls directly via LiteRT-LM tool calling.
#  Requirements (LiteRT-LM reads these to build the schema for Gemma 4):
#    • Docstrings with Args: section
#    • Type hints on all parameters and return type
#
#  Each factory call returns a fresh set of functions bound to the current
#  graph state and focal node — so the model can't accidentally affect the
#  wrong node even when running concurrently.
# ══════════════════════════════════════════════════════════════════════════════

def make_planner_tools(graph: Graph, focal_id: str) -> list:

    def create_task(label: str) -> str:
        """Add a new task node to the graph as a sub-component of the current goal.
        Returns the new node's ID so you can reference it in add_dependency calls.

        Args:
            label: Short, specific description of what this task covers.
        """
        node = Node(id=_uid(), type=NT.TASK, label=label)
        graph.add_node(node)
        graph.add_edge(Edge(id=_uid(), type=ET.PART_OF, src=node.id, dst=focal_id))
        log.info("  \u2699 create_task(label=%r) \u2192 %s", label, node.id[:8])
        _print_node(node)
        return f"Task created \u2014 id={node.id!r}  label={node.label!r}"

    def create_action(label: str, instruction: str) -> str:
        """Add an action node that the system executor will run.
        The current task node will automatically wait for this action to complete.
        Returns the new node's ID.

        Args:
            label: Short description of the action (e.g. 'Research write throughput').
            instruction: Full, self-contained instruction: what to research or compute.
        """
        node = Node(id=_uid(), type=NT.ACTION, label=label,
                    props={"instruction": instruction})
        graph.add_node(node)
        graph.add_edge(Edge(id=_uid(), type=ET.PART_OF, src=node.id, dst=focal_id))
        req = Edge(id=_uid(), type=ET.REQUIRES, src=focal_id, dst=node.id)
        graph.add_edge(req)
        log.info("  \u2699 create_action(label=%r) \u2192 %s", label, node.id[:8])
        _print_node(node)
        _print_edge(req, graph)
        return f"Action created \u2014 id={node.id!r}. Current task will wait for it."

    def create_synthesis_node(label: str) -> str:
        """Add a synthesis node that will write the final answer once all tasks finish.
        After calling this, call add_dependency to make it wait for each task node.
        Returns the new synthesis node's ID.

        Args:
            label: Short label for the synthesis node (e.g. 'Final recommendation').
        """
        node = Node(id=_uid(), type=NT.SYNTHESIS, label=label)
        graph.add_node(node)
        graph.add_edge(Edge(id=_uid(), type=ET.PART_OF, src=node.id, dst=focal_id))
        log.info("  \u2699 create_synthesis_node(label=%r) \u2192 %s", label, node.id[:8])
        _print_node(node)
        return f"Synthesis created \u2014 id={node.id!r}. Now call add_dependency for each task."

    def add_dependency(waiting_node_id: str, must_finish_first_id: str) -> str:
        """Make one node wait for another to complete before it can run.
        Use this to wire the synthesis node to wait for all task nodes.

        Args:
            waiting_node_id: ID of the node that should wait (e.g. the synthesis id).
            must_finish_first_id: ID of the node that must complete first (e.g. a task id).
        """
        if waiting_node_id not in graph.nodes:
            return f"Error: node {waiting_node_id!r} not found."
        if must_finish_first_id not in graph.nodes:
            return f"Error: node {must_finish_first_id!r} not found."
        edge = Edge(id=_uid(), type=ET.REQUIRES,
                    src=waiting_node_id, dst=must_finish_first_id)
        graph.add_edge(edge)
        log.info("  \u2699 add_dependency(%s, %s)", waiting_node_id[:8], must_finish_first_id[:8])
        _print_edge(edge, graph)
        wl = graph.nodes[waiting_node_id].label
        ml = graph.nodes[must_finish_first_id].label
        return f"Dependency set: {wl!r} now waits for {ml!r}."

    def resolve_current_node(output: str) -> str:
        """Mark the current node as complete with its final output or summary.

        Args:
            output: The result, summary, or answer for this node. Be specific and thorough.
        """
        graph.resolve(focal_id, output)
        log.info("  \u2699 resolve_current_node() \u2192 RESOLVED (%d chars)", len(output))
        _print_resolve(graph.nodes[focal_id])
        return f"Node {focal_id!r} resolved successfully."

    return [create_task, create_action, create_synthesis_node,
            add_dependency, resolve_current_node]


# ══════════════════════════════════════════════════════════════════════════════
#  4.  LLM PROCESSOR  (GOAL / TASK / SYNTHESIS)
#      Passes the node + neighborhood to Gemma 4 with the mutation tools.
#      The model calls tools; tools mutate the graph. No parsing required.
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(node: Node, graph: Graph) -> str:
    ctx = graph.subgraph_text(node.id)

    guides: dict[NT, str] = {
        NT.GOAL: """\
INSTRUCTIONS — you are processing a GOAL node:
1. Call create_task() 2–4 times to decompose the goal into concrete tasks.
2. Call create_synthesis_node() once — this will write the final answer later.
3. Call add_dependency(synthesis_id, task_id) for EVERY task you created.
   Use the IDs returned by create_task() and create_synthesis_node().
4. Call resolve_current_node("Decomposed into N tasks + synthesis.") to finish.""",

        NT.TASK: """\
INSTRUCTIONS — you are processing a TASK node:
Look at the graph neighborhood above carefully.

Case A — no resolved ACTION nodes visible yet:
  Call create_action() 1–2 times with clear, self-contained instructions.
  Do NOT call resolve_current_node yet. The task will automatically
  pause and wait for the actions to complete.

Case B — resolved ACTION nodes ARE present with outputs:
  Read all their outputs, synthesize what was found, then call
  resolve_current_node("Summary: <your synthesis here>").""",

        NT.SYNTHESIS: """\
INSTRUCTIONS — you are processing a SYNTHESIS node:
All required task nodes have now resolved. Their outputs are in the
neighborhood above. Read every resolved node's output carefully.
Write a comprehensive, well-structured final answer that addresses
the original goal completely.
Call resolve_current_node("<your full answer>").""",
    }

    guide = guides.get(node.type, "Reason about this node and call the appropriate tools.")

    return f"""\
You are the reasoning core of a graph-based AI agent.
The agent's memory and state live entirely in a live directed graph.
You mutate the graph by calling the tools provided.

━━ CURRENT NODE ━━
Type:  {node.type.upper()}
Label: {node.label!r}
ID:    {node.id}

━━ GRAPH NEIGHBORHOOD ━━
{ctx}

━━ WHAT TO DO ━━
{guide}

Think through what needs to happen, then call the tools."""


class LLMProcessor:
    def __init__(self, engine: litert_lm.Engine):
        self.engine = engine

    async def process(self, node: Node, graph: Graph) -> str:
        tools  = make_planner_tools(graph, node.id)
        prompt = _build_prompt(node, graph)

        def _run() -> str:
            with self.engine.create_conversation(tools=tools) as conv:
                resp = conv.send_message(prompt)
                return resp["content"][0]["text"]

        return await asyncio.to_thread(_run)


# ══════════════════════════════════════════════════════════════════════════════
#  5.  TOOL EXECUTOR  (ACTION nodes — streaming for live output)
# ══════════════════════════════════════════════════════════════════════════════

class ToolExecutor:
    def __init__(self, engine: litert_lm.Engine):
        self.engine = engine

    async def process(self, node: Node, graph: Graph) -> str:
        instruction = node.props.get("instruction", node.label)
        ctx         = graph.subgraph_text(node.id, hops=2)

        prompt = (f"Context from prior work in this task:\n{ctx}\n\n"
                  f"Your task: {instruction}\n\nBe specific and thorough.")

        log.info("  \u2699 ToolExecutor streaming node %s", node.id[:8])
        print(f"   \033[90m", end="", flush=True)   # dim for streamed text

        def _run() -> str:
            parts = []
            with self.engine.create_conversation() as conv:
                for chunk in conv.send_message_async(prompt):
                    text = chunk["content"][0]["text"]
                    print(text, end="", flush=True)
                    parts.append(text)
            return "".join(parts)

        output = await asyncio.to_thread(_run)
        log.info("  \u2190 ToolExecutor done  (%d chars)", len(output))
        print("\033[0m")   # reset colour
        return output


# ══════════════════════════════════════════════════════════════════════════════
#  6.  AGENT  —  scheduler + run loop
# ══════════════════════════════════════════════════════════════════════════════

class Agent:
    MAX_STEPS = 40

    def __init__(self, goal: str):
        self.graph  = Graph()
        self._step  = 0
        self.planner: LLMProcessor | None = None
        self.tools:   ToolExecutor | None = None

        root = Node(id=_uid(), type=NT.GOAL, label=goal, state=S.READY)
        self.graph.add_node(root)
        self.root = root.id

    async def run(self):
        _header(self.graph.nodes[self.root].label)
        log.info("Loading model: %s", MODEL_PATH)
        print(f"  Loading model\u2026  {MODEL_PATH}")

        with litert_lm.Engine(MODEL_PATH) as engine:
            self.planner = LLMProcessor(engine)
            self.tools   = ToolExecutor(engine)
            log.info("Model loaded")
            print("  \u2713 Model ready\n")

            while self._step < self.MAX_STEPS:
                ready = self.graph.ready()

                if not ready:
                    active  = sum(1 for n in self.graph.nodes.values() if n.state == S.ACTIVE)
                    pending = sum(1 for n in self.graph.nodes.values() if n.state == S.PENDING)
                    if active:
                        await asyncio.sleep(0.05)
                        continue
                    if pending:
                        log.warning("Deadlock - %d nodes stuck in PENDING", pending)
                        _err(f"Deadlock \u2014 {pending} nodes stuck in PENDING")
                    resolved = sum(1 for n in self.graph.nodes.values() if n.state == S.RESOLVED)
                    failed = sum(1 for n in self.graph.nodes.values() if n.state == S.FAILED)
                    log.info("Scheduler done: step=%d  %d resolved  %d failed",
                             self._step, resolved, failed)
                    break

                log.info("Schedule pass %d: %d node(s) ready", self._step + 1, len(ready))
                # Sequential for clarity; parallelise with asyncio.gather if needed
                for node in ready:
                    await self._tick(node)
                self._step += 1

        _summary(self.graph)

    async def _tick(self, node: Node):
        log.info("Node %s [%s] %r \u2192 ACTIVE", node.id[:8], node.type.value,
                 node.label[:60].replace("\n", "\\n"))
        self.graph.set_state(node.id, S.ACTIVE)
        _print_start(node)

        try:
            if node.type == NT.ACTION:
                # ToolExecutor streams the answer directly
                output = await self.tools.process(node, self.graph)
                self.graph.resolve(node.id, output)
                log.info("Node %s [ACTION] \u2192 RESOLVED  (%d chars)", node.id[:8], len(output))
                _print_resolve(self.graph.nodes[node.id])

            elif node.type in (NT.GOAL, NT.TASK, NT.SYNTHESIS):
                # LLMProcessor calls mutation tools; final text is the model's commentary
                reason = await self.planner.process(node, self.graph)
                if reason.strip():
                    _print_reason(reason)

        except Exception as exc:
            log.error("Node %s failed: %s", node.id[:8], exc)
            self.graph.set_state(node.id, S.FAILED)
            _err(f"{node.label!r}: {exc}")
            return

        # Post-processing: if node is still ACTIVE after the call
        n = self.graph.nodes[node.id]
        if n.state != S.ACTIVE:
            return   # already resolved or failed by tool call — done

        unresolved_reqs = [
            self.graph.edges[eid]
            for eid in self.graph._out.get(node.id, [])
            if (self.graph.edges[eid].type == ET.REQUIRES
                and self.graph.nodes[self.graph.edges[eid].dst].state != S.RESOLVED)
        ]
        if unresolved_reqs:
            # Model added actions / dependencies — node correctly waits
            self.graph.set_state(node.id, S.PENDING)
        elif node.type == NT.GOAL:
            # GOAL didn't call resolve_current_node — auto-resolve it
            self.graph.resolve(node.id, "Decomposed into tasks.")
            log.info("Node %s [GOAL] auto-resolved", node.id[:8])
            _print_resolve(self.graph.nodes[node.id])
        else:
            # Model neither resolved nor added deps — something went wrong
            log.warning("Node %s left ACTIVE without resolving or adding deps", node.id[:8])
            self.graph.set_state(node.id, S.FAILED)
            _err(f"{node.label!r} left ACTIVE without resolving or adding deps")


# ══════════════════════════════════════════════════════════════════════════════
#  7.  TERMINAL OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

_ICON = {NT.GOAL: "🎯", NT.TASK: "📋", NT.ACTION: "⚡", NT.SYNTHESIS: "🔮"}
_COL  = {"pending": "\033[90m", "ready": "\033[94m", "active": "\033[93m",
         "resolved": "\033[92m", "failed": "\033[91m"}
R = "\033[0m"

def _header(goal: str):
    print(f"\n{'═'*68}")
    print(f"  🧠  GRAPH REASONING AGENT  ·  LiteRT-LM / Gemma 4")
    print(f"  Goal: {goal}")
    print(f"{'═'*68}")

def _print_start(n: Node):
    print(f"\n{_COL['active']}▶ {n.type.upper():10}{R}  "
          f"{_ICON.get(n.type, '•')} {n.label!r}  \033[90m({n.id})\033[0m")

def _print_reason(r: str):
    short = r[:160].replace("\n", " ")
    print(f"   💭  {short}{'…' if len(r) > 160 else ''}")

def _print_node(n: Node):
    print(f"   + {_ICON.get(n.type, '•')} [{n.type}]  {n.label!r}")

def _print_edge(e: Edge, g: Graph):
    sl = g.nodes[e.src].label[:30] if e.src in g.nodes else e.src
    dl = g.nodes[e.dst].label[:30] if e.dst in g.nodes else e.dst
    print(f"   → [{e.type:8}]  {sl!r} → {dl!r}")

def _print_resolve(n: Node):
    preview = n.output[:90].replace("\n", " ") if n.output else ""
    print(f"   {_COL['resolved']}✓ RESOLVED{R}  {n.label!r}"
          f"{'  — ' + preview + '…' if preview else ''}")

def _err(msg: str):
    print(f"\033[91m  ✗  {msg}\033[0m")

def _summary(g: Graph):
    by_s: dict = {}
    for n in g.nodes.values():
        by_s.setdefault(n.state, []).append(n)

    print(f"\n{'─'*68}")
    print(f"  📊  GRAPH  ({len(g.nodes)} nodes · {len(g.edges)} edges)")
    print(f"{'─'*68}")
    for st, ns in sorted(by_s.items()):
        sample = "  ".join(f"{_ICON.get(n.type, '•')}{n.label[:22]}" for n in ns[:4])
        print(f"  {_COL.get(st, '')}{st.upper():12}{R}  {len(ns):2}   "
              f"{sample}{'…' if len(ns) > 4 else ''}")

    synths = [n for n in g.nodes.values()
              if n.type == NT.SYNTHESIS and n.state == S.RESOLVED]
    if synths:
        print(f"\n{'═'*68}\n  ✅  FINAL OUTPUT\n{'═'*68}")
        print(synths[-1].output)
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  8.  ENTRY
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
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
            tag = f"{_Color.cyan}{_Color.bold}[litert]{_Color.reset}"
            msg = msg.replace("\u2192 ACTIVE", f"{_Color.yellow}\u2192 ACTIVE{_Color.reset}")
            msg = msg.replace("\u2192 RESOLVED", f"{_Color.green}\u2192 RESOLVED{_Color.reset}")
            msg = msg.replace("\u2192 FAILED", f"{_Color.red}\u2192 FAILED{_Color.reset}")
            msg = msg.replace("\u2699 ", f"{_Color.magenta}\u2699 {_Color.reset}")
            msg = msg.replace("\u2190 ", f"{_Color.green}\u2190 {_Color.reset}")
            return f"{tag} {msg}"

    _log_ = logging.getLogger("litert_agent")
    _log_.setLevel(logging.INFO)
    _h_ = logging.StreamHandler(sys.stdout)
    _h_.setFormatter(_LogFormatter("%(message)s"))
    _log_.handlers.clear()
    _log_.addHandler(_h_)
    _log_.propagate = False
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Compare event sourcing vs traditional CRUD for a high-scale fintech "
        "system and provide a concrete architecture recommendation"
    )
    asyncio.run(Agent(goal).run())