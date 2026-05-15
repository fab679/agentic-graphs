import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class S(str, Enum):
    PENDING  = "pending"
    READY    = "ready"
    ACTIVE   = "active"
    RESOLVED = "resolved"
    FAILED   = "failed"


class NT(str, Enum):
    GOAL      = "goal"
    TASK      = "task"
    ACTION    = "action"
    SYNTHESIS = "synthesis"
    MESSAGE   = "message"
    TOOLCALL  = "toolcall"


class ET(str, Enum):
    REQUIRES    = "requires"
    PART_OF     = "part_of"
    PRODUCES    = "produces"
    HAS_MSG     = "has_msg"
    NEXT        = "next"
    CALLED      = "called"
    RETURNED    = "returned"
    SEEN_BEFORE = "seen_before"

# Node types excluded from subgraph_text() so LLM prompts stay lean
_SKIP_IN_SUBGRAPH = {NT.MESSAGE, NT.TOOLCALL}


@dataclass
class Node:
    id:    str
    type:  NT
    label: str
    state: S    = S.PENDING
    output: str = ""
    props: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class Edge:
    id:   str
    type: ET
    src:  str
    dst:  str


class Graph:
    """Directed graph that serves as the single source of truth
    for all agent state."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self._out:  dict[str, list[str]] = {}
        self._in:   dict[str, list[str]] = {}

    # ── writes ────────────────────────────────────────────────────────────

    def add_node(self, n: Node):
        self.nodes[n.id] = n
        self._out.setdefault(n.id, [])
        self._in.setdefault(n.id, [])
        self._recheck(n.id)

    def add_edge(self, e: Edge):
        self.edges[e.id] = e
        self._out.setdefault(e.src, []).append(e.id)
        self._in.setdefault(e.dst, []).append(e.id)
        if e.type == ET.REQUIRES:
            self._recheck(e.src)

    def resolve(self, nid: str, output: str = ""):
        n = self.nodes[nid]
        n.state, n.output = S.RESOLVED, output
        self._propagate(nid)

    def set_state(self, nid: str, s: S):
        self.nodes[nid].state = s

    # ── readiness ─────────────────────────────────────────────────────────

    def _recheck(self, nid: str):
        n = self.nodes[nid]
        if n.state in (S.ACTIVE, S.RESOLVED, S.FAILED):
            return
        reqs = [self.edges[eid] for eid in self._out.get(nid, [])
                if self.edges[eid].type == ET.REQUIRES]
        n.state = (S.READY
                   if all(self.nodes[e.dst].state == S.RESOLVED for e in reqs)
                   else S.PENDING)

    def _propagate(self, rid: str):
        for eid in self._in.get(rid, []):
            e = self.edges[eid]
            if e.type == ET.REQUIRES:
                self._recheck(e.src)

    # ── queries ───────────────────────────────────────────────────────────

    def ready(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.state == S.READY]

    def subgraph_text(self, focal_id: str, hops: int = 3) -> str:
        """Human-readable neighbourhood description suitable for LLM context.

        Skips internal node types (MESSAGE, TOOLCALL) to keep prompts lean.

        Traversal follows PART_OF edges from parent to child only, so
        subagent scopes don't bleed into sibling subtrees.
        """
        seen, frontier = set(), {focal_id}
        for _ in range(hops):
            nxt = set()
            for x in frontier:
                seen.add(x)
                for eid in self._out.get(x, []) + self._in.get(x, []):
                    e = self.edges[eid]
                    other = e.dst if e.src == x else e.src
                    if other in seen:
                        continue
                    # Don't follow PART_OF from a GOAL to its parent —
                    # prevents scope bleed between sibling subagents.
                    if (e.type == ET.PART_OF
                            and self.nodes.get(x, Node('','',label='',state=S.PENDING)).type == NT.GOAL
                            and e.src == x):
                        continue
                    nxt.add(other)
            frontier = nxt - seen
        seen |= frontier

        lines = ["NODES:"]
        for nid in sorted(seen):
            if nid not in self.nodes:
                continue
            n = self.nodes[nid]
            if n.type in _SKIP_IN_SUBGRAPH:
                continue
            marker = "  ← YOU ARE HERE" if nid == focal_id else ""
            lines.append(
                f"  [{n.type.upper()}] id={n.id!r}  state={n.state}"
                f"  label={n.label!r}{marker}"
            )
            if n.output:
                preview = n.output[:400] + ("…" if len(n.output) > 400 else "")
                lines.append(f"    output: {preview}")

        lines.append("\nEDGES (src --[type]--> dst):")
        for e in self.edges.values():
            if e.src in seen and e.dst in seen:
                sl = self.nodes[e.src].label if e.src in self.nodes else e.src
                dl = self.nodes[e.dst].label if e.dst in self.nodes else e.dst
                lines.append(f"  {sl!r} --[{e.type}]--> {dl!r}")

        return "\n".join(lines)

    # ── graph algorithms ──────────────────────────────────────────────────

    def topological_sort(self) -> list[Node]:
        """Return nodes in dependency order (prerequisites first).

        REQUIRES edges have semantics: src depends on dst.
        Uses Kahn's algorithm on the reversed graph.
        Raises ValueError if a cycle is detected.
        """
        rev_adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}

        for e in self.edges.values():
            if e.type == ET.REQUIRES:
                rev_adj[e.dst].append(e.src)
                in_degree[e.src] = in_degree.get(e.src, 0) + 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result: list[Node] = []
        while queue:
            nid = queue.pop(0)
            result.append(self.nodes[nid])
            for dep in rev_adj[nid]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        return result

    def has_cycle(self) -> bool:
        """Detect whether the graph contains a directed cycle."""
        visited: dict[str, int] = {nid: 0 for nid in self.nodes}  # 0=unvisited, 1=visiting, 2=done

        def _dfs(nid: str) -> bool:
            visited[nid] = 1
            for eid in self._out.get(nid, []):
                e = self.edges[eid]
                if e.type != ET.REQUIRES:
                    continue
                if visited[e.dst] == 1:
                    return True
                if visited[e.dst] == 0 and _dfs(e.dst):
                    return True
            visited[nid] = 2
            return False

        for nid in self.nodes:
            if visited[nid] == 0 and _dfs(nid):
                return True
        return False

    def subgraph(self, root_id: str, hops: int = 3) -> "Graph":
        """Extract a subgraph rooted at *root_id* (within *hops* edges).

        Returns a new Graph with copied nodes and edges.
        """
        seen: set[str] = set()
        frontier = {root_id}
        for _ in range(hops):
            nxt = set()
            for x in frontier:
                seen.add(x)
                for eid in self._out.get(x, []) + self._in.get(x, []):
                    e = self.edges[eid]
                    other = e.dst if e.src == x else e.src
                    if other not in seen:
                        nxt.add(other)
            frontier = nxt - seen
        seen |= frontier

        sg = Graph()
        for nid in seen:
            if nid in self.nodes:
                n = self.nodes[nid]
                sg.add_node(Node(id=n.id, type=n.type, label=n.label,
                                 state=n.state, output=n.output,
                                 props=dict(n.props), created_at=n.created_at))
        for e in self.edges.values():
            if e.src in seen and e.dst in seen:
                sg.add_edge(Edge(id=e.id, type=e.type, src=e.src, dst=e.dst))
        return sg

    def diff(self, other: "Graph") -> dict:
        """Compare two graphs. Returns added/removed/changed items."""
        my_ids = set(self.nodes)
        other_ids = set(other.nodes)
        added_nodes = [nid for nid in other_ids - my_ids]
        removed_nodes = [nid for nid in my_ids - other_ids]
        changed_nodes = []
        for nid in my_ids & other_ids:
            a, b = self.nodes[nid], other.nodes[nid]
            if (a.state, a.output, a.label) != (b.state, b.output, b.label):
                changed_nodes.append(nid)

        my_eids = set(self.edges)
        other_eids = set(other.edges)
        added_edges = [eid for eid in other_eids - my_eids]
        removed_edges = [eid for eid in my_eids - other_eids]

        return {
            "nodes_added": added_nodes,
            "nodes_removed": removed_nodes,
            "nodes_changed": changed_nodes,
            "edges_added": added_edges,
            "edges_removed": removed_edges,
        }

    def merge(self, other: "Graph") -> list[str]:
        """Merge another graph into this one. Returns list of new node IDs."""
        new_nodes = []
        for n in other.nodes.values():
            if n.id not in self.nodes:
                self.add_node(Node(id=n.id, type=n.type, label=n.label,
                                    state=n.state, output=n.output,
                                    props=dict(n.props), created_at=n.created_at))
                new_nodes.append(n.id)
        for e in other.edges.values():
            if e.id not in self.edges and e.src in self.nodes and e.dst in self.nodes:
                self.add_edge(Edge(id=e.id, type=e.type, src=e.src, dst=e.dst))
        return new_nodes

    # ── serialisation ─────────────────────────────────────────────────────

    def to_json(self) -> dict:
        return {
            "nodes": [
                {"id": n.id, "type": n.type.value, "label": n.label,
                 "state": n.state.value, "output": n.output,
                 "created_at": n.created_at}
                for n in self.nodes.values()
            ],
            "edges": [
                {"id": e.id, "type": e.type.value, "src": e.src, "dst": e.dst}
                for e in self.edges.values()
            ],
        }

    def save_json(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    def summary_stats(self) -> dict:
        by_s: dict = {}
        for n in self.nodes.values():
            by_s.setdefault(n.state, []).append(n)
        return {"nodes": len(self.nodes), "edges": len(self.edges),
                "by_state": by_s}


def _uid() -> str:
    return uuid.uuid4().hex[:8]
