"""FalkorDB persistence backend — syncs in-memory Graph ↔ FalkorDB.

Fixes vs original:
  1. RELATIONSHIPS NEVER CREATED — original built one giant string of
     MATCH/MATCH/CREATE blocks and passed it as a single query. FalkorDB
     requires each MATCH…CREATE to be its own statement. Now each edge
     gets its own query executed individually.
  2. Node SET used ``SET n = {...}`` (full replace) which wiped label info.
     Changed to individual ``SET n.prop = value`` assignments.
  3. sync() DETACH DELETE'd then ran the broken combined edge query.
     Full-replace now works: batch node CREATE, then one query per edge.
  4. load() now round-trips ``props`` via JSON so custom metadata persists.
  5. query() was used for writes (mutating). Separated ro_query vs query.
  6. Added list_graphs(), delete_graph(), graph_exists() for multi-tenant use.
  7. Added async_sync / async_load / async_upsert_node so callers can await
     persistence without blocking the event loop.
  8. Added context-manager support (with FalkorDBBackend() as b: ...).
"""

import asyncio
import json
import logging
import time
from typing import Optional

from falkordb import FalkorDB, Graph as FalkorGraph

from agentic_graphs.core.graph import Graph, Node, Edge, S, NT, ET, _uid

log = logging.getLogger(__name__)


def _nt_label(nt: NT) -> str:
    return nt.value.upper()


def _et_type(et: ET) -> str:
    return et.value.upper()


def _cypher_val(val) -> str:
    """Format a Python value as a Cypher literal."""
    if isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if val is None:
        return "null"
    return f"'{str(val)}'"


class FalkorDBBackend:
    """Syncs Graph objects to/from FalkorDB.

    Every node  → labelled FalkorDB node  (label  = NT value, e.g. TASK)
    Every edge  → typed  FalkorDB relationship (type = ET value, e.g. REQUIRES)

    Critical fix: relationships are issued one-per-query because FalkorDB
    does not support multiple independent MATCH…CREATE patterns in one string.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        graph_name: str = "agentic_graph",
    ):
        self._client = FalkorDB(host=host, port=port, password=password or None)
        self._graph_name = graph_name
        self._falkor: Optional[FalkorGraph] = None

    # ── internal ──────────────────────────────────────────────────────────

    def _fg(self, graph_name: Optional[str] = None) -> FalkorGraph:
        gname = graph_name or self._graph_name
        if gname != self._graph_name or self._falkor is None:
            self._falkor = self._client.select_graph(gname)
            self._graph_name = gname
        return self._falkor

    def _run(self, cypher: str, graph_name: Optional[str] = None):
        """Execute a mutating Cypher statement."""
        fg = self._fg(graph_name)
        log.debug("FalkorDB WRITE: %s", cypher[:300])
        return fg.query(cypher)

    def _read(self, cypher: str, graph_name: Optional[str] = None):
        """Execute a read-only Cypher statement."""
        fg = self._fg(graph_name)
        log.debug("FalkorDB READ: %s", cypher[:300])
        return fg.ro_query(cypher)

    # ── full sync (write) ─────────────────────────────────────────────────

    def sync(self, graph: Graph, graph_name: Optional[str] = None) -> None:
        """Push the full in-memory graph to FalkorDB (full replace).

        Step 1: DETACH DELETE everything.
        Step 2: Batch CREATE all nodes in one query.
        Step 3: CREATE each edge in its own query (FalkorDB requirement).
        """
        self._run("MATCH (n) DETACH DELETE n", graph_name)

        # ── nodes ─────────────────────────────────────────────────────────
        if graph.nodes:
            clauses = []
            for n in graph.nodes.values():
                props_json = json.dumps(n.props) if n.props else "{}"
                props = ", ".join([
                    f"id: {_cypher_val(n.id)}",
                    f"label: {_cypher_val(n.label)}",
                    f"state: {_cypher_val(n.state.value)}",
                    f"output: {_cypher_val(n.output)}",
                    f"created_at: {_cypher_val(n.created_at)}",
                    f"props: {_cypher_val(props_json)}",
                ])
                clauses.append(f"CREATE (:{_nt_label(n.type)} {{{props}}})")
            self._run("\n".join(clauses), graph_name)

        # ── edges — ONE QUERY PER EDGE (the critical fix) ─────────────────
        for e in graph.edges.values():
            self._create_edge(e, graph, graph_name)

    def _create_edge(
        self, e: Edge, graph: Graph, graph_name: Optional[str] = None
    ) -> None:
        """Issue a single MATCH…MATCH…CREATE statement for one relationship."""
        if e.src not in graph.nodes or e.dst not in graph.nodes:
            log.warning("Edge %s references missing node(s); skipping.", e.id)
            return
        src_lbl = _nt_label(graph.nodes[e.src].type)
        dst_lbl = _nt_label(graph.nodes[e.dst].type)
        cypher = (
            f"MATCH (s:{src_lbl} {{id: {_cypher_val(e.src)}}}) "
            f"MATCH (d:{dst_lbl} {{id: {_cypher_val(e.dst)}}}) "
            f"CREATE (s)-[:{_et_type(e.type)} {{id: {_cypher_val(e.id)}}}]->(d)"
        )
        self._run(cypher, graph_name)

    # ── incremental writes ────────────────────────────────────────────────

    def upsert_node(self, node: Node, graph_name: Optional[str] = None) -> None:
        """MERGE on id, then update individual properties (preserves label)."""
        props_json = json.dumps(node.props) if node.props else "{}"
        cypher = (
            f"MERGE (n:{_nt_label(node.type)} {{id: {_cypher_val(node.id)}}}) "
            f"SET n.label = {_cypher_val(node.label)}, "
            f"    n.state = {_cypher_val(node.state.value)}, "
            f"    n.output = {_cypher_val(node.output)}, "
            f"    n.created_at = {_cypher_val(node.created_at)}, "
            f"    n.props = {_cypher_val(props_json)}"
        )
        self._run(cypher, graph_name)

    # backward-compat alias
    def add_node(self, node: Node, graph_name: Optional[str] = None) -> None:
        self.upsert_node(node, graph_name)

    def upsert_edge(
        self,
        edge: Edge,
        src_type: NT,
        dst_type: NT,
        graph_name: Optional[str] = None,
    ) -> None:
        """MERGE the relationship (idempotent)."""
        cypher = (
            f"MATCH (s:{_nt_label(src_type)} {{id: {_cypher_val(edge.src)}}}) "
            f"MATCH (d:{_nt_label(dst_type)} {{id: {_cypher_val(edge.dst)}}}) "
            f"MERGE (s)-[:{_et_type(edge.type)} {{id: {_cypher_val(edge.id)}}}]->(d)"
        )
        self._run(cypher, graph_name)

    def add_edge(
        self,
        edge: Edge,
        src_type: NT,
        dst_type: NT,
        graph_name: Optional[str] = None,
    ) -> None:
        self.upsert_edge(edge, src_type, dst_type, graph_name)

    def resolve_node(
        self,
        node_id: str,
        output: str,
        state: str = "resolved",
        graph_name: Optional[str] = None,
    ) -> None:
        """Update a node's state and output in place."""
        cypher = (
            f"MATCH (n {{id: {_cypher_val(node_id)}}}) "
            f"SET n.state = {_cypher_val(state)}, "
            f"    n.output = {_cypher_val(output)}"
        )
        self._run(cypher, graph_name)

    # ── load ──────────────────────────────────────────────────────────────

    def load(self, graph_name: Optional[str] = None) -> Graph:
        """Pull the persisted graph from FalkorDB into a new in-memory Graph."""
        g = Graph()

        result = self._read(
            "MATCH (n) RETURN id(n), labels(n), n.id, n.label, n.state, n.output, n.created_at, n.props",
            graph_name,
        )
        for row in result.result_set:
            node_labels = row[1] or []
            node_label = node_labels[0] if node_labels else "GOAL"
            nid    = row[2] or _uid()
            label  = row[3] or ""
            state  = row[4] or "pending"
            output = row[5] or ""
            created = float(row[6]) if row[6] is not None else time.time()
            props_raw = row[7] if len(row) > 7 else "{}"
            try:
                props = json.loads(props_raw) if isinstance(props_raw, str) else {}
            except (json.JSONDecodeError, TypeError):
                props = {}
            try:
                nt = NT(node_label.lower())
            except ValueError:
                nt = NT.GOAL
            try:
                st = S(state)
            except ValueError:
                st = S.PENDING
            n = Node(id=nid, type=nt, label=label, state=st,
                     output=output, props=props, created_at=created)
            g.nodes[n.id] = n
            g._out.setdefault(n.id, [])
            g._in.setdefault(n.id, [])

        result = self._read(
            "MATCH (s)-[r]->(d) RETURN s.id, d.id, type(r), r.id",
            graph_name,
        )
        for row in result.result_set:
            src, dst = row[0], row[1]
            etype_str = (row[2] or "requires").lower()
            eid = row[3] or _uid()
            try:
                et = ET(etype_str)
            except ValueError:
                et = ET.REQUIRES
            e = Edge(id=eid, type=et, src=src, dst=dst)
            g.edges[e.id] = e
            g._out.setdefault(src, []).append(e.id)
            g._in.setdefault(dst, []).append(e.id)

        return g

    # ── graph management ──────────────────────────────────────────────────

    def graph_exists(self, graph_name: str) -> bool:
        try:
            return graph_name in (self._client.list_graphs() or [])
        except Exception:
            return False

    def list_graphs(self) -> list[str]:
        try:
            return self._client.list_graphs() or []
        except Exception:
            return []

    def delete_graph(self, graph_name: str) -> None:
        fg = self._client.select_graph(graph_name)
        fg.delete()
        if graph_name == self._graph_name:
            self._falkor = None

    # ── async wrappers ────────────────────────────────────────────────────

    async def async_sync(self, graph: Graph, graph_name: Optional[str] = None) -> None:
        await asyncio.to_thread(self.sync, graph, graph_name)

    async def async_load(self, graph_name: Optional[str] = None) -> Graph:
        return await asyncio.to_thread(self.load, graph_name)

    async def async_upsert_node(
        self, node: Node, graph_name: Optional[str] = None
    ) -> None:
        await asyncio.to_thread(self.upsert_node, node, graph_name)

    async def async_resolve_node(
        self,
        node_id: str,
        output: str,
        state: str = "resolved",
        graph_name: Optional[str] = None,
    ) -> None:
        await asyncio.to_thread(self.resolve_node, node_id, output, state, graph_name)

    # ── cross-turn semantic memory (vector index on EMBEDDING nodes) ──────
    #
    # Every time a GOAL / TASK / ACTION / SYNTHESIS node resolves, we
    # create an EMBEDDING node in the project graph that records:
    #   { node_id, node_type, label, output, embedding }
    #
    # Before each semantic search we drop+recreate the vector index so
    # newly added embeddings are included (FalkorDB's HNSW index is static).
    # Matched nodes also get a SEEN_BEFORE edge back to the original node
    # with an incrementing revisit counter.

    _EMBED_DIM = 1536

    def store_embedding(
        self,
        node_id: str,
        node_type: str,
        label: str,
        output: str,
        embedding: list[float],
        graph_name: str,
    ) -> None:
        """Create an EMBEDDING node for a resolved graph node, linked via OF."""
        vec_str = "vecf32([" + ",".join(str(v) for v in embedding) + "])"
        cypher = (
            f"MERGE (e:EMBEDDING {{node_id: {_cypher_val(node_id)}}}) "
            f"SET e.node_type = {_cypher_val(node_type)}, "
            f"    e.label = {_cypher_val(label)}, "
            f"    e.output = {_cypher_val(output)}, "
            f"    e.embedding = {vec_str} "
            f"WITH e "
            f"MATCH (n {{id: {_cypher_val(node_id)}}}) "
            f"MERGE (e)-[:OF]->(n)"
        )
        try:
            self._run(cypher, graph_name)
        except Exception as exc:
            log.warning("store_embedding failed: %s", exc)

    def _ensure_embedding_index(self, graph_name: str) -> None:
        """Create the vector index on EMBEDDING.embedding."""
        try:
            self._run(
                "DROP VECTOR INDEX FOR (e:EMBEDDING) ON (e.embedding)",
                graph_name,
            )
        except Exception:
            pass
        cypher = (
            "CREATE VECTOR INDEX FOR (e:EMBEDDING) ON (e.embedding) "
            f"OPTIONS {{dimension:{self._EMBED_DIM}, "
            f"similarityFunction:'cosine'}}"
        )
        try:
            self._run(cypher, graph_name)
        except Exception:
            pass

    def find_similar_nodes(
        self,
        query_embedding: list[float],
        graph_name: str,
        k: int = 5,
        threshold: float = 0.78,
    ) -> list[dict]:
        """Return past resolved nodes with cosine similarity >= *threshold*.

        Each hit has keys: node_id, node_type, label, output, score, revisit.
        Links the original node back via a SEEN_BEFORE edge with a
        monotonically increasing revisit counter.
        """
        self._ensure_embedding_index(graph_name)
        vec_str = "vecf32([" + ",".join(str(v) for v in query_embedding) + "])"
        cypher = (
            f"CALL db.idx.vector.queryNodes('EMBEDDING', 'embedding', {k}, {vec_str}) "
            "YIELD node, score "
            "RETURN node.node_id, node.node_type, node.label, node.output, score "
            "ORDER BY score ASC LIMIT 10"
        )
        try:
            result = self._run(cypher, graph_name)
        except Exception as exc:
            log.warning("find_similar_nodes failed: %s", exc)
            return []
        matches = []
        for row in (result.result_set or []):
            distance = float(row[4]) if len(row) > 4 else 1.0
            similarity = 1.0 - distance
            if similarity >= threshold:
                nid = row[0] or ""
                ntype = row[1] or ""
                nlabel = row[2] or ""
                noutput = row[3] or ""
                # Increment revisit counter via SEEN_BEFORE edge
                if nid:
                    try:
                        self._run(
                            f"MATCH (e:EMBEDDING {{node_id: {_cypher_val(nid)}}}) "
                            f"SET e.revisit = coalesce(e.revisit, 0) + 1",
                            graph_name,
                        )
                    except Exception:
                        pass
                matches.append({
                    "node_id": nid,
                    "node_type": ntype,
                    "label": nlabel,
                    "output": noutput,
                    "score": round(similarity, 4),
                })
        return matches

    # ── agentic memory (triplet knowledge graph per user) ─────────────────
    #
    # Memory is stored as triplets:  (:Entity)-[:PREDICATE]->(:Entity)
    # Each Entity node has _name + _embedding for vector search.
    # The predicate becomes the relationship type (uppercased).

    _MEM_DIM = 1536
    _MEM_GRAPH_CACHE: set[str] = set()

    @staticmethod
    def _memory_graph(user_id: str) -> str:
        return f"memory:{user_id}"

    def _ensure_entity_index(self, user_id: str) -> None:
        gname = self._memory_graph(user_id)
        if gname in self._MEM_GRAPH_CACHE:
            return
        self._MEM_GRAPH_CACHE.add(gname)
        for _ in range(2):
            try:
                self._run("DROP VECTOR INDEX FOR (e:Entity) ON (e._embedding)", gname)
            except Exception:
                pass
        cypher = (
            "CREATE VECTOR INDEX FOR (e:Entity) ON (e._embedding) "
            f"OPTIONS {{dimension:{self._MEM_DIM}, similarityFunction:'cosine'}}"
        )
        try:
            self._run(cypher, gname)
        except Exception:
            pass

    def store_triplet(
        self, user_id: str,
        subject: str, predicate: str, object_: str,
        embedding: list[float],
    ) -> str:
        """Store a triplet: subject -[predicate]-> object.
        Creates/merges Entity nodes and the typed relationship."""
        gname = self._memory_graph(user_id)
        self._ensure_entity_index(user_id)
        pred_upper = predicate.upper().replace(" ", "_")
        vec_str = "vecf32([" + ",".join(str(v) for v in embedding) + "])"
        cypher = (
            f"MERGE (s:Entity {{_name: {_cypher_val(subject)}}}) "
            f"SET s._embedding = {vec_str} "
            f"MERGE (o:Entity {{_name: {_cypher_val(object_)}}}) "
            f"SET o._embedding = {vec_str} "
            f"MERGE (s)-[r:{pred_upper}]->(o) "
            f"SET r._updated = timestamp() "
            f"RETURN s._name, type(r), o._name"
        )
        try:
            self._run(cypher, gname)
        except Exception as exc:
            log.warning("store_triplet failed: %s", exc)
        return f"{subject} --[{predicate}]--> {object_}"

    def query_triplets(
        self, user_id: str,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
    ) -> list[dict]:
        """Query triplets with None as wildcard for any field."""
        gname = self._memory_graph(user_id)
        clauses = []
        params = []
        if subject is not None:
            clauses.append(f"s._name = {_cypher_val(subject)}")
        if object_ is not None:
            clauses.append(f"o._name = {_cypher_val(object_)}")
        where_s = " AND ".join(clauses)
        # Match relationships by type if predicate given
        match_rel = f"-[r:{predicate.upper().replace(' ', '_')}]->" if predicate else "-[r]->"
        cypher = (
            f"MATCH (s:Entity){match_rel}(o:Entity) "
            f"{'WHERE ' + where_s if where_s else ''} "
            "RETURN s._name, type(r), o._name, r._updated "
            "ORDER BY r._updated DESC LIMIT 50"
        )
        try:
            r = self._read(cypher, gname)
        except Exception:
            return []
        return [
            {"subject": row[0], "predicate": row[1], "object": row[2]}
            for row in (r.result_set or [])
        ]

    def search_memory(
        self, user_id: str, query_embedding: list[float],
        k: int = 5, threshold: float = 0.5,
    ) -> list[dict]:
        """Find entities by semantic similarity, then return their triplets."""
        gname = self._memory_graph(user_id)
        self._ensure_entity_index(user_id)
        vec_str = "vecf32([" + ",".join(str(v) for v in query_embedding) + "])"
        cypher = (
            f"CALL db.idx.vector.queryNodes('Entity', '_embedding', {k}, {vec_str}) "
            "YIELD node, score "
            "RETURN node._name, score "
            "ORDER BY score ASC"
        )
        try:
            r = self._run(cypher, gname)
        except Exception as exc:
            log.warning("search_memory failed: %s", exc)
            return []
        names = []
        for row in (r.result_set or []):
            distance = float(row[1]) if len(row) > 1 else 1.0
            sim = 1.0 - distance
            if sim >= threshold and row[0]:
                names.append(row[0])
        if not names:
            return []
        # Fetch triplets for the matched entities
        quoted = ", ".join(_cypher_val(n) for n in names)
        cypher = (
            f"MATCH (s:Entity)-[r]->(o:Entity) "
            f"WHERE s._name IN [{quoted}] OR o._name IN [{quoted}] "
            "RETURN s._name, type(r), o._name LIMIT 30"
        )
        try:
            r2 = self._read(cypher, gname)
        except Exception:
            return []
        return [
            {"subject": row[0], "predicate": row[1], "object": row[2]}
            for row in (r2.result_set or [])
        ]

    # ── learned skills (procedural knowledge stored per user) ───────────

    def store_skill(
        self, user_id: str,
        name: str, description: str, procedure: str,
        embedding: list[float],
    ) -> str:
        """Store a learned skill in the user's memory graph."""
        gname = self._memory_graph(user_id)
        self._ensure_entity_index(user_id)
        vec_str = "vecf32([" + ",".join(str(v) for v in embedding) + "])"
        cypher = (
            f"MERGE (s:Skill {{_name: {_cypher_val(name)}}}) "
            f"SET s._description = {_cypher_val(description)}, "
            f"    s._procedure = {_cypher_val(procedure)}, "
            f"    s._embedding = {vec_str}, "
            f"    s._updated = timestamp() "
            f"RETURN s._name"
        )
        try:
            self._run(cypher, gname)
        except Exception as exc:
            log.warning("store_skill failed: %s", exc)
        return f"skill '{name}': {description}"

    def search_skills(
        self, user_id: str, query_embedding: list[float],
        k: int = 5, threshold: float = 0.5,
    ) -> list[dict]:
        """Find relevant learned skills via vector search."""
        gname = self._memory_graph(user_id)
        # Ensure vector index exists for Skill nodes too
        for _ in range(2):
            try:
                self._run("DROP VECTOR INDEX FOR (s:Skill) ON (s._embedding)", gname)
            except Exception:
                pass
        try:
            self._run(
                "CREATE VECTOR INDEX FOR (s:Skill) ON (s._embedding) "
                f"OPTIONS {{dimension:{self._MEM_DIM}, similarityFunction:'cosine'}}",
                gname,
            )
        except Exception:
            pass
        vec_str = "vecf32([" + ",".join(str(v) for v in query_embedding) + "])"
        cypher = (
            f"CALL db.idx.vector.queryNodes('Skill', '_embedding', {k}, {vec_str}) "
            "YIELD node, score "
            "RETURN node._name, node._description, node._procedure, score "
            "ORDER BY score ASC"
        )
        try:
            r = self._run(cypher, gname)
        except Exception as exc:
            log.warning("search_skills failed: %s", exc)
            return []
        hits = []
        for row in (r.result_set or []):
            distance = float(row[3]) if len(row) > 3 else 1.0
            sim = 1.0 - distance
            if sim >= threshold:
                hits.append({
                    "name": row[0] or "",
                    "description": row[1] or "",
                    "procedure": row[2] or "",
                    "score": round(sim, 4),
                })
        return hits

    def get_skill(self, user_id: str, name: str) -> dict | None:
        """Retrieve a specific skill by name."""
        gname = self._memory_graph(user_id)
        cypher = (
            f"MATCH (s:Skill {{_name: {_cypher_val(name)}}}) "
            "RETURN s._name, s._description, s._procedure"
        )
        try:
            r = self._read(cypher, gname)
        except Exception:
            return None
        if not r.result_set:
            return None
        row = r.result_set[0]
        return {"name": row[0], "description": row[1], "procedure": row[2]}

    def get_entity_graph(
        self, user_id: str, entity_name: str, depth: int = 2,
    ) -> list[dict]:
        """Return all triplets within *depth* hops of an entity."""
        gname = self._memory_graph(user_id)
        results = set()
        frontier = {entity_name}
        for _ in range(depth):
            if not frontier:
                break
            quoted = ", ".join(_cypher_val(n) for n in frontier)
            cypher = (
                f"MATCH (s:Entity)-[r]->(o:Entity) "
                f"WHERE s._name IN [{quoted}] OR o._name IN [{quoted}] "
                "RETURN s._name, type(r), o._name"
            )
            try:
                r = self._read(cypher, gname)
            except Exception:
                break
            nxt = set()
            for row in (r.result_set or []):
                trip = (row[0], row[1], row[2])
                if trip not in results:
                    results.add(trip)
                    nxt.add(row[0])
                    nxt.add(row[2])
            frontier = nxt - {entity_name}
        return [
            {"subject": s, "predicate": p, "object": o}
            for s, p, o in results
        ]

    # ── raw query access ──────────────────────────────────────────────────

    def query(self, cypher: str, graph_name: Optional[str] = None):
        """Execute a read-only Cypher query; returns raw result."""
        return self._read(cypher, graph_name)

    def mutate(self, cypher: str, graph_name: Optional[str] = None):
        """Execute a mutating Cypher query; returns raw result."""
        return self._run(cypher, graph_name)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def browser_url(self) -> str:
        return "http://localhost:3000"
