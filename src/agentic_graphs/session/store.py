"""
session/store.py — Thread persistence via FalkorDB.

Every Thread is stored as a graph named  "meta:thread:<thread_id>".
Within that graph:

  (:THREAD {id, name, user_id, created_at, metadata_json})
      \\--[:HAS_TURN]-->
         (:TURN {id, index, user_message, assistant_reply,
                 graph_name, status, created_at, metadata_json})

This means:
  - All thread metadata and turn history lives in FalkorDB (no separate DB).
  - The FalkorDB browser shows the full conversation graph alongside
    the execution graphs.
  - Querying "all threads for user X" is a single Cypher query.
  - Turn execution graphs are separate named graphs (thread:X:turn:N)
    so they don't pollute the metadata graph.
"""

from __future__ import annotations

import json
import logging
import time

from agentic_graphs.core.falkordb_backend import FalkorDBBackend, _cypher_val
from agentic_graphs.session.models import Thread, Turn, TurnStatus, _uid

log = logging.getLogger(__name__)

_META_PREFIX = "meta:thread:"


def _thread_graph_name(thread_id: str) -> str:
    return f"{_META_PREFIX}{thread_id}"


def _turn_graph_name(thread_id: str, turn_index: int) -> str:
    return f"project:{thread_id}"


class ThreadStore:
    """Persists and retrieves Threads and Turns from FalkorDB.

    Usage::

        store = ThreadStore(FalkorDBBackend())

        thread = store.create_thread("My chat", user_id="alice")
        turn   = store.add_turn(thread.id, "What is 2+2?")

        # after agent runs:
        store.resolve_turn(thread.id, turn.id, reply="4", status=TurnStatus.DONE)

        # retrieve later:
        thread = store.get_thread(thread.id)
        for t in thread.turns:
            print(t.user_message, "->", t.assistant_reply)
    """

    def __init__(self, backend: FalkorDBBackend):
        self._b = backend

    # -- thread CRUD -------------------------------------------------------

    def create_thread(
        self,
        name: str = "",
        user_id: str = "",
        thread_id: str | None = None,
        metadata: dict | None = None,
    ) -> Thread:
        tid = thread_id or _uid()
        now = time.time()
        thread = Thread(
            id=tid,
            name=name or f"Thread {tid[:6]}",
            user_id=user_id,
            created_at=now,
            metadata=metadata or {},
        )
        self._write_thread_node(thread)
        return thread

    def _write_thread_node(self, thread: Thread) -> None:
        gname = _thread_graph_name(thread.id)
        meta_json = _cypher_val(json.dumps(thread.metadata))
        cypher = (
            f"MERGE (t:THREAD {{id: {_cypher_val(thread.id)}}}) "
            f"SET t.name = {_cypher_val(thread.name)}, "
            f"    t.user_id = {_cypher_val(thread.user_id)}, "
            f"    t.created_at = {thread.created_at}, "
            f"    t.metadata_json = {meta_json}"
        )
        self._b.mutate(cypher, gname)

    def get_thread(self, thread_id: str) -> Thread | None:
        gname = _thread_graph_name(thread_id)
        if not self._b.graph_exists(gname):
            return None

        # Read thread node
        r = self._b.query(
            f"MATCH (t:THREAD {{id: {_cypher_val(thread_id)}}}) "
            "RETURN t.id, t.name, t.user_id, t.created_at, t.metadata_json",
            gname,
        )
        if not r.result_set:
            return None
        row = r.result_set[0]
        thread = Thread(
            id=row[0],
            name=row[1] or "",
            user_id=row[2] or "",
            created_at=float(row[3] or time.time()),
            metadata=_safe_json(row[4]),
        )

        # Read turns ordered by index
        tr = self._b.query(
            "MATCH (t:THREAD)-[:HAS_TURN]->(u:TURN) "
            "RETURN u.id, u.index, u.user_message, u.assistant_reply, "
            "       u.graph_name, u.status, u.created_at, u.metadata_json "
            "ORDER BY u.index ASC",
            gname,
        )
        for trow in tr.result_set:
            thread.turns.append(Turn(
                id=trow[0] or _uid(),
                index=int(trow[1] or 0),
                user_message=trow[2] or "",
                assistant_reply=trow[3] or "",
                graph_name=trow[4] or "",
                status=_safe_status(trow[5]),
                created_at=float(trow[6] or time.time()),
                metadata=_safe_json(trow[7]),
            ))
        return thread

    def list_threads(self, user_id: str | None = None, limit: int = 100) -> list[dict]:
        """Return summary dicts for all threads (optionally filtered by user)."""
        results = []
        try:
            all_graphs = self._b.list_graphs()
        except Exception:
            return []

        for gname in all_graphs:
            if not gname.startswith(_META_PREFIX):
                continue
            try:
                cypher = (
                    "MATCH (t:THREAD) RETURN t.id, t.name, t.user_id, t.created_at"
                )
                r = self._b.query(cypher, gname)
                if not r.result_set:
                    continue
                row = r.result_set[0]
                if user_id and row[2] != user_id:
                    continue
                # Count turns
                tc = self._b.query(
                    "MATCH (t:THREAD)-[:HAS_TURN]->(u:TURN) RETURN count(u)", gname
                )
                turn_count = int(tc.result_set[0][0]) if tc.result_set else 0
                results.append({
                    "id":          row[0],
                    "name":        row[1] or "",
                    "user_id":     row[2] or "",
                    "created_at":  float(row[3] or 0),
                    "turn_count":  turn_count,
                })
            except Exception as exc:
                log.warning("Error reading thread %s: %s", gname, exc)
            if len(results) >= limit:
                break

        results.sort(key=lambda x: x["created_at"], reverse=True)
        return results

    def delete_thread(self, thread_id: str, delete_turn_graphs: bool = True) -> None:
        """Delete a thread and optionally all its execution graphs."""
        thread = self.get_thread(thread_id)
        if thread and delete_turn_graphs:
            for turn in thread.turns:
                if turn.graph_name and self._b.graph_exists(turn.graph_name):
                    try:
                        self._b.delete_graph(turn.graph_name)
                    except Exception:
                        pass
        gname = _thread_graph_name(thread_id)
        if self._b.graph_exists(gname):
            self._b.delete_graph(gname)

    # -- turn CRUD ---------------------------------------------------------

    def add_turn(
        self,
        thread_id: str,
        user_message: str,
        metadata: dict | None = None,
    ) -> Turn:
        """Create and persist a new pending Turn on the given Thread."""
        thread = self.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"Thread {thread_id!r} not found")

        index = thread.turn_count
        turn_id = _uid()
        graph_name = _turn_graph_name(thread_id, index)

        turn = Turn(
            id=turn_id,
            index=index,
            user_message=user_message,
            graph_name=graph_name,
            status=TurnStatus.PENDING,
            metadata=metadata or {},
        )
        self._write_turn_node(thread_id, turn)
        return turn

    def _write_turn_node(self, thread_id: str, turn: Turn) -> None:
        gname = _thread_graph_name(thread_id)
        meta_json = _cypher_val(json.dumps(turn.metadata))
        # Create/update the TURN node
        self._b.mutate(
            f"MERGE (u:TURN {{id: {_cypher_val(turn.id)}}}) "
            f"SET u.index = {turn.index}, "
            f"    u.user_message = {_cypher_val(turn.user_message)}, "
            f"    u.assistant_reply = {_cypher_val(turn.assistant_reply)}, "
            f"    u.graph_name = {_cypher_val(turn.graph_name)}, "
            f"    u.status = {_cypher_val(turn.status.value)}, "
            f"    u.created_at = {turn.created_at}, "
            f"    u.metadata_json = {meta_json}",
            gname,
        )
        # Connect THREAD -> HAS_TURN -> TURN
        self._b.mutate(
            f"MATCH (t:THREAD {{id: {_cypher_val(thread_id)}}}), "
            f"      (u:TURN {{id: {_cypher_val(turn.id)}}}) "
            f"MERGE (t)-[:HAS_TURN]->(u)",
            gname,
        )

    def resolve_turn(
        self,
        thread_id: str,
        turn_id: str,
        reply: str,
        status: TurnStatus = TurnStatus.DONE,
    ) -> None:
        """Update the turn's reply and status after the agent finishes."""
        gname = _thread_graph_name(thread_id)
        self._b.mutate(
            f"MATCH (u:TURN {{id: {_cypher_val(turn_id)}}}) "
            f"SET u.assistant_reply = {_cypher_val(reply)}, "
            f"    u.status = {_cypher_val(status.value)}",
            gname,
        )

    def update_turn_status(
        self, thread_id: str, turn_id: str, status: TurnStatus
    ) -> None:
        gname = _thread_graph_name(thread_id)
        self._b.mutate(
            f"MATCH (u:TURN {{id: {_cypher_val(turn_id)}}}) "
            f"SET u.status = {_cypher_val(status.value)}",
            gname,
        )

    def get_turn(self, thread_id: str, turn_id: str) -> Turn | None:
        thread = self.get_thread(thread_id)
        if thread is None:
            return None
        return next((t for t in thread.turns if t.id == turn_id), None)

    def turn_graph_name(self, thread_id: str, turn_index: int) -> str:
        return _turn_graph_name(thread_id, turn_index)


# -- helpers -------------------------------------------------------------------

def _safe_json(raw) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _safe_status(raw) -> TurnStatus:
    try:
        return TurnStatus(raw or "pending")
    except ValueError:
        return TurnStatus.PENDING
