"""
session/session.py — Session: the live handle between a user and an agent.

A Session bridges the conversational layer (Threads / Turns) and the
graph-execution layer (Agent / scheduler).  The key insight is:

    Every user message -> a GOAL node in a fresh Graph.
    The Graph is executed by the agent scheduler.
    The synthesised reply is stored back on the Turn.
    Historical turns are injected as LLM context for the next turn.

This gives you:
  - Full per-turn graph traceability in FalkorDB.
  - Chat history automatically woven into LLM context.
  - Clean resumability: recreate a Session from a Thread to continue.

Usage::

    session = await Session.create(
        llm=OpenAILLM(),
        agent_class=MyAgent,
        thread_name="Support chat",
        user_id="alice",
        backend=FalkorDBBackend(),
    )

    reply = await session.chat("What is the capital of Kenya?")
    reply = await session.chat("And the population?")

    # Retrieve history later:
    thread = store.get_thread(session.thread_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Type, AsyncIterator

from agentic_graphs.core.graph import Graph, Node, NT, S, _uid as graph_uid
from agentic_graphs.core.falkordb_backend import FalkorDBBackend
from agentic_graphs.llm.base import LLM, Message
from agentic_graphs.agent.base import Agent
from agentic_graphs.session.models import Thread, Turn, TurnStatus, SessionConfig, _uid
from agentic_graphs.session.store import ThreadStore

log = logging.getLogger(__name__)


class Session:
    """A live conversation session backed by a Thread in FalkorDB.

    Create via ``Session.create(...)`` (async factory).
    Send messages via ``await session.chat(text)``.
    """

    def __init__(
        self,
        llm: LLM,
        agent_class: Type[Agent],
        thread: Thread,
        store: ThreadStore,
        config: SessionConfig | None = None,
        agent_kwargs: dict | None = None,
    ):
        self.llm = llm
        self.agent_class = agent_class
        self.thread = thread
        self.store = store
        self.config = config or SessionConfig()
        self._agent_kwargs = agent_kwargs or {}
        self._current_agent: Agent | None = None

    # -- factory -----------------------------------------------------------

    @classmethod
    async def create(
        cls,
        llm: LLM,
        agent_class: Type[Agent],
        backend: FalkorDBBackend,
        thread_name: str = "",
        user_id: str = "",
        thread_id: str | None = None,
        config: SessionConfig | None = None,
        agent_kwargs: dict | None = None,
        metadata: dict | None = None,
    ) -> "Session":
        """Create a new Session (and Thread) or resume an existing one.

        Pass ``thread_id`` to resume an existing thread from FalkorDB.
        """
        store = ThreadStore(backend)

        if thread_id:
            thread = await asyncio.to_thread(store.get_thread, thread_id)
            if thread is None:
                raise ValueError(f"Thread {thread_id!r} not found in FalkorDB")
        else:
            thread = await asyncio.to_thread(
                store.create_thread,
                thread_name or f"Session {_uid()[:6]}",
                user_id,
                None,
                metadata,
            )

        return cls(llm, agent_class, thread, store, config, agent_kwargs)

    # -- properties --------------------------------------------------------

    @property
    def thread_id(self) -> str:
        return self.thread.id

    @property
    def turn_count(self) -> int:
        return self.thread.turn_count

    # -- main chat interface -----------------------------------------------

    async def chat(self, message: str, metadata: dict | None = None) -> str:
        """Send a user message, execute the agent graph, return the reply.

        This is the primary public interface.  Internally it:
          1. Creates a Turn and persists it (status=PENDING).
          2. Builds a fresh Graph with the message as the GOAL node.
          3. Injects prior conversation turns as LLM context.
          4. Runs the agent scheduler.
          5. Stores the reply on the Turn (status=DONE).
          6. Returns the reply string.
        """
        # 1. Persist the turn (PENDING)
        turn = await asyncio.to_thread(
            self.store.add_turn, self.thread.id, message, metadata
        )
        await asyncio.to_thread(
            self.store.update_turn_status,
            self.thread.id, turn.id, TurnStatus.RUNNING,
        )
        self.thread.turns.append(turn)

        # Auto-name the thread from the first message
        if self.thread.turn_count == 1 and self.thread.name.startswith("Session "):
            new_name = message[:60] + ("\u2026" if len(message) > 60 else "")
            self.thread.name = new_name
            await asyncio.to_thread(self.store._write_thread_node, self.thread)

        # 2. Build the agent with this turn's graph name
        history = self.thread.history_messages(self.config.max_history_turns)[:-1]

        agent = self.agent_class(
            llm=self.llm,
            goal=message,
            graph_name=turn.graph_name,
            max_iterations=self.config.max_graph_iterations,
            history_messages=history,
            **self._agent_kwargs,
        )
        agent.attach_backend(self.store._b)
        self._current_agent = agent

        # 3+4. Cross-turn semantic memory: find similar past GOALs and inject
        # their resolved outputs as extra context for the LLM.
        try:
            similar_context = []
            try:
                emb = await self.llm.embed(message)
                past = self.store._b.find_similar_goals(
                    emb, turn.graph_name, k=3, threshold=0.82,
                )
                for p in past:
                    if p["output"]:
                        similar_context.append(
                            f"[Past: {p['label']}]\n{p['output']}"
                        )
            except (NotImplementedError, Exception):
                pass

            if similar_context:
                context_block = (
                    "\n\n━━ RELATED PAST EXPERIENCE ━━\n"
                    + "\n\n".join(similar_context)
                    + "\n\n━━ END PAST EXPERIENCE ━━"
                )
                # Inject into the agent's history so the LLM sees it
                agent.history_messages.append({
                    "role": "system",
                    "content": context_block,
                })
                log.info("Injected %d similar past GOAL(s) as context",
                         len(similar_context))

            # Persist initial GOAL, then run (incremental upserts only)
            if agent._root_id and agent._root_id in agent.graph.nodes:
                agent._persist_node(agent.graph.nodes[agent._root_id])
            reply = await agent.run()

            # Store this GOAL's embedding for future cross-turn retrieval
            if agent._root_id:
                try:
                    emb = await self.llm.embed(message)
                    self.store._b.store_goal_embedding(
                        agent._root_id, emb, turn.graph_name,
                    )
                except (NotImplementedError, Exception):
                    pass
        except Exception as exc:
            log.error("Turn %s failed: %s", turn.id, exc)
            await asyncio.to_thread(
                self.store.resolve_turn,
                self.thread.id, turn.id,
                f"[error: {exc}]", TurnStatus.ERROR,
            )
            turn.assistant_reply = f"[error: {exc}]"
            turn.status = TurnStatus.ERROR
            raise

        # 5. Persist the reply
        await asyncio.to_thread(
            self.store.resolve_turn,
            self.thread.id, turn.id, reply, TurnStatus.DONE,
        )
        turn.assistant_reply = reply
        turn.status = TurnStatus.DONE

        return reply

    # -- history access ----------------------------------------------------

    def history(self) -> list[dict]:
        """Return all turns as a list of dicts (suitable for JSON APIs)."""
        return [t.to_dict() for t in self.thread.turns]

    def messages(self) -> list[dict]:
        """Return flat OpenAI-style message list from the full thread."""
        return self.thread.history_messages(max_turns=9999)

    async def load_turn_graph(self, turn_index: int):
        """Load the full execution graph for a given turn from FalkorDB.

        The graph includes GOAL/TASK/ACTION/SYNTHESIS nodes plus the
        MESSAGE/TOOLCALL traceability subgraph with HAS_MSG, NEXT,
        CALLED, and RETURNED relationships — all as first-class queryable
        nodes and edges.
        """
        if turn_index < 0 or turn_index >= len(self.thread.turns):
            raise IndexError(f"Turn index {turn_index} out of range "
                             f"(0-{len(self.thread.turns) - 1})")
        gname = self.thread.turns[turn_index].graph_name
        return await asyncio.to_thread(self.store._b.load, gname)

    async def refresh(self) -> None:
        """Reload the thread from FalkorDB (picks up changes from other processes)."""
        refreshed = await asyncio.to_thread(self.store.get_thread, self.thread.id)
        if refreshed:
            self.thread = refreshed
