"""
session/models.py — Core data models for threads, sessions, and turns.

Design philosophy:
  A *Thread* is a named, persistent conversation context owned by a user.
  Each human message sent to a Thread creates a *Turn*. A Turn holds the raw
  user text AND the Graph that was built and executed to answer it.  This
  means you get full graph-level traceability for every reply — not just
  a flat message log.

  A *Session* is an ephemeral in-process handle to an active Thread. It holds
  the LLM, the agent class, and the live Graph for the current Turn. When the
  session is closed (or the process restarts) the state is fully recoverable
  from FalkorDB via the Thread's turn history.

  Hierarchy:
      User --< Thread --< Turn --< Graph (nodes + edges in FalkorDB)

  FalkorDB graph names follow the pattern:
      thread:<thread_id>:turn:<turn_index>
  e.g. "thread:abc123:turn:0042"

  This means every turn's graph is independently browsable in the FalkorDB UI.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


def _uid() -> str:
    return uuid.uuid4().hex[:12]


class TurnStatus(str, Enum):
    PENDING   = "pending"    # created, graph not yet running
    RUNNING   = "running"    # scheduler in flight
    DONE      = "done"       # all nodes resolved
    ERROR     = "error"      # scheduler hit an unrecoverable failure


@dataclass
class Turn:
    """One human->assistant exchange within a Thread.

    Attributes:
        id:           Unique turn identifier (hex).
        index:        Sequential position within the thread (0-based).
        user_message: The raw text the user sent.
        assistant_reply: The synthesised text reply (populated after run).
        graph_name:   FalkorDB graph key storing this turn's execution graph.
        status:       Lifecycle state.
        created_at:   Unix timestamp (float).
        metadata:     Arbitrary caller-supplied key-value pairs.
    """
    id:               str
    index:            int
    user_message:     str
    assistant_reply:  str       = ""
    graph_name:       str       = ""
    status:           TurnStatus = TurnStatus.PENDING
    created_at:       float     = field(default_factory=time.time)
    metadata:         dict      = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "index":           self.index,
            "user_message":    self.user_message,
            "assistant_reply": self.assistant_reply,
            "graph_name":      self.graph_name,
            "status":          self.status.value,
            "created_at":      self.created_at,
            "metadata":        self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(
            id              = d["id"],
            index           = d["index"],
            user_message    = d["user_message"],
            assistant_reply = d.get("assistant_reply", ""),
            graph_name      = d.get("graph_name", ""),
            status          = TurnStatus(d.get("status", "pending")),
            created_at      = d.get("created_at", time.time()),
            metadata        = d.get("metadata", {}),
        )

    # Convenience: produce an OpenAI-style message pair for LLM context injection
    def to_messages(self) -> list[dict]:
        msgs = [{"role": "user", "content": self.user_message}]
        if self.assistant_reply:
            msgs.append({"role": "assistant", "content": self.assistant_reply})
        return msgs


@dataclass
class Thread:
    """A named, persistent conversation thread.

    Attributes:
        id:         Unique thread identifier.
        name:       Human-readable title (defaults to first message snippet).
        user_id:    Opaque owner identifier (your app's user ID or email).
        turns:      Ordered list of Turns (newest last).
        created_at: Unix timestamp.
        metadata:   Arbitrary caller-supplied key-value pairs.
    """
    id:         str
    name:       str
    user_id:    str               = ""
    turns:      list[Turn]        = field(default_factory=list)
    created_at: float             = field(default_factory=time.time)
    metadata:   dict              = field(default_factory=dict)

    # -- accessors ---------------------------------------------------------

    @property
    def last_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def get_turn(self, index: int) -> Turn | None:
        if 0 <= index < len(self.turns):
            return self.turns[index]
        return None

    def history_messages(self, max_turns: int = 20) -> list[dict]:
        """Return a flat list of OpenAI-style messages from recent turns.

        Used to inject conversation context into the next LLM call.
        """
        msgs: list[dict] = []
        for t in self.turns[-max_turns:]:
            msgs.extend(t.to_messages())
        return msgs

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "name":       self.name,
            "user_id":    self.user_id,
            "turns":      [t.to_dict() for t in self.turns],
            "created_at": self.created_at,
            "metadata":   self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Thread":
        t = cls(
            id         = d["id"],
            name       = d.get("name", ""),
            user_id    = d.get("user_id", ""),
            created_at = d.get("created_at", time.time()),
            metadata   = d.get("metadata", {}),
        )
        t.turns = [Turn.from_dict(td) for td in d.get("turns", [])]
        return t


@dataclass
class SessionConfig:
    """Runtime configuration for a Session."""
    max_history_turns: int   = 20    # how many past turns to inject as context
    max_graph_iterations: int = 50   # scheduler iteration cap per turn
    retry_failed_nodes: bool  = True
    stream: bool              = False  # reserved for future streaming support
    metadata: dict            = field(default_factory=dict)
