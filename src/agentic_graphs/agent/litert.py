"""LiteRTAgent — Agent subclass for LiteRT-LM on-device inference.

LiteRT's ``Engine.create_conversation(tools=...)`` manages the entire tool-call
loop internally (call → execute → feed result → continue).  This agent
forwards tool implementations to the LLM before each ``generate()`` call so
that ALL tools (built-in graph mutation tools + user tools) are available
to the LiteRT SDK.

Tool calls are captured from the LiteRT conversation and persisted to the
FalkorDB graph as TOOLCALL nodes for full traceability.
"""

from __future__ import annotations

from agentic_graphs.agent.base import Agent
from agentic_graphs.core.graph import Node, Edge, S, NT, ET, _uid


class LiteRTAgent(Agent):
    """Agent subclass for use with ``LiteRTLLM``.

    Automatically forwards the full set of tool implementations (built-in
    graph mutation tools + user-defined tools) to the LLM before every
    ``generate()`` call, enabling the LiteRT SDK to handle the tool-call
    loop internally.
    """

    async def _call_llm(self, node: Node) -> str:
        _, tool_fns = self.build_tools(node)
        if hasattr(self.llm, "set_tool_fns"):
            self.llm.set_tool_fns(tool_fns)
        result = await super()._call_llm(node)

        # Persist tool calls as TOOLCALL nodes in FalkorDB
        if self._backend:
            self._persist_toolcalls(node)

        return result

    def _persist_toolcalls(self, node: Node) -> None:
        """Create TOOLCALL nodes in the graph from the captured tool call history."""
        try:
            from agentic_graphs.llm.litert import _tool_call_history
        except ImportError:
            return

        history = list(_tool_call_history)
        if not history:
            return

        for entry in history:
            tc_node = Node(
                id=_uid(),
                type=NT.TOOLCALL,
                label=entry["name"],
                state=S.RESOLVED,
                output=entry.get("result") or entry.get("error") or "",
                props={
                    "arguments": entry["arguments"],
                    "result": entry.get("result"),
                    "error": entry.get("error"),
                },
            )
            self.graph.add_node(tc_node)
            self._persist_node(tc_node)
            e = Edge(id=_uid(), type=ET.CALLED, src=node.id, dst=tc_node.id)
            self.graph.add_edge(e)
            self._persist_edge(e, self.graph.nodes[e.src].type,
                              self.graph.nodes[e.dst].type)
            if self._root_id and self._root_id != node.id:
                pe = Edge(id=_uid(), type=ET.PART_OF,
                          src=tc_node.id, dst=self._root_id)
                self.graph.add_edge(pe)
                self._persist_edge(pe, self.graph.nodes[pe.src].type,
                                  self.graph.nodes[pe.dst].type)
