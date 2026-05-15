"""Legacy re-export — prefer ``from agentic_graphs import ...``."""
import sys
sys.path.insert(0, "src")
from agentic_graphs.core.graph import S, NT, ET, Node, Edge, Graph, _uid

__all__ = ["S", "NT", "ET", "Node", "Edge", "Graph", "_uid"]
