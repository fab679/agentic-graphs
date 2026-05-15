"""
agentic_graphs — A graph-native framework for building AI agents.

Public API:
    Graph, Node, Edge, S, NT, ET    — graph primitives
    tool, Tool                      — @tool decorator with auto-schema
    LLM                             — abstract LLM interface
    OpenAILLM                       — OpenAI provider (always available)
    AnthropicLLM                    — Claude provider  (pip install anthropic)
    GeminiLLM                       — Gemini provider  (pip install google-generativeai)
    GroqLLM                         — Groq provider    (pip install groq)
    OllamaLLM                       — Ollama provider  (local, pip install httpx)
    AzureOpenAILLM                  — Azure OpenAI     (pip install openai)
    LiteRTLLM                       — On-device LiteRT (pip install litert-lm)
    Agent                           — base agent class
    LiteRTAgent                     — agent for LiteRT on-device inference
    run_scheduler, process_node     — functional scheduler
    collect_answer, set_sync_hook   — scheduler utilities
    FalkorDBBackend                 — FalkorDB persistence (fixed)
    Session                         — conversation session
    Thread, Turn, TurnStatus        — session data models
    ThreadStore                     — thread persistence
    SessionConfig                   — session runtime config
"""

from agentic_graphs.core.graph import Graph, Node, Edge, S, NT, ET, _uid
from agentic_graphs.core.tool import tool, Tool
from agentic_graphs.core.falkordb_backend import FalkorDBBackend
from agentic_graphs.llm.base import LLM
from agentic_graphs.llm.openai import OpenAILLM
from agentic_graphs.agent.base import Agent
from agentic_graphs.agent.litert import LiteRTAgent
from agentic_graphs.agent.scheduler import (
    run_scheduler, process_node, collect_answer, set_sync_hook,
)
from agentic_graphs.session.models import Thread, Turn, TurnStatus, SessionConfig
from agentic_graphs.session.store import ThreadStore
from agentic_graphs.session.session import Session

__all__ = [
    # graph
    "Graph", "Node", "Edge", "S", "NT", "ET", "_uid",
    # tools
    "tool", "Tool",
    # LLM
    "LLM", "OpenAILLM",
    # providers (lazy — only imported when accessed)
    "AnthropicLLM", "GeminiLLM", "GroqLLM",
    "OllamaLLM", "AzureOpenAILLM", "LiteRTLLM",
    # agent
    "Agent", "LiteRTAgent",
    # scheduler
    "run_scheduler", "process_node", "collect_answer", "set_sync_hook",
    # persistence
    "FalkorDBBackend",
    # session
    "Session", "Thread", "Turn", "TurnStatus", "ThreadStore", "SessionConfig",
]


def __getattr__(name: str):
    _providers = {
        "AnthropicLLM":   ("agentic_graphs.llm.anthropic",   "AnthropicLLM"),
        "GeminiLLM":      ("agentic_graphs.llm.gemini",      "GeminiLLM"),
        "GroqLLM":        ("agentic_graphs.llm.groq",        "GroqLLM"),
        "OllamaLLM":      ("agentic_graphs.llm.ollama",      "OllamaLLM"),
        "AzureOpenAILLM": ("agentic_graphs.llm.azureopenai", "AzureOpenAILLM"),
        "LiteRTLLM":      ("agentic_graphs.llm.litert",      "LiteRTLLM"),
    }
    if name in _providers:
        import importlib
        mod_path, cls_name = _providers[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    raise AttributeError(f"module 'agentic_graphs' has no attribute {name!r}")
