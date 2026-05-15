"""LLM providers for agentic_graphs.

Available providers (import only the ones whose SDK you have installed):

    from agentic_graphs.llm import OpenAILLM       # pip install openai
    from agentic_graphs.llm import AnthropicLLM    # pip install anthropic
    from agentic_graphs.llm import GeminiLLM       # pip install google-generativeai
    from agentic_graphs.llm import GroqLLM         # pip install groq
    from agentic_graphs.llm import OllamaLLM       # pip install httpx  (needs local Ollama)
    from agentic_graphs.llm import AzureOpenAILLM  # pip install openai
    from agentic_graphs.llm import LiteRTLLM       # pip install litert-lm
"""

from agentic_graphs.llm.base import LLM, Message, ToolCall, Chunk
from agentic_graphs.llm.openai import OpenAILLM

__all__ = [
    "LLM", "Message", "ToolCall", "Chunk",
    "OpenAILLM",
    # Lazy imports below — only fail at usage time if SDK missing
    "AnthropicLLM",
    "GeminiLLM",
    "GroqLLM",
    "OllamaLLM",
    "AzureOpenAILLM",
    "LiteRTLLM",
]


def __getattr__(name: str):
    _map = {
        "AnthropicLLM":   ("agentic_graphs.llm.anthropic",   "AnthropicLLM"),
        "GeminiLLM":      ("agentic_graphs.llm.gemini",      "GeminiLLM"),
        "GroqLLM":        ("agentic_graphs.llm.groq",        "GroqLLM"),
        "OllamaLLM":      ("agentic_graphs.llm.ollama",      "OllamaLLM"),
        "AzureOpenAILLM": ("agentic_graphs.llm.azureopenai", "AzureOpenAILLM"),
        "LiteRTLLM":      ("agentic_graphs.llm.litert",      "LiteRTLLM"),
    }
    if name in _map:
        import importlib
        mod_path, cls_name = _map[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    raise AttributeError(f"module 'agentic_graphs.llm' has no attribute {name!r}")
