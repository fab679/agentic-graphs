"""Google Gemini provider via google-generativeai SDK."""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

from agentic_graphs.llm.base import LLM, Message, Chunk

try:
    import google.generativeai as genai
    from google.generativeai.types import content_types
except ImportError as exc:
    raise ImportError(
        "Install the Google AI SDK: pip install google-generativeai"
    ) from exc


def _to_gemini_tools(tools: list[dict]) -> list:
    """Convert OpenAI-style tool schemas to Gemini FunctionDeclarations."""
    declarations = []
    for t in tools:
        fn = t.get("function", t)
        declarations.append(
            genai.protos.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        k: genai.protos.Schema(
                            type=genai.protos.Type[v.get("type", "string").upper()]
                        )
                        for k, v in fn.get("parameters", {})
                            .get("properties", {}).items()
                    },
                    required=fn.get("parameters", {}).get("required", []),
                ),
            )
        )
    return [genai.Tool(function_declarations=declarations)]


def _messages_to_gemini(messages: list[Message]) -> tuple[str, list[dict]]:
    system = ""
    history = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system = m.get("content") or ""
            continue
        if role == "tool":
            history.append({
                "role": "function",
                "parts": [{"function_response": {
                    "name": "tool_result",
                    "response": {"result": m.get("content") or ""},
                }}],
            })
            continue
        gemini_role = "model" if role == "assistant" else "user"
        history.append({"role": gemini_role, "parts": [{"text": m.get("content") or ""}]})
    return system, history


class GeminiLLM(LLM):
    """Google Gemini provider (gemini-2.0-flash, gemini-2.5-pro, etc.)."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        embed_model: str = "models/embedding-001",
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.model = model
        self.embed_model = embed_model
        self._timeout = timeout
        genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY", ""))
        self._model = genai.GenerativeModel(model)

    async def embed(self, text: str) -> list[float]:
        import asyncio
        def _sync():
            result = genai.embed_content(model=self.embed_model, content=text)
            return result["embedding"]
        return await asyncio.to_thread(_sync)

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        import asyncio
        system, history = _messages_to_gemini(messages)

        def _sync_call():
            m = genai.GenerativeModel(
                self.model,
                system_instruction=system or None,
                tools=_to_gemini_tools(tools) if tools else None,
            )
            chat = m.start_chat(history=history[:-1] if history else [])
            last = history[-1]["parts"][0]["text"] if history else ""
            return chat.send_message(last)

        response = await asyncio.wait_for(
            asyncio.to_thread(_sync_call), timeout=self._timeout,
        )

        tool_calls = []
        text = ""
        for part in response.parts:
            if hasattr(part, "text") and part.text:
                text += part.text
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_calls.append({
                    "id": fc.name,
                    "type": "function",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(dict(fc.args)),
                    },
                })

        result: Message = {"role": "assistant", "content": text}
        if tool_calls:
            result["tool_calls"] = tool_calls

        # Capture usage if available
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            result["usage"] = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0),
            }

        return result

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[Chunk]:
        # Gemini streaming is sync; fall back to non-streaming
        msg = await self.generate(messages, tools)
        yield {"content": msg.get("content") or "", "done": True}
