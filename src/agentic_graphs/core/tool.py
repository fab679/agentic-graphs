"""@tool decorator — auto-generates OpenAI-compatible JSON schema from
type hints and docstrings, then registers the function in a global registry.

Fix applied: UnionType and Optional were referenced before their try/except
import block, causing NameError at decoration time on Python < 3.10.
Both imports now appear before _pytype_to_jsontype.

Usage:

    @tool
    def add(a: float, b: float) -> float:
        \"\"\"Add two numbers together.

        Args:
            a: First number.
            b: Second number.
        \"\"\"
        return a + b

    add.schema          # → OpenAI tool definition dict
    add.tool_call({"a": 1, "b": 2})  # → "3.0"
"""

import inspect
import re
from typing import get_type_hints

# ── compatibility imports (must come before _pytype_to_jsontype) ──────────────
try:
    from types import UnionType          # Python 3.10+
except ImportError:
    UnionType = None                     # type: ignore[assignment,misc]

try:
    from typing import Optional, Union
except ImportError:                      # pragma: no cover
    Optional = None                      # type: ignore[assignment]
    Union = None                         # type: ignore[assignment]

# ─────────────────────────────────────────────────────────────────────────────

_TYPE_MAP = {
    str:        "string",
    int:        "integer",
    float:      "number",
    bool:       "boolean",
    type(None): "null",
}

_PARAM_DESC_RE = re.compile(r"^\s*(\w+)\s*:\s*(.+)$", re.MULTILINE)


def _pytype_to_jsontype(t: type) -> str:
    origin = getattr(t, "__origin__", None)
    # Handle Union[X, None] / Optional[X] / X | None (Python 3.10+)
    is_union = (
        (UnionType is not None and isinstance(t, UnionType))
        or origin is Union
    )
    if is_union:
        non_none = [a for a in getattr(t, "__args__", []) if a is not type(None)]
        return _pytype_to_jsontype(non_none[0]) if non_none else "string"
    return _TYPE_MAP.get(t, "string")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Return (description, {param_name: description})."""
    if not doc:
        return "", {}
    parts = doc.strip().split("\n\n", 1)
    desc = parts[0].strip().replace("\n", " ")
    param_descs: dict[str, str] = {}
    if len(parts) > 1 and "Args:" in parts[1]:
        for line in parts[1].split("\n"):
            m = _PARAM_DESC_RE.match(line)
            if m:
                param_descs[m.group(1)] = m.group(2).strip()
    return desc, param_descs


def _build_schema(fn) -> dict:
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    desc, param_descs = _parse_docstring(fn.__doc__)

    properties: dict = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls", "return"):
            continue
        t = hints.get(name, str)
        jsontype = _pytype_to_jsontype(t)
        prop: dict = {"type": jsontype}
        if name in param_descs:
            prop["description"] = param_descs[name]
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class Tool:
    """A registered tool with auto-generated schema and callable interface."""

    def __init__(self, fn):
        self.fn = fn
        self.schema = _build_schema(fn)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def tool_call(self, args: dict) -> str:
        """Execute the tool with the given arguments; always returns str."""
        result = self.fn(**args)
        return "" if result is None else str(result)

    @property
    def name(self) -> str:
        return self.fn.__name__

    @property
    def description(self) -> str:
        return self.schema["function"]["description"]


_REGISTRY: dict[str, Tool] = {}


def tool(fn=None, *, name: str | None = None):
    """Decorator that registers a function as a Tool.

    Can be used bare ``@tool`` or parametrised ``@tool(name="my_name")``.
    """
    def wrapper(f):
        t = Tool(f)
        key = name or f.__name__
        _REGISTRY[key] = t
        return t

    if fn is not None:
        return wrapper(fn)
    return wrapper


def get_tool(name: str) -> Tool | None:
    return _REGISTRY.get(name)


def list_tools() -> list[Tool]:
    return list(_REGISTRY.values())


def tool_schemas(*tools: Tool) -> list[dict]:
    """Return OpenAI-compatible schema dicts for the given tools."""
    if tools:
        return [t.schema for t in tools]
    return [t.schema for t in _REGISTRY.values()]
