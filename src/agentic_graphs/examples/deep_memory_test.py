#!/usr/bin/env python3
"""Deep memory test — exercises triplets + skills, then verifies FalkorDB storage.

Usage:
    uv run python -m agentic_graphs.examples.deep_memory_test
"""

import asyncio

from agentic_graphs import Agent, OpenAILLM, tool
from agentic_graphs.core.falkordb_backend import FalkorDBBackend
from agentic_graphs.llm.base import Message


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


_TOOLS = {"multiply": multiply, "add": add}
_SCHEMAS = [multiply.schema, add.schema]
_USER = "deep_test_alice"
_PROJECT = "deep_memory_test"


def _verify(backend: FalkorDBBackend, label: str, condition: bool):
    status = "✓" if condition else "✗"
    print(f"  {status} {label}")
    return condition


async def main():
    backend = None
    try:
        backend = FalkorDBBackend()
    except Exception as e:
        print(f"FalkorDB unavailable: {e}")
        return

    # ── clean slate ─────────────────────────────────────────────────────
    for g in [_PROJECT, backend._memory_graph(_USER)]:
        try:
            backend._fg(g).query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

    history: list[Message] = []
    passed = 0
    total = 0

    # ── Turn 1: introduce user ──────────────────────────────────────────
    msg = "My name is Alice, I live in London, and I'm allergic to peanuts"
    print(f"\nTURN 1: {msg}")
    history.append({"role": "user", "content": msg})
    agent = Agent(
        OpenAILLM(), msg,
        graph_name=_PROJECT, user_id=_USER,
        history_messages=history[:-1],
    )
    if backend:
        agent.attach_backend(backend)
    await agent.run()
    history.append({"role": "assistant", "content": "ok"})

    # ── Verify triplets after turn 1 ────────────────────────────────────
    fg = backend._fg(backend._memory_graph(_USER))
    r = fg.query("MATCH (s:Entity)-[r]->(o:Entity) RETURN s._name, type(r), o._name")
    facts = {(row[0], row[1], row[2]) for row in r.result_set}
    total += 1
    e = ("Alice", "LIVES_IN", "London") in facts or any(
        "alice" in str(row[0]).lower() and "london" in str(row[2]).lower()
        for row in r.result_set
    )
    passed += _verify(backend, "Alice lives in London triplet stored", e)
    total += 1
    e2 = any("peanut" in str(row[2]).lower() for row in r.result_set)
    passed += _verify(backend, "Peanut allergy triplet stored", e2)

    # ── Turn 2: another personal fact + simple calc ─────────────────────
    msg = "I also love Italian food. What is 15 * 4?"
    print(f"\nTURN 2: {msg}")
    history.append({"role": "user", "content": msg})
    agent = Agent(
        OpenAILLM(), msg,
        graph_name=_PROJECT, user_id=_USER,
        history_messages=history[:-1],
        extra_action_tools=_TOOLS,
        extra_action_schemas=_SCHEMAS,
    )
    if backend:
        agent.attach_backend(backend)
    await agent.run()
    history.append({"role": "assistant", "content": "ok"})

    # ── Verify ──────────────────────────────────────────────────────────
    r = fg.query("MATCH (s:Entity)-[r]->(o:Entity) RETURN s._name, type(r), o._name")
    facts = {(row[0], row[1], row[2]) for row in r.result_set}
    total += 1
    e3 = any("alice" in str(row[0]).lower() and "italian" in str(row[2]).lower()
             for row in r.result_set)
    passed += _verify(backend, "Alice loves Italian food triplet stored", e3)
    print(f"    Current triplets: {len(facts)}")

    # ── Turn 3: a multi-step problem → should create a skill ────────────
    msg = "A box weighs 8 kg. Convert to grams (1 kg = 1000 g)"
    print(f"\nTURN 3: {msg}")
    history.append({"role": "user", "content": msg})
    agent = Agent(
        OpenAILLM(), msg,
        graph_name=_PROJECT, user_id=_USER,
        history_messages=history[:-1],
        extra_action_tools=_TOOLS,
        extra_action_schemas=_SCHEMAS,
    )
    if backend:
        agent.attach_backend(backend)
    await agent.run()
    history.append({"role": "assistant", "content": "ok"})

    # ── Verify skill created ────────────────────────────────────────────
    r = fg.query("MATCH (s:Skill) RETURN s._name, s._description, s._procedure")
    total += 1
    has_skill = len(r.result_set) > 0
    passed += _verify(backend, "At least one Skill node stored", has_skill)
    for row in r.result_set:
        print(f"    Skill: {row[0]}")
    if not has_skill:
        print("    (Agent may not have called create_skill — checking triplets instead)")

    # ── Turn 4: similar problem → should find skill ─────────────────────
    msg = "A parcel weighs 3.5 kg. Convert to grams."
    print(f"\nTURN 4: {msg}")
    history.append({"role": "user", "content": msg})
    agent = Agent(
        OpenAILLM(), msg,
        graph_name=_PROJECT, user_id=_USER,
        history_messages=history[:-1],
        extra_action_tools=_TOOLS,
        extra_action_schemas=_SCHEMAS,
    )
    if backend:
        agent.attach_backend(backend)
    await agent.run()
    history.append({"role": "assistant", "content": "ok"})

    # ── Query memory about Alice ────────────────────────────────────────
    agent = Agent(
        OpenAILLM(), "What do you know about Alice?",
        graph_name=_PROJECT, user_id=_USER,
    )
    if backend:
        agent.attach_backend(backend)
    reply = await agent.run()

    total += 1
    knows_name = "alice" in reply.lower()
    knows_london = "london" in reply.lower()
    knows_peanut = "peanut" in reply.lower() or "allerg" in reply.lower()
    knows_italian = "italian" in reply.lower()
    passed += _verify(backend, f"Remembers name ({knows_name})", knows_name)
    total += 1
    passed += _verify(backend, f"Remembers London ({knows_london})", knows_london)
    total += 1
    passed += _verify(backend, f"Remembers peanut allergy ({knows_peanut})", knows_peanut)
    total += 1
    passed += _verify(backend, f"Remembers Italian food ({knows_italian})", knows_italian)
    print(f"    Reply: {reply[:200]}")

    # ── Final database dump ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL DATABASE STATE — memory:{_USER}")
    print(f"{'='*60}")

    r = fg.query("MATCH (s:Entity)-[r]->(o:Entity) RETURN s._name, type(r), o._name")
    print(f"\nTRIPLETS ({len(r.result_set)}):")
    for row in r.result_set:
        print(f"  {row[0]} --[{row[1]}]--> {row[2]}")

    r = fg.query("MATCH (s:Skill) RETURN s._name, LEFT(s._description, 60), LEFT(s._procedure, 80)")
    print(f"\nSKILLS ({len(r.result_set)}):")
    for row in r.result_set:
        print(f"  {row[0]}")
        print(f"    Desc: {row[1]}")
        print(f"    Proc: {row[2]}...")

    # Check entity embeddings exist
    r = fg.query("MATCH (e:Entity) WHERE e._embedding IS NOT NULL RETURN count(e) as n")
    emb_count = r.result_set[0][0] if r.result_set else 0
    r = fg.query("MATCH (e:Entity) RETURN count(e) as n")
    ent_count = r.result_set[0][0] if r.result_set else 0
    print(f"\nEntity nodes: {ent_count} (all with embeddings: {emb_count == ent_count})")

    r = fg.query("MATCH (s:Skill) WHERE s._embedding IS NOT NULL RETURN count(s) as n")
    sk_emb = r.result_set[0][0] if r.result_set else 0
    r = fg.query("MATCH (s:Skill) RETURN count(s) as n")
    sk_count = r.result_set[0][0] if r.result_set else 0
    print(f"Skill nodes: {sk_count} (all with embeddings: {sk_emb == sk_count})")

    print(f"\n{'='*60}")
    print(f"RESULT: {passed}/{total} checks passed")
    if passed == total:
        print("ALL CHECKS PASSED — memory system working correctly")
    else:
        print(f"{total - passed} check(s) failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
