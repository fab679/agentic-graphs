#!/usr/bin/env python3
"""Session debugger — inspect threads, turns, and full execution graphs.

Connects to FalkorDB and displays:
  - All threads (or a specific thread via --thread <id>)
  - Every turn with its user message and reply
  - Every execution-graph node with its state, output, and the full
    LLM message transcript (tool calls + results)

Usage:
    uv run python -m agentic_graphs.examples.debug_session
    uv run python -m agentic_graphs.examples.debug_session --thread <id>
    uv run python -m agentic_graphs.examples.debug_session --user <user_id>
    uv run python -m agentic_graphs.examples.debug_session --host localhost --port 6379
    uv run python -m agentic_graphs.examples.debug_session --raw   # list all FalkorDB graph names
"""

import asyncio
import sys

from agentic_graphs.core.falkordb_backend import FalkorDBBackend
from agentic_graphs.session import ThreadStore


# ── ANSI colours ────────────────────────────────────────────────────────────

C = type("C", (), {
    "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m",
    "magenta": "\033[35m", "red": "\033[91m", "dim": "\033[2m",
    "bold": "\033[1m", "reset": "\033[0m",
})()


def _snap(s: str, n: int = 80) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "..."


def _pp(val: str, color: str = "") -> str:
    return f"{color}{val}{C.reset}"


# ── display helpers ─────────────────────────────────────────────────────────

def _show_thread_summary(t: dict):
    print(f"\n  {C.bold}{C.cyan}{'═'*68}{C.reset}")
    print(f"  {C.bold}THREAD{C.reset}  {C.cyan}{t['id']}{C.reset}")
    print(f"  {C.dim}name{C.reset}       {t['name']}")
    print(f"  {C.dim}user_id{C.reset}    {t['user_id']}")
    print(f"  {C.dim}turns{C.reset}       {t['turn_count']}")
    print(f"  {C.dim}created_at{C.reset}  {t['created_at']}")


def _show_turn(turn, graph=None):
    status_color = {
        "pending": C.yellow, "running": C.yellow,
        "done": C.green, "error": C.red,
    }.get(turn.status.value, C.dim)

    print(f"\n    {C.bold}{'─'*60}{C.reset}")
    print(f"    {C.bold}TURN {turn.index}{C.reset}  "
          f"{status_color}{turn.status.value}{C.reset}  "
          f"{C.dim}{turn.id[:10]}...{C.reset}")
    print(f"    {C.dim}graph{C.reset}  {turn.graph_name}")
    print(f"    {C.dim}user{C.reset}   {_pp(turn.user_message, C.yellow)}")
    if turn.assistant_reply:
        print(f"    {C.dim}reply{C.reset}  {_snap(turn.assistant_reply, 120)}")
    else:
        print(f"    {C.dim}reply{C.reset}  {C.dim}(none){C.reset}")

    if graph:
        print()
        _show_graph(graph)


def _show_graph(graph):
    # Separate execution nodes (GOAL/TASK/ACTION/SYNTHESIS) from
    # traceability subgraph (MESSAGE/TOOLCALL)
    exec_nodes = [n for n in graph.nodes.values()
                  if n.type.value not in ("message", "toolcall")]
    trace_nodes = [n for n in graph.nodes.values()
                   if n.type.value in ("message", "toolcall")]

    print(f"      {C.bold}{C.cyan}Execution graph: {len(exec_nodes)} nodes, "
          f"{len(graph.edges)} edges{C.reset}")

    for n in sorted(exec_nodes, key=lambda x: x.type.value):
        state_color = {
            "pending": C.dim, "ready": "blue", "active": C.yellow,
            "resolved": C.green, "failed": C.red,
        }.get(n.state.value, C.dim)

        created = f"  {C.dim}@{n.created_at:.1f}{C.reset}" if n.created_at else ""
        print(f"\n      {C.bold}[{n.type.value.upper()}]{C.reset}  "
              f"{state_color}{n.state.value}{C.reset}  "
              f"{C.dim}{n.id[:10]}...{C.reset}{created}")
        print(f"      {C.dim}label{C.reset}  {n.label}")

        if n.output:
            out_preview = _snap(n.output, 200)
            print(f"      {C.dim}output{C.reset} {out_preview}")

    # Show LLM message transcript as a sequential story
    if trace_nodes:
        msgs_by_src: dict[str, list] = {}
        for e in graph.edges.values():
            if e.type.value == "has_msg":
                msgs_by_src.setdefault(e.src, []).append(e.dst)

        for src_id, msg_ids in msgs_by_src.items():
            src = graph.nodes.get(src_id)
            src_label = f"[{src.type.value.upper()}] {src.label}" if src else src_id

            # Sort messages by idx from props
            def _msg_sort_key(mid):
                m = graph.nodes.get(mid)
                return (m.props.get("idx", 0) if m else 0)

            msg_ids.sort(key=_msg_sort_key)

            print(f"\n      {C.dim}LLM transcript for {C.reset}{C.bold}"
                  f"{src_label}{C.reset}{C.dim}:{C.reset}")
            for mid in msg_ids:
                m = graph.nodes.get(mid)
                if not m:
                    continue
                role = m.label
                content = m.output  # stored as first-class attribute, not props

                if role == "system":
                    print(f"        {C.dim}SYS  {_snap(content, 120)}{C.reset}")
                elif role == "user":
                    print(f"        {C.yellow}USR  {_snap(content, 120)}{C.reset}")
                elif role == "assistant":
                    print(f"        {C.cyan}AST  {_snap(content, 120)}{C.reset}")
                    # Tool calls linked from this message
                    for e in graph.edges.values():
                        if e.type.value == "called" and e.src == mid:
                            tc = graph.nodes.get(e.dst)
                            if tc:
                                args = _snap(tc.output, 80)
                                result = tc.props.get("result", "")
                                label = f"  \u21b3 {tc.label}({args})"
                                if result:
                                    label += f"  \u2190 {_snap(result, 60)}"
                                print(f"        {C.magenta}{label}{C.reset}")
                elif role == "tool":
                    print(f"        {C.green}RST  {_snap(content, 120)}{C.reset}")
            print()


# ── diagnostics ─────────────────────────────────────────────────────────────

def _check_falkordb(backend) -> bool:
    """Check FalkorDB connectivity and return True if reachable."""
    try:
        graphs = backend.list_graphs()
        print(f"  {C.green}\u2713 FalkorDB reachable{C.reset}  "
              f"{C.dim}{backend._client}{C.reset}")
        print(f"  {C.dim}Available graphs: {len(graphs)}{C.reset}")
        if graphs:
            meta = [g for g in graphs if g.startswith("meta:thread:")]
            exec_graphs = [g for g in graphs if not g.startswith("meta:thread:")]
            if meta:
                print(f"  {C.dim}  thread metadata: {len(meta)}{C.reset}")
            if exec_graphs:
                print(f"  {C.dim}  execution graphs: {len(exec_graphs)}{C.reset}")
        return True
    except Exception as exc:
        print(f"  {C.red}\u2717 FalkorDB unreachable{C.reset}")
        print(f"  {C.dim}Error: {exc}{C.reset}")
        print(f"\n  {C.yellow}Make sure FalkorDB is running:{C.reset}")
        print("    docker run -d -p 6379:6379 -p 3000:3000 "
              "--name falkordb falkordb/falkordb:latest")
        print(f"  {C.yellow}Then open http://localhost:3000 in your browser.{C.reset}")
        return False


# ── main ────────────────────────────────────────────────────────────────────

async def main():
    thread_id = None
    user_id = None
    host = "localhost"
    port = 6379
    raw = False

    if "--raw" in sys.argv:
        raw = True
    if "--host" in sys.argv:
        idx = sys.argv.index("--host")
        host = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else host
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        port = int(sys.argv[idx + 1]) if len(sys.argv) > idx + 1 else port
    if "--thread" in sys.argv:
        idx = sys.argv.index("--thread")
        thread_id = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None
    if "--user" in sys.argv:
        idx = sys.argv.index("--user")
        user_id = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None

    print(f"\n  {C.bold}{C.cyan}FalkorDB Session Debugger{C.reset}")
    print(f"  {C.dim}{host}:{port}{C.reset}")

    backend = FalkorDBBackend(host=host, port=port)

    if raw:
        if not _check_falkordb(backend):
            return
        graphs = backend.list_graphs()
        print(f"\n  All FalkorDB graphs ({len(graphs)}):")
        for g in graphs:
            if g.startswith("meta:thread:"):
                tid = g[len("meta:thread:"):]
                print(f"  {C.cyan}\u2502  {g}{C.reset}  {C.dim}(thread {tid[:16]}...){C.reset}")
            else:
                print(f"  {C.dim}\u2502  {g}{C.reset}")
        print()
        return
    store = ThreadStore(backend)

    if not _check_falkordb(backend):
        return

    if thread_id:
        try:
            thread = store.get_thread(thread_id)
        except Exception as exc:
            print(f"  {C.red}Error reading thread {thread_id!r}: {exc}{C.reset}")
            return
        if thread is None:
            print(f"  {C.red}Thread {thread_id!r} not found{C.reset}")
            return
        threads_list = [{"id": thread.id, "name": thread.name,
                         "user_id": thread.user_id, "created_at": thread.created_at,
                         "turn_count": thread.turn_count}]
    else:
        threads_list = store.list_threads(user_id=user_id)
        if not threads_list:
            print(f"  {C.yellow}No threads found{C.reset}")
            print(f"  {C.dim}(run chat_session.py or math_agent --chat first){C.reset}")
            return

    for td in threads_list:
        _show_thread_summary(td)
        thread = store.get_thread(td["id"])
        if not thread:
            continue

        # Load from project graph (all turns share one graph)
        project_graph_name = f"project:{td['id']}"
        try:
            project_graph = backend.load(project_graph_name)
            if project_graph.nodes:
                print(f"  {C.dim}Project graph: {len(project_graph.nodes)} nodes, "
                      f"{len(project_graph.edges)} edges{C.reset}")
        except Exception:
            project_graph = None

        for turn in thread.turns:
            if turn.graph_name:
                try:
                    graph = backend.load(turn.graph_name)
                except Exception as exc:
                    graph = project_graph  # fall back to project graph
                    if graph is None:
                        failed_msg = (
                            f"    {C.red}(failed to load execution graph {turn.graph_name}: {exc})"
                            f"{C.reset}"
                        )
                        print(failed_msg)
            else:
                graph = project_graph

            _show_turn(turn, graph=graph)

        # Show agentic memory for this user
        user_id = td.get("user_id", "")
        if user_id:
            mem_graph_name = f"memory:{user_id}"
            try:
                mg = backend._fg(mem_graph_name)
                r_tri = mg.query(
                    "MATCH (s:Entity)-[r]->(o:Entity) RETURN count(*) as n"
                )
                r_sk = mg.query("MATCH (s:Skill) RETURN count(*) as n")
                tri_count = r_tri.result_set[0][0] if r_tri.result_set else 0
                sk_count = r_sk.result_set[0][0] if r_sk.result_set else 0
                if tri_count or sk_count:
                    print(f"\n  {C.bold}{C.cyan}Memory: {mem_graph_name}{C.reset}")
                    print(f"  {C.dim}Triplets: {tri_count}, Skills: {sk_count}{C.reset}")
                    if tri_count:
                        r = mg.query(
                            "MATCH (s:Entity)-[r]->(o:Entity) "
                            "RETURN s._name, type(r), o._name LIMIT 5"
                        )
                        for row in r.result_set:
                            print(f"    {row[0]} --[{row[1]}]--> {row[2]}")
            except Exception:
                pass

    print(f"\n  {C.bold}{C.cyan}{'═'*68}{C.reset}\n")


if __name__ == "__main__":
    asyncio.run(main())
