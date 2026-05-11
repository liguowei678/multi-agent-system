"""
FastAPI server wrapping the LangGraph multi-agent system.
Serves static/index.html at / and exposes REST API endpoints.
"""

import asyncio
import json
import uuid
import sqlite3
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from config.settings import SQLITE_PATH


# ── State ────────────────────────────────────────────────────────────────────
_graph = None


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    import aiosqlite
    from main import build_graph, chat
    from config.settings import SQLITE_PATH
    async_conn = await aiosqlite.connect(SQLITE_PATH)
    _graph = build_graph(conn=async_conn)
    print("Graph loaded, ready to serve.")
    yield
    _graph = None


app = FastAPI(title="Multi-Agent Dashboard", lifespan=lifespan)

STATIC = Path(__file__).parent / "static"


# ── Chat request model ────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


# ── Message serialization ─────────────────────────────────────────────────────
def _serialize_state(state: dict, thread_id: str) -> dict:
    """Convert raw AgentState (with LangChain message objects) to JSON-safe dict."""
    messages = []
    for m in state.get("messages", []):
        try:
            role = None
            content = ""
            extra = {}

            type_name = getattr(m, "type", None) or getattr(m, "role", None)

            if type_name in ("human", "user"):
                role = "user"
                content = m.content if hasattr(m, "content") else str(m)
            elif type_name in ("ai", "assistant"):
                role = "ai"
                content = m.content if hasattr(m, "content") else str(m)
                if not content and hasattr(m, "reasoning_content"):
                    content = m.reasoning_content or ""
                # Detect agent from content patterns
                agent = _infer_agent(content)
                if agent:
                    extra["agent"] = agent
                # Pass through tool_calls if present
                if hasattr(m, "tool_calls") and m.tool_calls:
                    extra["tool_calls"] = [
                        {"name": tc.get("name", ""), "args": tc.get("args", {})}
                        for tc in m.tool_calls
                    ]
            elif type_name == "tool":
                role = "tool"
                content = str(m.content)[:3000] if hasattr(m, "content") else str(m)
                tool_name = getattr(m, "name", "") or getattr(m, "tool_call_id", "")
                extra["tool_name"] = tool_name
            elif type_name == "system":
                continue  # skip system messages in chat UI
            else:
                # Generic LangChain message — try .content
                content = m.content if hasattr(m, "content") else str(m)
                if not content or len(content) < 5:
                    continue
                role = "ai"
                agent = _infer_agent(content)
                if agent:
                    extra["agent"] = agent

            if role and content:
                msg = {"role": role, "content": content[:5000]}
                msg.update(extra)
                messages.append(msg)
        except Exception:
            messages.append({"role": "error", "content": "(unsupported message)"})

    # Extract final reply
    reply = state.get("final_output", "")
    if not reply:
        tasks = state.get("completed_tasks", [])
        if tasks:
            reply = tasks[-1].get("result", "")

    return {
        "thread_id": thread_id,
        "reply": reply,
        "intent": state.get("intent", ""),
        "current_agent": state.get("current_agent", ""),
        "iteration_count": state.get("iteration_count", 0),
        "completed_tasks": state.get("completed_tasks", []),
        "pending_tasks": state.get("pending_tasks", []),
        "need_human_approval": state.get("need_human_approval", False),
        "reflection_insights": state.get("reflection_insights", []),
        "messages": messages,
    }


def _infer_agent(content: str) -> str:
    """Attempt to infer which agent generated this message content."""
    if not content:
        return ""
    if '"intent"' in content[:200] or 'intent' in content[:100]:
        return "supervisor"
    if 'action' in content[:100] and ('continue' in content or 'review' in content):
        return "supervisor"
    if '"score"' in content[:200] or '审查' in content[:100]:
        return "reviewer"
    return ""


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    index_path = STATIC / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "Frontend not built")
    return FileResponse(index_path)


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if _graph is None:
        raise HTTPException(503, "Graph not initialized")

    thread_id = req.thread_id or f"session-{uuid.uuid4().hex[:8]}"

    try:
        import asyncio
        from main import chat as run_chat
        state = await asyncio.to_thread(run_chat, _graph, req.message, thread_id)
        return _serialize_state(state, thread_id)
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Agent execution failed")


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """SSE streaming endpoint: token-level events from astream_events."""
    if _graph is None:
        raise HTTPException(503, "Graph not initialized")

    thread_id = req.thread_id or f"session-{uuid.uuid4().hex[:8]}"

    async def event_generator():
        from main import chat_stream_events
        try:
            async for event in chat_stream_events(_graph, req.message, thread_id):
                if event.get("event") == "complete":
                    raw = event.get("state", {})
                    event["state"] = _serialize_state(raw, thread_id)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            traceback.print_exc()
            yield f"data: {json.dumps({'event':'error','error':str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sessions")
async def list_sessions():
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cur = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        )
        rows = [row[0] for row in cur.fetchall()]
        sessions = [{"thread_id": tid, "message_count": 0, "last_updated": ""} for tid in rows]
        conn.close()
        return {"sessions": sessions}
    except Exception:
        return {"sessions": []}


@app.get("/api/sessions/{thread_id}")
async def get_session(thread_id: str):
    if _graph is None:
        raise HTTPException(503, "Graph not initialized")
    try:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = _graph.get_state(config)
        if not snapshot or not snapshot.values:
            raise HTTPException(404, f"Session {thread_id} not found")
        state = dict(snapshot.values)
        return _serialize_state(state, thread_id)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(404, f"Session {thread_id} not found")


@app.delete("/api/sessions/{thread_id}")
async def delete_session(thread_id: str):
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cur = conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        return {"status": "deleted", "rows": deleted}
    except Exception:
        return {"status": "error", "rows": 0}
