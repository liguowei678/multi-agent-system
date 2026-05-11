
import asyncio
import aiosqlite
import sqlite3
import subprocess
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import os
import sys
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, SQLITE_PATH, TAVILY_API_KEY
from config.mcp_servers import MCP_SERVERS
from state.graph_state import AgentState, initial_state
import queue as _queue_mod
from langchain_core.messages import AIMessage, AIMessageChunk

# ── Streaming token channel ─────────────────────────────────────────────────
_token_queue = None  # module-level, set per streaming request

def _push_token(token: str, agent: str):
    q = _token_queue
    if q is not None:
        q.put({"event": "token", "token": token, "agent": agent})

class StreamWrapper:
    """LLM wrapper: intercept invoke() → stream(), push tokens, return full msg."""
    def __init__(self, llm, agent_name=""):
        self._llm = llm
        self._agent = agent_name

    def invoke(self, messages, **kwargs):
        full = None
        for chunk in self._llm.stream(messages, **kwargs):
            full = chunk if full is None else full + chunk
            c = getattr(chunk, "content", "")
            if c:
                _push_token(c, self._agent)
        if full is None:
            return AIMessage(content="")
        return full

    def bind_tools(self, tools):
        bound = self._llm.bind_tools(tools)
        return StreamWrapper(bound, self._agent)

    @property
    def name(self):
        return getattr(self._llm, "name", "")

from agents.supervisor import supervisor_node, supervisor_routing
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.coder import coder_node
from agents.reviewer import reviewer_node
from mcp_bridge.client_manager import MCPClientManager


def build_graph(conn=None):
    """Build the agent graph. Pass aiosqlite.Connection for async streaming support."""
    # Inject Tavily API key into env so MCP subprocesses inherit it
    if TAVILY_API_KEY:
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

    # Supervisor 用 temperature=0 确保路由确定性
    llm_supervisor = StreamWrapper(ChatOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL, temperature=0, streaming=True,
    ), "supervisor")
    # 对话生成用 temperature=0.3 (自然) — 每个 agent 独立 wrapper 推送正确名称
    _base_llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL, temperature=0.3, streaming=True,
    )
    llm = StreamWrapper(_base_llm, "")  # 通用，具体 agent 在传参时覆写

    # Load MCP tools — each server connects independently, failures don't cascade
    mcp = MCPClientManager(MCP_SERVERS)
    all_tools = mcp.get_tools()
    print(f"MCP: {len(all_tools)} 个工具已加载")

    # 启动时执行记忆生命周期维护 (archive >10天 + forget >30天)
    try:
        from memory.memory_tool import scheduled_archive
        archived = scheduled_archive()
        if archived:
            print(f"记忆维护: {len(archived)} 条降冷")
        from memory.memory_manager import MemoryManager
        deleted = MemoryManager().scheduled_forget()
        if deleted:
            print(f"记忆维护: {len(deleted)} 条删除")
    except Exception:
        pass  # 维护失败不影响主流程

    @tool
    def open_browser(url: str) -> str:
        """在系统默认浏览器中打开指定网址。例如: open_browser('https://www.huya.com')"""
        import webbrowser
        try:
            webbrowser.open(url)
            return f"已在默认浏览器中打开 {url}"
        except Exception as e:
            return f"打开浏览器失败: {e}"

    @tool
    def run_python(code: str) -> str:
        """Execute Python. ONLY for calculations/math, NOT for searching memory/database/files."""
        try:
            r = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, timeout=15,
                encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
            return out.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: timeout (15s)"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def rag_search(query: str) -> str:
        """[内部优先] 搜索公司私有知识库。用于: 制度/报销/请假/出差/薪资/合同/政策/流程/规范/标准。
        公司内部问题必须优先用此工具，不要用公网搜索。"""
        from rag.client import RAGClient
        try:
            result = RAGClient().search(query)
            return result if result else "RAG 无匹配结果"
        except Exception as e:
            return f"RAG 不可用: {e}"

    # 按 Agent 拆分工具箱 — 减少 LLM 选择混乱
    _tool_map = {t.name: t for t in all_tools}
    _search_names = {"tavily_search", "tavily_extract", "tavily_crawl", "tavily_map",
                     "tavily_research", "rag_search", "open_browser",
                     "browser_init", "navigate", "get_content", "click", "type",
                     "wait", "browser_close", "solve_captcha", "random_scroll",
                     "find_selector", "save_content_as_markdown"}
    _file_names = {"read_file", "read_text_file", "read_media_file", "read_multiple_files",
                   "write_file", "edit_file", "create_directory", "list_directory",
                   "list_directory_with_sizes", "directory_tree", "move_file",
                   "search_files", "get_file_info", "list_allowed_directories"}
    _code_names = {"run_python"}

    coder_tools = [run_python] + [_tool_map[n] for n in _file_names if n in _tool_map]
    @tool
    def search_graph(node_name: str) -> str:
        """查询知识图谱中的结构化关系。用于: 用户偏好/不喜欢什么/喜欢什么/与谁有关。例如: search_graph('Peter')"""
        from memory.memory_tool import search_graph as sg
        results = sg(node_name)
        return str(results) if results else "图谱无匹配"

    all_tools_list = [rag_search, open_browser, run_python, search_graph] + [_tool_map[n] for n in _tool_map]
    researcher_tools = [rag_search, open_browser, search_graph] + [_tool_map[n] for n in _search_names if n in _tool_map]
    simple_tools = [open_browser]
    analyst_tools = researcher_tools

    TOOL_SYSTEM_PROMPT = (
        "你是工具执行器。用 open_browser 打开网址，用 run_python 执行代码，"
        "用 rag_search 搜内部知识，用 tavily_search 搜公网，用 filesystem 读写文件。"
        "直接执行，不要说你做不到。"
    )

    # Shared helper: invoke LLM with tools and execute any tool calls
    def _invoke_with_tools_simple(prompt: str, tool_list: list, caller_llm):
        """返回 (result_string, tool_messages, ai_msg)"""
        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
        llm_with_tools = caller_llm.bind_tools(tool_list)
        ai_msg = llm_with_tools.invoke([SystemMessage(content=TOOL_SYSTEM_PROMPT), HumanMessage(content=prompt)])
        tool_msgs = []
        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            tool_map = {t.name: t for t in tool_list}
            results = []
            for tc in ai_msg.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                tool = tool_map.get(tool_name)
                if tool:
                    try:
                        r = tool.invoke(tool_args)
                        results.append(f"[{tool_name}]: {r}")
                        tool_msgs.append(ToolMessage(content=str(r)[:2000], tool_call_id=tc.get("id","")))
                    except Exception as e:
                        results.append(f"[{tool_name}] Error: {e}")
                        tool_msgs.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc.get("id","")))
            if results:
                return "\n".join(results), tool_msgs, ai_msg
        content = getattr(ai_msg, "content", "")
        reasoning = getattr(ai_msg, "reasoning_content", "")
        return (content or reasoning or "").strip(), tool_msgs, ai_msg

    # Simple-task tool executor — takes tools as param for minimal toolset
    def make_tool_executor(executor_tools, agent_llm):
        def tool_executor_node(state: AgentState) -> AgentState:
            msgs = state.get("messages", [])
            user_msg = "无任务"
            for m in reversed(msgs):
                role = getattr(m, "type", None) or getattr(m, "role", None)
                if role in ("human", "user"):
                    user_msg = m.content if hasattr(m, "content") else str(m)
                    break
            # 对话上下文: 记忆系统 + 最近12轮对话 + compact压缩
            from memory.memory_tool import search as mem_search, add as mem_add
            from memory.compact import compact_messages, check_context_pressure

            mem_results = mem_search(user_msg, limit=3)
            mem_text = "\n".join(str(m.get("content",""))[:300] for m in mem_results) if mem_results else "无"

            # 保留最近16条(≈8轮), 超出部分摘要压缩
            all_msgs = list(msgs)
            if len(all_msgs) > 16:
                ctx_msgs = compact_messages(all_msgs, agent_llm)
            else:
                ctx_msgs = all_msgs  # 最近12条

            recent = []
            for m in ctx_msgs:
                c = getattr(m, "content", "") if hasattr(m, "content") else str(m)
                if c and len(c) > 5:
                    recent.append(c[:300])
            history = "\n".join(recent) if recent else "无历史"

            prompt = f"记忆系统:\n{mem_text}\n\n对话历史:\n{history}\n\n当前: {user_msg}\n请处理。"
            result, tool_msgs, ai_msg = _invoke_with_tools_simple(prompt, executor_tools, agent_llm)
            if not result or (result.startswith("{") and "intent" in result[:30]):
                chat_msg = agent_llm.invoke(f"记忆系统:\n{mem_text}\n\n对话历史:\n{history}\n\n用户说: {user_msg}\n请用中文回复。")
                result = (
                    getattr(chat_msg, "content", "") or
                    getattr(chat_msg, "reasoning_content", "") or
                    ""
                )
                if not result.strip():
                    result = "你好！有什么可以帮你的？"
            # 写入记忆系统
            mem_add({"task": user_msg[:200], "result": result[:500]})
            state["messages"] = list(state.get("messages", [])) + [ai_msg] + tool_msgs
            tasks = state.get("completed_tasks", [])
            state["completed_tasks"].append({
                "agent": "tool_executor", "task": str(user_msg)[:100],
                "result": result, "status": "done",
            })
            return state
        return tool_executor_node

    # 确保 state 必需字段存在 (checkpoint 首次加载时可能缺失)
    def _ensure_state(state: dict):
        for key, default in [("completed_tasks", []), ("pending_tasks", []),
                              ("iteration_count", 0), ("intent", ""),
                              ("reflection_insights", []), ("last_reflection_round", 0)]:
            if key not in state:
                state[key] = default

    # 包装函数: 自动补全缺失的 state 字段
    def _wrap(fn):
        def wrapped(state):
            _ensure_state(state)
            return fn(state)
        return wrapped

    # 每个 agent 用独立 StreamWrapper，推送正确的 agent 名
    llm_researcher = StreamWrapper(ChatOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL, temperature=0, streaming=True,
    ), "researcher")
    llm_analyst = StreamWrapper(_base_llm, "analyst")
    llm_coder = StreamWrapper(_base_llm, "coder")
    llm_reviewer = StreamWrapper(_base_llm, "reviewer")
    llm_tool = StreamWrapper(_base_llm, "tool_executor")

    # Build the graph
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", _wrap(lambda s: supervisor_node(s, llm_supervisor, all_tools_list)))
    builder.add_node("tool_executor", _wrap(make_tool_executor(simple_tools, llm_tool)))
    builder.add_node("researcher", _wrap(lambda s: researcher_node(s, llm_researcher, researcher_tools)))
    builder.add_node("analyst", _wrap(lambda s: analyst_node(s, llm_analyst, analyst_tools)))
    builder.add_node("coder", _wrap(lambda s: coder_node(s, llm_coder, coder_tools)))
    builder.add_node("reviewer", _wrap(lambda s: reviewer_node(s, llm_reviewer, all_tools_list)))

    # Edges
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", supervisor_routing)

    # All worker nodes loop back to supervisor
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")
    builder.add_edge("coder", "supervisor")
    builder.add_edge("reviewer", "supervisor")
    builder.add_edge("tool_executor", "supervisor")

    # Checkpointer: async conn for streaming, sync fallback for CLI
    if conn is not None:
        checkpointer = AsyncSqliteSaver(conn)
        # setup is lazy — first async op (aget_state etc.) calls it internally
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver(sqlite3.connect(SQLITE_PATH, check_same_thread=False))
        checkpointer.setup()
    return builder.compile(
        checkpointer=checkpointer,
        # interrupt_before=["tool_executor"],  # 生产环境启用: 工具执行前需人类批准
    )


def chat(graph, user_input: str, thread_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    input_state = initial_state()
    input_state["messages"] = [{"role": "user", "content": user_input}]
    state = graph.invoke(input_state, config)
    return state


def chat_stream(graph, user_input: str, thread_id: str = "default"):
    """流式版本: 每经过一个图节点就 yield 一次状态快照。"""
    config = {"configurable": {"thread_id": thread_id}}
    input_state = initial_state()
    input_state["messages"] = [{"role": "user", "content": user_input}]

    for chunk in graph.stream(input_state, config):
        node_name = list(chunk.keys())[0]
        node_state = chunk[node_name]
        yield {
            "event": "node",
            "node": node_name,
            "current_agent": node_state.get("current_agent", ""),
            "completed_tasks": [
                {"agent": t.get("agent", ""), "task": str(t.get("task", ""))[:120],
                 "status": t.get("status", ""), "review_score": t.get("review_score")}
                for t in node_state.get("completed_tasks", [])
            ],
            "pending_tasks": [
                {"agent": t.get("agent", ""), "task": str(t.get("task", ""))[:120],
                 "status": t.get("status", "pending")}
                for t in node_state.get("pending_tasks", [])
            ],
            "iteration_count": node_state.get("iteration_count", 0),
            "intent": node_state.get("intent", ""),
        }

    # 最后推送完整 state（server 层负责序列化）
    final_state = graph.get_state(config)
    if final_state and final_state.values:
        yield {"event": "complete", "state": dict(final_state.values)}


async def chat_stream_events(graph, user_input: str, thread_id: str = "default"):
    """异步流式: graph events + LLM token 队列交错推送。"""
    global _token_queue
    config = {"configurable": {"thread_id": thread_id}}
    input_state = initial_state()
    input_state["messages"] = [{"role": "user", "content": user_input}]

    tq = _queue_mod.Queue()
    _token_queue = tq

    try:
        active_node = None
        async for event in graph.astream_events(input_state, config, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_chain_start" and name not in ("LangGraph", "supervisor_routing"):
                active_node = name
                yield {"event": "agent_start", "agent": name}

            elif kind == "on_chain_end" and name == active_node:
                out = event.get("data", {}).get("output", {})
                yield {"event": "agent_end", "agent": name,
                       "current_agent": out.get("current_agent", name) if isinstance(out, dict) else name}
                active_node = None

            # 在每个 graph event 后 drain token 队列
            while not tq.empty():
                yield tq.get_nowait()

        # Drain remaining tokens
        while not tq.empty():
            yield tq.get_nowait()

        # 推送最终 state（async context 用 aget_state）
        final_state = await graph.aget_state(config)
        if final_state and final_state.values:
            yield {"event": "complete", "state": dict(final_state.values)}
    finally:
        _token_queue = None


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)
    graph = build_graph()
    print("Agent 已就绪。输入 'exit' 退出。\n")

    tid = "session-1"
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        result = chat(graph, user_input, tid)
        # 输出: final_output → 最后一条 completed_task → 兜底
        reply = result.get("final_output", "")
        if not reply:
            tasks = result.get("completed_tasks", [])
            if tasks:
                reply = tasks[-1].get("result", "")
        if not reply:
            reply = "（请重新提问）"
        print(f"Agent: {reply}\n")
