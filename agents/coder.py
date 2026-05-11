from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from state.graph_state import AgentState
from memory.memory_tool import search, add

CODER_PROMPT = """你是编程执行员。基于需求生成可运行代码。

能力: 文件读写、代码生成、命令执行
约束:
- 生成代码前先检查当前环境 (文件结构、已有代码、依赖)
- 代码必须包含错误处理，关键步骤加断言
- 执行后报告: {"code_summary": "...", "execution_result": "...", "files_changed": [...]}
- 输出必须经 Reviewer 审查后才算完成
- 禁止: 删除用户文件、修改非项目路径、安装全局依赖
- 不在用户项目中写注释，除非逻辑确实非显而易见

## 重要性判断
只有以下情况标记 importance=high（否则为 low）:
- 修复了之前失败的 bug
- 生成了涉及安全/核心逻辑的关键代码
- 产出用于生产环境的代码

## 记忆指令
- 执行前: memory_search(task_keywords) 查历史解决过的类似问题
- 执行后: memory_add({solution, files_changed, timestamp})"""


def coder_node(state: AgentState, llm, tools: list) -> AgentState:
    pending = state.get("pending_tasks", [])
    task = pending[0] if pending else {"task": "write_code"}
    task_desc = task.get("task", str(task))

    recent_msgs = state.get("messages", [])[-5:]
    context = "\n".join(
        f"[{type(m).__name__}] {str(m.content)[:500]}"
        for m in recent_msgs if hasattr(m, "content")
    ) if recent_msgs else "无上游需求"

    history = search(task_desc, limit=3)
    history_context = (
        "\n".join(str(h.get("content", "")) for h in history)
        if history else "无历史记录"
    )

    system_msg = SystemMessage(content=CODER_PROMPT)
    user_msg = HumanMessage(
        content=f"编程任务: {task_desc}\n需求上下文: {context}\n历史解决记录: {history_context}"
    )

    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke([system_msg, user_msg])

    result_text = ""
    tool_msgs = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_map = {t.name: t for t in tools}
        for tc in response.tool_calls:
            t = tool_map.get(tc.get("name", ""))
            if t:
                try:
                    r = t.invoke(tc.get("args", {}))
                    result_text += f"[{t.name}]: {r}\n"
                    tool_msgs.append(ToolMessage(content=str(r)[:2000], tool_call_id=tc.get("id","")))
                except Exception as e:
                    result_text += f"[{t.name}] Error: {e}\n"
                    tool_msgs.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc.get("id","")))
    if not result_text:
        result_text = str(response.content) if hasattr(response, "content") else str(response)

    # importance=high only if fixing a bug or generating critical code
    high_markers = ["修复", "bug", "安全", "关键", "核心", "生产"]
    importance = "high" if any(m in result_text for m in high_markers) else "low"
    add({"task": task_desc, "solution": result_text[:500]}, importance=importance)

    completed = state.get("completed_tasks", [])
    completed.append({
        "agent": "coder", "task": task_desc,
        "result": result_text, "status": "done",
        "needs_review": True, "importance": importance,
    })
    state["completed_tasks"] = completed

    state["messages"] = list(state.get("messages", [])) + [response] + tool_msgs
    state["pending_tasks"] = pending[1:] if len(pending) > 1 else []
    state["current_agent"] = "coder"
    return state
