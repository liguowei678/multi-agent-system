from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from state.graph_state import AgentState
from memory.memory_tool import search, add

ANALYST_PROMPT = """你是分析员/写手。基于上游信息做结构化加工输出。

能力: 整理分析、撰写报告、生成方案
约束:
- 输入来自 Researcher 或其他上游，整理为结构化输出
- 输出格式: 技术报告 / 方案建议 / JSON / Markdown
- 必须标注信息源，不确定处标 [待验证]
- 可调用 memory_search() 获取历史经验
- 禁止编造事实，禁止未经验证的断言

## 重要性判断
只有以下情况标记 importance=high（否则为 low）:
- 结论涉及用户画像/个人偏好/习惯
- 结论涉及公司政策/制度/规范
- 产出了关键策略建议或决策依据

## 记忆指令
- 执行前: memory_search(task_keywords) 获取相关知识
- 执行后: memory_add({analysis, conclusions, timestamp})"""


def analyst_node(state: AgentState, llm, tools: list) -> AgentState:
    pending = state.get("pending_tasks", [])
    task = pending[0] if pending else {"task": "analyze"}
    task_desc = task.get("task", str(task))

    recent_msgs = state.get("messages", [])[-5:]
    context = "\n".join(
        f"[{type(m).__name__}] {str(m.content)[:500]}"
        for m in recent_msgs if hasattr(m, "content")
    ) if recent_msgs else "无上游输入"

    history = search(task_desc, limit=3)
    history_context = (
        "\n".join(str(h.get("content", "")) for h in history)
        if history else "无历史经验"
    )

    system_msg = SystemMessage(content=ANALYST_PROMPT)
    user_msg = HumanMessage(
        content=f"任务: {task_desc}\n上游信息: {context}\n历史经验: {history_context}\n请分析并输出结构化结果。"
    )

    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke([system_msg, user_msg])

    # Execute tool calls (if any)
    tool_results = ""
    tool_msgs = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_map = {t.name: t for t in tools}
        for tc in response.tool_calls:
            t = tool_map.get(tc.get("name", ""))
            if t:
                try:
                    r = t.invoke(tc.get("args", {}))
                    tool_results += f"[{t.name}]: {r}\n"
                    tool_msgs.append(ToolMessage(content=str(r)[:2000], tool_call_id=tc.get("id","")))
                except Exception as e:
                    tool_results += f"[{t.name}] Error: {e}\n"
                    tool_msgs.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc.get("id","")))

    # 工具执行后, 让LLM基于上游上下文+工具结果合成最终报告
    synthesis_prompt = f"上游信息: {context}\n工具补充结果: {tool_results if tool_results else '无'}\n任务: {task_desc}\n请直接生成最终报告(不要再调工具)。"
    final_response = llm.invoke([SystemMessage(content=ANALYST_PROMPT), HumanMessage(content=synthesis_prompt)])
    result_text = final_response.content if hasattr(final_response, "content") else str(final_response)

    # importance=high only if analysis contains strategic/policy/personal insights
    high_markers = ["用户画像", "偏好", "策略建议", "个人习惯", "关键结论"]
    importance = "high" if any(m in result_text for m in high_markers) else "low"
    add({"task": task_desc, "analysis": result_text[:500]}, importance=importance)

    completed = state.get("completed_tasks", [])
    completed.append({
        "agent": "analyst", "task": task_desc,
        "result": result_text, "status": "done",
        "consolidated": False, "importance": importance,
    })
    state["completed_tasks"] = completed

    state["messages"] = list(state.get("messages", [])) + [response] + tool_msgs
    state["pending_tasks"] = pending[1:] if len(pending) > 1 else []
    state["current_agent"] = "analyst"
    return state
