from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from state.graph_state import AgentState
from memory.memory_tool import search, add

RESEARCHER_PROMPT = """你是信息采集员。只做一件事: 调用工具获取信息。

## 工具选择 SOP（严格按优先级执行）

### 判断逻辑
**优先判断**: 问题是关于企业/公司内部事务，还是外部公网信息？
  - 涉及公司/企业内部: 报销/请假/出差/制度/流程/标准/规范/政策/规定/员工/部门/薪资 → 只用 RAG
  - 涉及外部对象: 开源库/名人/公开新闻/行业技术/编程语言 → 只用 Tavily

1. 公司内部事务（报销/出差/请假/制度/流程/标准/员工/部门等）
   → 只调 rag_search，不调 Tavily
   → 无结果 → 告知 Supervisor

2. 外部公网信息（技术新闻、开源库、名人、行业动态、编程问题）
   → 只调 tavily_search，不调 RAG

3. 浏览网页 → browser 工具
4. 文件操作 → filesystem 工具
5. 代码执行 → run_python
6. 确定无法判断归属 → 两个都调，RAG 结果标 [内部]，Tavily 标 [公网]

## 约束
- 只用工具获取信息，不对信息做分析或总结
- 输出: {"source": "工具名", "raw_result": "...", "confidence": 1-5}
- 工具调用失败报告原因，最多重试 2 次
- 你是原料供应商，你的下游是 Analyst
- RAG 结果标注 [内部知识]，Tavily 结果标注 [公网信息]

## 重要性判断
以下发现标记 importance=high（否则为 low）:
- 涉及用户个人偏好/习惯/不喜欢的东西
- 涉及公司内部制度/政策/规定/流程
- 找到了关键证据或核心信息

## 记忆指令
- 执行前: memory_search(task_keywords) 查历史类似检索
- 执行后: memory_add({task, result, timestamp})"""


def researcher_node(state: AgentState, llm, tools: list) -> AgentState:
    pending = state.get("pending_tasks", [])
    task = pending[0] if pending else {"task": "unknown"}
    task_desc = task.get("task", str(task))

    # 获取用户原始问题（不是 Supervisor 改写的任务描述）
    msgs = state.get("messages", [])
    user_question = task_desc
    for m in reversed(msgs):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role in ("human", "user"):
            user_question = m.content if hasattr(m, "content") else str(m)
            break

    history = search(user_question, limit=3)
    history_context = (
        "\n".join(str(h.get("content", "")) for h in history)
        if history else "无相关历史记录"
    )

    system_msg = SystemMessage(content=RESEARCHER_PROMPT)
    user_msg = HumanMessage(
        content=f"任务: {task_desc}\n历史相关记录: {history_context}\n请调用工具采集信息。"
    )

    llm_with_tools = llm.bind_tools(tools)
    ai_msg = llm_with_tools.invoke([system_msg, user_msg])

    # 硬编码: RAG 永远先执行, LLM 没选也强制插入
    result_text = ""
    tool_msgs = []
    tool_map = {t.name: t for t in tools}
    calls = list(ai_msg.tool_calls) if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls else []

    # LLM 没选 RAG → 强制插到最前面
    if not any("rag" in tc.get("name","") for tc in calls):
        rag_tool = tool_map.get("rag_search")
        if rag_tool:
            calls.insert(0, {"name": "rag_search", "args": {"query": user_question}})

    if calls:
        rag_result = ""   # RAG 结果单独存, 无结果时不显示
        tavily_result = ""
        rag_ok = False
        for tc in calls:
            name = tc.get("name", "")
            if rag_ok and "tavily" in name:
                continue
            t = tool_map.get(name)
            if t:
                try:
                    r = t.invoke(tc.get("args", {}))
                    tool_msgs.append(ToolMessage(content=str(r)[:2000], tool_call_id=tc.get("id","")))
                    if "rag" in name:
                        r_str = str(r)
                        no_result = any(kw in r_str for kw in ["无匹配", "不可用", "没有找到", "无法回答", "无相关", "无结果"])
                        if no_result:
                            rag_result = ""  # RAG 无结果 → 不显示
                        else:
                            rag_result = f"[{t.name}]: {r}\n"
                            rag_ok = True
                    elif "tavily" in name:
                        tavily_result = f"[{t.name}]: {r}\n"
                    else:
                        result_text += f"[{t.name}]: {r}\n"
                except Exception as e:
                    result_text += f"[{t.name}] Error: {e}\n"
                    tool_msgs.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc.get("id","")))
        # RAG 有结果才显示, 否则只显示 Tavily
        result_text = rag_result + tavily_result + result_text
    else:
        result_text = str(ai_msg.content) if hasattr(ai_msg, "content") else str(ai_msg)

    high_markers = ["偏好", "不喜欢", "喜欢", "讨厌", "习惯", "个人", "关键", "核心"]
    importance = "high" if any(m in result_text for m in high_markers) else "low"
    add({"task": task_desc, "result": result_text[:1000], "tool": "mcp"},
        importance=importance)

    completed = state.get("completed_tasks", [])
    completed.append({
        "agent": "researcher", "task": task_desc,
        "result": result_text, "status": "done",
        "importance": importance,
    })
    state["completed_tasks"] = completed

    msgs = state.get("messages", [])
    state["messages"] = list(msgs) + [ai_msg] + tool_msgs
    state["pending_tasks"] = pending[1:] if len(pending) > 1 else []
    state["current_agent"] = "researcher"
    return state
