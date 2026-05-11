import re
from langchain_core.messages import SystemMessage, HumanMessage
from state.graph_state import AgentState
from memory.memory_tool import search, add

REVIEWER_PROMPT = """你是质量审查员。对上游输出做独立验证。

能力: 完整性检查、正确性验证、安全性审查
约束:
- 检查维度: 完整性/正确性/安全性/是否符合需求
- 输出: {"pass": true/false, "score": 1-5, "issues": [...], "suggestion": "..."}
- score < 3 → 必须打回重做
- score 3-4 → 可选修正
- score 5 → 直接通过
- Code Agent 输出是必审项
- 只做判定和建议，禁止修改原始内容

## 记忆指令
- 执行前: memory_search(similar_review) 查历史审查记录
- 执行后: memory_add({review_result, score, timestamp})
- score ≤ 2 的失败案例标记 importance=high"""


def reviewer_node(state: AgentState, llm, tools: list) -> AgentState:
    completed = state.get("completed_tasks", [])
    if not completed:
        return state
    target = completed[-1]

    task_desc = target.get("task", "")
    result = target.get("result", "")

    history = search(f"review:{task_desc}", limit=3)
    history_context = (
        "\n".join(str(h.get("content", "")) for h in history)
        if history else "无历史审查记录"
    )

    system_msg = SystemMessage(content=REVIEWER_PROMPT)
    user_msg = HumanMessage(
        content=f"审查任务: {task_desc}\n输出内容: {str(result)[:2000]}\n历史审查记录: {history_context}\n请给出评分和建议。"
    )

    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke([system_msg, user_msg])

    result_text = str(response.content) if hasattr(response, "content") else str(response)

    score = 3
    try:
        match = re.search(r'"score"\s*:\s*(\d)', result_text)
        if match:
            score = int(match.group(1))
    except Exception:
        pass

    passed = score >= 3
    target["reviewed"] = True
    target["review_score"] = score
    target["review_pass"] = passed

    if score >= 3:
        state["final_output"] = result
    if score <= 2:
        target["status"] = "failed"
        state["need_human_approval"] = True

    add({"review_task": task_desc, "score": score, "passed": passed},
        importance="high" if score <= 2 else "low")

    state["messages"] = list(state.get("messages", [])) + [response]
    state["current_agent"] = "reviewer"
    return state
