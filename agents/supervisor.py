import json
import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.constants import END
from state.graph_state import AgentState
from memory.memory_tool import search, add
from memory.reflection import reflection_check, reflect
from config.settings import MAX_ITERATIONS


SUPERVISOR_PROMPT = """你是中心调度器。你统一管理多个专家Agent协同完成用户任务。

你的职责:
1. 意图识别: 判断用户请求是"简单任务"还是"复杂任务"
   - 简单任务: 自我介绍/聊天/陈述偏好/纯计算/代码执行/打开网页/关于对话本身的回忆问题 → tool_executor
     注意: "我不喜欢X"是陈述偏好，"我刚才问了什么/打开了谁的直播"是回忆对话，都不是搜索请求
   - 复杂任务: 用户要求搜索新信息/查资料/多步推理 → 派 researcher
2. 任务规划: 将复杂任务拆解为有序子任务列表
3. 动态调度: 根据进度和Agent返回结果，决定下一步
4. 质量把关: 判断任务是否完成，是否需回滚重做
5. 反思决策: 检测到重复失败/矛盾记忆/高重要性标记时，触发记忆整合

可用子Agent:
- researcher: 信息采集，调用MCP工具获取外部数据 (负责RAG搜索、公网搜索、文件读取)
- analyst: 加工分析，结构化输出报告/方案
- coder: 编程执行，生成和运行代码
- reviewer: 质量审查，独立验证输出质量

调度原则:
- 需要查资料/搜索/读文件 → 必须先派 researcher，不得判为简单任务
- 需要分析整理 → 派 analyst (在researcher之后)
- 需要编程 → 派 coder，task描述为"写Python脚本做X"的形式(coder用run_python执行)
- coder之后必须经过reviewer
- 最终审查 → 派 reviewer
- 重要: 用户要求"搜索X然后写脚本/生成报告" → 必须拆成 researcher→coder→analyst 三步，不得合成一步

输出格式:
如果是简单任务: {"intent": "simple", "route": "tool_executor", "reason": "..."}
如果是复杂任务: {"intent": "complex", "plan": [{"agent": "researcher", "task": "..."}, ...], "reason": "..."}"""

PROGRESS_PROMPT = """你是调度器。检查执行结果，决定下一步。

关键规则:
- **"review" 仅当 pending_summary 为空时才能使用**。只要待执行列表里还有未完成的 Agent，必须用 "continue" 并按顺序派发下一个。
- 不要把 researcher 返回的搜索结果当成最终答案。researcher 之后如果有 analyst/coder，必须等它们也完成。
- "全部完成" = 计划里所有 Agent 都执行完毕并产出结果，不是 researcher 搜到数据就叫完成。

选项:
1. 全部完成(pending_summary 为空) → {{"action": "review", "reason": "..."}}
2. 还有未执行任务 → {{"action": "continue", "next_agent": "researcher|analyst|coder|reviewer", "task": "...", "reason": "..."}}
3. 结果不理想需重做 → {{"action": "retry", "reason": "..."}}

当前已完成任务: {completed_summary}
当前待执行任务: {pending_summary}"""


def _parse_json(text: str, default: dict) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return default


def _messages_to_text(msgs: list, llm=None) -> str:
    """对话历史转纯文本: 保留最近16条(≈8轮)原始, 更早的摘要压缩。"""
    from memory.compact import compact_messages
    all_msgs = list(msgs)
    if len(all_msgs) > 16 and llm:
        all_msgs = compact_messages(all_msgs, llm)
    lines = []
    for m in all_msgs:
        role = getattr(m, "type", None) or getattr(m, "role", "unknown")
        content = getattr(m, "content", "") or ""
        if role in ("human", "user"):
            lines.append(f"用户: {content}")
        elif role in ("ai", "assistant") and content:
            lines.append(f"AI: {content[:300]}")
        elif role == "tool" and content:
            lines.append(f"[工具]: {content[:300]}")
        elif role == "system" and content:
            lines.append(f"[摘要]: {content[:300]}")
    return "\n".join(lines) if lines else "（无历史）"


def supervisor_node(state: AgentState, llm, tools: list) -> AgentState:
    iteration = state.get("iteration_count", 0) + 1
    state["iteration_count"] = iteration

    msgs = state.get("messages", [])
    user_request = ""
    for m in msgs:
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role in ("human", "user"):
            user_request = m.content if hasattr(m, "content") else str(m)

    # === FIRST ENTRY: Intent classification ===
    if iteration == 1:
        history_str = _messages_to_text(msgs, llm)
        system_msg = SystemMessage(content=SUPERVISOR_PROMPT)
        user_msg = HumanMessage(content=f"对话历史:\n{history_str}\n\n请根据以上对话历史，判断用户最新意图并决定下一步行动。")
        response = llm.invoke([system_msg, user_msg])
        response_text = response.content if hasattr(response, "content") else str(response)
        decision = _parse_json(response_text, {"intent": "simple", "route": "tool_executor"})

        state["intent"] = decision.get("intent", "simple")
        # 复杂任务: 单独调 LLM 做拆解 (与意图分类分离)
        if state["intent"] == "complex":
            decomp_prompt = f"""分配Agent处理以下需求。可选: researcher(搜索), coder(写Python脚本用run_python执行), analyst(整理报告), reviewer(审查)。

硬性规则:
1. 搜索类需求必须: researcher → analyst → reviewer（不可跳过 analyst）
2. 涉及写代码: 在 analyst 前加 coder
3. 搜索结果≠最终答案，analyst 负责整理成结构化输出

用户需求: {user_request}
输出JSON: {{"agents": ["agent1","agent2","agent3"], "reason":"..."}}"""
            resp2 = llm.invoke([HumanMessage(content=decomp_prompt)])
            text2 = resp2.content if hasattr(resp2, "content") else str(resp2)
            agent_list = _parse_json(text2, {"agents": ["researcher", "analyst", "reviewer"]}).get("agents", ["researcher", "analyst", "reviewer"])
            # 代码兜底: 有 researcher 就必须跟 analyst
            if "researcher" in agent_list and "analyst" not in agent_list:
                agent_list.insert(agent_list.index("researcher") + 1, "analyst")
            if len(agent_list) >= 2 and "reviewer" not in agent_list:
                agent_list.append("reviewer")
            pending = state.get("pending_tasks", [])
            task_templates = {
                "researcher": user_request[:100],
                "coder": "写Python脚本用run_python执行 — " + user_request[:80],
                "analyst": "基于前面researcher和coder的结果，整理生成一份完整报告(不要再搜索新信息)",
                "reviewer": "审查最终输出质量和正确性",
            }
            for agent in agent_list:
                task_desc = task_templates.get(agent, user_request[:100])
                pending.append({"agent": agent, "task": task_desc, "status": "pending"})
            state["pending_tasks"] = pending

        state["messages"] = list(msgs) + [AIMessage(content=response_text)]
        state["current_agent"] = "supervisor"
        return state

    # === SUBSEQUENT ENTRIES: Progress check + reflection ===
    completed = state.get("completed_tasks", [])
    pending = state.get("pending_tasks", [])
    completed_summary = "\n".join(
        f"- [{c.get('agent', '?')}] {c.get('task', '?')}: {c.get('status', '?')}"
        for c in completed[-5:]
    ) if completed else "无"
    pending_summary = "\n".join(
        f"- [{p.get('agent', '?')}] {p.get('task', '?')} ({p.get('status', 'pending')})"
        for p in pending[:5]
    ) if pending else "无"
    progress_msg = PROGRESS_PROMPT.format(
        completed_summary=completed_summary,
        pending_summary=pending_summary,
    )
    system_msg = SystemMessage(content=progress_msg)
    user_msg = HumanMessage(content=f"原始需求: {user_request}\n请检查进度并决定下一步。")
    response = llm.invoke([system_msg, user_msg])
    response_text = response.content if hasattr(response, "content") else str(response)
    decision = _parse_json(response_text, {"action": "review"})
    state["last_progress_action"] = decision.get("action", "review")

    # 代码层兜底: LLM 说 "review" 但还有待执行任务时, 强制继续
    if decision.get("action") == "review":
        still_pending = [t for t in pending if t.get("status") == "pending"]
        if still_pending:
            next_task = still_pending[0]
            decision = {
                "action": "continue",
                "next_agent": next_task.get("agent", "reviewer"),
                "task": next_task.get("task", ""),
                "reason": f"代码兜底: 还有 {len(still_pending)} 个待执行任务, 不能 review"
            }
            state["last_progress_action"] = "continue"

    # Redis 计数器: 每轮 +1, 满8轮触发 consolidate
    try:
        from memory import redis_cache
        r = redis_cache._get()
        if r:
            cnt = r.incr("consolidate_counter")
            if cnt >= 8:
                r.delete("consolidate_counter")
                reflection_result = reflect(state, ["safety_net"], llm)
                existing = state.get("reflection_insights", [])
                existing.extend(reflection_result.get("insights", []))
                state["reflection_insights"] = existing
    except Exception:
        pass

    # Reflection check (原有逻辑保持不变)
    triggers = reflection_check(state)
    if triggers:
        reflection_result = reflect(state, triggers, llm)
        existing = state.get("reflection_insights", [])
        existing.extend(reflection_result["insights"])
        state["reflection_insights"] = existing
        state["last_reflection_round"] = iteration

    state["messages"] = list(msgs) + [AIMessage(content=response_text or "")]
    state["current_agent"] = "supervisor"
    return state


def supervisor_routing(state: AgentState):
    """Conditional edge routing. Returns next node name or END."""
    iteration = state.get("iteration_count", 0)

    if iteration >= MAX_ITERATIONS:
        return "reviewer"

    # First entry — route based on intent
    if iteration == 1:
        if state.get("intent") == "simple":
            return "tool_executor"
        pending = state.get("pending_tasks", [])
        if pending:
            task = pending[0]
            task["status"] = "dispatched"
            return task.get("agent", "reviewer")
        return END

    # Simple task done → skip reviewer, go straight to END
    if state.get("intent") == "simple":
        return END

    # Subsequent rounds — check if more pending tasks
    pending = [t for t in state.get("pending_tasks", []) if t.get("status") == "pending"]
    if pending:
        task = pending[0]
        task["status"] = "dispatched"
        return task.get("agent", "reviewer")

    # No more pending tasks — final review or end
    completed = state.get("completed_tasks", [])
    if completed and not completed[-1].get("review_pass"):
        return "reviewer"
    return END
