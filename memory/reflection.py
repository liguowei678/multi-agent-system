from memory.memory_tool import consolidate, mark_protected, search
from config.settings import REFLECTION_SAFETY_NET


def reflection_check(state: dict) -> list[str]:
    """Check if reflection should be triggered. Returns list of trigger reasons."""
    triggers = []

    # Trigger 1: same agent failed same task >= 2 times
    completed = state.get("completed_tasks", [])
    failed_by_agent = {}
    for t in completed:
        if t.get("status") == "failed":
            agent = t.get("agent", "unknown")
            task_name = t.get("task", "")
            key = f"{agent}:{task_name}"
            failed_by_agent[key] = failed_by_agent.get(key, 0) + 1
    if any(v >= 2 for v in failed_by_agent.values()):
        triggers.append("plan_may_be_wrong")

    # Trigger 2: any sub-agent flagged importance=high and not yet consolidated
    for t in completed:
        if t.get("importance") == "high" and not t.get("consolidated"):
            triggers.append("urgent_consolidate")
            break

    # Trigger 3: state has memory_contradiction flag
    if state.get("memory_contradiction"):
        triggers.append("memory_conflict")

    # Trigger 4: safety net — too many rounds since last reflection
    last_reflection = state.get("last_reflection_round", 0)
    current_round = state.get("iteration_count", 0)
    if current_round - last_reflection >= REFLECTION_SAFETY_NET:
        triggers.append("safety_net")

    return triggers


def reflect(state: dict, triggers: list[str], llm=None) -> dict:
    """Execute reflection: LLM抽取结构化知识 → Neo4j + plan adjustment."""
    insights = []

    # Consolidate: LLM 从对话中提取 (subject, relation, object) → Neo4j
    if "urgent_consolidate" in triggers or "safety_net" in triggers:
        cr = consolidate(llm=llm)
        if cr["consolidated"] > 0:
            insights.append(f"extracted {cr['consolidated']} relations to Neo4j")
        for t in state.get("completed_tasks", []):
            if t.get("importance") == "high":
                t["consolidated"] = True

    if "plan_may_be_wrong" in triggers:
        insights.append("detected repeated failures — consider adjusting task plan")

    if "memory_conflict" in triggers:
        insights.append("memory contradiction detected — old memories may need update")

    return {
        "insights": insights,
        "last_reflection_round": state.get("iteration_count", 0),
    }
