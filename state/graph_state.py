from typing import Any, Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    pending_tasks: list[dict]
    completed_tasks: list[dict]
    current_agent: str
    iteration_count: int
    rollback_count: int
    memory_context: str
    final_output: str
    need_human_approval: bool
    human_decision: str
    tool_call_cache: dict[str, Any]
    last_reflection_round: int
    reflection_insights: list[str]


def initial_state() -> AgentState:
    return {
        "messages": [],
        "intent": "",
        "pending_tasks": [],
        "completed_tasks": [],
        "current_agent": "",
        "iteration_count": 0,
        "rollback_count": 0,
        "memory_context": "",
        "final_output": "",
        "need_human_approval": False,
        "human_decision": "",
        "tool_call_cache": {},
        "last_reflection_round": 0,
        "reflection_insights": [],
    }
