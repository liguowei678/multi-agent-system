import tiktoken
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config.settings import (
    TOKEN_MONITOR_THRESHOLD, TOKEN_WARN_THRESHOLD,
    TOKEN_FORCE_THRESHOLD, COMPACT_KEEP_RECENT
)


def count_tokens(messages: list, model: str = "gpt-4") -> int:
    """Count total tokens across all messages."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        total += len(enc.encode(content))
    return total


def check_context_pressure(messages: list, context_limit: int = 65536) -> dict:
    """Check if we need compaction. Returns pressure level + stats."""
    tokens = count_tokens(messages)
    ratio = tokens / context_limit
    if ratio >= TOKEN_FORCE_THRESHOLD:
        return {"pressure": "force", "token_count": tokens, "ratio": ratio}
    if ratio >= TOKEN_WARN_THRESHOLD:
        return {"pressure": "warn", "token_count": tokens, "ratio": ratio}
    if ratio >= TOKEN_MONITOR_THRESHOLD:
        return {"pressure": "monitor", "token_count": tokens, "ratio": ratio}
    return {"pressure": "normal", "token_count": tokens, "ratio": ratio}


def compact_messages(messages: list, llm) -> list:
    """Keep last N raw messages, summarize older ones. Returns compressed list."""
    if len(messages) <= COMPACT_KEEP_RECENT:
        return messages

    recent = list(messages[-COMPACT_KEEP_RECENT:])
    older = list(messages[:-COMPACT_KEEP_RECENT])

    older_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI' if isinstance(m, AIMessage) else 'System'}: {m.content}"
        for m in older
    )

    summary_prompt = (
        "Summarize the following conversation history in Chinese, keeping key facts, "
        "decisions, user preferences, and action items. Be concise:\n\n" + older_text
    )
    summary_msg = llm.invoke([HumanMessage(content=summary_prompt)])
    summary_content = f"[历史摘要] {summary_msg.content}"

    return [SystemMessage(content=summary_content)] + recent
