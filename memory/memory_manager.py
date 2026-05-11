import uuid
from datetime import datetime
from memory.stores.sqlite_store import SQLiteStore
from memory.stores.qdrant_store import QdrantStore
from memory.stores.neo4j_store import Neo4jStore
from config.settings import HOT_DAYS, WARM_DAYS


class MemoryManager:
    def __init__(self):
        self.sqlite = SQLiteStore()
        self.qdrant = None   # lazy init
        self.neo4j = None    # lazy init

    def _get_qdrant(self):
        if self.qdrant is None:
            try:
                self.qdrant = QdrantStore()
            except Exception:
                pass
        return self.qdrant

    def _get_neo4j(self):
        if self.neo4j is None:
            try:
                self.neo4j = Neo4jStore()
            except Exception:
                pass
        return self.neo4j

    def classify(self, content: dict) -> str:
        """Classify memory: episodic | semantic | perceptual | working"""
        if "relation" in content or "preference" in content:
            return "semantic"
        # task 优先于 tool — 确保任务记忆进 Qdrant 可被检索
        if "event" in content or "task" in content:
            return "episodic"
        if "tool" in content:
            return "working"
        return "episodic"

    def route(self, memory_id: str, thread_id: str, memory_type: str,
              content: dict, importance: str = "low") -> None:
        """Write to SQLite + Qdrant (immediate). Neo4j deferred to consolidate()."""
        self.sqlite.add(memory_id, thread_id, memory_type, content, importance)

        # Qdrant: immediate for semantic search within current conversation
        if memory_type in ("episodic", "perceptual", "semantic"):
            try:
                qdrant = self._get_qdrant()
                if qdrant:
                    # 构建可读的文本描述而非 dict 字符串
                    task = content.get("task", "")
                    result = content.get("result", "")
                    readable = f"{task} → {result}"[:500]
                    qdrant.add(memory_id, readable, {
                        "thread_id": thread_id,
                        "type": memory_type,
                        "importance": importance,
                        "created_at": datetime.utcnow().isoformat(),
                        "summary": readable,
                    })
            except Exception:
                pass

    EXTRACT_PROMPT = """分析对话,判断是否有值得存入知识图谱的结构化信息。

准入标准(满足任一即抽取):
1. 用户偏好/好恶/习惯/禁忌 ("不喜欢弹窗""习惯晚上跑步""最讨厌开会")
2. 人际关系/归属/身份 ("我是前端组长""张三是李四的上级""我叫彼特")
3. 对事物的明确态度或评价 ("我觉得这个方案太复杂了")

不抽取(输出[]):
- 普通问答、查询、搜索 ("陪产假几天""什么是RAG")
- 闲聊、问候、敷衍 ("好的""嗯""知道了")
- 一次性任务 ("打开网页""创建文件""帮我算一下")
- 无法确定关系或实体模糊的陈述

若有,输出JSON: [{"subject":"实体","relation":"关系","object":"对象"}]
若无,输出: []

对话:
{conversation}"""

    def consolidate(self, thread_id: str, llm=None) -> dict:
        """LLM语义抽取 → Neo4j. 从对话中提取结构化关系."""
        results = {"consolidated": 0, "memory_ids": []}
        if not llm:
            return results

        recent = self.sqlite.search(thread_id=thread_id, limit=10)
        if not recent:
            return results

        # 拼对话文本
        lines = []
        for r in recent:
            c = r.get("content", {})
            task = c.get("task", "") if isinstance(c, dict) else ""
            res = c.get("result", "") if isinstance(c, dict) else str(c)
            if task: lines.append(f"用户: {task[:200]}")
            if res: lines.append(f"系统: {str(res)[:300]}")
        conversation = "\n".join(lines)

        try:
            from langchain_core.messages import HumanMessage
            import json, re
            prompt_text = self.EXTRACT_PROMPT.replace("{conversation}", conversation)
            resp = llm.invoke([HumanMessage(content=prompt_text)])
            text = resp.content if hasattr(resp, "content") else str(resp)
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                triples = json.loads(match.group())
                neo4j = self._get_neo4j()
                for t in triples:
                    if isinstance(t, dict) and "subject" in t and "relation" in t:
                        try:
                            if neo4j:
                                neo4j.add_relationship(
                                    t["subject"], t["relation"], t.get("object", ""),
                                    "AgentMemory_Preference",
                                    {"source": "llm_extraction"}
                                )
                            results["consolidated"] += 1
                        except Exception:
                            pass
            # 抽取成功 → 保护相关SQLite记录(不归档不删除)
            if results["consolidated"] > 0:
                for r in recent:
                    self.sqlite.mark_protected(r["id"])
        except Exception:
            pass

        return results

    # ===== 冷热迁移 ====

    def archive(self, memory_id: str) -> None:
        """Move from warm to cold: remove Qdrant index, keep SQLite record."""
        qdrant = self._get_qdrant()
        if qdrant:
            try:
                qdrant.delete(memory_id)
            except Exception:
                pass
        self.sqlite.mark_cold(memory_id)

    def warm_up(self, memory_id: str) -> None:
        """Move from cold back to warm: re-index in Qdrant."""
        records = self.sqlite.search(thread_id=None, limit=100)
        for r in records:
            if r["id"] == memory_id:
                qdrant = self._get_qdrant()
                if qdrant:
                    try:
                        qdrant.add(memory_id, str(r["content"]), {"thread_id": r["thread_id"], "type": r["event_type"]})
                    except Exception:
                        pass
                self.sqlite.warm_up(memory_id)
                break

    def scheduled_archive(self) -> list[str]:
        """Find warm records >10 days without access → archive to cold."""
        candidates = self.sqlite.get_cold_candidates(days=WARM_DAYS)
        to_archive = [c["id"] for c in candidates if c["protected"] == 0]
        for mid in to_archive:
            self.archive(mid)
        return to_archive

    def mark_protected(self, memory_id: str) -> None:
        """Mark a record as protected — never archived or deleted."""
        self.sqlite.mark_protected(memory_id)

    # ===== 遗忘机制 ====

    def scheduled_forget(self) -> list[str]:
        """Permanently delete cold records that have been archived for >30 days."""
        # Cold records older than 30 days → final deletion
        old_cold = self.sqlite.get_cold_candidates(days=30)
        to_delete = [c["id"] for c in old_cold if c["cold_label"] == "cold" and c["protected"] == 0]
        for mid in to_delete:
            self.sqlite.delete(mid)
        return to_delete
