import json
import uuid
from memory.memory_manager import MemoryManager
from memory import redis_cache

_manager: MemoryManager = None


def _get_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager


def add(content: dict, thread_id: str = "default", importance: str = "low") -> str:
    """Write a memory. Returns memory_id."""
    memory_id = str(uuid.uuid4())
    mm = _get_manager()
    memory_type = mm.classify(content)
    try:
        mm.route(memory_id, thread_id, memory_type, content, importance)
    except Exception:
        pass
    # 写入热缓存
    redis_cache.set(f"mem:{memory_id}", json.dumps({"content": content, "type": memory_type}, ensure_ascii=False))
    return memory_id


def search(query: str, thread_id: str = None, limit: int = 5) -> list[dict]:
    """Four-tier search: hot(Redis) → warm(Qdrant) → cold(SQLite archive) → deep(SQLite)."""
    cache_key = f"search:{query}:{limit}"

    # 1. 热层: Redis
    cached = redis_cache.get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    # 2. 温层: Qdrant
    mm = _get_manager()
    qdrant = mm._get_qdrant()
    results = []
    if qdrant:
        try:
            results = qdrant.search(query, limit)
        except Exception:
            pass

    # 3. Neo4j 图查询: 结构化关系 (Qdrant未命中优先查图谱)
    if not results:
        neo4j = mm._get_neo4j()
        if neo4j:
            try:
                graph_results = neo4j.search_relations(query, depth=2)
                results = [{"id": "neo4j", "content": r, "type": "semantic_graph"} for r in graph_results]
            except Exception:
                pass

    # 3b. 双字词硬匹配兜底 (Neo4j也未命中时)
    if not results and qdrant:
        try:
            q_words = [query[i:i+2] for i in range(len(query)-1)]
            pts, _ = qdrant.client.scroll(collection_name="agent_memory", limit=50, with_payload=True)
            for p in pts:
                text = str(p.payload.get('summary', ''))
                if sum(1 for w in q_words if w in text) >= 2:
                    results.append({"id": p.id, "content": text, "type": "qdrant_fallback"})
                    if len(results) >= limit:
                        break
        except Exception:
            pass

    # 4. 冷层: SQLite 归档数据
    if not results:
        rows = mm.sqlite.search_cold(query, limit)
        results = [{"id": r["id"], "content": r["content"], "type": r["event_type"]} for r in rows]
        for r in rows:
            mm.warm_up(r["id"])

    # 5. 兜底: SQLite 全部记录
    if not results:
        rows = mm.sqlite.search(thread_id=thread_id, limit=limit)
        results = [{"id": r["id"], "content": r["content"], "type": r["event_type"]} for r in rows]

    # 回填热层
    redis_cache.set(cache_key, json.dumps(results, ensure_ascii=False), ttl=3600)
    return results


def archive(memory_id: str) -> None:
    """Move to cold: remove Qdrant index, keep SQLite record."""
    _get_manager().archive(memory_id)


def mark_protected(memory_id: str) -> None:
    """Mark as protected — never archived or deleted."""
    _get_manager().mark_protected(memory_id)


def scheduled_archive() -> list[str]:
    """Archive warm records >10 days without access."""
    return _get_manager().scheduled_archive()


def forget(memory_id: str) -> None:
    """Permanently delete from all stores (cold records only)."""
    _get_manager().scheduled_forget()


def consolidate(thread_id: str = "default", llm=None) -> dict:
    """LLM 从对话中抽取结构化知识 → Neo4j."""
    return _get_manager().consolidate(thread_id, llm)


def search_graph(node_name: str, depth: int = 2) -> list[dict]:
    """Search Neo4j graph for relationships."""
    mm = _get_manager()
    neo4j = mm._get_neo4j()
    if neo4j:
        try:
            return neo4j.search_relations(node_name, depth=depth)
        except Exception:
            pass
    return []
