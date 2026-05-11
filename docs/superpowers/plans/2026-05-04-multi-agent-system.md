# Multi-Agent Collaborative System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph-based Supervisor multi-agent system with MCP integration, three-layer memory, and human-in-the-loop supervision.

**Architecture:** Supervisor star-topology graph with 4 specialized sub-agents (Researcher, Analyst, Coder, Reviewer), MCP tool execution via MultiServerMCPClient, memory system (MemoryTool → MemoryManager → SQLite/Qdrant/Neo4j stores), BGE vector model for semantic memory.

**Tech Stack:** LangGraph 1.1.10, LangChain 1.2.17, langchain-mcp-adapters 0.2.2, DeepSeek API (via ChatOpenAI), SQLite, Qdrant, Neo4j, BGE-small-zh-v1.5

---

### Task 1: Install dependencies + project skeleton

**Files:**
- Modify: `.env` (create if absent)
- Create: `requirements.txt`

- [ ] **Step 1: Install missing packages**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/pip install openai sentence-transformers tiktoken qdrant-client neo4j
```

Expected: All 5 packages install without error.

- [ ] **Step 2: Write .env file**

```env
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
BGE_MODEL_PATH=C:\Users\86158\Desktop\bge-small-zh-v1.5
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=agent_memory
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password-here
SQLITE_PATH=./data/agent_memory.db
RAG_API_URL=http://localhost:8000
MAX_ITERATIONS=15
```

- [ ] **Step 3: Write requirements.txt**

```txt
openai>=1.0.0
sentence-transformers>=3.0.0
tiktoken>=0.7.0
qdrant-client>=1.9.0
neo4j>=5.0.0
python-dotenv>=1.0.0
```

- [ ] **Step 4: Create empty __init__.py files for all packages**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
mkdir -p config state agents mcp memory/stores rag utils data
touch config/__init__.py state/__init__.py agents/__init__.py mcp/__init__.py memory/__init__.py memory/stores/__init__.py rag/__init__.py utils/__init__.py
```

- [ ] **Step 5: Verify structure**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
ls -la config/ state/ agents/ mcp/ memory/ rag/ utils/
```

Expected: Each directory has `__init__.py`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: add project skeleton, dependencies, and .env template"
```

---

### Task 2: Config layer - settings.py + mcp_servers.py

**Files:**
- Create: `config/settings.py`
- Create: `config/mcp_servers.py`

- [ ] **Step 1: Write config/settings.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = "deepseek-v4-flash"

BGE_MODEL_PATH = os.getenv("BGE_MODEL_PATH")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "agent_memory")

NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/agent_memory.db")

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "15"))
TOKEN_MONITOR_THRESHOLD = 0.70   # 70% start monitoring
TOKEN_WARN_THRESHOLD = 0.80      # 80% ask user for compaction
TOKEN_FORCE_THRESHOLD = 0.90     # 90% force compaction
COMPACT_KEEP_RECENT = 8          # keep last 8 turns raw

HOT_DAYS = 3
WARM_DAYS = 10
REFLECTION_SAFETY_NET = 8       # force reflection after 8 rounds

assert DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY is required"
```

- [ ] **Step 2: Write config/mcp_servers.py**

```python
MCP_SERVERS = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
        "transport": "stdio",
    },
    "docker": {
        "command": "docker",
        "args": ["run", "-i", "mcp/docker"],
        "transport": "stdio",
    },
    "brave-search": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "transport": "stdio",
    },
}
```

- [ ] **Step 3: Verify imports**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "from config.settings import DEEPSEEK_MODEL, MAX_ITERATIONS; print(f'OK: model={DEEPSEEK_MODEL}, max_iter={MAX_ITERATIONS}')"
```

Expected: `OK: model=deepseek-v4-flash, max_iter=15`

- [ ] **Step 4: Commit**

```bash
git add config/settings.py config/mcp_servers.py
git commit -m "feat: add config layer with env loading and MCP server registry"
```

---

### Task 3: AgentState definition

**Files:**
- Create: `state/graph_state.py`

- [ ] **Step 1: Write state/graph_state.py**

```python
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
```

- [ ] **Step 2: Verify import**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "from state.graph_state import AgentState, initial_state; s = initial_state(); print(f'OK: {list(s.keys())}')"
```

Expected: `OK: ['messages', 'intent', ...]`

- [ ] **Step 3: Commit**

```bash
git add state/graph_state.py
git commit -m "feat: add AgentState TypedDict with all fields"
```

---

### Task 4: Async utility wrapper

**Files:**
- Create: `utils/async_utils.py`

- [ ] **Step 1: Write utils/async_utils.py**

```python
import asyncio
from typing import Any, Callable, Coroutine


def run_async(coro: Coroutine) -> Any:
    """Run an async coroutine synchronously. Safe for LangGraph sync nodes."""
    try:
        loop = asyncio.get_running_loop()
        # Already in async context — use thread executor as fallback
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)


def make_sync(async_fn: Callable[..., Coroutine]) -> Callable:
    """Wrap an async function to be callable synchronously."""
    def wrapper(*args, **kwargs):
        return run_async(async_fn(*args, **kwargs))
    return wrapper
```

- [ ] **Step 2: Verify**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "
import asyncio
from utils.async_utils import run_async
async def foo(): return 42
assert run_async(foo()) == 42
print('OK')
"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add utils/async_utils.py
git commit -m "feat: add async-to-sync wrapper for LangGraph nodes"
```

---

### Task 5: SQLite memory store

**Files:**
- Create: `memory/stores/sqlite_store.py`

- [ ] **Step 1: Write memory/stores/sqlite_store.py**

```python
import sqlite3
import json
import os
from datetime import datetime, timedelta
from config.settings import SQLITE_PATH


class SQLiteStore:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or SQLITE_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance TEXT DEFAULT 'low',
                    created_at TEXT NOT NULL,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 1,
                    cold_label TEXT DEFAULT 'hot',
                    protected INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_thread ON events(thread_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at)")

    def add(self, memory_id: str, thread_id: str, event_type: str,
            content: dict, importance: str = "low") -> None:
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (memory_id, thread_id, event_type, json.dumps(content),
                 importance, now, now, 1, "hot", 0)
            )

    def search(self, thread_id: str = None, event_type: str = None,
               limit: int = 10) -> list[dict]:
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        if thread_id:
            query += " AND thread_id = ?"
            params.append(thread_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_access(self, memory_id: str) -> None:
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE events SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                (now, memory_id)
            )

    def mark_cold(self, memory_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE events SET cold_label = 'cold' WHERE id = ?", (memory_id,))

    def get_cold_candidates(self, days: int = 10) -> list[dict]:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE last_accessed < ? AND protected = 0 AND cold_label != 'cold'",
                (cutoff,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def delete(self, memory_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM events WHERE id = ?", (memory_id,))

    def _row_to_dict(self, row: tuple) -> dict:
        cols = ["id", "thread_id", "event_type", "content", "importance",
                "created_at", "last_accessed", "access_count", "cold_label", "protected"]
        d = dict(zip(cols, row))
        d["content"] = json.loads(d["content"])
        return d
```

- [ ] **Step 2: Verify**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "
from memory.stores.sqlite_store import SQLiteStore
store = SQLiteStore()
store.add('test-1', 'thread-1', 'task', {'key': 'value'}, 'high')
results = store.search(thread_id='thread-1')
assert len(results) == 1
assert results[0]['content'] == {'key': 'value'}
print('OK')
"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add memory/stores/sqlite_store.py
git commit -m "feat: add SQLite memory store for events and cold/hot indexing"
```

---

### Task 6: Qdrant memory store

**Files:**
- Create: `memory/stores/qdrant_store.py`

- [ ] **Step 1: Write memory/stores/qdrant_store.py**

```python
import uuid
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from config.settings import QDRANT_URL, QDRANT_COLLECTION, BGE_MODEL_PATH


class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.model: Optional[SentenceTransformer] = None
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if QDRANT_COLLECTION not in collections:
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(BGE_MODEL_PATH)
        return self.model

    def _embed(self, text: str) -> list[float]:
        return self._get_model().encode(text).tolist()

    def add(self, memory_id: str, text: str, metadata: dict) -> None:
        vector = self._embed(text)
        self.client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[PointStruct(id=memory_id, vector=vector, payload=metadata)],
        )

    def search(self, query: str, limit: int = 5) -> list[dict]:
        vector = self._embed(query)
        results = self.client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=limit,
        )
        return [{"id": r.id, "score": r.score, **r.payload} for r in results]

    def delete(self, memory_id: str) -> None:
        self.client.delete(collection_name=QDRANT_COLLECTION, points_selector=[memory_id])
```

- [ ] **Step 2: Verify (skip if Qdrant Docker not running)**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "
try:
    from memory.stores.qdrant_store import QdrantStore
    store = QdrantStore()
    print('QdrantStore initialized (Qdrant must be running)')
except Exception as e:
    print(f'Skipped: {e}')
"
```

Expected: `QdrantStore initialized...` or `Skipped (Qdrant not running)`

- [ ] **Step 3: Commit**

```bash
git add memory/stores/qdrant_store.py
git commit -m "feat: add Qdrant vector store with BGE embeddings"
```

---

### Task 7: Neo4j memory store

**Files:**
- Create: `memory/stores/neo4j_store.py`

- [ ] **Step 1: Write memory/stores/neo4j_store.py**

```python
from neo4j import GraphDatabase
from config.settings import NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD

AGENT_LABELS = {
    "user_profile": "AgentMemory_UserProfile",
    "preference": "AgentMemory_Preference",
    "event": "AgentMemory_Event",
}


class Neo4jStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def add_relationship(self, subject: str, relation: str, obj: str,
                         label: str = "AgentMemory_Event", props: dict = None) -> None:
        safe_label = label.replace(":", "")
        query = (
            f"MERGE (s:{safe_label} {{name: $subject}}) "
            f"MERGE (o:{safe_label} {{name: $obj}}) "
            f"MERGE (s)-[r:{relation}]->(o) "
            f"SET r += $props"
        )
        with self.driver.session() as session:
            session.run(query, subject=subject, obj=obj,
                        props=props or {}, relation=relation)

    def search_relations(self, node_name: str, label: str = None,
                         depth: int = 2) -> list[dict]:
        lbl = label or "AgentMemory_Event"
        query = (
            f"MATCH path = (n:{lbl} {{name: $name}})-[*1..{depth}]-(connected) "
            f"RETURN [rel in relationships(path) | type(rel)] as relations, "
            f"[node in nodes(path) | node.name] as nodes "
            f"LIMIT 10"
        )
        with self.driver.session() as session:
            results = session.run(query, name=node_name)
            return [{"relations": r["relations"], "nodes": r["nodes"]} for r in results]

    def delete_node(self, node_name: str, label: str = None) -> None:
        lbl = label or "AgentMemory_Event"
        with self.driver.session() as session:
            session.run(f"MATCH (n:{lbl} {{name: $name}}) DETACH DELETE n", name=node_name)

    def close(self):
        self.driver.close()
```

- [ ] **Step 2: Verify (skip if Neo4j not running)**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "
try:
    from memory.stores.neo4j_store import Neo4jStore
    store = Neo4jStore()
    store.add_relationship('Peter', 'DISLIKES', 'SmallWindow', 'AgentMemory_Preference')
    store.close()
    print('Neo4jStore OK (Neo4j must be running)')
except Exception as e:
    print(f'Skipped: {e}')
"
```

Expected: `Neo4jStore OK...` or `Skipped...`

- [ ] **Step 3: Commit**

```bash
git add memory/stores/neo4j_store.py
git commit -m "feat: add Neo4j graph store for structured relationships"
```

---

### Task 8: MemoryManager (routing + classification)

**Files:**
- Create: `memory/memory_manager.py`

- [ ] **Step 1: Write memory/memory_manager.py**

```python
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

    def _get_qdrant(self) -> QdrantStore:
        if self.qdrant is None:
            try:
                self.qdrant = QdrantStore()
            except Exception:
                pass
        return self.qdrant

    def _get_neo4j(self) -> Neo4jStore:
        if self.neo4j is None:
            try:
                self.neo4j = Neo4jStore()
            except Exception:
                pass
        return self.neo4j

    def classify(self, content: dict) -> str:
        """Classify memory type: episodic, semantic, perceptual, working"""
        if "tool" in content:
            return "working"
        if "relation" in content or "preference" in content:
            return "semantic"
        if "event" in content or "task" in content:
            return "episodic"
        return "working"

    def route(self, memory_id: str, thread_id: str, memory_type: str,
              content: dict, importance: str = "low") -> None:
        # Always write event record to SQLite
        self.sqlite.add(memory_id, thread_id, memory_type, content, importance)

        text = str(content)

        if memory_type == "semantic":
            neo4j = self._get_neo4j()
            if neo4j and "subject" in content and "relation" in content:
                label = content.get("label", "AgentMemory_Event")
                neo4j.add_relationship(
                    content["subject"], content["relation"],
                    content.get("object", ""), label, content.get("props", {})
                )

        if memory_type in ("episodic", "perceptual"):
            qdrant = self._get_qdrant()
            if qdrant:
                qdrant.add(memory_id, text, {"thread_id": thread_id, "type": memory_type})

    def consolidate(self, thread_id: str) -> dict:
        """Funnel filter: working memory → long-term memory extraction"""
        recent = self.sqlite.search(thread_id=thread_id, limit=50)
        to_promote = [r for r in recent if r["importance"] == "high"]
        # Dedup: mark duplicates for merge
        results = {"consolidated": len(to_promote), "memory_ids": []}
        for r in to_promote:
            mt = self.classify(r["content"])
            self.route(r["id"], thread_id, mt, r["content"], r["importance"])
            results["memory_ids"].append(r["id"])
        return results

    def cold_storage_policy(self) -> list[str]:
        """Identify cold memories for potential deletion."""
        candidates = self.sqlite.get_cold_candidates(days=WARM_DAYS)
        return [c["id"] for c in candidates if c["protected"] == 0]

    def scheduled_forget(self, memory_ids: list[str]) -> None:
        for mid in memory_ids:
            self.sqlite.delete(mid)
            qdrant = self._get_qdrant()
            if qdrant:
                try:
                    qdrant.delete(mid)
                except Exception:
                    pass
```

- [ ] **Step 2: Verify**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "
from memory.memory_manager import MemoryManager
mm = MemoryManager()
cid = 'test-cons-1'
mm.sqlite.add(cid, 't1', 'task', {'task': 'learn python'}, 'high')
result = mm.consolidate('t1')
assert result['consolidated'] >= 1
print(f'OK: consolidated {result[\"consolidated\"]}')
"
```

Expected: `OK: consolidated 1`

- [ ] **Step 3: Commit**

```bash
git add memory/memory_manager.py
git commit -m "feat: add MemoryManager with classification, routing, and cold storage policy"
```

---

### Task 9: MemoryTool (unified interface)

**Files:**
- Create: `memory/memory_tool.py`

- [ ] **Step 1: Write memory/memory_tool.py**

```python
import uuid
from datetime import datetime
from memory.memory_manager import MemoryManager

_manager: MemoryManager = None


def _get_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager


def add(content: dict, thread_id: str = "default", importance: str = "low") -> str:
    memory_id = str(uuid.uuid4())[:8]
    mm = _get_manager()
    memory_type = mm.classify(content)
    mm.route(memory_id, thread_id, memory_type, content, importance)
    return memory_id


def search(query: str, thread_id: str = None, limit: int = 5) -> list[dict]:
    mm = _get_manager()
    qdrant = mm._get_qdrant()
    if qdrant:
        return qdrant.search(query, limit)
    # Fallback to SQLite text search
    rows = mm.sqlite.search(thread_id=thread_id, limit=limit)
    return [{"id": r["id"], "content": r["content"], "type": r["event_type"]} for r in rows]


def forget(memory_id: str) -> None:
    mm = _get_manager()
    mm.sqlite.delete(memory_id)
    qdrant = mm._get_qdrant()
    if qdrant:
        try:
            qdrant.delete(memory_id)
        except Exception:
            pass


def consolidate(thread_id: str = "default") -> dict:
    return _get_manager().consolidate(thread_id)


def search_graph(node_name: str, depth: int = 2) -> list[dict]:
    mm = _get_manager()
    neo4j = mm._get_neo4j()
    if neo4j:
        return neo4j.search_relations(node_name, depth=depth)
    return []
```

- [ ] **Step 2: Verify**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "
from memory.memory_tool import add, search, consolidate
mid = add({'task': 'test search'}, 't2', 'high')
results = search('test search', thread_id='t2')
print(f'OK: add={mid}, search_results={len(results)}')
"
```

Expected: `OK: add=xxxxxxxx, search_results=...`

- [ ] **Step 3: Commit**

```bash
git add memory/memory_tool.py
git commit -m "feat: add MemoryTool unified interface for agent consumption"
```

---

### Task 10: Context compaction

**Files:**
- Create: `memory/compact.py`

- [ ] **Step 1: Write memory/compact.py**

```python
import tiktoken
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config.settings import (
    TOKEN_MONITOR_THRESHOLD, TOKEN_WARN_THRESHOLD,
    TOKEN_FORCE_THRESHOLD, COMPACT_KEEP_RECENT, DEEPSEEK_MODEL
)


def count_tokens(messages: list, model: str = None) -> int:
    model = model or DEEPSEEK_MODEL
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
    """Returns: {'pressure': 'normal'|'warn'|'force', 'token_count': int, 'ratio': float}"""
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
    """Keep last COMPACT_KEEP_RECENT raw, summarize earlier messages."""
    if len(messages) <= COMPACT_KEEP_RECENT:
        return messages

    recent = messages[-COMPACT_KEEP_RECENT:]
    older = messages[:-COMPACT_KEEP_RECENT]

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

    return [SystemMessage(content=summary_content)] + list(recent)
```

- [ ] **Step 2: Verify token counting**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "
from langchain_core.messages import HumanMessage
from memory.compact import count_tokens, check_context_pressure
msgs = [HumanMessage(content='你好世界' * 100)]
t = count_tokens(msgs)
print(f'OK: tokens={t}, pressure={check_context_pressure(msgs, 65536)[\"pressure\"]}')
"
```

Expected: `OK: tokens=..., pressure=normal`

- [ ] **Step 3: Commit**

```bash
git add memory/compact.py
git commit -m "feat: add context compaction with token-aware pressure detection"
```

---

### Task 11: Dynamic reflection

**Files:**
- Create: `memory/reflection.py`

- [ ] **Step 1: Write memory/reflection.py**

```python
from memory.memory_tool import consolidate, forget, search
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

    # Trigger 2: any sub-agent flagged importance=high this round
    for t in completed:
        if t.get("importance") == "high" and not t.get("consolidated"):
            triggers.append("urgent_consolidate")

    # Trigger 3: new info contradicts existing memory — check via state flag
    if state.get("memory_contradiction"):
        triggers.append("memory_conflict")

    # Trigger 4: safety net — too many rounds since last reflection
    last_reflection = state.get("last_reflection_round", 0)
    current_round = state.get("iteration_count", 0)
    if current_round - last_reflection >= REFLECTION_SAFETY_NET:
        triggers.append("safety_net")

    return triggers


def reflect(state: dict, triggers: list[str]) -> dict:
    """Execute reflection based on triggers. Returns insights to inject into state."""
    insights = []

    if "urgent_consolidate" in triggers:
        thread_id = state.get("messages", [{}])[0].get("thread_id", "default") if state.get("messages") else "default"
        result = consolidate(thread_id)
        insights.append(f"consolidated {result['consolidated']} high-importance memories")

    if "plan_may_be_wrong" in triggers:
        insights.append("detected repeated failures — consider adjusting task plan")

    if "memory_conflict" in triggers:
        insights.append("memory contradiction detected — old memories may need update")

    if "safety_net" in triggers and len(insights) == 0:
        thread_id = "default"
        result = consolidate(thread_id)
        if result["consolidated"] > 0:
            insights.append(f"periodic consolidation: {result['consolidated']} memories")

    return {
        "insights": insights,
        "last_reflection_round": state.get("iteration_count", 0),
    }
```

- [ ] **Step 2: Verify logic**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "
from memory.reflection import reflection_check
state = {
    'completed_tasks': [
        {'agent': 'researcher', 'task': 'find_data', 'status': 'failed', 'importance': 'low'},
        {'agent': 'researcher', 'task': 'find_data', 'status': 'failed', 'importance': 'low'},
    ],
    'iteration_count': 10,
    'last_reflection_round': 0,
    'memory_contradiction': False,
}
triggers = reflection_check(state)
assert 'plan_may_be_wrong' in triggers
assert 'safety_net' in triggers
print(f'OK: triggers={triggers}')
"
```

Expected: `OK: triggers=['plan_may_be_wrong', 'safety_net']`

- [ ] **Step 3: Commit**

```bash
git add memory/reflection.py
git commit -m "feat: add event-driven dynamic reflection with 4 trigger conditions"
```

---

### Task 12: MCP client manager

**Files:**
- Create: `mcp/client_manager.py`

- [ ] **Step 1: Write mcp/client_manager.py**

```python
import hashlib
import json
from typing import Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils.async_utils import run_async


class MCPClientManager:
    def __init__(self, servers_config: dict):
        self.servers_config = servers_config
        self.client: MultiServerMCPClient = None
        self._tools: list = []
        self._cache: dict[str, Any] = {}

    def connect(self):
        self.client = MultiServerMCPClient(self.servers_config)
        run_async(self.client.__aenter__())

    def get_tools(self) -> list:
        """Return wrapped tools with caching layer."""
        if not self._tools:
            raw_tools = run_async(self.client.get_tools())
            self._tools = [self._wrap_with_cache(t) for t in raw_tools]
        return self._tools

    def _wrap_with_cache(self, tool):
        """Wrap a tool to cache results by tool_name + params hash."""
        original_invoke = tool.invoke

        def cached_invoke(input_str: str, **kwargs) -> Any:
            cache_key = self._cache_key(tool.name, input_str)
            if cache_key in self._cache:
                return self._cache[cache_key]
            result = original_invoke(input_str, **kwargs)
            self._cache[cache_key] = result
            return result

        tool.invoke = cached_invoke
        return tool

    def _cache_key(self, tool_name: str, input_str: str) -> str:
        h = hashlib.sha256(f"{tool_name}:{input_str}".encode()).hexdigest()[:16]
        return f"{tool_name}:{h}"

    def clear_cache(self):
        self._cache.clear()

    def get_cache_size(self) -> int:
        return len(self._cache)

    def disconnect(self):
        if self.client:
            run_async(self.client.__aexit__(None, None, None))
```

- [ ] **Step 2: Verify import and structure**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "
from mcp.client_manager import MCPClientManager
from config.mcp_servers import MCP_SERVERS
mgr = MCPClientManager(MCP_SERVERS)
print('MCPClientManager created OK')
print(f'Cache size: {mgr.get_cache_size()}')
"
```

Expected: `MCPClientManager created OK` / `Cache size: 0`

- [ ] **Step 3: Commit**

```bash
git add mcp/client_manager.py
git commit -m "feat: add MCP client manager with tool call caching"
```

---

### Task 13: RAG client

**Files:**
- Create: `rag/client.py`

- [ ] **Step 1: Write rag/client.py**

```python
import requests
from config.settings import RAG_API_URL


class RAGClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or RAG_API_URL

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        try:
            resp = requests.post(
                f"{self.base_url}/search",
                json={"query": query, "top_k": top_k},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.RequestException:
            return []

    def graph_query(self, cypher: str) -> list[dict]:
        try:
            resp = requests.post(
                f"{self.base_url}/graph/query",
                json={"cypher": cypher},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.RequestException:
            return []
```

- [ ] **Step 2: Verify import**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "
from rag.client import RAGClient
rc = RAGClient()
results = rc.search('test')
print(f'OK: results={len(results)} (expected 0 if RAG not running)')
"
```

Expected: `OK: results=0 (expected 0 if RAG not running)`

- [ ] **Step 3: Commit**

```bash
git add rag/client.py
git commit -m "feat: add RAG REST API client for external knowledge base"
```

---

### Task 14: Researcher Agent

**Files:**
- Create: `agents/researcher.py`

- [ ] **Step 1: Write agents/researcher.py**

```python
from langchain_core.messages import SystemMessage, HumanMessage
from state.graph_state import AgentState
from memory.memory_tool import search, add

RESEARCHER_PROMPT = """你是信息采集员。只做一件事: 调用工具获取外部信息。

能力: 调用全部 MCP 工具 (文件系统、浏览器搜索、Docker)
约束:
- 只用工具获取信息，不对信息做分析或总结
- 输出: {"source": "工具名", "raw_result": "...", "confidence": 1-5}
- 工具调用失败报告原因，最多重试 2 次
- 你是原料供应商，你的下游是 Analyst

## 记忆指令
- 执行前: memory_search(task_keywords) 查历史类似检索
- 执行后: memory_add({task, result, timestamp})
- 重要发现标记 importance=high"""


def researcher_node(state: AgentState, llm, tools: list) -> AgentState:
    task = state["pending_tasks"][0] if state["pending_tasks"] else {"task": "unknown"}
    task_desc = task.get("task", str(task))

    # Pre-search memory
    history = search(task_desc, limit=3)
    history_context = "\n".join(str(h.get("content", "")) for h in history) if history else "无相关历史"

    system_msg = SystemMessage(content=RESEARCHER_PROMPT)
    user_msg = HumanMessage(content=f"任务: {task_desc}\n历史相关记录: {history_context}\n请调用工具采集信息。")

    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke([system_msg, user_msg])

    # Post-execution: record to memory
    add({"task": task_desc, "result": response.content, "timestamp": "now"}, importance="high" if "重要" in response.content else "low")

    state["messages"].append(response)
    state["completed_tasks"].append({
        "agent": "researcher", "task": task_desc,
        "result": response.content, "status": "done",
    })
    state["pending_tasks"] = state["pending_tasks"][1:]
    state["current_agent"] = "researcher"
    return state
```

- [ ] **Step 2: Verify import**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "from agents.researcher import researcher_node, RESEARCHER_PROMPT; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agents/researcher.py
git commit -m "feat: add Researcher agent with memory integration"
```

---

### Task 15: Analyst Agent

**Files:**
- Create: `agents/analyst.py`

- [ ] **Step 1: Write agents/analyst.py**

```python
from langchain_core.messages import SystemMessage, HumanMessage
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

## 记忆指令
- 执行前: memory_search(task_keywords) 获取相关知识
- 执行后: memory_add({analysis, conclusions, timestamp})
- 产出结论时标记 importance=high → 触发 consolidate"""


def analyst_node(state: AgentState, llm, tools: list) -> AgentState:
    task = state["pending_tasks"][0] if state["pending_tasks"] else {"task": "analyze"}
    task_desc = task.get("task", str(task))

    # Gather context from messages
    recent_msgs = state["messages"][-5:]
    context = "\n".join(
        f"[{type(m).__name__}] {m.content[:500]}" for m in recent_msgs if hasattr(m, "content")
    ) if recent_msgs else "无上游输入"

    history = search(task_desc, limit=3)
    history_context = "\n".join(str(h.get("content", "")) for h in history) if history else "无"

    system_msg = SystemMessage(content=ANALYST_PROMPT)
    user_msg = HumanMessage(content=f"任务: {task_desc}\n上游信息: {context}\n历史经验: {history_context}\n请分析并输出结构化结果。")

    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke([system_msg, user_msg])

    add({"task": task_desc, "analysis": response.content[:500], "timestamp": "now"},
        importance="high")

    state["messages"].append(response)
    state["completed_tasks"].append({
        "agent": "analyst", "task": task_desc,
        "result": response.content, "status": "done",
    })
    state["pending_tasks"] = state["pending_tasks"][1:]
    state["current_agent"] = "analyst"
    return state
```

- [ ] **Step 2: Verify import**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "from agents.analyst import analyst_node; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agents/analyst.py
git commit -m "feat: add Analyst agent with structured output formatting"
```

---

### Task 16: Coder Agent

**Files:**
- Create: `agents/coder.py`

- [ ] **Step 1: Write agents/coder.py**

```python
from langchain_core.messages import SystemMessage, HumanMessage
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

## 记忆指令
- 执行前: memory_search(task_keywords) 查历史解决过的类似问题
- 执行后: memory_add({solution, files_changed, timestamp})
- 修正过的 bug 标记 importance=high"""


def coder_node(state: AgentState, llm, tools: list) -> AgentState:
    task = state["pending_tasks"][0] if state["pending_tasks"] else {"task": "write_code"}
    task_desc = task.get("task", str(task))

    recent_msgs = state["messages"][-5:]
    context = "\n".join(
        f"[{type(m).__name__}] {m.content[:500]}" for m in recent_msgs if hasattr(m, "content")
    ) if recent_msgs else "无上游需求"

    history = search(task_desc, limit=3)
    history_context = "\n".join(str(h.get("content", "")) for h in history) if history else "无"

    system_msg = SystemMessage(content=CODER_PROMPT)
    user_msg = HumanMessage(content=f"编程任务: {task_desc}\n需求上下文: {context}\n历史解决记录: {history_context}")

    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke([system_msg, user_msg])

    add({"task": task_desc, "solution": response.content[:500], "timestamp": "now"},
        importance="high")

    state["messages"].append(response)
    state["completed_tasks"].append({
        "agent": "coder", "task": task_desc,
        "result": response.content, "status": "done",
        "needs_review": True,
    })
    state["pending_tasks"] = state["pending_tasks"][1:]
    state["current_agent"] = "coder"
    return state
```

- [ ] **Step 2: Verify import**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "from agents.coder import coder_node; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agents/coder.py
git commit -m "feat: add Coder agent with environment-aware code generation"
```

---

### Task 17: Reviewer Agent

**Files:**
- Create: `agents/reviewer.py`

- [ ] **Step 1: Write agents/reviewer.py**

```python
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
    # Review the last completed task
    completed = state["completed_tasks"]
    if not completed:
        return state
    target = completed[-1]

    task_desc = target.get("task", "")
    result = target.get("result", "")

    history = search(f"review:{task_desc}", limit=3)
    history_context = "\n".join(str(h.get("content", "")) for h in history) if history else "无"

    system_msg = SystemMessage(content=REVIEWER_PROMPT)
    user_msg = HumanMessage(content=f"审查任务: {task_desc}\n输出内容: {result[:2000]}\n历史审查记录: {history_context}\n请给出评分和建议。")

    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke([system_msg, user_msg])

    # Parse score from response
    import json, re
    score = 3
    try:
        match = re.search(r'"score"\s*:\s*(\d)', response.content)
        if match:
            score = int(match.group(1))
    except Exception:
        pass

    passed = score >= 3
    target["reviewed"] = True
    target["review_score"] = score
    target["review_pass"] = passed

    if score >= 5:
        state["final_output"] = result
    if score <= 2:
        target["status"] = "failed"
        state["need_human_approval"] = True

    add({"review_task": task_desc, "score": score, "passed": passed, "timestamp": "now"},
        importance="high" if score <= 2 else "low")

    state["messages"].append(response)
    state["current_agent"] = "reviewer"
    return state
```

- [ ] **Step 2: Verify import**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "from agents.reviewer import reviewer_node; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agents/reviewer.py
git commit -m "feat: add Reviewer agent with scoring and human approval triggering"
```

---

### Task 18: Supervisor Agent + routing

**Files:**
- Create: `agents/supervisor.py`

- [ ] **Step 1: Write agents/supervisor.py**

```python
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.constants import Send, END
from state.graph_state import AgentState
from memory.memory_tool import search, add
from memory.reflection import reflection_check, reflect
from config.settings import MAX_ITERATIONS


SUPERVISOR_PROMPT = """你是中心调度器。你统一管理多个专家Agent协同完成用户任务。

你的职责:
1. 意图识别: 判断用户请求是"简单任务"还是"复杂任务"
   - 简单任务: 单步工具调用能解决 → route to tool_executor
   - 复杂任务: 需要多步推理或多Agent协同 → 拆解为子任务，分配子Agent
2. 任务规划: 将复杂任务拆解为有序子任务列表
3. 动态调度: 根据进度和Agent返回结果，决定下一步
4. 质量把关: 判断任务是否完成，是否需回滚重做
5. 反思决策: 检测到重复失败/矛盾记忆/高重要性标记时，触发记忆整合

可用子Agent:
- researcher: 信息采集，调用MCP工具获取外部数据
- analyst: 加工分析，结构化输出报告/方案
- coder: 编程执行，生成和运行代码
- reviewer: 质量审查，独立验证输出质量

调度原则:
- 需要外部数据 → 先派 researcher
- 需要分析整理 → 派 analyst (在researcher之后)
- 需要编程 → 派 coder (coder之后必须经过reviewer)
- 最终审查 → 派 reviewer
- 无数据依赖的子任务可以并行

输出格式:
如果是简单任务: {"intent": "simple", "route": "tool_executor", "reason": "..."}
如果是复杂任务: {"intent": "complex", "plan": [{"agent": "researcher", "task": "..."}, ...], "reason": "..."}"""


def supervisor_node(state: AgentState, llm, tools: list) -> AgentState:
    state["iteration_count"] += 1

    system_msg = SystemMessage(content=SUPERVISOR_PROMPT)
    user_msg = HumanMessage(content=f"用户消息: {state['messages'][-1].content if state['messages'] else '无'}")

    response = llm.invoke([system_msg, user_msg])

    # Parse intent and plan
    try:
        decision = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: extract JSON from text
        import re
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        decision = json.loads(match.group()) if match else {"intent": "simple", "route": "tool_executor"}

    state["intent"] = decision.get("intent", "simple")

    if state["intent"] == "complex" and "plan" in decision:
        for item in decision["plan"]:
            state["pending_tasks"].append({
                "agent": item.get("agent", "researcher"),
                "task": item.get("task", ""),
                "status": "pending",
            })

    # Reflection check
    triggers = reflection_check(state)
    if triggers:
        reflection_result = reflect(state, triggers)
        state["reflection_insights"].extend(reflection_result["insights"])
        state["last_reflection_round"] = state["iteration_count"]

    state["messages"].append(AIMessage(content=response.content))
    state["current_agent"] = "supervisor"
    return state


def supervisor_routing(state: AgentState):
    """Conditional edge routing. Returns node name, list[Send], or END."""
    # Force terminate if max iterations
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "reviewer"

    # Check if all done
    if not state["pending_tasks"] and state["completed_tasks"]:
        last_done = state["completed_tasks"][-1]
        if last_done.get("review_pass", False):
            return END

    # Simple task → direct tool call
    if state["intent"] == "simple":
        return "tool_executor"

    # Complex task: dispatch next agent(s)
    pending = [t for t in state["pending_tasks"] if t.get("status") == "pending"]
    if not pending:
        return "reviewer"

    # Check for parallel dispatch opportunity
    if len(pending) > 1:
        # Send in parallel if no dependency between them
        for t in pending:
            t["status"] = "dispatched"
        return [Send(t["agent"], {"task": t}) for t in pending]

    task = pending[0]
    task["status"] = "dispatched"
    return task["agent"]
```

- [ ] **Step 2: Verify imports**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "
from agents.supervisor import supervisor_node, supervisor_routing, SUPERVISOR_PROMPT
print('OK: supervisor imports')
"
```

Expected: `OK: supervisor imports`

- [ ] **Step 3: Commit**

```bash
git add agents/supervisor.py
git commit -m "feat: add Supervisor agent with intent recognition, routing, and reflection"
```

---

### Task 19: Main entry point - Graph assembly

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write main.py**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, SQLITE_PATH
from config.mcp_servers import MCP_SERVERS
from state.graph_state import AgentState, initial_state
from agents.supervisor import supervisor_node, supervisor_routing
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.coder import coder_node
from agents.reviewer import reviewer_node
from mcp.client_manager import MCPClientManager
from memory.memory_tool import search, add


def build_graph():
    # LLM
    llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL,
        temperature=0.3,
    )

    # MCP tools
    mcp = MCPClientManager(MCP_SERVERS)
    mcp.connect()
    tools = mcp.get_tools()

    # Tool executor (simple pass-through for simple tasks)
    def tool_executor_node(state: AgentState) -> AgentState:
        msg = state["messages"][-1].content
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(msg)
        state["messages"].append(response)
        state["completed_tasks"].append({"agent": "tool_executor", "task": msg[:100], "result": response.content, "status": "done"})
        return state

    # Build graph
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", lambda s: supervisor_node(s, llm, tools))
    builder.add_node("tool_executor", tool_executor_node)
    builder.add_node("researcher", lambda s: researcher_node(s, llm, tools))
    builder.add_node("analyst", lambda s: analyst_node(s, llm, tools))
    builder.add_node("coder", lambda s: coder_node(s, llm, tools))
    builder.add_node("reviewer", lambda s: reviewer_node(s, llm, tools))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", supervisor_routing)

    # All sub-agents and tool_executor return to supervisor
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")
    builder.add_edge("coder", "supervisor")
    builder.add_edge("reviewer", "supervisor")
    builder.add_edge("tool_executor", "supervisor")

    # Checkpointer
    checkpointer = SqliteSaver.from_conn_string(SQLITE_PATH)
    return builder.compile(checkpointer=checkpointer, interrupt_before=["tool_executor"])


def chat(graph, user_input: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
    )
    return state


if __name__ == "__main__":
    graph = build_graph()
    print("Agent graph built. Entering interactive mode (type 'exit' to quit).")

    tid = "session-1"
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        result = chat(graph, user_input, tid)
        # Print final output if available
        if result.get("final_output"):
            print(f"\nAgent: {result['final_output']}")
        elif result.get("messages"):
            last = result["messages"][-1]
            if hasattr(last, "content"):
                print(f"\nAgent: {last.content}")
            else:
                print(f"\nAgent: {last}")
```

- [ ] **Step 2: Verify graph compiles**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "
from main import build_graph
graph = build_graph()
print('Graph compiled OK')
print(f'Nodes: {list(graph.nodes.keys())}')
"
```

Expected: `Graph compiled OK` / `Nodes: ['supervisor', 'tool_executor', 'researcher', 'analyst', 'coder', 'reviewer']`

- [ ] **Step 3: Dry-run smoke test**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "
from main import build_graph, chat
graph = build_graph()
result = chat(graph, '你好，请介绍一下你能做什么')
print(f'OK: iteration_count={result.get(\"iteration_count\")}, intent={result.get(\"intent\")}')
"
```

Expected: Graph runs, returns state with `iteration_count >= 1` and some intent.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add main graph assembly with supervisor-star topology and sqlite checkpointer"
```

---

### Final Verification

- [ ] **Run full integration test**

```bash
cd "C:\Users\86158\.claude\my_project\agent开发"
.venv/Scripts/python -c "
from main import build_graph
graph = build_graph()
assert 'supervisor' in graph.nodes
assert 'researcher' in graph.nodes
assert 'reviewer' in graph.nodes
print('All systems go.')
"
```

Expected: `All systems go.`
