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

    def mark_protected(self, memory_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE events SET protected = 1 WHERE id = ?", (memory_id,))

    def search_cold(self, keyword: str, limit: int = 5) -> list[dict]:
        """Full-text search on cold/archived records."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE cold_label = 'cold' AND content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{keyword}%", limit)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def warm_up(self, memory_id: str) -> None:
        """Move a cold record back to hot."""
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE events SET cold_label = 'hot', last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                (now, memory_id)
            )

    def delete(self, memory_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM events WHERE id = ?", (memory_id,))

    def _row_to_dict(self, row: tuple) -> dict:
        cols = ["id", "thread_id", "event_type", "content", "importance",
                "created_at", "last_accessed", "access_count", "cold_label", "protected"]
        d = dict(zip(cols, row))
        d["content"] = json.loads(d["content"])
        return d
