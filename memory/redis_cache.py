"""Redis hot cache layer. Ephemeral — Qdrant/SQLite are source of truth."""
import json
import redis
from config.settings import REDIS_HOST, REDIS_PORT, REDIS_DB

_client = None


def _get() -> redis.Redis:
    global _client
    if _client is None:
        try:
            _client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                                  decode_responses=True, socket_connect_timeout=2)
            _client.ping()
        except Exception:
            _client = False  # 连不上就降级
    return _client


def get(key: str) -> str | None:
    r = _get()
    if not r:
        return None
    try:
        return r.get(key)
    except Exception:
        return None


def set(key: str, value: str, ttl: int = 259200) -> None:  # 默认 3 天
    r = _get()
    if not r:
        return
    try:
        r.setex(key, ttl, value)
    except Exception:
        pass


def delete(pattern: str) -> None:
    r = _get()
    if not r:
        return
    try:
        for k in r.scan_iter(pattern):
            r.delete(k)
    except Exception:
        pass
