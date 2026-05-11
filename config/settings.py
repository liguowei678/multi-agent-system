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

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "1"))  # 用 db 1，不和 RAG 的 db 0 混

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "15"))
TOKEN_MONITOR_THRESHOLD = 0.70
TOKEN_WARN_THRESHOLD = 0.80
TOKEN_FORCE_THRESHOLD = 0.90
COMPACT_KEEP_RECENT = 16  # ≈8轮对话

HOT_DAYS = 3
WARM_DAYS = 10
REFLECTION_SAFETY_NET = 8

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

assert DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY is required"
