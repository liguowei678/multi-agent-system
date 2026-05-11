import requests
from config.settings import RAG_API_URL


class RAGClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or RAG_API_URL).rstrip("/")

    def search(self, query: str, top_k: int = 5) -> str:
        """混合检索: 语义 + 知识图谱。返回 LLM 生成的答案。"""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/qa-hybrid-graphrag/query",
                json={"query": query, "top_k": top_k},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            # 返回 LLM 生成的答案，附上检索耗时
            answer = data.get("answer", "")
            retrieval_time = data.get("retrieval_time", 0)
            if answer:
                return f"[RAG 内部知识] {answer} (检索耗时: {retrieval_time:.1f}s)"
            return "RAG 无匹配结果"
        except requests.RequestException as e:
            return f"RAG 不可用: {e}"

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """纯语义搜索，不走图谱。"""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/qa/simple/semantic/query",
                json={"query": query, "top_k": top_k},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", data.get("documents", []))
        except requests.RequestException:
            return []

    def graph_query(self, cypher: str) -> list[dict]:
        """Neo4j 图查询已有知识图谱。"""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/graphrag/query",
                json={"cypher": cypher},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.RequestException:
            return []
