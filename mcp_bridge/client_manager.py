import hashlib
from typing import Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import StructuredTool
from utils.async_utils import run_async


class MCPClientManager:
    def __init__(self, servers_config: dict):
        self.servers_config = servers_config
        self._tools: list = []
        self._cache: dict[str, Any] = {}

    def get_tools(self) -> list:
        """Return tools from all MCP servers. Each connects independently."""
        if self._tools:
            return self._tools

        for server_name, server_config in self.servers_config.items():
            try:
                single_config = {server_name: server_config}
                client = MultiServerMCPClient(single_config)
                raw_tools = run_async(client.get_tools())
                wrapped = [self._wrap_with_cache(t) for t in raw_tools]
                self._tools.extend(wrapped)
                print(f"MCP [{server_name}]: {len(raw_tools)} 个工具已加载")
            except Exception as e:
                print(f"MCP [{server_name}]: 跳过 — {e}")

        return self._tools

    def _wrap_with_cache(self, tool):
        """Wrap an async MCP tool with sync invoke + result caching."""
        original_ainvoke = tool.ainvoke  # MCP tools are async-only

        def cached_func(**kwargs) -> Any:
            raw = str(sorted(kwargs.items()))
            cache_key = self._cache_key(tool.name, raw)
            if cache_key in self._cache:
                return self._cache[cache_key]
            result = run_async(original_ainvoke(kwargs))
            self._cache[cache_key] = result
            return result

        return StructuredTool.from_function(
            func=cached_func,
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
        )

    def _cache_key(self, tool_name: str, input_str: str) -> str:
        h = hashlib.sha256(f"{tool_name}:{input_str}".encode()).hexdigest()[:16]
        return f"{tool_name}:{h}"

    def clear_cache(self):
        self._cache.clear()

    def get_cache_size(self) -> int:
        return len(self._cache)
