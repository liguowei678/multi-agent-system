MCP_SERVERS = {
    # 本地文件系统 — 读写文件、目录操作 (14个工具)
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
        "transport": "stdio",
    },
    # Tavily 搜索 — 网页搜索 + 内容提取 (替代 brave-search)
    "tavily-search": {
        "command": "npx",
        "args": ["-y", "tavily-mcp@latest"],
        "transport": "stdio",
    },
    # 浏览器操控 — Puppeteer 控制浏览器
    "browser": {
        "command": "npx",
        "args": ["-y", "puppeteer-mcp-server"],
        "transport": "stdio",
    },
}
