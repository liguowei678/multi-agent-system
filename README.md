# Supervisor 多智能体协同问答助手

## 概述

基于 LangGraph 的 Supervisor 多智能体系统+RAG内部知识库检索。Supervisor(temperature=0) 统一调度 Researcher/Analyst/Coder/Reviewer四个子Agent，意图分类与任务拆解分离为两次独立LLM 调用；集成 MCP 协议（filesystem + Tavily 搜索 + Browser）共 27 个外部工具，stdio传输；五层记忆检索（Redis→Qdrant→Neo4j→关键词兜底→SQLite），LLM 语义抽取三元组写入知识图谱，采用冷热处理机制，记忆 10 天降冷 30 天删除含保护标记；上下文压缩保留最近 16条消息+摘要；动态反思自动触发送检；FastAPI + SSE 逐 token 流式输出，StreamWrapper 透明拦截 LLM 调用零侵入。

## 技术栈

- **Agent 框架**: LangGraph 1.1.10 + LangChain 1.2.17
- **MCP**: langchain-mcp-adapters 0.2.2, MultiServerMCPClient, stdio 传输
- **LLM**: DeepSeek API (deepseek-v4-flash), ChatOpenAI 兼容接口；Supervisor/Researcher 用 temperature=0，其余 0.3
- **记忆存储**: Redis(热缓存) + Qdrant(语义向量) + Neo4j(知识图谱) + SQLite(事件日志)
- **向量模型**: BGE-small-zh-v1.5 (本地), SentenceTransformers 加载
- **Web 前端**: 纯 HTML/CSS/JS 单文件, Dark Theme 三栏布局, SSE 流式接收
- **API 层**: FastAPI + uvicorn, lifespan 模式加载 graph

## 项目结构

```
agent开发/
├── .env
├── main.py                           # 入口: 构建graph, 原生工具, 启动对话
├── server.py                         # FastAPI HTTP/SSE 服务端
├── static/
│   └── index.html                    # 单文件前端 (CSS Grid + 原生JS)
├── config/
│   ├── settings.py                   # 环境变量, 模型配置, 阈值常量
│   └── mcp_servers.py                # MCP 服务器注册 (stdio)
├── state/
│   └── graph_state.py                # AgentState (TypedDict, 16字段)
├── agents/
│   ├── supervisor.py                 # 调度+路由+意图分类+进度检查+反思
│   ├── researcher.py                 # 信息采集(SOP+强制RAG优先+拦截Tavily)
│   ├── analyst.py                    # 分析/写作
│   ├── coder.py                      # 编程执行
│   └── reviewer.py                   # 质量审查(score 1-5)
├── mcp_bridge/
│   └── client_manager.py             # MultiServerMCPClient, 独立连接, 工具缓存
├── memory/
│   ├── memory_tool.py                # 统一接口: add/search/forget/consolidate
│   ├── memory_manager.py             # 分类/路由/LLM抽取/升降冷/遗忘
│   ├── redis_cache.py                # Redis热缓存 (TTL 3天)
│   ├── compact.py                    # 上下文压缩 (保留16条≈8轮+摘要)
│   ├── reflection.py                 # 动态反思 (LLM传参)
│   └── stores/
│       ├── sqlite_store.py           # 事件日志+时间线
│       ├── qdrant_store.py           # 语义记忆 (collection: agent_memory)
│       └── neo4j_store.py            # 图谱关系 (label: AgentMemory_Preference)
├── rag/
│   └── client.py                     # RAG REST API (混合检索)
└── utils/
    └── async_utils.py                # asyncio→sync 桥接
```

## 环境配置

### .env

```
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
BGE_MODEL_PATH=C:\Users\86158\Desktop\bge-small-zh-v1.5
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=agent_memory
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=xxx
SQLITE_PATH=./data/agent_memory.db
RAG_API_URL=http://localhost:8000
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=1
TAVILY_API_KEY=tvly-dev-xxx
MAX_ITERATIONS=15
```

### MCP 服务器 (config/mcp_servers.py)

```python
MCP_SERVERS = {
    "filesystem":    {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."], "transport": "stdio"},
    "tavily-search": {"command": "npx", "args": ["-y", "tavily-mcp@latest"], "transport": "stdio"},
    "browser":       {"command": "npx", "args": ["-y", "puppeteer-mcp-server"], "transport": "stdio"},
}
```

### 工具清单 (总计 31)

| 来源 | 工具 | 数量 |
|------|------|------|
| filesystem MCP | read/write/edit/list/search 等 | 14 |
| tavily-search MCP | tavily_search/extract/crawl/map/research | 5 |
| browser MCP | browser_init/navigate/get_content 等 | 8 |
| 原生 open_browser | Python webbrowser 模块 | 1 |
| 原生 run_python | subprocess 执行 Python | 1 |
| 原生 rag_search | RAG HTTP 混合检索 | 1 |
| 原生 search_graph | Neo4j 图谱关系查询 | 1 |

### Agent 工具分配 (避免选择混乱)

| Agent | 工具 | 数量 |
|-------|------|------|
| tool_executor | open_browser | 1 |
| researcher | rag_search + open_browser + search_graph + tavily + browser MCP | 18 |
| coder | run_python + filesystem MCP | 15 |
| supervisor | 全部 (仅规划, 不执行) | 31 |

## 图拓扑

```
START → supervisor
          ├─► tool_executor   简单(聊天/回忆/打开网页)
          ├─► researcher      信息检索(RAG/Tavily/Browser/图谱)
          ├─► analyst         加工输出
          ├─► coder           编程执行
          └─► reviewer        质量审查 → END/打回

Supervisor 路由:
  iteration=1, intent=simple  → tool_executor
  iteration=1, intent=complex → researcher, 拆子任务
  iteration>1, simple → END
  iteration>1, complex → 进度检查 → reviewer / 继续
```

### 复杂任务案例: 搜索 + 代码执行 + 报告生成

**用户输入**: "帮我搜索LangGraph最新版本的变化，然后写个Python脚本检查当前环境安装的版本是否符合最新要求，把检查结果整理成报告"

**执行流程**:
```
Supervisor 意图分类 → complex
  → 任务拆解(单独LLM调用): researcher + coder + analyst + reviewer
  → 派 researcher → [Tavily搜索] → 获取PyPI版本列表(1.1.10/1.2.0a3等)
  → 派 coder      → [run_python: pip show langgraph] → 当前版本 1.1.10
  → 派 analyst    → [合成上游结果] → 生成结构化Markdown报告
  → 派 reviewer   → [审查] → 通过
```

**关键设计**: 意图分类和任务拆解分离为两次独立LLM调用，确保拆解可靠性。

## AgentState

```python
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
```

## Agent 设计

### Supervisor — 意图分类 + 调度

- temperature=0 确保路由确定性
- `_messages_to_text()` 将最近16条消息(≈8轮)转为纯文本传给 LLM
- 输出 JSON: `{"intent":"simple/complex", "plan":[...]}`
- 进度检查只对复杂任务；简单任务直接 END

### Researcher — 信息采集

强制 RAG 优先执行机制:
```
1. LLM 自由选择工具(含RAG/Tavily/Browser/图谱)
2. 代码层强制: 如果LLM没选RAG → 插入RAG到最前面
3. RAG有结果 → 拦截Tavily, 跳过公网搜索
4. RAG无结果 → Tavily兜底, 不显示RAG的空结果
```

工具选择 SOP: 内部知识(制度/报销等)→RAG; 外部信息→Tavily; 浏览网页→Browser; 文件→filesystem

### Analyst / Coder / Reviewer

- Analyst: 结构化输出报告, 可调 rag_search/tavily 补搜
- Coder: 写代码+执行(run_python), 输出必须经 Reviewer
- Reviewer: 独立验证, score 1-5, ≤2 触发人类监督

### importance 判断 (关键词匹配)

| Agent | high 条件 | 默认 |
|-------|----------|------|
| Researcher | 偏好/不喜欢/喜欢/讨厌/习惯/个人/关键/核心 | low |
| Analyst | 用户画像/偏好/策略建议/个人习惯/关键结论 | low |
| Coder | 修复/bug/安全/关键/核心/生产 | low |
| Reviewer | score ≤ 2 失败案例 | low |

## 记忆系统

### 五层检索 (每次任务都执行)

```
search(query)
  ├─ 1. Redis 热层    (TTL 3天, <1ms)
  ├─ 2. Qdrant 温层   (语义向量, <10ms)
  ├─ 3. Neo4j 图谱    (关系+节点模糊匹配, <10ms)
  ├─ 4. 关键词兜底     (双字词硬匹配Qdrant summary)
  ├─ 5. SQLite 全量   (兜底)
  └─ 结果回填 Redis
```

### Neo4j 图查询

- 写入: `consolidate()` → LLM 从对话提取三元组 → `AgentMemory_Preference` label
- 读取: `search_relations()` 支持查询词包含节点名或关系类型
- "我讨厌什么" → CONTAINS "讨厌" → 匹配关系类型 → 返回 `(用户)-[讨厌]->(弹窗广告)`

### 记忆生命周期

**A类: protected=1 (永久)**
```
LLM抽取成功 → Neo4j写入 → mark_protected → 永不降冷/删除
```

**B类: protected=0 (30天窗口)**
```
add → SQLite+Qdrant+Redis
  → 10天无访问 → 降冷 (Qdrant删索引, SQLite标cold)
  → 30天 → 删除
```

降冷是删除前的20天反悔期。冷数据被检索→自动升温回Qdrant。

### 跨会话记忆

checkpoint 清空后, tool_executor 仍通过 `mem_search()` 查到 Qdrant/Neo4j 中的持久记忆。

### 对话记忆

- 保留最近16条消息(≈8轮)原始 + 更早的 compact.py 摘要压缩
- 每个对话独立 thread_id, checkpoint 按 thread 隔离

## 知识图谱整合 (LLM语义抽取)

### 触发机制

Redis 计数器: 每次 supervisor 运行 +1, 满8触发 consolidate → 归零。
约每3-4个用户消息触发一次。

### 抽取流程

```
consolidate(thread_id, llm):
  1. 取 SQLite 最近10条
  2. 拼成 "用户: xxx\n系统: xxx" 文本
  3. LLM + EXTRACT_PROMPT 判断是否有结构化知识
     - 有: 输出 [{"subject":"实体","relation":"关系","object":"对象"}]
     - 无: 输出 []
  4. 写入 Neo4j (AgentMemory_Preference)
  5. 成功 → mark_protected (对应SQLite记录不降冷/不删除)
```

### 抽取准入标准

- 准入: 用户偏好/好恶/习惯/禁忌, 人际关系/归属/身份, 明确态度或评价
- 不准入: 普通问答/查询/闲聊, 暂时性信息, 一次性任务

## RAG 集成

`rag/client.py` 通过 `POST /api/v1/qa-hybrid-graphrag/query` 调用 Docker 里的 RAG 服务。
Agent 内存数据与 RAG 知识库隔离: Qdrant `agent_memory` collection vs `default`, Neo4j `AgentMemory_Preference` label vs RAG 业务 labels.

## 记忆维护

每次启动 (`build_graph()`) 自动执行:
- `scheduled_archive()`: >10天 + protected=0 → 降冷
- `scheduled_forget()`: cold + >30天 + protected=0 → 删除

## Web 前端 & SSE 流式输出

### 整体架构

```
浏览器 (static/index.html)
  │  fetch POST /api/chat/stream (Accept: text/event-stream)
  ▼
server.py (FastAPI)
  │  async def event_generator():
  │    async for event in chat_stream_events(graph, msg, tid):
  │      yield f"data: {json}\n\n"
  │  return StreamingResponse(..., media_type="text/event-stream")
  ▼
main.py → chat_stream_events()
  │  graph.astream_events(input, config, version="v2")
  │  每个节点 on_chain_start/end → agent_start/agent_end 事件
  │  每个 token (via StreamWrapper) → 推入 queue.Queue → yield token 事件
  │  完成 → aget_state() → yield complete 事件
  ▼
StreamWrapper (LLM 透明包装)
  │  拦截 llm.invoke() → 内部用 llm.stream()
  │  每个 chunk.content → _push_token(token, agent_name)
  │  chunk 累加 (AIMessageChunk.__add__) → 返回完整 AIMessage
```

### SSE 是什么

Server-Sent Events。HTTP 长连接，服务器单向推送，浏览器 `EventSource` / `fetch + ReadableStream` 接收。

与 WebSocket 区别:
| | SSE | WebSocket |
|------|------|------|
| 方向 | 服务器→客户端 | 双向 |
| 协议 | HTTP/1.1 标准 | ws:// 升级 |
| 断线重连 | 浏览器自动 | 需手动 |
| 适用场景 | 流式输出、通知 | 聊天、游戏 |

Agent 流式场景天然适合 SSE——用户发一次请求，服务端持续推送 agent 执行过程。

### Token 级流式实现

**问题**: Graph 节点是 sync 函数, 内部 `llm.invoke()` 对 graph 来说是黑盒, `astream_events` 拿不到 token。

**方案**: `StreamWrapper` 透明包装——不改任何 agent 代码:

```python
class StreamWrapper:
    def __init__(self, llm, agent_name):
        self._llm = llm        # ChatOpenAI(streaming=True)
        self._agent = agent_name

    def invoke(self, messages, **kwargs):
        """拦截 invoke, 内部用 stream, 逐 token 推送"""
        full = None
        for chunk in self._llm.stream(messages, **kwargs):
            full = chunk if full is None else full + chunk  # AIMessageChunk 累加
            if chunk.content:
                _push_token(chunk.content, self._agent)     # → queue.Queue
        return full  # 完整 AIMessage (含 tool_calls)

    def bind_tools(self, tools):  # 保持 tool calling 兼容
        return StreamWrapper(self._llm.bind_tools(tools), self._agent)
```

**token 传输通道**: `queue.Queue` (线程安全). LLM stream 在 thread pool 执行 → `put()` token → async event loop 取走 → yield SSE.

**每个 agent 独立 wrapper**: `llm_supervisor`, `llm_researcher`, `llm_analyst`, `llm_coder`, `llm_reviewer`, `llm_tool`, 保证推送给前端的 agent 名正确。

### 前端关键设计

- **三栏布局**: 左侧会话列表(260px) | 中间聊天区 | 右侧详情面板(380px, 可折叠)
- **JSON 过滤**: `isSupervisorJson()` 检测 supervisor 决策 JSON, 不渲染到聊天区
- **Token 实时渲染**: supervisor 的 token 不实时显示(输出的是 JSON), 其他 agent 逐 token 追加到 live bubble
- **迷你进度条**: 嵌入聊天流, 默认折叠, 展开显示时间线, 完成后保留为绿色 ✓
- **打字机动画**: 最终回复 >80字时逐字显示, 每个 interval 2字符/25ms
- **会话管理**: localStorage 持久化 threadId, 切换/删除会话

### 启动方式

```bash
uvicorn server:app --host 0.0.0.0 --port 8001
# 浏览器打开 http://localhost:8001
```

## 注意事项

- 密钥不进代码 (.env 已 gitignore)
- Qdrant/Neo4j/Redis 需 Docker 容器已启动, 失败时降级运行
- BGE 模型首次加载慢, 结果缓存到内存
- mcp_bridge/ 目录名避免与 Python mcp 包冲突
- 浏览器 MCP 需 Chrome 完全关闭后使用
- 删 checkpoint 命令: `del data\agent_memory.db`
