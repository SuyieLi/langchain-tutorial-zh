# 10 · LangGraph 工作流编排

> 从"黑盒 Agent"到"可编程状态机"：LangGraph 的节点、边、状态与高级模式。
> 上一篇：[[09-Agent智能体]] ｜ 下一篇：[[11-生产实践与LangSmith]]

---

## 1. 为什么需要 LangGraph

`create_agent` 帮你做了标准 Agent 循环，但真实业务往往需要**更精细的控制**：

| 需求 | 普通 Agent 能行吗 |
|------|-------------------|
| 固定的业务顺序（审核→生成→发送） | ❌ 流程不由模型决定 |
| 人类审批环节（AI 生成 → 人工确认 → 再执行） | ❌ |
| 流程中断后恢复（服务器重启接着跑） | ❌ |
| 多个 Agent 协作分工 | ❌ |

**LangGraph = 用"图"来编排 LLM 工作流**：节点（做什么）+ 边（怎么走）+ 状态（数据流）。

> [!note] 定位
> LangChain = 高层抽象（自动挡）；**LangGraph = 低层编排（手动挡）**。
> `create_agent` 的底层就是一个 LangGraph 图。会了 LangGraph，你就能"拆开看"并定制它。

---

## 2. 核心概念：Node / Edge / State

```
           ┌─────────────────┐
  START →  │  节点A (LLM)     │ ──正常路径──→ ┌──────────────┐
           └─────────────────┘               │  节点B (工具)  │ → END
                   │ 条件边(if)              └──────────────┘
                   ▼ (不满足条件)
              ┌────────────┐
              │  节点C (兜底) │ → END
              └────────────┘
```

| 概念 | 说明 | 类比 |
|------|------|------|
| **State（状态）** | 贯穿全程的数据容器（TypedDict） | 数据库记录 |
| **Node（节点）** | 一个执行单元（函数），读状态、写状态 | 函数 |
| **Edge（边）** | 节点间连接 | 调用关系 |
| **条件边** | 根据状态决定走哪条路 | if/else |
| **Checkpointer** | 每一步状态快照（持久化） | 存档点 |

---

## 3. 第一个 LangGraph 应用（5 分钟上手）

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

model = init_chat_model("gpt-4o-mini")

# ① 定义状态：图里流动的数据
class State(TypedDict):
    messages: list          # 消息列表（贯穿全程）
    sentiment: str          # 自定义字段

# ② 节点：普通 Python 函数（读 state → 返回更新）
def analyze_sentiment(state: State) -> State:
    """节点1：情感分析"""
    resp = model.invoke([
        SystemMessage("判断用户情绪：positive / negative / neutral，只输出一个词"),
        HumanMessage(state["messages"][-1].content),
    ])
    return {"sentiment": resp.content.strip()}   # 只返回要更新的字段

def generate_reply(state: State) -> State:
    """节点2：生成回复（根据情感定制风格）"""
    style = {"positive": "热情", "negative": "安慰", "neutral": "客观"}[state["sentiment"]]
    resp = model.invoke([
        SystemMessage(f"用{style}的语气回复用户"),
        HumanMessage(state["messages"][-1].content),
    ])
    return {"messages": [resp]}                  # 追加回复到消息列表

# ③ 建图：节点 + 边
graph = StateGraph(State)
graph.add_node("analyze", analyze_sentiment)
graph.add_node("reply", generate_reply)
graph.add_edge(START, "analyze")     # 入口 → 分析
graph.add_edge("analyze", "reply")   # 分析 → 回复
graph.add_edge("reply", END)         # 回复 → 出口

app = graph.compile()                # 编译成可运行图

# ④ 运行
result = app.invoke({"messages": [HumanMessage("今天项目终于上线了！")]})
print(result["sentiment"])           # positive
print(result["messages"][-1].content)  # 热情回复
```

> [!tip] 节点返回值的约定
> 节点函数返回的 dict **只包含要更新的字段**，LangGraph 自动合并进全局状态。
> 返回 `{"messages": [resp]}` 表示"追加消息"，返回 `{"sentiment": x}` 表示"更新字段"。

---

## 4. 条件边：让流程"智能分叉"

```python
from langgraph.graph import StateGraph, START, END

def route_by_sentiment(state: State) -> str:
    """根据情感决定走哪条路（返回目标节点名）"""
    if state["sentiment"] == "negative":
        return "comfort"          # 负面 → 安慰节点
    return "normal_reply"         # 其他 → 普通回复

graph = StateGraph(State)
graph.add_node("analyze", analyze_sentiment)
graph.add_node("comfort", comfort_node)        # 安慰
graph.add_node("normal_reply", normal_node)    # 普通回复
graph.add_edge(START, "analyze")
graph.add_conditional_edges(
    "analyze",
    route_by_sentiment,                        # 路由函数
    {"comfort": "comfort", "normal_reply": "normal_reply"},  # 返回名 → 节点
)
graph.add_edge("comfort", END)
graph.add_edge("normal_reply", END)
```

**条件边 = 让流程拥有 if/else 的能力**。路由函数可以很简单（读字段），也可以让模型来决定（LLM 路由）。

---

## 5. 循环：LangGraph 的"循环"能力

Agent 的工具循环本质上就是"环"。手动实现一个：

```python
def should_continue(state: State) -> str:
    """最后一条消息是工具调用 → 继续循环；否则 → 结束"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END

graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")    # 工具执行完回到 agent（形成环）
```

> [!note] 与旧版区别
> 1.0 起，`create_react_agent`（langgraph.prebuilt）已废弃，迁移到 `create_agent`（langchain.agents）。
> 但 LangGraph 的底层 API（StateGraph / Node / Edge）不变，学一次永久受用。

---

## 6. 持久化与断点（生产级能力）

### 6.1 Checkpointer：自动存档 + 断点续跑

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()              # 生产用 PostgreSQL/SQLite 版本
app = graph.compile(checkpointer=checkpointer)

# 每次调用带 thread_id（会话标识），状态自动持久化
config = {"configurable": {"thread_id": "thread-1"}}
app.invoke({"messages": [HumanMessage("第一步")]}, config)

# 服务器重启后，仍可用同一 thread_id 恢复上下文
app.invoke({"messages": [HumanMessage("第二步")]}, config)
```

> 这是 LangGraph 1.0 主打的 **Durable State（持久状态）**：进程崩溃、重启都不丢上下文。

### 6.2 Human-in-the-loop：人工审批

```python
# ① 在敏感节点前打断：compile 时指定
app = graph.compile(checkpointer=checkpointer, interrupt_before=["send_email"])

# ② 第一次运行：会在 send_email 前暂停
result = app.invoke(initial_input, config)
print("已暂停，等待人工确认...")

# ③ 人工确认后：继续执行
app.invoke(None, config)   # 从断点继续
```

| 人机协同模式 | 用途 |
|--------------|------|
| `interrupt_before` | 关键动作前暂停等审批 |
| 修改状态后继续 | 人工修改 AI 生成的方案再执行 |
| 审核 Agent | 审核通过才放行 |

> 高危场景（发邮件、转账、发布）**必须**加人工审批——这是生产底线。

---

## 7. 多 Agent 协作

复杂系统 = 多个专职 Agent 分工：

```python
# 模式：Supervisor（主管）
# 主管 Agent 决定把任务分派给哪个子 Agent（研究员 / 写手 / 审查员）
```

```python
# 伪代码示意
graph.add_node("supervisor", supervisor_agent)   # 调度者
graph.add_node("researcher", researcher_agent)   # 查资料
graph.add_node("writer", writer_agent)           # 写文章
graph.add_node("reviewer", reviewer_agent)       # 审稿

# 主管动态路由到子 Agent，子 Agent 完成后回到主管
graph.add_conditional_edges("supervisor", route_to_agent, AGENT_MAP)
```

---

## 8. LangGraph 选型决策表

| 需求 | 方案 |
|------|------|
| 标准工具循环 | `create_agent`（[[09-Agent智能体]]）就够 |
| 固定业务流程（审核→生成→发送） | LangGraph 自定义图 |
| 需要断点续跑 / 多日任务 | LangGraph + Checkpointer |
| 需要人工审批 | LangGraph + interrupt |
| 多 Agent 分工 | LangGraph Supervisor 模式 |
| 只是问答 / 翻译 | 普通 LCEL 链（[[05-LCEL表达式语言]]），别用图 |

> [!warning] 复杂度警示
> LangGraph 强大但**有学习成本**。先从 `create_agent` 开始，遇到"控制不了流程"再升级到 LangGraph。

---

## ✅ 动手练习

1. 搭一个"翻译 → 检测语言 → 按语言调整风格"的三节点图；
2. 给图加一个条件边：当检测到负面情绪时走安慰分支；
3. 用 InMemorySaver 做 checkpointer，分两次调用验证状态保持；
4. 用 `interrupt_before` 实现"人工确认后才输出最终结果"；
5. （进阶）模仿 Supervisor 模式，构建"研究员+写手"双 Agent 协作图。

---

🏷️ `#LangGraph` `#工作流` `#状态机` `#HITL`

[[README|← 返回学习路径总览]] ｜ [[11-生产实践与LangSmith|下一篇：生产实践 →]]
