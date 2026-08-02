# 09 · Agent 智能体

> 从"一问一答"到"自主完成任务"：Agent 决策循环、create_agent 实战与设计原则。
> 上一篇：[[08-RAG检索增强生成]] ｜ 下一篇：[[10-LangGraph工作流编排]]

---

## 1. 什么是 Agent

**Agent（智能体）= LLM（大脑）+ 工具（手脚）+ 循环（自主决策）**。

普通链（Chain）：问题 → 固定流程 → 答案（流程写死，模型无决策权）

```
Agent（1.0 的 ReAct 循环）：
思考(Reason) → 行动(Act，可选调工具) → 观察(Observe) → 再思考 → ... → 完成输出
```

| 对比 | 普通链 | Agent |
|------|--------|-------|
| 流程 | 写死（prompt → model → parser） | 模型自主决定 |
| 工具 | 一般不用 | 核心能力 |
| 场景 | 固定任务（翻译、摘要） | 开放任务（"帮我订个行程"） |
| 可控性 | 高 | 需约束（system_prompt、工具集） |

> [!note] ReAct 名字的由来
> ReAct = **Re**asoning + **Act**ing（推理 + 行动）论文（Yao et al., 2022）。
> 核心思想：让模型"边想边做"，把思考过程显式输出，再根据工具结果继续推理。

---

## 2. `create_agent`：1.0 的统一 Agent 接口

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# ① 定义工具
@tool
def get_weather(city: str) -> str:
    """查询指定城市天气。"""
    return f"{city} 晴，25°C"

@tool
def search_flights(origin: str, dest: str, date: str) -> str:
    """查询航班信息。"""
    return f"{origin}→{dest} {date}：09:00 有航班，¥680"

# ② 初始化模型
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# ③ 创建 Agent
agent = create_agent(
    model=model,
    tools=[get_weather, search_flights],
    system_prompt="你是旅行规划助手。查询信息时使用工具，信息不足主动追问。",
)

# ④ 调用（输入格式：messages 列表）
result = agent.invoke({
    "messages": [{"role": "user", "content": "帮我看看明天北京到上海的航班，顺便查下上海天气"}]
})
print(result["messages"][-1].content)   # 最终回答
```

### 调用约定（必须记住）

| 项 | 约定 |
|----|------|
| 输入 | `{"messages": [{"role": "user", "content": "..."}]}` 或消息对象列表 |
| 输出 | `result["messages"]` 是**完整消息轨迹**（含思考、工具调用、最终回答） |
| 最终回答 | `result["messages"][-1].content` |

> [!tip] 调试利器
> `print(result["messages"])` 能看到 Agent 每一步的完整轨迹：模型思考、工具调用、工具结果。
> 排错先看消息轨迹，比只看最终答案有用得多。

---

## 3. Agent 内部到底发生了什么

`create_agent` 底层是 **LangGraph 图**（[[10-LangGraph工作流编排]]），核心结构：

```
        ┌──────────────────────────────┐
        │          model 节点           │
        │   (思考 + 决定:回答 or 调工具) │
        └──────────────┬───────────────┘
                       │
        有 tool_calls？ ┤ 无 → 输出最终回答 ✅
                       │ 有
                       ▼
              ┌──────────────────┐
              │  tools 节点       │
              │  (执行工具,回填结果)│
              └─────────┬────────┘
                        │
                        └──→ 回到 model 节点（循环）
```

1. **model 节点**：模型读全部消息 → 决定"直接回答"还是"调用工具"
2. 若返回 `tool_calls` → 进入 **tools 节点**执行工具
3. 工具结果作为 `ToolMessage` 追加 → 回到 model 节点
4. 直到模型不再调工具 → 输出最终回答

> 循环可能多轮：一个任务里模型可能连续调用 3、5、10 次工具。

---

## 4. Agent 的进阶能力

### 4.1 结构化输出（让 Agent 返回固定结构）

```python
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy

class TripPlan(BaseModel):
    destination: str
    flight: str
    weather_tip: str
    cost_estimate: float

agent = create_agent(
    model=model,
    tools=[get_weather, search_flights],
    system_prompt="你是旅行规划助手。",
    response_format=ToolStrategy(TripPlan),   # ← Agent 最终输出必须符合该结构
)

result = agent.invoke({"messages": [{"role": "user", "content": "规划明天上海一日游"}]})
# result["structured_response"]  → TripPlan 对象
```

> 1.0 将结构化输出整合进 Agent 主循环，比旧版"多一次 LLM 调用"更快更省。

### 4.2 多轮对话 + 记忆

```python
# Agent 天然接收 messages 历史，直接多轮即可
messages = [
    {"role": "user", "content": "我喜欢海滩旅行"},
    {"role": "assistant", "content": "好的，我记下了！"},
    {"role": "user", "content": "推荐个目的地"},
]
result = agent.invoke({"messages": messages})
```

### 4.3 流式输出

```python
for chunk in agent.stream({"messages": [{"role": "user", "content": "规划明天行程"}]}):
    print(chunk)     # 每一节点/每一步的增量
```

---

## 5. Agent 设计原则（避坑指南）

### 5.1 工具设计（最重要）

- **数量克制**：5–10 个为宜，过多模型选不准
- **命名直观**：`get_weather` 优于 `w1`
- **docstring 写清"何时用"**：`"仅当用户要求退款时调用"`
- **返回友好错误**：工具内部捕获异常，返回 `"未找到该订单，请确认单号"` 而非崩溃

### 5.2 Prompt 设计

- system_prompt 明确：**身份、任务、工具使用规则、边界**
- 关键约束写进 prompt：`"信息不足时主动追问，不要猜测"`
- 禁止行为写清楚：`"不要编造航班价格"`

### 5.3 安全与成本（生产必看）

| 风险 | 对策 |
|------|------|
| 工具被滥用（模型乱调 API） | 最小权限工具集、参数校验、人工审批（HITL） |
| 无限循环烧钱 | 设置最大迭代轮数、超时、预算上限 |
| 幻觉 | 强制"无资料不回答"、结构化输出校验 |
| Prompt 注入 | 工具输入与用户输入隔离、内容过滤 |

> HITL（Human-in-the-loop，人机协同）是 LangGraph 的核心能力，见 [[10-LangGraph工作流编排]]。

---

## 6. 何时用 Agent，何时用普通链（决策树）

```
任务流程固定？（如：翻译/摘要/格式化）
 ├─ 是 → 普通 LCEL 链（更稳更省）✅
 └─ 否 → 需要外部信息/执行动作/多步决策？
       ├─ 是 → Agent（create_agent）✅
       └─ 否 → 复杂多步状态流程？
             └─ 是 → LangGraph 自定义图
```

> [!warning] 反模式
> **能用链解决的就别上 Agent。** Agent 的灵活性换来了不确定性和更高的 token 消耗。
> 固定流程 = 链；开放任务 = Agent；复杂状态机 = LangGraph。

---

## 7. Agentic RAG（进阶预告）

给 RAG 检索器穿上 Agent 外壳：

- Agent 自主决定**要不要检索**（简单问题直接答）
- 自主决定**检索几次、检索什么**（多轮细化查询）
- 检索结果不够好 → 改写查询再检索

```python
@tool
def search_knowledge_base(query: str) -> str:
    """在知识库中检索资料。"""
    return format_docs(retriever.invoke(query))

agent = create_agent(model, tools=[search_knowledge_base],
                     system_prompt="先判断是否需要检索资料，需要再调用工具。")
```

> 这就是"Agentic RAG"——RAG 与 Agent 的结合，2024 年后最流行的架构之一。

---

## ✅ 动手练习

1. 用 `create_agent` + 2–3 个自定义工具，构建一个"旅游助手"并完整对话；
2. 打印 `result["messages"]`，逐条观察"思考 → 调工具 → 观察 → 回答"轨迹；
3. 构造一个需要**连续多次工具调用**的任务，观察循环次数与 token 消耗；
4. 用 `response_format=ToolStrategy(...)` 让 Agent 输出结构化结果；
5. （进阶）给 Agent 加"最大迭代轮数"限制，防止死循环。

---

🏷️ `#LangChain` `#Agent` `#智能体` `#ReAct`

[[README|← 返回学习路径总览]] ｜ [[10-LangGraph工作流编排|下一篇：LangGraph →]]
