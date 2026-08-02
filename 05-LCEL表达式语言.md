# 05 · LCEL 表达式语言

> LangChain 的灵魂：用 `|` 把组件串成管道，组合式编程构建复杂应用。
> 上一篇：[[04-输出解析与结构化输出]] ｜ 下一篇：[[06-记忆与多轮对话]]

---

## 1. LCEL 是什么

**LCEL（LangChain Expression Language，LangChain 表达式语言）** 是一种声明式编程范式：

> 用 `|`（管道符）把 LangChain 组件串联起来，前一个的输出自动成为后一个的输入。

```python
# 三件套：模板 → 模型 → 纯文本解析
chain = prompt | model | StrOutputParser()
```

**为什么它如此重要？**

| 传统写法（❌） | LCEL（✅） |
|----------------|-----------|
| 手动调用每个组件，if/else 处理 | 一行声明管道 |
| 同步/异步/流式/批量要分别写 | 四种调用方式**自动全部支持** |
| 中间步骤难观察 | 天然可追踪（配合 LangSmith） |
| 组件难复用 | 每个链本身也是组件，可嵌套组合 |

> [!note] 官方原话
> LCEL 的三大承诺：**先发制人的并行**、**自动重试与回退**、**对异步/流式/批量的第一等支持**。

---

## 2. Runnable 协议：LCEL 的地基

LCEL 能生效，是因为每个组件都实现了 **`Runnable` 接口**——统一提供：

```
invoke()     单次调用
stream()     流式输出
batch()      批量处理
ainvoke()    异步单次
astream()    异步流式
abatch()     异步批量
```

**任何组件（Prompt / Model / Parser / Retriever / 自定义 Runnable）都长一个样**，所以才能无缝 `|` 拼接。

### 2.1 RunnableLambda：把任意 Python 函数变成 Runnable

```python
from langchain_core.runnables import RunnableLambda

def add_header(text: str) -> str:
    return f"[翻译任务] {text}"

chain = RunnableLambda(add_header) | model | StrOutputParser()
# 普通函数 → Runnable，混入管道
```

### 2.2 RunnablePassthrough：原样透传（RAG 里很常用）

```python
from langchain_core.runnables import RunnablePassthrough

# 透传 = 什么都不改，只是"经过"——常用于同时保留多个变量
chain = (
    {"question": RunnablePassthrough()}     # 问题原样传入
    | RunnableLambda(print_debug)           # 可以打印调试
)
```

> 在 [[08-RAG检索增强生成]] 中，`RunnablePassthrough.assign()` 用来"一边检索文档、一边保留用户问题"。

---

## 3. 管道核心规则（必须吃透）

### 3.1 `|` 的语义：输出 → 输入

```python
chain = prompt | model | parser
# 输入 dict → prompt 渲染成 messages → model 输出 AIMessage → parser 输出纯文本
```

### 3.2 输入/输出的格式约定

| 组件 | 输入 | 输出 |
|------|------|------|
| `PromptTemplate` | dict（模板变量） | PromptValue |
| `ChatPromptTemplate` | dict | messages |
| `ChatModel` | messages / str | AIMessage |
| `StrOutputParser` | AIMessage | str |
| `PydanticOutputParser` | AIMessage | Pydantic 对象 |

> [!warning] 最常见的报错：KeyError / 参数不匹配
> `prompt | model` 拼接时，prompt 输出 messages，正好是 model 的输入——所以顺序不能乱。
> 若某个环节输入结构不对（如 model 收到 dict），会立刻抛 `InvalidInputError`。善用报错信息定位。

### 3.3 dict 同时驱动多个变量

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}   # 并行：检索 + 透传
    | prompt
    | model
    | StrOutputParser()
)
# 注意：{} 内是"并行执行"的，互不等待（LCEL 的自动并行）
```

---

## 4. 实战组合：一个完整的翻译链

```python
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = init_chat_model("gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业翻译，把{target}翻译得地道自然。"),
    ("human", "{text}"),
])

chain = prompt | model | StrOutputParser()

# 单次
print(chain.invoke({"target": "英语", "text": "机器学习改变了世界"}))

# 流式（打字机效果）
for chunk in chain.stream({"target": "日语", "text": "机器学习改变了世界"}):
    print(chunk, end="", flush=True)

# 批量
results = chain.batch([
    {"target": "英语", "text": "你好"},
    {"target": "法语", "text": "谢谢"},
])

# 异步
import asyncio
async def main():
    r = await chain.ainvoke({"target": "英语", "text": "异步真香"})
    print(r)
asyncio.run(main())
```

**同一个 chain，四种调用方式零修改。** 这就是 LCEL 的威力。

---

## 5. 高级：合并、分支与条件路由

### 5.1 并行合并多个链：`RunnableParallel`

```python
from langchain_core.runnables import RunnableParallel

chain1 = prompt_a | model | StrOutputParser()   # 写摘要
chain2 = prompt_b | model | StrOutputParser()   # 提取关键词

parallel = RunnableParallel(summary=chain1, keywords=chain2)
result = parallel.invoke({"doc": "..."})
# {"summary": "...", "keywords": "..."}  ← 两条链并行跑
```

### 5.2 条件路由：`RunnableBranch`（或 Dict 路由）

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "代码" in x["question"], coding_chain),   # 条件1 → 链1
    (lambda x: "翻译" in x["question"], translate_chain),# 条件2 → 链2
    default_chain,                                        # 兜底
)
```

> [!tip] 路由的另一选择
> 也可以让**模型自己决定**走哪条路（RouterPrompt / 工具调用实现），
> 复杂场景交给 [[09-Agent智能体]] / [[10-LangGraph工作流编排]] 更优雅。

### 5.3 可配置链：`.configurable_fields()` / `.configurable_alternatives()`

```python
chain = prompt | model | StrOutputParser()

# 运行时切换模型（不重写代码）
config = {"configurable": {"model": "claude-sonnet-4-5"}}
result = chain.invoke({"text": "hi"}, config=config)
```

---

## 6. 调试技巧

### 6.1 打断管道观察中间结果

```python
from langchain_core.runnables import RunnableLambda

def debug(x):
    print("=== 中间结果 ===")
    print(x)
    return x

chain = prompt | RunnableLambda(debug) | model | StrOutputParser()
```

### 6.2 结合 LangSmith 追踪

设置 `LANGSMITH_TRACING=true` 后，每次调用自动记录完整链路（每步的输入输出、耗时、token 数），可视化调试：

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_xxx
```

> 📖 详见 [[11-生产实践与LangSmith]]。

---

## ✅ 动手练习

1. 搭一个"笑话 → 翻译成英语 → 判断幽默等级"的三段链；
2. 用 `RunnableParallel` 让模型同时输出"摘要"和"关键词"，打印结果；
3. 用 `RunnableBranch` 实现"问代码走代码链，问翻译走翻译链"；
4. 给链加上 `.configurable_fields()`，运行两次分别用不同模型；
5. 用 `RunnableLambda` 插入调试，观察 prompt 输出长什么样。

---

🏷️ `#LangChain` `#LCEL` `#Runnable` `#管道`

[[README|← 返回学习路径总览]] ｜ [[06-记忆与多轮对话|下一篇：记忆 →]]
