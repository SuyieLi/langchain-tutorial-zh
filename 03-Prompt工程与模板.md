# 03 · Prompt 工程与模板

> PromptTemplate 的完整体系：变量、消息模板、消息占位符、Few-shot、输出格式化。
> 上一篇：[[02-模型调用基础]] ｜ 下一篇：[[04-输出解析与结构化输出]]

---

## 1. 为什么需要 Prompt 模板

直接写死 Prompt 的坏处：

```python
# ❌ 反模式：每次拼接字符串，易错、难维护
prompt = f"把下面这段话翻译成英文：{text}"
```

用模板的好处：

- ✅ **变量与结构分离**，一处修改全局生效
- ✅ 内置变量校验（缺变量直接报错，而不是生成坏 Prompt）
- ✅ 可复用、可组合（模板套模板）
- ✅ 是 LCEL 链式编程（[[05-LCEL表达式语言]]）的标准组件

---

## 2. 基础模板：`PromptTemplate`

### 2.1 字符串模板

```python
from langchain_core.prompts import PromptTemplate

# 方式一：from_template（用 {} 占位）
prompt = PromptTemplate.from_template(
    "你是{domain}领域的专家。请回答下面的问题：\n{question}"
)

# 方式二：构造函数（用变量名定义）
prompt = PromptTemplate(
    template="翻译成{target_lang}：{text}",
    input_variables=["target_lang", "text"],
)

# 渲染：把变量填进去
formatted = prompt.format(target_lang="英语", text="你好世界")
print(formatted)
# 翻译成英语：你好世界
```

### 2.2 缺失变量会怎样？

```python
prompt.format(target_lang="英语")   # ❌ KeyError: 'text'——提前发现错误，而不是发给模型
```

> [!note] 两种占位符语法
> `{variable}` 是最通用的；`{variable:format}` 可指定格式（如 `{name:0.2f}`）。实际开发用前者即可。

---

## 3. Chat 消息模板：`ChatPromptTemplate`（最常用）

聊天模型接收的是**消息列表**，所以需要能区分 system/user/assistant 角色的模板：

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位{domain}专家，回答要专业、简洁。"),
    ("human", "请解释一下：{question}"),
])

# 渲染成消息列表
messages = prompt.format_messages(domain="金融科技", question="什么是量化交易？")
# [SystemMessage(...), HumanMessage(...)]

# 直接接入模型
resp = model.invoke(messages)
```

**每个元素是 (角色, 模板) 二元组**，支持的角色：

| 角色 | 含义 |
|------|------|
| `system` | 系统设定（身份、规则、风格） |
| `human` | 用户输入 |
| `ai` | 模型的历史回复（多轮对话用） |
| `placeholder` | 动态消息插槽（见下节） |

---

## 4. 消息占位符：`MessagesPlaceholder`（多轮对话的关键）

**痛点**：多轮对话时，历史消息条数不定，无法用固定模板描述。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    MessagesPlaceholder("history"),        # ← 历史消息在这里动态插入
    ("human", "{question}"),
])

# 使用时传入 history
result = prompt.format_messages(
    history=[
        ("human", "我叫小明"),
        ("ai", "你好小明！"),
    ],
    question="我叫什么名字？",
)
```

> [!tip] 记忆口诀
> **固定消息用元组，动态消息用 `MessagesPlaceholder`**。
> `MessagesPlaceholder(variable_name="history")` 等价于简写 `("placeholder", "{history}")`。

`MessagesPlaceholder` 是 [[06-记忆与多轮对话]] 的基石——对话历史可以无限长，模板永远只写一次。

---

## 5. Few-shot 示例：给模型"打样"

### 5.1 普通 Few-shot（用变量塞示例）

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是电商客服，回答要友好。"),
    ("human", "示例1：用户问『多久发货』→ 答『亲，48小时内发货哦~』\n{examples}"),
    ("human", "{question}"),
])
```

### 5.2 结构化 Few-shot：`FewShotChatMessagePromptTemplate`（进阶）

更规范的做法是把示例做成结构化列表：

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

examples = [
    {"input": "你们的退货政策是？", "output": "7 天无理由退货，运费由我们承担！"},
    {"input": "产品保修多久？", "output": "整机保修 1 年，关键部件 3 年。"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{store}的客服，用示例中的风格回答问题。"),
    few_shot_prompt,                       # ← 示例整体作为一部分
    ("human", "{input}"),
])
```

---

## 6. Prompt 组合：`+` 拼接与条件选择

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# ① 字符串模板可以直接 + 拼接
full = PromptTemplate.from_template("系统信息：{sys}\n") + PromptTemplate.from_template("用户问题：{q}")

# ② 按条件路由（动态选择模板）
def route_to_prompt(question: str):
    if "代码" in question:
        return coding_prompt
    return general_prompt
```

> 模板组合、路由是高级 Prompt 管理的起点，配合 LangGraph 可做复杂路由逻辑（[[10-LangGraph工作流编排]]）。

---

## 7. 提示词工程核心原则（通用方法论）

框架之外，写 Prompt 本身有方法论。**这四条要刻进 DNA：**

1. **角色 + 任务 + 约束 + 示例** = 好 Prompt 四件套
   ```python
   ("system", "你是一位资深律师（角色）。帮我分析合同风险（任务）。"
              "只输出正式法律意见，不要闲聊（约束）。以下是参考格式：{example}（示例）")
   ```
2. **给模型"思考空间"**：复杂推理任务让模型先分析再回答（CoT，思维链）
3. **负面指令用"不要"说清楚**：`不要编造数据，不知道就说不知道`
4. **具体优于抽象**：`用 200 字以内` 优于 `回答简短点`

> [!note] 什么是好的测试集？
> 提示词改 10 个版本不如做 1 套测试集。把你的提问记录成测试样例，配合 LangSmith 评估
> （[[11-生产实践与LangSmith]]），让"调 Prompt"从玄学变成科学。

---

## ✅ 动手练习

1. 用 `ChatPromptTemplate` 做一个"翻译官"模板：system 指定语言方向，human 放原文；
2. 用 `MessagesPlaceholder` 构造一个包含 2 轮历史的多轮对话模板并渲染；
3. 把电商客服做成 Few-shot 版本，对比有无示例时模型回答的差别；
4. （进阶）思考你的实际业务里，哪些固定 Prompt 值得抽象成模板复用？

---

🏷️ `#LangChain` `#Prompt工程` `#模板`

[[README|← 返回学习路径总览]] ｜ [[04-输出解析与结构化输出|下一篇：输出解析 →]]
