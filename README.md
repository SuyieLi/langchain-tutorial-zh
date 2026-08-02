# 🧠 LangChain 学习笔记（基于 v1.0）

> 从入门到进阶的系统性学习路径，笔记与可运行代码配套。
> 基于 **LangChain 1.0**（2025-10 正式发布），兼顾 0.x 旧版兼容知识。

---

## 🗺️ 学习路径总览（MOC）

### 第一阶段：地基（入门）

| 笔记 | 主题 | 状态 |
|------|------|------|
| [[00-学习路径与生态全景]] | 生态全景、1.0 包结构、学习路线图 | ✅ |
| [[01-环境搭建与第一个程序]] | 安装、API Key、Hello World | ✅ |
| [[02-模型调用基础]] | Chat Models、`init_chat_model`、多厂商切换 | ✅ |
| [[03-Prompt工程与模板]] | PromptTemplate、MessagePlaceholder、Few-shot | ✅ |
| [[04-输出解析与结构化输出]] | Output Parser、结构化输出（Pydantic） | ✅ |

### 第二阶段：进阶（构建应用）

| 笔记 | 主题 | 状态 |
|------|------|------|
| [[05-LCEL表达式语言]] | `\|` 管道、Runnable、链式组合、流式 | ✅ |
| [[06-记忆与多轮对话]] | 对话历史、Message History、内存存储 | ✅ |
| [[07-工具与函数调用]] | `@tool`、Tool Calling、多工具协作 | ✅ |
| [[08-RAG检索增强生成]] | 加载→切分→向量化→检索→生成 全流程 | ✅ |

### 第三阶段：高级（生产级智能体）

| 笔记 | 主题 | 状态 |
|------|------|------|
| [[09-Agent智能体]] | `create_agent`、ReAct 循环、Agent 设计 | ✅ |
| [[10-LangGraph工作流编排]] | 图编排、状态、持久化、人机协同 | ✅ |
| [[11-生产实践与LangSmith]] | 可观测性、评估、部署、成本控制 | ✅ |
| [[12-常见问题与避坑指南]] | FAQ、报错排查、性能优化 | ✅ |

---

## 📂 目录结构

```
LangChain/
├── README.md                  # 本文件（学习路径 MOC）
├── 00~12-*.md                 # 13 篇分主题笔记
└── code/                      # 配套可运行代码
    ├── requirements.txt
    ├── 01_hello_langchain.py
    ├── 02_models.py
    ├── 03_prompt_templates.py
    ├── 04_structured_output.py
    ├── 05_lcel.py
    ├── 06_memory.py
    ├── 07_tools.py
    ├── 08_rag.py
    ├── 09_agent.py
    ├── 10_langgraph_workflow.py
    └── 11_langsmith_eval.py
```

---

## 🚀 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. 安装依赖（见笔记 01）
pip install -r code/requirements.txt

# 3. 配置 API Key
# Windows (PowerShell): $env:OPENAI_API_KEY="sk-..."
# macOS/Linux:         export OPENAI_API_KEY="sk-..."

# 4. 运行第一个程序
python code/01_hello_langchain.py
```

> 💡 国内用户可改用 DeepSeek / 通义千问 / 智谱 GLM 等，代码中通过 `init_chat_model` 一行切换，详见 [[02-模型调用基础]]。

---

## 📌 版本说明

本教程基于 **LangChain 1.0**（Python 3.10+）。核心 API 变化：

- ❌ `from langchain.chat_models import ChatOpenAI` → ✅ `from langchain.chat_models import init_chat_model`
- ❌ `AgentExecutor` / `create_react_agent` → ✅ `create_agent`（基于 LangGraph）
- ❌ `from langchain.llms import OpenAI`（旧 LLM 类）→ ✅ 统一使用 Chat Models
- ✅ 新增 `content_blocks` 标准内容块（支持推理痕迹、引用、工具调用）

遇到 0.x 时代的老教程/老代码时，请对照 [[12-常见问题与避坑指南]] 中的迁移对照表。

---

## 🏷️ 标签

`#LangChain` `#LLM` `#AI编程` `#学习笔记` `#Obsidian`
