# LangChain 简介

LangChain 是一个用于构建大语言模型（LLM）应用的开发框架。
它的核心理念是提供统一的抽象接口，让开发者可以轻松切换不同的模型提供商。

## 核心组件

1. 模型（Chat Models）：统一封装不同厂商的聊天模型接口。
2. Prompt 模板：把提示词和变量分离，方便复用。
3. 输出解析器：把模型输出转换为结构化数据。
4. 工具（Tools）：让模型能够调用外部函数和 API。
5. 记忆（Memory）：管理多轮对话的历史上下文。

## LCEL 表达式语言

LCEL（LangChain Expression Language）是框架的灵魂，
它允许用竖线（|）把组件串联成管道，例如：prompt | model | parser。

## Agent 智能体

Agent 让模型能够自主决策：思考是否调用工具、调用哪个工具，
然后根据工具结果继续推理，直到完成任务。1.0 版本使用 create_agent 创建智能体。