# -*- coding: utf-8 -*-
"""
05 · LCEL 表达式语言 —— 管道、RunnableParallel、RunnableBranch、RunnableLambda
对应笔记: 05-LCEL表达式语言.md
"""
import asyncio
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnableBranch,
    RunnablePassthrough,
)

# 自动读取项目根目录的 .env（默认走 DeepSeek）
load_dotenv()
api_key = os.getenv("API_KEY", "")
model_name = os.getenv("MODEL_NAME", "deepseek-chat")
model_provider = os.getenv("MODEL_PROVIDER", "openai")
base_url = os.getenv("BASE_URL", "https://api.deepseek.com/v1")

model = init_chat_model(
    model_name,
    model_provider=model_provider,
    api_key=api_key,
    base_url=base_url,
)

translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业翻译，把{target}翻译得地道自然。"),
    ("human", "{text}"),
])
translate_chain = translate_prompt | model | StrOutputParser()


def part1_basic_chain():
    """最简三件套 + 四种调用方式"""
    print("=== Part 1 · 三件套管道 ===")
    chain = translate_prompt | model | StrOutputParser()

    # invoke
    print("invoke:", chain.invoke({"target": "英语", "text": "机器学习改变了世界"}))

    # stream
    print("stream: ", end="")
    for chunk in chain.stream({"target": "日语", "text": "机器学习改变了世界"}):
        print(chunk, end="", flush=True)
    print()

    # batch
    results = chain.batch([
        {"target": "英语", "text": "你好"},
        {"target": "法语", "text": "谢谢"},
    ])
    print("batch:", results)

    # 异步
    async def async_call():
        return await chain.ainvoke({"target": "英语", "text": "异步调用"})
    print("async:", asyncio.run(async_call()))


def part2_runnable_lambda():
    """RunnableLambda: 普通函数接入管道"""
    print("\n=== Part 2 · RunnableLambda ===")

    # 入参是字典，取出input字段
    def add_header(data: dict) -> dict:
        raw_text = data["input"]
        processed_text = f"[翻译任务] {raw_text}"
        # 返回 translate_chain 需要的字典格式！
        return {
            "target": "英语",
            "text": processed_text
        }

    chain = RunnableLambda(add_header) | translate_chain
    print(chain.invoke({"input": "你好，世界"}))


def part3_parallel():
    """RunnableParallel: 并行执行多条链"""
    print("\n=== Part 3 · RunnableParallel ===")
    summary_prompt = ChatPromptTemplate.from_messages([("human", "用一句话总结：{doc}")])
    keyword_prompt = ChatPromptTemplate.from_messages([("human", "提取3个关键词（逗号分隔）：{doc}")])

    chain1 = summary_prompt | model | StrOutputParser()
    chain2 = keyword_prompt | model | StrOutputParser()

    parallel = RunnableParallel(summary=chain1, keywords=chain2)
    result = parallel.invoke({"doc": "LangChain 是一个用于构建大语言模型应用的开发框架，它提供了统一的抽象接口。"})
    print("summary:", result["summary"])
    print("keywords:", result["keywords"])


def part4_branch():
    """RunnableBranch: 条件路由"""
    print("\n=== Part 4 · RunnableBranch ===")
    coding_prompt = ChatPromptTemplate.from_messages([("human", "你是编程专家，回答：{question}")])
    cooking_prompt = ChatPromptTemplate.from_messages([("human", "你是美食专家，回答：{question}")])

    coding_chain = coding_prompt | model | StrOutputParser()
    cooking_chain = cooking_prompt | model | StrOutputParser()

    # 适配器：统一输入格式，适配 translate_chain
    translate_adapter = RunnableLambda(lambda x: {
        "target": "中文",
        "text": x["question"]
    })
    wrapped_translate_chain = translate_adapter | translate_chain

    branch = RunnableBranch(
        (lambda x: "代码" in x["question"] or "python" in x["question"].lower(), coding_chain),
        (lambda x: "菜" in x["question"] or "吃" in x["question"], cooking_chain),
        wrapped_translate_chain,  # 使用包装后的链
    )

    print("代码问题 →", branch.invoke({"question": "python 的装饰器是什么？"})[:60])
    print("美食问题 →", branch.invoke({"question": "红烧肉怎么做？"})[:60])
    # 测试兜底分支，触发翻译链
    print("兜底翻译 →", branch.invoke({"question": "今天天气很好"}))

def part5_passthrough():
    """RunnablePassthrough: 原样透传 + assign"""
    print("\n=== Part 5 · RunnablePassthrough ===")
    chain = (
        {"question": RunnablePassthrough(), "note": RunnableLambda(lambda x: "已记录问题")}
        | RunnableLambda(lambda d: f"问题: {d['question']} | {d['note']}")
    )
    print(chain.invoke("LangChain 是什么？"))


def main():
    part1_basic_chain()
    part2_runnable_lambda()
    part3_parallel()
    part4_branch()
    part5_passthrough()


if __name__ == "__main__":
    main()
