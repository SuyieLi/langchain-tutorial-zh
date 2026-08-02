# -*- coding: utf-8 -*-
"""
03 · Prompt 模板 —— ChatPromptTemplate / MessagesPlaceholder / Few-shot
对应笔记: 03-Prompt工程与模板.md
"""
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)


def part1_basic():
    """基础模板"""
    print("=== Part 1 · 基础模板 ===")
    prompt = PromptTemplate.from_template(
        "你是{domain}领域的专家。请回答下面的问题：\n{question}"
    )
    formatted = prompt.format(domain="金融科技", question="什么是量化交易？")
    print(formatted)

    # 缺少变量会报错（提前发现错误）
    try:
        prompt.format(domain="金融科技")
    except KeyError as e:
        print(f"\n[校验生效] 缺少变量时报错: {e}")


def part2_chat_prompt():
    """聊天消息模板"""
    print("\n=== Part 2 · ChatPromptTemplate ===")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位{domain}专家，回答要专业、简洁。"),
        ("human", "请解释一下：{question}"),
    ])
    messages = prompt.format_messages(domain="金融科技", question="什么是量化交易？")
    for m in messages:
        print(f"[{m.type}] {m.content}")


def part3_placeholder():
    """消息占位符（多轮对话关键）"""
    print("\n=== Part 3 · MessagesPlaceholder ===")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个乐于助人的助手。"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    messages = prompt.format_messages(
        history=[
            ("human", "我叫小明"),
            ("ai", "你好小明！"),
        ],
        question="我叫什么名字？",
    )
    for m in messages:
        print(f"[{m.type}] {m.content}")


def part4_few_shot():
    """Few-shot 示例"""
    print("\n=== Part 4 · Few-shot ===")
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
        few_shot_prompt,
        ("human", "{input}"),
    ])
    messages = final_prompt.format_messages(store="电子产品商城", input="你们能开发票吗？")
    for m in messages:
        print(f"[{m.type}] {m.content}")


def main():
    part1_basic()
    part2_chat_prompt()
    part3_placeholder()
    part4_few_shot()


if __name__ == "__main__":
    main()
