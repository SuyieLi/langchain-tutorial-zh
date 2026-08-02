# -*- coding: utf-8 -*-
"""
02 · 模型调用基础 —— 消息、四种调用方式、结构化输出、嵌入
对应笔记: 02-模型调用基础.md
"""
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field


def section(title: str):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def part1_messages():
    """消息结构: system / human 消息"""
    section("Part 1 · 消息结构")
    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    messages = [
        SystemMessage("你是一个资深的Python导师，回答要简洁、带代码示例。"),
        HumanMessage("什么是装饰器？"),
    ]
    resp = model.invoke(messages)
    print(resp.content[:200])
    print("\n消息类型:", type(resp).__name__)


def part2_invoke_stream_batch():
    """四种调用方式"""
    section("Part 2 · invoke / stream / batch")
    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    # stream: 流式输出（打字机效果）
    print("--- stream 流式 ---")
    for chunk in model.stream("用一句话介绍自己"):
        print(chunk.content, end="", flush=True)
    print()

    # batch: 批量
    print("\n--- batch 批量 ---")
    resps = model.batch(["1+1等于几", "2+2等于几"])
    for r in resps:
        print(r.content)


def part3_structured():
    """结构化输出"""
    section("Part 3 · 结构化输出")
    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    class MovieReview(BaseModel):
        title: str = Field(description="电影名")
        rating: float = Field(description="评分，1-10 分")
        summary: str = Field(description="一句话剧情简介")
        tags: list[str] = Field(description="标签列表，2-5 个")

    review = model.with_structured_output(MovieReview).invoke("评价电影《星际穿越》")
    print(f"电影: {review.title}")
    print(f"评分: {review.rating}")
    print(f"简介: {review.summary}")
    print(f"标签: {review.tags}")
    print("转 dict:", review.model_dump())


def part4_embeddings():
    """嵌入模型"""
    section("Part 4 · 嵌入模型")
    try:
        from langchain.embeddings import init_embeddings
        embeddings = init_embeddings("openai:text-embedding-3-small")
        vec = embeddings.embed_query("LangChain 是什么？")
        print(f"向量维度: {len(vec)}")
        print(f"前 5 个值: {vec[:5]}")
    except Exception as e:
        print(f"（嵌入模型需额外配置，跳过）{e}")


def main():
    part1_messages()
    part2_invoke_stream_batch()
    part3_structured()
    part4_embeddings()


if __name__ == "__main__":
    main()
