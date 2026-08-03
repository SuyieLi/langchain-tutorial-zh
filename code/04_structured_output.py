# -*- coding: utf-8 -*-
"""
04 · 结构化输出 —— with_structured_output / PydanticOutputParser / 嵌套结构
对应笔记: 04-输出解析与结构化输出.md
"""
import os

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# 自动读取项目根目录的 .env（缺失也不会报错）
load_dotenv()

class MovieReview(BaseModel):
    title: str = Field(description="电影名")
    rating: float = Field(description="评分，1-10 分")
    summary: str = Field(description="一句话剧情简介")
    tags: list[str] = Field(description="标签列表，2-5 个")


def part1_with_structured():
    """最简方式: with_structured_output"""
    print("=== Part 1 · with_structured_output ===")
    model = init_chat_model(
        model_name,
        model_provider=model_provider,
        api_key=api_key,
        base_url=base_url,
    )

    review = model.with_structured_output(
        MovieReview,
        method="function_calling"  # 强制走function call，不使用json_schema
    ).invoke("评价电影《星际穿越》")
    print(f"类型: {type(review).__name__}")
    print(f"电影: {review.title} | 评分: {review.rating}")
    print(f"简介: {review.summary}")
    print(f"标签: {review.tags}")


def part2_parser_manual():
    """手动组装: PromptTemplate + PydanticOutputParser（理解原理）"""
    print("\n=== Part 2 · 手动组装 OutputParser ===")
    model = init_chat_model(
        model_name,
        model_provider=model_provider,
        api_key=api_key,
        base_url=base_url,
    )
    parser = PydanticOutputParser(pydantic_object=MovieReview)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你输出严格的 JSON。{format_instructions}"),
        ("human", "{input}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    # 看模型实际收到的格式指令（理解框架在背后做什么）
    print("--- 自动注入的格式指令 ---")
    print(parser.get_format_instructions()[:300])
    print("--- 调用结果 ---")
    result = (prompt | model | parser).invoke({"input": "评价电影《肖申克的救赎》"})
    print(f"{result.title} | {result.rating} | {result.tags}")


def part3_nested():
    """嵌套结构与枚举约束"""
    print("\n=== Part 3 · 嵌套结构 ===")
    model = init_chat_model(
        model_name,
        model_provider=model_provider,
        api_key=api_key,
        base_url=base_url,
    )

    class Paper(BaseModel):
        title: str
        year: int
        field: Literal["NLP", "CV", "RL", "其他"]

    class PaperList(BaseModel):
        papers: list[Paper]

    result = model.with_structured_output(
        PaperList,
        method="function_calling"  # 强制走function call，不使用json_schema
    ).invoke("列举 3 篇经典 Transformer 相关论文")
    for p in result.papers:
        print(f"  {p.year} [{p.field}] {p.title}")


def part4_json_parser():
    """JsonOutputParser: 只要 JSON dict 不需要校验"""
    print("\n=== Part 4 · JsonOutputParser ===")
    model = init_chat_model(
        model_name,
        model_provider=model_provider,
        api_key=api_key,
        base_url=base_url,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("human", "输出 JSON 格式：{input}"),
    ])
    result = (prompt | model | JsonOutputParser()).invoke(
        {"input": "评价电影《盗梦空间》，字段: title, rating, summary, tags"}
    )
    print(f"类型: {type(result).__name__}")
    print(result)


def main():
    part1_with_structured()
    part2_parser_manual()
    part3_nested()
    part4_json_parser()


if __name__ == "__main__":
    # 从环境变量读取配置，带默认值（默认走DeepSeek）
    api_key = os.getenv("API_KEY", "")
    model_name = os.getenv("MODEL_NAME", "deepseek-chat")
    model_provider = os.getenv("MODEL_PROVIDER", "openai")
    base_url = os.getenv("BASE_URL", "https://api.deepseek.com/v1")

    main()
