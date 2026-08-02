# -*- coding: utf-8 -*-
"""
09 · Agent 智能体 —— create_agent、轨迹观察、结构化输出
对应笔记: 09-Agent智能体.md
"""
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from pydantic import BaseModel, Field

model = init_chat_model("gpt-4o-mini", model_provider="openai")


# ---------- 工具 ----------
@tool
def get_weather(city: str) -> str:
    """查询指定城市天气。"""
    return f"{city} 晴，26°C，微风。"


@tool
def search_flights(origin: str, dest: str, date: str) -> str:
    """查询航班信息。参数: 出发地、目的地、日期(YYYY-MM-DD)。"""
    return f"{origin}→{dest} {date}：09:00 航班 ¥680，14:30 航班 ¥560"


@tool
def search_hotels(city: str, date: str) -> str:
    """查询酒店信息。参数: 城市、日期(YYYY-MM-DD)。"""
    return f"{city} {date}：锦江之星 ¥280/晚，全季酒店 ¥420/晚"


def part1_basic_agent():
    """基础 Agent + 轨迹观察"""
    print("=== Part 1 · 基础 Agent ===")
    agent = create_agent(
        model=model,
        tools=[get_weather, search_flights, search_hotels],
        system_prompt="你是旅行规划助手。查询信息时使用工具，信息不足时主动追问。",
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "帮我规划明天（2026-08-03）北京到上海的行程：查一下航班和上海天气"}]
    })

    print("--- 完整消息轨迹（Agent 决策过程）---")
    for m in result["messages"]:
        name = type(m).__name__
        content = str(getattr(m, "content", "") or "")[:100]
        calls = getattr(m, "tool_calls", None)
        print(f"[{name}] {content}")
        if calls:
            for c in calls:
                print(f"        ↳ 调用工具: {c['name']}({c['args']})")

    print(f"\n=== 最终回答 ===")
    print(result["messages"][-1].content)


def part2_structured_agent():
    """结构化输出的 Agent"""
    print("\n=== Part 2 · Agent 结构化输出 ===")

    class TripPlan(BaseModel):
        destination: str = Field(description="目的地")
        flight_info: str = Field(description="航班信息")
        hotel_info: str = Field(description="酒店信息")
        weather_tip: str = Field(description="天气提醒")
        cost_estimate: float = Field(description="预估总花费（元）")

    agent = create_agent(
        model=model,
        tools=[get_weather, search_flights, search_hotels],
        system_prompt="你是旅行规划助手，规划完成后输出结构化行程。",
        response_format=ToolStrategy(TripPlan),
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "规划明天（2026-08-03）深圳到广州的一日游"}]
    })

    plan = result["structured_response"]
    print(f"目的地: {plan.destination}")
    print(f"航班: {plan.flight_info}")
    print(f"酒店: {plan.hotel_info}")
    print(f"天气: {plan.weather_tip}")
    print(f"预估花费: ¥{plan.cost_estimate}")


def part3_stream():
    """流式输出"""
    print("\n=== Part 3 · Agent 流式 ===")
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="你是天气助手。",
    )
    for chunk in agent.stream({
        "messages": [{"role": "user", "content": "成都天气怎么样？"}]
    }):
        # chunk 里包含节点事件，打印增量文本
        text = getattr(chunk, "content", None)
        if text:
            print(text, end="", flush=True)
    print()


def main():
    part1_basic_agent()
    part2_structured_agent()
    part3_stream()


if __name__ == "__main__":
    main()
