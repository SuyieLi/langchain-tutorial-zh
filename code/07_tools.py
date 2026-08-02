# -*- coding: utf-8 -*-
"""
07 · 工具与函数调用 —— @tool / bind_tools 手动循环 / create_agent
对应笔记: 07-工具与函数调用.md
"""
import datetime

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool

model = init_chat_model("gpt-4o-mini", model_provider="openai")


# ---------- 工具定义 ----------
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气情况。"""
    return f"{city} 今天晴，25°C，适合出行。"


@tool
def get_time() -> str:
    """获取当前时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，如 '1+2*3'。"""
    return str(eval(expression, {"__builtins__": {}}, {}))


def part1_bind_tools():
    """手动工具循环（理解原理）"""
    print("=== Part 1 · 手动 bind_tools 循环 ===")
    model_with_tools = model.bind_tools([get_weather])
    tool_map = {"get_weather": get_weather}

    messages = [HumanMessage("北京天气怎么样？")]
    resp = model_with_tools.invoke(messages)
    print("模型返回的工具调用意图:")
    print(f"  tool_calls = {resp.tool_calls}")

    # 执行循环
    while resp.tool_calls:
        messages.append(resp)
        for call in resp.tool_calls:
            result = tool_map[call["name"]].invoke(call["args"])
            print(f"执行工具 {call['name']}({call['args']}) → {result}")
            # 关键: tool_call_id 必须匹配
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
        resp = model_with_tools.invoke(messages)

    print("最终回答:", resp.content)


def part2_create_agent():
    """框架方案: create_agent"""
    print("\n=== Part 2 · create_agent（自动处理循环）===")
    agent = create_agent(
        model=model,
        tools=[get_weather, get_time, calculate],
        system_prompt="你是一个全能助手，需要外部信息或计算时请使用工具。",
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "现在是几点？帮我算一下 (15+7)*3，顺便查一下深圳天气？"}]
    })

    # 查看完整轨迹（调试利器）
    print("--- 消息轨迹 ---")
    for m in result["messages"]:
        role = type(m).__name__
        content = getattr(m, "content", "") or ""
        calls = getattr(m, "tool_calls", None)
        print(f"[{role}] {str(content)[:80]}")
        if calls:
            print(f"        ↳ 工具调用: {[c['name'] for c in calls]}")

    print("\n最终回答:", result["messages"][-1].content)


def main():
    part1_bind_tools()
    part2_create_agent()


if __name__ == "__main__":
    main()
