# -*- coding: utf-8 -*-
"""
10 · LangGraph 工作流 —— 节点/边/状态、条件边、检查点、人机协同
对应笔记: 10-LangGraph工作流编排.md
"""
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

model = init_chat_model("gpt-4o-mini", model_provider="openai")


# ---------- 状态 ----------
class State(TypedDict):
    messages: list        # 消息列表
    sentiment: str        # 情感标签
    reply_style: str      # 回复风格


# ---------- 节点 ----------
def analyze_sentiment(state: State) -> State:
    """节点1: 情感分析"""
    print(f"\n→ [analyze] 输入消息: {state['messages'][-1].content[:40]}...")
    resp = model.invoke([
        SystemMessage("判断用户情绪，只输出一个词：positive / negative / neutral"),
        HumanMessage(state["messages"][-1].content),
    ])
    sentiment = resp.content.strip()
    print(f"  [analyze] 判断结果: {sentiment}")
    return {"sentiment": sentiment}


def generate_reply(state: State) -> State:
    """节点2: 生成回复"""
    style = {"positive": "热情活泼", "negative": "温柔安慰", "neutral": "客观专业"}
    chosen = style.get(state["sentiment"], "客观专业")
    print(f"→ [reply] 风格: {chosen}")
    resp = model.invoke([
        SystemMessage(f"用{chosen}的语气回复用户，不超过2句话"),
        HumanMessage(state["messages"][-1].content),
    ])
    return {"messages": [resp], "reply_style": chosen}


def comfort_reply(state: State) -> State:
    """负面情绪专用回复"""
    print("→ [comfort] 负面情绪，走安慰分支")
    resp = model.invoke([
        SystemMessage("用户心情不好，请温暖地安慰TA，并给一个实用小建议，不超过2句话"),
        HumanMessage(state["messages"][-1].content),
    ])
    return {"messages": [resp], "reply_style": "comfort"}


# ---------- 条件边 ----------
def route_by_sentiment(state: State) -> str:
    """负面 → comfort 节点，其他 → 普通回复"""
    if state["sentiment"] == "negative":
        return "comfort"
    return "normal"


def build_graph(interrupt_before: list[str] | None = None):
    """构建工作流图"""
    graph = StateGraph(State)
    graph.add_node("analyze", analyze_sentiment)
    graph.add_node("reply", generate_reply)
    graph.add_node("comfort", comfort_reply)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges(
        "analyze",
        route_by_sentiment,
        {"comfort": "comfort", "normal": "reply"},
    )
    graph.add_edge("reply", END)
    graph.add_edge("comfort", END)

    checkpointer = InMemorySaver() if interrupt_before else None
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def part1_basic():
    """基础图 + 条件边"""
    print("=" * 60)
    print("Part 1 · 基础工作流（情感分析 → 条件路由 → 回复）")
    print("=" * 60)
    app = build_graph()

    # 场景A: 正面情绪 → 热情回复
    r1 = app.invoke({"messages": [HumanMessage("今天项目终于上线了！")]})
    print(f"\n[正面] 情感={r1['sentiment']} | 回复: {r1['messages'][-1].content}")

    # 场景B: 负面情绪 → 安慰分支
    r2 = app.invoke({"messages": [HumanMessage("又被客户骂了，好沮丧...")]})
    print(f"\n[负面] 情感={r2['sentiment']} | 回复: {r2['messages'][-1].content}")


def part2_checkpoint():
    """检查点: 持久化 + 断点续跑 + 人机协同"""
    print("\n" + "=" * 60)
    print("Part 2 · Checkpoint（状态持久化）")
    print("=" * 60)
    app = build_graph()
    config = {"configurable": {"thread_id": "thread-1"}}

    # 第一次调用
    r1 = app.invoke({"messages": [HumanMessage("第一步：告诉我今天的任务")]}, config)
    print(f"第1次调用回复: {r1['messages'][-1].content[:40]}...")

    # 第二次调用（同一 thread_id，状态延续）
    r2 = app.invoke({"messages": [HumanMessage("第二步：总结一下刚才的任务")]}, config)
    print(f"第2次调用回复: {r2['messages'][-1].content[:40]}...")
    print("→ 状态通过 checkpointer 保持，跨调用共享消息历史")


def part3_hitl():
    """人机协同: 在关键节点前打断"""
    print("\n" + "=" * 60)
    print("Part 3 · Human-in-the-loop（人工确认）")
    print("=" * 60)
    app = build_graph(interrupt_before=["reply", "comfort"])

    config = {"configurable": {"thread_id": "thread-h1"}}
    # 第一次运行：在 reply 节点前暂停
    result = app.invoke({"messages": [HumanMessage("我想吐槽一下今天遇到的bug")]}, config)
    print("⏸ 已暂停（等待人工确认）...")
    print(f"  当前情感: {result['sentiment']}")
    print(f"  graph_state 检查点: {list(app.get_state(config).next)}")

    # 人工确认后：继续执行
    print("✅ 人工确认，继续执行...")
    final = app.invoke(None, config)
    print(f"最终回复: {final['messages'][-1].content[:60]}")


def main():
    part1_basic()
    part2_checkpoint()
    part3_hitl()


if __name__ == "__main__":
    main()
