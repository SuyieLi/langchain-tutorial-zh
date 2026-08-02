# -*- coding: utf-8 -*-
"""
06 · 记忆与多轮对话 —— RunnableWithMessageHistory / trim_messages
对应笔记: 06-记忆与多轮对话.md
"""
from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.messages import trim_messages

model = init_chat_model("gpt-4o-mini", model_provider="openai")

# 会话存储: session_id -> 历史对象
store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """按 session_id 取/建历史（可替换为数据库实现做持久化）"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def build_chain(with_trimmer: bool = False):
    """构建带记忆的链"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是健谈的助手，回答不超过 3 句话。"),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    chain = prompt | model | StrOutputParser()

    if with_trimmer:
        trimmer = trim_messages(
            max_tokens=200,
            strategy="last",
            token_counter=model,
            include_system=True,
            allow_partial=False,
        )
        chain = trimmer | chain

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def demo_memory():
    """演示: 带记忆的多轮对话"""
    print("=== 演示: 多轮对话记忆（session 隔离）===")
    chain = build_chain()

    # 用户A 会话
    r1 = chain.invoke({"input": "我叫小明，我喜欢旅游"},
                      config={"configurable": {"session_id": "user-A"}})
    print(f"[user-A] 第1轮: {r1}")

    # 用户B 会话（记忆隔离: 不应知道小明）
    r2 = chain.invoke({"input": "我叫什么名字？"},
                      config={"configurable": {"session_id": "user-B"}})
    print(f"[user-B] 第1轮: {r2}  ← 记忆隔离，不知道小明")

    # 用户A 继续（应该记得）
    r3 = chain.invoke({"input": "我叫什么名字？我爱好什么？"},
                      config={"configurable": {"session_id": "user-A"}})
    print(f"[user-A] 第2轮: {r3}  ← 记得上下文")

    # 查看历史存储
    print(f"\n历史存储的 session 数量: {len(store)}")
    print(f"user-A 历史消息: {len(store['user-A'].messages)} 条")


def demo_trim():
    """演示: trim_messages 裁剪"""
    print("\n=== 演示: trim_messages 裁剪 ===")
    trimmer = trim_messages(
        max_tokens=200,
        strategy="last",
        token_counter=model,
        include_system=True,
    )
    fake_messages = [
        {"role": "user", "content": f"这是第{i}条很长的历史消息，内容" + "。" * 100}
        for i in range(10)
    ]
    trimmed = trimmer.invoke(fake_messages)
    print(f"原始消息数: {len(fake_messages)}")
    print(f"裁剪后消息数: {len(trimmed)}")


def main():
    demo_memory()
    demo_trim()


if __name__ == "__main__":
    main()
