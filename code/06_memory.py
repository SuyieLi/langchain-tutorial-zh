# -*- coding: utf-8 -*-
"""
06 · 记忆与多轮对话 —— RunnableWithMessageHistory / trim_messages
对应笔记: 06-记忆与多轮对话.md
"""
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.messages import trim_messages

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


def _count_tokens(messages) -> int:
    """简易 token 计数器（按字符数近似估算）。

    DeepSeek 等非官方 OpenAI 模型名无法使用 model.get_num_tokens_from_messages()
    （会抛 NotImplementedError），这里用字符数近似替代，足以演示 trim_messages
    的裁剪逻辑。生产环境若需精确计数，可换用 tiktoken 或模型自带的 tokenizer。
    """
    total = 0
    for m in messages:
        content = m.content if hasattr(m, "content") else m.get("content", "")
        total += len(str(content))
    return total


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
            token_counter=_count_tokens,
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
        token_counter=_count_tokens,
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
