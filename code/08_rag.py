# -*- coding: utf-8 -*-
"""
08 · RAG 检索增强生成 —— 加载→切分→向量化→检索→生成 全流程
对应笔记: 08-RAG检索增强生成.md

运行准备:
  1. 在 code/data/ 下放几个 .txt/.md 知识文件（或修改下面的路径）
  2. 需要配置 embedding 的 API Key（OPENAI_API_KEY 即可，text-embedding-3-small）
  3. 首次运行会向量化入库；之后复用 chroma_db 目录
"""
import os

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- 配置 ----------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "openai:text-embedding-3-small"

model = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = init_embeddings(EMBEDDING_MODEL)

# 准备示例知识文件（如不存在则创建）
os.makedirs(DATA_DIR, exist_ok=True)
sample_file = os.path.join(DATA_DIR, "langchain_intro.md")
if not os.path.exists(sample_file):
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write("""
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
""".strip())


def build_vector_store() -> Chroma:
    """加载 → 切分 → 向量化 → 入库（幂等）"""
    # 1. 加载
    from langchain_community.document_loaders import TextLoader
    docs = TextLoader(sample_file, encoding="utf-8").load()
    print(f"[1] 加载文档: {len(docs)} 篇")

    # 2. 切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=30,
        separators=["\n\n", "\n", "。", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[2] 切分为: {len(chunks)} 块 (chunk_size=200)")

    # 3+4. 向量化入库
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    print(f"[3/4] 已向量化入库: {DB_DIR}")
    return vector_store


def load_or_build_store() -> Chroma:
    """已存在库则直接加载，否则重建"""
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("检测到已有向量库，直接加载...")
        return Chroma(persist_directory=DB_DIR, embedding=embeddings)
    return build_vector_store()


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    # 向量库
    vector_store = load_or_build_store()

    # 5. 检索器
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # 6. 生成链
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是知识库问答助手。只能依据以下资料回答，"
                   "资料不足时回答'资料中没有相关信息'。\n\n资料：\n{context}"),
        ("human", "问题：{question}"),
    ])

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # 测试 1: 资料里有的问题
    q1 = "LangChain 的核心组件有哪些？"
    print(f"\n=== 问题1: {q1} ===")
    print(rag_chain.invoke(q1))

    # 测试 2: 资料里没有的问题（验证不胡说）
    q2 = "2026年世界杯冠军是谁？"
    print(f"\n=== 问题2: {q2}（资料外，应拒绝回答）===")
    print(rag_chain.invoke(q2))


if __name__ == "__main__":
    main()
