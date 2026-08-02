# -*- coding: utf-8 -*-
"""
11 · LangSmith 评估 —— 创建数据集 + 批量评估（示意）
对应笔记: 11-生产实践与LangSmith.md

运行前提:
  1. 注册 LangSmith 获取 API Key: https://smith.langchain.com
  2. 设置环境变量:
     LANGSMITH_API_KEY=lsv2_xxx
     LANGSMITH_TRACING=true
  3. 本脚本中的 evaluate 调用需要 langsmith 依赖（已包含在 requirements.txt）
"""
import os

from langsmith import Client
from langsmith.evaluation import evaluate

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 被测系统: 一个简单的问答链
model = init_chat_model("gpt-4o-mini", model_provider="openai")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是严谨的助手，回答不超过两句话。"),
    ("human", "{question}"),
])
chain = prompt | model | StrOutputParser()


def create_dataset_if_missing():
    """创建评估数据集（已存在则复用）"""
    client = Client()

    existing = [d for d in client.list_datasets() if d.name == "qa_demo"]
    if existing:
        print("数据集已存在，复用:", existing[0].name)
        return existing[0]

    dataset = client.create_dataset(
        dataset_name="qa_demo",
        description="LangChain 学习评估示例数据集",
    )
    examples = [
        {"inputs": {"question": "LangChain 是什么？"},
         "outputs": {"answer": "用于构建大语言模型应用的开发框架。"}},
        {"inputs": {"question": "1+1 等于几？"},
         "outputs": {"answer": "2"}},
        {"inputs": {"question": "Python 中 print 的作用？"},
         "outputs": {"answer": "向控制台输出内容。"}},
    ]
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
    )
    print("已创建数据集，共", len(examples), "条示例")
    return dataset


def llm_grader(run, example) -> dict:
    """评估器: 用模型评判回答是否与参考答案一致"""
    from langsmith.evaluators import llm_evaluator

    result = llm_evaluator(
        prediction=run.outputs["output"],
        reference=example.outputs["answer"],
        rubric="""判断预测答案与参考答案语义是否一致。
        一致给 1.0，部分一致给 0.5，完全不一致给 0.0。
        只输出分数。""",
    )
    return {"key": "correctness", "score": result}


def main():
    print("=" * 60)
    print("LangSmith 评估演示")
    print("=" * 60)

    if not os.environ.get("LANGSMITH_API_KEY"):
        print("\n⚠️  未检测到 LANGSMITH_API_KEY")
        print("   先注册 LangSmith 并设置环境变量，或在 app.smith.langchain.com 上操作。")
        print("   本脚本其余部分仍可参考注释理解评估流程。")
        return

    dataset = create_dataset_if_missing()

    # 跑评估: 用数据集逐条调用被测链，并用评估器打分
    results = evaluate(
        lambda x: chain.invoke(x["question"]),   # 被测系统（接受 dict，返回 dict）
        data=dataset.name,
        evaluators=[llm_grader],
        experiment_prefix="langchain-learning-demo",
    )
    print("\n评估完成:")
    print(results)


if __name__ == "__main__":
    main()
