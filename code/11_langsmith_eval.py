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

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 自动读取项目根目录的 .env（默认走 DeepSeek）
load_dotenv()
api_key = os.getenv("API_KEY", "")
model_name = os.getenv("MODEL_NAME", "deepseek-chat")
model_provider = os.getenv("MODEL_PROVIDER", "openai")
base_url = os.getenv("BASE_URL", "https://api.deepseek.com/v1")

# 被测系统: 一个简单的问答链
model = init_chat_model(
    model_name,
    model_provider=model_provider,
    api_key=api_key,
    base_url=base_url,
)
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


def _local_grade(prediction: str, reference: str) -> float:
    """本地规则评估器：基于字符重叠度 + 关键词包含打分（0~1）。

    不调用任何 LLM，零成本。用于无 LangSmith 时演示评估流程。
    生产环境建议用 LLM 评判（如 llm_grader）或注册 LangSmith。
    """
    if not reference:
        return 0.0
    pred_chars = set(prediction)
    ref_chars = set(reference)
    overlap = len(pred_chars & ref_chars) / len(ref_chars) if ref_chars else 0.0
    # 参考答案被预测包含 → 视为基本正确
    if reference in prediction:
        overlap = max(overlap, 0.85)
    return round(min(overlap, 1.0), 2)


def run_local_eval():
    """无 LangSmith Key 时的本地评估演示。

    用 DeepSeek（或 .env 配置的模型）作为被测系统，本地规则评估器打分，
    终端打印评估表格。不依赖任何云平台，只需被测模型可用。
    """
    examples = [
        {"question": "LangChain 是什么？", "answer": "用于构建大语言模型应用的开发框架。"},
        {"question": "1+1 等于几？", "answer": "2"},
        {"question": "Python 中 print 的作用？", "answer": "向控制台输出内容。"},
    ]

    print("（本地评估模式：被测模型 = DeepSeek，评估器 = 本地规则，不调用云平台）\n")
    header = f"{'#':<3} {'问题':<22} {'模型回答':<34} {'参考答案':<22} {'得分'}"
    print(header)
    print("-" * len(header))

    scores = []
    for i, ex in enumerate(examples, 1):
        try:
            pred = chain.invoke({"question": ex["question"]})
        except Exception as e:
            pred = f"[调用失败: {type(e).__name__}]"
        score = _local_grade(str(pred), ex["answer"])
        scores.append(score)
        q = ex["question"][:20]
        p = str(pred).replace("\n", " ")[:32]
        r = ex["answer"][:20]
        print(f"{i:<3} {q:<22} {p:<34} {r:<22} {score}")

    print("-" * len(header))
    avg = round(sum(scores) / len(scores), 2) if scores else 0
    print(f"平均得分: {avg} / 1.00\n")
    print("提示: 这是本地规则评估演示。要使用完整的 LangSmith 评估平台")
    print("     （可视化、数据集管理、LLM 评判、实验对比），请注册")
    print("     https://smith.langchain.com 并在 .env 设置 LANGSMITH_API_KEY。")


def main():
    print("=" * 60)
    print("LangSmith 评估演示")
    print("=" * 60)

    if not os.environ.get("LANGSMITH_API_KEY"):
        print("\n⚠️  未检测到 LANGSMITH_API_KEY")
        print("   说明: DeepSeek 可作为「被测模型」，但 LangSmith 是独立的评估平台，")
        print("   需在 https://smith.langchain.com 单独注册获取 Key（与 DeepSeek Key 无关）。")
        print("   → 已切换到「本地评估模式」演示评估流程（无需 LangSmith）。\n")
        run_local_eval()
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
