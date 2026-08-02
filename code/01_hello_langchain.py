# -*- coding: utf-8 -*-
"""
01 · Hello LangChain —— 第一个程序
对应笔记: 01-环境搭建与第一个程序.md

运行前:
  1. 安装依赖: pip install -r requirements.txt
  2. 配置密钥: 复制 .env.example 为 .env，填入你的 API Key
     - 默认使用 DeepSeek（国内模型，便宜好用），无需科学上网
     - 想用 OpenAI / 通义 / Ollama，改 .env 里的 MODEL_NAME / MODEL_PROVIDER / BASE_URL 即可

.env 写法示例:
  API_KEY=sk-你的deepseek密钥
  MODEL_NAME=deepseek-chat
  MODEL_PROVIDER=openai            # DeepSeek 兼容 OpenAI 协议，所以 provider 仍是 openai
  BASE_URL=https://api.deepseek.com/v1
"""
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 自动读取项目根目录的 .env（缺失也不会报错）
load_dotenv()


def main():
    # 1. 从环境变量读取配置，带默认值（默认走 DeepSeek）
    api_key = os.getenv("API_KEY", "")
    model_name = os.getenv("MODEL_NAME", "deepseek-chat")
    model_provider = os.getenv("MODEL_PROVIDER", "openai")
    base_url = os.getenv("BASE_URL", "https://api.deepseek.com/v1")

    # 2. 友好校验：远程模型（非 Ollama）必须有 Key，才给出明确指引
    #    Ollama 是本地模型，完全不需要 Key，放行。
    if model_provider != "ollama" and not api_key:
        raise SystemExit(
            "\n❌ 未检测到 API Key。\n"
            "请按以下步骤配置:\n"
            "  1) 复制 .env.example 为 .env\n"
            "  2) 在 .env 中填入 API_KEY（以及模型相关配置）\n"
            "  3) 重新运行 python code/01_hello_langchain.py\n\n"
            "如果想用「免 Key 的本地模型」，把 .env 改成:\n"
            "  MODEL_PROVIDER=ollama\n"
            "  MODEL_NAME=qwen2.5:7b\n"
            "  BASE_URL=http://localhost:11434\n"
            "  API_KEY=\n"
            "（需先安装 Ollama 并拉取模型，见下方说明）\n"
        )

    # 3. 初始化模型（1.0 统一入口）
    #    - 远程模型（DeepSeek/OpenAI/通义）：provider 多为 openai，配 base_url + api_key
    #    - 本地 Ollama：provider 用 ollama，无需 api_key，base_url 指向本地服务
    if model_provider == "ollama":
        init_kwargs = {}
        if base_url:
            init_kwargs["base_url"] = base_url
        model = init_chat_model(
            model_name,
            model_provider="ollama",
            **init_kwargs,
        )
    else:
        model = init_chat_model(
            model_name,
            model_provider=model_provider,
            api_key=api_key,
            base_url=base_url,
        )

    # 4. 调用模型
    resp = model.invoke("用一句话介绍 LangChain")

    # 5. 打印结果
    print("=== 模型回复 ===")
    print(resp.content)

    # 看完整响应对象结构
    print("\n=== 响应类型 ===")
    print(type(resp))
    print("usage:", resp.usage_metadata)


if __name__ == "__main__":
    main()
