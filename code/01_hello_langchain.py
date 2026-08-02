# -*- coding: utf-8 -*-
"""
01 · Hello LangChain —— 第一个程序
对应笔记: 01-环境搭建与第一个程序.md

运行前:
  1. 安装依赖: pip install -r requirements.txt
  2. 配置 API Key:
     Windows PowerShell: $env:OPENAI_API_KEY = "sk-xxx"
     macOS/Linux:        export OPENAI_API_KEY="sk-xxx"
     国内模型(DeepSeek等) 见下方"国内模型配置"注释块
"""
from langchain.chat_models import init_chat_model


def main():
    # 1. 初始化模型（1.0 统一入口，自动读取环境变量 OPENAI_API_KEY）
    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    # 2. 调用模型
    resp = model.invoke("用一句话介绍 LangChain")

    # 3. 打印结果
    print("=== 模型回复 ===")
    print(resp.content)

    # 看完整响应对象结构
    print("\n=== 响应类型 ===")
    print(type(resp))
    print("usage:", resp.usage_metadata)


# ===================== 国内模型配置（二选一） =====================
#
# 方案 A: 环境变量方式（代码无需改动）
#   Windows PowerShell:
#     $env:OPENAI_API_KEY = "sk-你的deepseek-key"
#     $env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
#   macOS/Linux:
#     export OPENAI_API_KEY="sk-你的deepseek-key"
#     export OPENAI_BASE_URL="https://api.deepseek.com/v1"
#
# 方案 B: 代码中显式指定
#   model = init_chat_model(
#       "deepseek-chat",
#       model_provider="openai",
#       base_url="https://api.deepseek.com/v1",
#   )
#
# 本地 Ollama（免费离线）:
#   model = init_chat_model(
#       "qwen2.5:7b", model_provider="ollama",
#       base_url="http://localhost:11434/",
#   )
# ==================================================================


if __name__ == "__main__":
    main()
