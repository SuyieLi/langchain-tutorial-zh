# 11 · 生产实践与 LangSmith

> 从 Demo 到生产：可观测性、评估、部署、成本与安全——AI 应用的工程化之道。
> 上一篇：[[10-LangGraph工作流编排]] ｜ 下一篇：[[12-常见问题与避坑指南]]

---

## 1. 生产环境的"三大件"

一个生产级 LLM 应用 = 应用本身 + **可观测性** + **评估体系**：

```
┌─────────────────────────────────────────┐
│  你的应用（LCEL 链 / Agent / LangGraph） │
├─────────────────────────────────────────┤
│  LangSmith：链路追踪 + 数据集 + 离线评估 │
│  LangSmith：在线监控 + 反馈收集          │
│  安全：PII 过滤、成本限额、速率限制      │
└─────────────────────────────────────────┘
```

> [!note] 残酷真相
> 模型输出的质量**不是恒定的**——Prompt 小改动、模型小更新都会漂移。
> 没有评估体系，"感觉还行"迟早翻车。**评估不是可选项，是生产底线。**

---

## 2. LangSmith：一站式可观测平台

### 2.1 是什么

LangChain 官方出品的**开发平台**，核心四件事：

| 能力 | 说明 |
|------|------|
| **Tracing（追踪）** | 自动记录每次调用的完整链路（每步输入/输出/耗时/token） |
| **Datasets（数据集）** | 沉淀测试样例，管理评估数据 |
| **Evaluation（评估）** | 离线跑评估，量化 Prompt 改动的效果 |
| **Monitoring（监控）** | 生产在线监控、延迟/成本/错误率 |

### 2.2 开启追踪（5 秒接入）

```bash
pip install langsmith
```

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "lsv2_你的key"
os.environ["LANGSMITH_PROJECT"] = "my-langchain-app"   # 项目分组

# 之后所有 invoke 自动上报，无需改业务代码！
```

> 💡 自托管版（Self-Hosted）适合数据不出域的场景。

### 2.3 追踪能帮你看到什么

一次 Agent 调用的 Trace 会显示：

```
1. LLM 调用 #1       (system prompt + 用户问题)     12 tokens
2. 工具调用          get_weather(city="北京")        0 tokens
3. LLM 调用 #2       (加上工具结果)                   45 tokens
4. 最终输出          "北京今天晴，25°C"
   ↓ 每一步的耗时、token 成本、异常一目了然
```

> 排错铁律：**生产问题先看 Trace，再改代码**。Trace 能立刻暴露"是哪一步出错、多花了多少 token"。

---

## 3. 评估（Evaluation）：把"调 Prompt"变成科学

### 3.1 基本流程

```
① 准备数据集（输入 + 期望输出/参考答案）
② 运行你的链/Agent，批量产生输出
③ 用"评判器"打分：LLM 评判 / 精确匹配 / 自定义指标
④ 对比不同版本（Prompt v1 vs v2）→ 数字说话
```

### 3.2 用 LangSmith 跑一次评估（示意）

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# ① 创建数据集（也可在网页上上传）
dataset = client.create_dataset(dataset_name="qa_eval")
client.create_examples(dataset_id=dataset.id, inputs=[{"question": "..."}],
                       outputs=[{"answer": "..."}])

# ② 定义评估器（这里用 LLM 打分）
def correctness(run, example) -> dict:
    from langsmith.evaluators import llm_evaluator
    return llm_evaluator(prediction=run.outputs["output"],
                         reference=example.outputs["answer"],
                         rubric="判断回答是否正确、完整")

# ③ 批量评估
results = evaluate(
    lambda x: my_chain.invoke(x["question"]),   # 被测系统
    data="qa_eval",                             # 数据集
    evaluators=[correctness],
)
print(results)   # 平均分、每例得分
```

> 完整可运行示例见 `code/11_langsmith_eval.py`。

### 3.3 评估维度

| 维度 | 评判方式 | 说明 |
|------|----------|------|
| 正确性 | LLM 打分 / 精确匹配 | 答案对不对 |
| 事实性（幻觉） | LLM 对照资料核查 | 有没有胡编 |
| 相关性 | LLM / 检索指标 | 检索结果相不相关 |
| 延迟 / 成本 | 平台统计 | 性能指标 |
| 人类反馈 | 线上 thumbs up/down | 真实用户满意度 |

> [!tip] 评估的最佳实践
> 1. **先建数据集，再改 Prompt**——否则改了都不知道好没好转；
> 2. 数据集 30–50 条起步，覆盖边界情况（刁钻问题、模糊问题）；
> 3. 每次改动 Prompt 都重跑一遍评估，形成"回归测试"习惯；
> 4. RAG 类应用把**检索质量**和**生成质量**分开评估。

---

## 4. 部署模式

| 方案 | 场景 | 说明 |
|------|------|------|
| **LangGraph Server**（官方） | 生产首选 | 把图/Agent 暴露为 REST API，自带持久化、人机协同接口、监控 |
| FastAPI 自封装 | 快速上线 | `chain.invoke()` 包一层 HTTP |
| Serverless（云函数） | 事件驱动 | 注意冷启动延迟 |
| 容器化 + K8s | 大规模 | 标准云原生 |

```python
# LangGraph Server 部署（server 用 pyproject.toml 声明图入口）
# 命令：
#   langgraph dev      # 本地开发（含可视化 Studio）
#   langgraph deploy   # 部署到 LangSmith 云 / 自托管
```

```python
# 或用 FastAPI 手动封装（示意）
from fastapi import FastAPI
app = FastAPI()

@app.post("/chat")
def chat(body: dict):
    return {"answer": chain.invoke(body["question"])}
```

---

## 5. 成本控制三板斧

```
① 模型选择：重活大模型（gpt-4o / claude-sonnet），轻活小模型（gpt-4o-mini）
② 缓存：相同问题命中缓存（LangSmith 提供 LLM 缓存、自定义缓存）
③ 输入压缩：trim_messages 裁剪历史、RAG 只检索 TopK
```

| 手段 | 效果 |
|------|------|
| 缓存常见问题 | 可省 30%+ 成本 |
| 小模型处理简单分支 | 成本降一个数量级 |
| 控制 max_tokens | 防"废话连篇"烧钱 |
| 会话级预算限额 | 防止单用户异常消耗 |

---

## 6. 安全清单（上线前逐项过）

- [ ] **Prompt 注入防护**：用户输入和系统指令隔离；工具调用前校验参数
- [ ] **PII 脱敏**：手机号、身份证等敏感信息过滤后再进 LLM
- [ ] **输出过滤**：内容安全审核（不当内容拦截）
- [ ] **工具权限最小化**：Agent 只能碰必要的系统（[[09-Agent智能体]]）
- [ ] **人工审批**：高风险动作（转账/发信/发布）HITL（[[10-LangGraph工作流编排]]）
- [ ] **限流与配额**：防滥用、防刷
- [ ] **API Key 安全**：服务端持有，永不下发前端
- [ ] **数据合规**：用户数据留存策略、知情同意

---

## 7. 生产上线检查单（Checklist）

```markdown
□ 可观测性已接入（LangSmith Tracing）
□ 评估数据集 ≥ 30 条，含边界用例
□ 核心指标已跑基准（正确率、延迟、成本）
□ Prompt 有版本管理（改动走评估）
□ 缓存已启用
□ 预算上限、速率限制已配置
□ 敏感操作有人工审批
□ 错误处理：模型超时/限流的重试与降级
□ 日志与监控告警已配置
□ API Key 只存服务端
```

---

## ✅ 动手练习

1. 接入 LangSmith Tracing，跑一次 Agent，在网页上观察 Trace 链路；
2. 创建 10 条问答数据集，跑一次评估，记录得分；
3. 修改你的 Prompt 一处细节，重跑评估对比分数——体验"数据驱动调优"；
4. 计算你日常 demo 的 token 成本，用缓存/裁剪做一次优化；
5. 对照安全清单，给你的应用做一次自查。

---

🏷️ `#LangChain` `#LangSmith` `#生产实践` `#LLMOps`

[[README|← 返回学习路径总览]] ｜ [[12-常见问题与避坑指南|下一篇：避坑指南 →]]
