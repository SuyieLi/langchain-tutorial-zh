# 08 · RAG 检索增强生成

> 给模型装上"外接大脑"：从文档加载到向量检索的完整 RAG 流水线。
> 上一篇：[[07-工具与函数调用]] ｜ 下一篇：[[09-Agent智能体]]

---

## 1. 什么是 RAG

**RAG（Retrieval-Augmented Generation，检索增强生成）**：回答前先检索相关资料，再把资料连同问题一起给模型。

```
传统问答：问题 → 模型（靠训练知识）→ 回答（可能过时/编造）
RAG 问答：问题 → ①检索相关资料 → ②问题+资料 → 模型 → 有据可依的回答
```

### 为什么需要 RAG（对比三种方案）

| 方案 | 优点 | 缺点 |
|------|------|------|
| 直接问模型 | 零成本 | 知识过时、幻觉（胡编）、无私有知识 |
| 微调（Fine-tuning） | 知识"内化" | 贵、慢、需高质量数据集、**不适合高频更新知识** |
| **RAG** ✅ | 便宜、快、知识实时更新、可溯源 | 检索质量决定上限 |

> [!tip] 选型口诀
> **知识会变 / 量大 / 要溯源 → RAG；能力风格改变（如模仿某作家文风）→ 微调。**
> 90% 的"知识问答"需求，RAG 是首选。

### RAG 全流程（五步）

```
文档加载(Load) → 文本切分(Split) → 向量化(Embed) → 向量入库与检索(Retrieve) → 生成(Generate)
```

---

## 2. 第一步：文档加载（Load）

LangChain 有几百种文档加载器（`langchain_community.document_loaders`）：

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader

# 纯文本
docs = TextLoader("notes.txt").load()

# PDF
docs = PyPDFLoader("report.pdf").load()

# 网页（注意：新版的网页加载器在 langchain 中）：
# from langchain.document_loaders import WebBaseLoader
# docs = WebBaseLoader("https://docs.langchain.com").load()

# 每个文档对象：page_content（正文）+ metadata（来源、页码等）
print(docs[0].page_content[:200])
print(docs[0].metadata)
```

| 常见加载器 | 用途 |
|-----------|------|
| `TextLoader` | .txt / .md |
| `PyPDFLoader` | PDF |
| `WebBaseLoader` | 网页抓取 |
| `CSVLoader` / `JSONLoader` | 表格/JSON |
| `DirectoryLoader` | 批量加载目录 |
| 各种数据库/平台加载器 | Notion、Confluence、S3… |

> ⚠️ 用 `load()` 默认一次全部读进内存，大文档用 `lazy_load()` 迭代式读取。

---

## 3. 第二步：文本切分（Split）

模型上下文有限，文档太长必须切成"块（chunk）"。**切分质量直接决定检索质量**。

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块目标字符数
    chunk_overlap=50,      # 相邻块重叠字符数（保住上下文连贯）
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],  # 按优先级切
)

chunks = splitter.split_documents(docs)
print(len(chunks), chunks[0].page_content[:100])
```

### 切分策略选型

| 策略 | 场景 |
|------|------|
| `RecursiveCharacterTextSplitter`（默认） | 通用文本，按段落/句子递归切 |
| `CharacterTextSplitter` | 简单字符切分 |
| `MarkdownHeaderTextSplitter` | Markdown 按标题切（保留层级） |
| 按 Token 切 | 精确控制 token 数 |
| **语义切分**（SemanticChunker，进阶） | 按语义边界切，更准但更慢 |

> [!tip] 参数经验值
> - `chunk_size` 500–1500 字符比较常见；
> - `chunk_overlap` 设为 chunk 的 10–20%；
> - 生产环境**用你真实的测试集调参**，别凭感觉（配合 [[11-生产实践与LangSmith]] 评估）。

---

## 4. 第三步：向量化 + 入库（Embed + Store）

### 4.1 向量化

```python
from langchain.embeddings import init_embeddings

embeddings = init_embeddings("openai:text-embedding-3-small")
# 中文场景也常用：text2vec / bge-m3（本地免费）等
```

### 4.2 向量库（Vector Store）

```python
from langchain_chroma import Chroma        # 轻量，本地文件存储，学习首选
# 其他：FAISS（内存）、Qdrant、Pinecone、Milvus（生产）、PGVector（Postgres）...

# 建库入库：一次把文档向量化并存入
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",       # 本地持久化目录
)

# 之后重启用 load 恢复（不用重新向量化）
# vector_store = Chroma(persist_directory="./chroma_db", embedding=embeddings)
```

| 向量库 | 特点 | 场景 |
|--------|------|------|
| Chroma | 零配置、本地文件 | 学习、原型 ✅ |
| FAISS | 内存级、极快 | 中等规模单机 |
| Qdrant / Milvus | 分布式、高并发 | 生产级 |
| PGVector | 复用 Postgres | 已有 PG 的团队 |

---

## 5. 第四步：检索（Retrieve）

```python
# 直接检索（相似度 TopK）
retriever = vector_store.as_retriever(
    search_type="similarity",      # 相似度检索
    search_kwargs={"k": 4},        # 返回 4 块
)

docs = retriever.invoke("LangChain 支持哪些向量库？")
for d in docs:
    print("—", d.page_content[:80])
```

### 检索类型对比

| 类型 | 原理 | 适合 |
|------|------|------|
| `similarity` | 向量余弦相似度 TopK | 通用默认 |
| `mmr` | 相似 + 多样性去重 | 结果内容高度重复时 |
| `similarity_score_threshold` | 相似度超阈值才返回 | 防止不相关内容混入 |

> [!note] 一句话理解向量检索
> 文本 → 向量（一串数字），向量近 = 语义近。
> "金融科技"和"FinTech"虽文字不同，向量却很近——这就是它比关键词搜索强的地方。

---

## 6. 第五步：生成（Generate）—— 组装 RAG 链

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是知识库问答助手。只能依据以下资料回答，资料不足时回答'资料中没有相关信息'。"
               "\n\n资料：\n{context}"),
    ("human", "问题：{question}"),
])

# 经典 RAG 链（LCEL 并行：检索 + 透传问题）
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("LangChain 支持哪些向量库？")
print(answer)
```

**注意 `{"context": retriever, "question": RunnablePassthrough()}` 的妙处**：
- `retriever` 并行执行检索 → 得到 docs 列表；
- `RunnablePassthrough()` 原样透传用户问题；
- 两者自动合并为 `{"context": [...docs], "question": "..."}` 喂给 prompt。

---

## 7. 让答案"可溯源"：返回引用

生产环境必须让用户看到答案来自哪里：

```python
from langchain_core.runnables import RunnableParallel

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnableParallel(
        context=retriever | RunnableLambda(format_docs),   # 格式化资料
        question=RunnablePassthrough(),
    )
    | prompt
    | model
    | StrOutputParser()
)

# 想同时返回资料来源？把检索和生成做成两个输出
chain_with_sources = RunnableParallel(
    answer=rag_chain,                      # 生成答案
    sources=lambda x: [d.metadata for d in retriever.invoke(x["question"])],  # 来源
)
```

---

## 8. RAG 进阶话题（了解目录）

| 主题 | 一句话说明 | 学习路径 |
|------|-----------|----------|
| 混合检索 | 向量 + 关键词（BM25）互补 | 检索质量优化 |
| 重排序（Rerank） | 检索后让模型/模型再排序 | 检索质量优化 |
| 查询改写 | 把问题改写成更适合检索的形式 | 检索质量优化 |
| 多路召回 | 多种检索方式结果合并 | 检索质量优化 |
| **Agentic RAG** | Agent 自主决定检索什么、检索几次 | 结合 [[09-Agent智能体]] |
| 图 RAG（GraphRAG） | 知识图谱增强检索 | 高级专题 |

> [!tip] RAG 优化的三板斧（按性价比排序）
> 1. **优化切分**（chunk 大小/重叠/语义切分）——最容易见效；
> 2. **优化 Prompt**（严格约束"只能依据资料"）；
> 3. **重排序**（检索 20 条 → 重排取 5 条）。

---

## ✅ 动手练习

1. 准备 1–2 个你自己的文本文件（如笔记/论文），完整跑一遍五步 RAG 流水线；
2. 问一个"资料里明明有"的问题，验证回答正确；再问"资料里没有"的，验证模型不说谎；
3. 修改 chunk_size（200 vs 1000），对比检索效果差异，记录观察；
4. 把检索方式换成 `mmr`，对比结果多样性；
5. （进阶）实现"返回引用来源"的版本，打印答案对应的文档出处。

---

🏷️ `#LangChain` `#RAG` `#向量检索` `#Embedding`

[[README|← 返回学习路径总览]] ｜ [[09-Agent智能体|下一篇：Agent →]]
