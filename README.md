# 企业知识库问答系统（enterprise_qa_system）

基于 **RAG（检索增强生成）** 的企业知识库问答系统，采用「FAQ 精确匹配 + RAG 语义检索」两级架构，对企业内部文档（PDF / Word / PPT / Markdown / 图片等）提供精准、可溯源的智能问答。

## 核心亮点

- **两级检索架构**：第一级 FAQ 精确匹配（jieba 分词 + BM25 + softmax 门控），命中直接返回、不走大模型；未命中再走第二级 RAG 语义检索。
- **混合检索**：BGE-M3 同时产出稠密 + 稀疏双向量，结合 `WeightedRanker` 做混合召回，兼顾语义与关键词。
- **父子块切分**：父块（大）保上下文、子块（小）做定位，检索用子块、生成用父块，解决"切大了定位粗、切小了上下文断"的矛盾。
- **精排重排**：BGE-Reranker 对召回结果二次精排，显著提升答案相关度。
- **查询分类 + 策略选择**：BERT 做问题分类，LLM 按问题类型动态选检索策略。
- **多格式文档解析**：8 种 Loader，PDF / Word / PPT / 图片走 OCR（RapidOCR）识别图片文字。
- **流式问答**：FastAPI + 前端单页，流式输出、支持多轮对话记忆。

## 技术栈

| 层 | 技术 |
|---|---|
| 接口 | FastAPI、WebSocket |
| 数据 | MySQL、Redis、Milvus（向量库） |
| 检索 | jieba、BM25、BGE-M3（dense + sparse）、BGE-Reranker |
| 分类 | BERT（bert-base-chinese） |
| 解析 | LangChain Loader、RapidOCR |
| 大模型 | 通义千问 qwen-plus（DashScope OpenAI 兼容接口） |
| 评测 | RAGAS（五指标） |

## 架构

```
[离线] CSV → MySQL(FAQ)  │  多格式文档 → 8 种 Loader → 父子块 → BGE-M3 → Milvus

[在线] 请求 → FastAPI(问候正则)
        → ① FAQ 精确匹配: jieba 分词 → BM25 → softmax + 阈值 → Redis/MySQL
             │ 命中 → 直接返回（不走大模型）
             │ 未命中 ↓
        → ② RAG 语义检索: BERT 分类 → LLM 选策略 → 混合检索（稠密+稀疏）
             → 父子块去重 → BGE-Reranker 精排 → qwen-plus 生成
        → 对话历史存 MySQL（5 轮）
[离线] RAGAS 五指标评测
```

## 目录结构

```
enterprise_qa_system/
├── base/                  # 基础设施：配置（config.py）+ 日志（logger.py）
├── mysql_qa/              # 第一级 FAQ（第 2、3 步）
│   ├── db/                #   MySQL 客户端
│   ├── cache/             #   Redis 客户端
│   ├── retrieval/         #   BM25 检索
│   ├── utils/             #   jieba 分词
│   └── main.py            #   FAQ 独立入口
├── rag_qa/                # 第二级 RAG
│   ├── core/              #   RAG 核心 + BERT 查询分类
│   ├── enterprise_document_loaders/  # 8 种文档加载器（PDF/Word/PPT/图片走 OCR）
│   ├── enterprise_text_spliter/  #   父子块切分
│   ├── rag_assessment/    #   RAGAS 评测
│   ├── models/            #   模型权重（BGE-M3 / BGE-Reranker / BERT，需自行下载）
│   └── data/              #   知识库数据
├── static/                # 前端单页（聊天界面）
├── app.py                 # FastAPI 入口
├── main.py                # 集成系统入口（FAQ + RAG）
├── config.ini             # 配置文件（示例值，真实密钥放 .env）
├── docker-compose.yml     # 容器编排（示例）
├── .env.example           # 环境变量模板（复制为 .env 后填真实值）
└── test/                  # 测试脚本
```

## 快速开始

### 1. 环境依赖

- Python 3.10+
- MySQL 8.0+
- Redis
- Milvus 2.4+

### 2. 启动依赖服务

可本机安装，或用 Docker 启动（按需）：

```bash
# MySQL（端口 3306）、Redis（端口 16379）、Milvus（端口 19530）
docker compose up -d
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入真实的数据库密码、DashScope API Key 等
```

> `.env` 已被 `.gitignore` 排除，**绝不会被提交**。DashScope API Key 在[阿里云百炼控制台](https://bailian.console.aliyun.com/)申请。

### 4. 下载模型权重

`rag_qa/models/` 下的模型需自行下载（体积大，未纳入仓库）：

| 模型 | 用途 | 下载地址 |
|---|---|---|
| BGE-M3 | 稠密+稀疏向量 | `BAAI/bge-m3`（HuggingFace） |
| BGE-Reranker-Large | 精排重排 | `BAAI/bge-reranker-large` |
| bert-base-chinese | 查询分类 | `google-bert/bert-base-chinese` |

### 5. 准备 FAQ 数据（导入 MySQL）

```bash
python mysql_qa/db/mysql_client.py   # 建表 + 导入 data/企业知识问答.csv
```

> 重复执行会自动清空旧数据再导入，不会产生重复记录。

### 6. 构建向量库（导入文档到 Milvus）

```bash
# 首次运行需先在 Milvus 中创建数据库 enterprise（可用 Attu 界面 http://localhost:30000 创建，或执行）：
python -c "from pymilvus import connections, utility; connections.connect(host='localhost', port='19530'); utility.create_database('enterprise')"

# 遍历 rag_qa/data/ 下各知识分类，完成「解析 → 父子块切分 → BGE-M3 向量化 → 写入 Milvus」
python rag_qa/core/vector_store.py
```

### 7. 启动服务

```bash
python app.py
# 浏览器访问 http://localhost:18080
```

## 效果

- 混合检索 + BGE-Reranker 精排，复杂问题召回率显著提升。
- 基于 RAGAS 五指标（忠实度、答案相关性、上下文精度、上下文召回等）离线评测，形成「检索优化 → 生成优化」闭环。

## License

个人学习 / 面试演示项目。
