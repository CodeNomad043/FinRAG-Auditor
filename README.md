# 📑 FinRAG-Auditor: 工业级智能财务审计工作台

\<p align="center"\>
\<img src="[https://www.google.com/search?q=https://img.shields.io/badge/Model-Qwen2.5--7B-blue](https://www.google.com/search?q=https://img.shields.io/badge/Model-Qwen2.5--7B-blue)" alt="Model"\>
\<img src="[https://www.google.com/search?q=https://img.shields.io/badge/Framework-LlamaIndex-green](https://www.google.com/search?q=https://img.shields.io/badge/Framework-LlamaIndex-green)" alt="Framework"\>
\<img src="[https://www.google.com/search?q=https://img.shields.io/badge/RAG-Two--Stage](https://www.google.com/search?q=https://img.shields.io/badge/RAG-Two--Stage)\_Retrieval-orange" alt="RAG"\>
\<img src="[https://www.google.com/search?q=https://img.shields.io/badge/Observability-Langfuse-purple](https://www.google.com/search?q=https://img.shields.io/badge/Observability-Langfuse-purple)" alt="Langfuse"\>
\</p\>

## 📌 项目愿景 (Project Vision)

在金融与审计场景中，传统的通用大语言模型（LLM）往往面临三大痛点：**“格式幻觉（无法输出稳定的 JSON）”**、**“信息冗余（废话过多）”以及“长文档检索偏离（相似财务科目混淆）”**。

本项目 **FinRAG-Auditor** 旨在打造一个端到端的生产级 AI 财务审计助手。系统通过对 **Qwen2.5-7B** 进行领域指令微调（SFT），结合 **Two-Stage RAG（双阶检索增强生成）** 架构，并接入 **Langfuse** 全链路监控，最终在 Streamlit 工作台中实现了多格式财报的即传即审、高精度溯源与 100% 稳定的结构化数据提取。

## 🎬 核心功能演示 (Showcase)

### 1\. 动态财报审计与数据溯源 (Streamlit UI)

支持用户动态上传 PDF 财报，系统实时构建索引。通过双阶检索精准提取财务指标，并提供 **Source Trace（溯源片段）** 展示，确保审计结论的绝对可信。

\<p align="center"\>
\<video src="[https://www.google.com/search?q=./Fin-RAG.mp4](https://www.google.com/search?q=./Fin-RAG.mp4)" controls="controls" width="80%" autoplay loop muted\>\</video\>
\</p\>
*(注：由于 GitHub Markdown 限制，如果视频无法直接播放，请点击 [Fin-RAG.mp4](https://www.google.com/search?q=./Fin-RAG.mp4) 下载或在线预览)*

### 2\. 微调前后指令遵循度对比 (A/B Test)

在 Phase 2 中，我们构建了多模对比工作台。左侧为原始 Base 模型（格式发散），右侧为 LoRA 微调后的模型（严格输出 JSON 且无冗余）。

\<p align="center"\>
\<img src="./model\_comparison.png" alt="Model Comparison" width="80%"\>
\</p\>

-----

## 🏗️ 核心系统架构 (System Architecture)

本项目并非简单的 API 调用，而是从底层微调到上层应用的完整闭环：

1.  **底座模型**：Qwen/Qwen2.5-7B-Instruct (兼顾算力与推理性能)
2.  **微调框架**：HuggingFace PEFT + LoRA + bitsandbytes (4-bit 量化)
3.  **RAG 引擎 (LlamaIndex)**：
      - **粗筛 (Coarse Retrieval)**：BAAI/bge-small-zh-v1.5 向量检索 (Top-12)
      - **精排 (Re-ranking)**：FlagEmbedding Reranker Base 交叉编码打分 (Top-3)
4.  **全链路监控**：Langfuse (Token 追踪、检索质量评估、耗时监控)
5.  **前端交互**：Streamlit (动态多文档状态管理与交互式审计)

-----

## 🗺️ 项目演进路线 (Evolution Roadmap)

本项目采用敏捷开发模式，历经四个核心阶段实现工程闭环：

### ✅ Phase 1: 领域指令微调 (Domain-Specific Fine-Tuning)

  * **痛点**：通用模型在执行结构化提取任务时，经常夹杂“好的，我已经为您提取...”等废话，导致下游 JSON 解析器崩溃。
  * **实现**：构建高质量财务指标提取数据集。采用 LoRA 策略冻结底座参数，仅训练旁路矩阵（Rank=8, Alpha=16），在保留基座泛化能力的同时注入“无冗余输出”的领域约束。权重采用 Safetensors 格式持久化，实现 Zero-copy 高速加载。

### ✅ Phase 2: 双引擎对比与 UI 评测 (Comparative Evaluation)

  * **痛点**：需在单张消费级显卡（24GB）上验证微调效果。
  * **实现**：引入 `bitsandbytes` 4-bit 量化加载 Base Model，显存占用压降至 6GB。利用 Peft 的 `with model.disable_adapter():` 隔离上下文，实现单底座、多 LoRA 权重的零延迟切换与直观的 A/B 测试对比。

### ✅ Phase 3: 双阶 RAG 架构引入 (Two-Stage RAG Pipeline)

  * **痛点**：财报篇幅极长（数百页），且存在大量相似词汇（如“研发费用”与“研发信用额度”），传统向量检索极易产生语义漂移。
  * **实现**：弃用单调的 Embedding 检索，引入 **Two-Stage Retrieval** 架构。首先通过 `BGE-Small` 扩大召回范围（Top-12），随后引入基于 Cross-Encoder 架构的 `BGE-Reranker-Base` 进行精准交叉打分，截取最相关的 Top-3 喂入大模型，将财务核心数据的提取准确率提升至极高水平。

### ✅ Phase 4: 生产级部署与全链路监控 (Production & Observability)

  * **痛点**：Demo 难以落地，无法追踪模型的“思考过程”和 Token 成本。
  * **实现**：
      * 使用 Streamlit 重构前端，实现内存级别的动态 PDF 上传与即时 RAG 索引构建功能。
      * 增加 **“Source Nodes（参考源片段）”** UI 模块，解决审计场景核心的“可追溯性”问题。
      * 无缝接入 Langfuse 埋点，实现每一次检索延迟、节点评分、Token 消耗的云端可视化监控。

-----

## 🛠️ 核心工程采坑笔记 (Engineering Insights)

作为一次完整的 AI 落地实践，本项目解决了一系列棘手的工程痛点：

1.  **依赖冲突与环境隔离**：在引入精排机制时，遇到了 `FlagEmbedding` 与最新版 `transformers` (XLMRobertaTokenizer) 的不兼容问题。通过对底层依赖进行版本锁定和精准降级，确保了交叉编码器架构的稳定运行。
2.  **长文本表格截断问题**：财务报表包含大量跨行表格。在 LlamaIndex 处理文档时，由于默认 `chunk_size` 过小导致关键财务数值被物理切断。通过将 `chunk_size` 调优至 768 并增加 `chunk_overlap=50`，有效维持了上下文连续性。
3.  **状态污染与显存管理**：在 Streamlit 支持多文档动态上传后，早期的向量空间容易发生不同财报之间的数据污染。通过设计 `session_state` 绑定临时文件路径，并在每次重构索引前执行内存清理，实现了严密的多文件隔离。

-----

## 🚀 快速开始 (Quick Start)

### 1\. 环境准备

```bash
git clone https://github.com/你的用户名/FinRAG-Auditor.git
cd FinRAG-Auditor
pip install -r requirements.txt
# 核心依赖包含: llama-index, streamlit, peft, langfuse, FlagEmbedding 等
```

### 2\. 环境变量配置

在根目录创建 `.env` 文件，配置 Langfuse 监控平台密钥：

```env
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_HOST="https://cloud.langfuse.com"
```

### 3\. 启动审计工作台

确保已下载对应的底座模型与 LoRA 权重，然后在终端运行：

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

访问 `http://localhost:8501` 即可体验。通过侧边栏上传任意 PDF 财报文件，点击构建索引后，即可发起带有数据溯源的深度审计查询。
