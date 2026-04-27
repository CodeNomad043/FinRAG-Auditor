# 📑 FinRAG-Auditor: 工业级智能财务审计工作台

<p align="center">
  <img src="https://img.shields.io/badge/Model-Qwen2.5--7B-blue" alt="Model">
  <img src="https://img.shields.io/badge/Framework-LlamaIndex-green" alt="Framework">
  <img src="https://img.shields.io/badge/RAG-Two--Stage_Retrieval-orange" alt="RAG">
  <img src="https://img.shields.io/badge/Observability-Langfuse-purple" alt="Langfuse">
</p>

## 📌 项目愿景 (Project Vision)

在金融与审计场景中，传统的通用大语言模型（LLM）往往面临三大痛点：**“格式幻觉（无法输出稳定的 JSON）”**、**“信息冗余（废话过多）”以及“长文档检索偏离（相似财务科目混淆）”**。

本项目 **FinRAG-Auditor** 旨在打造一个端到端的生产级 AI 财务审计助手。系统通过对 **Qwen2.5-7B** 进行领域指令微调（SFT），结合 **Two-Stage RAG（双阶检索增强生成）** 架构，并接入 **Langfuse** 全链路监控，最终在 Streamlit 工作台中实现了多格式财报的即传即审、高精度溯源与 100% 稳定的结构化数据提取。

---

## 🎬 核心功能演示 (Showcase)

### 1. 动态财报审计与数据溯源 (Streamlit UI)

支持用户动态上传 PDF 财报，系统实时构建索引。通过双阶检索精准提取财务指标，并提供 **Source Trace（溯源片段）** 展示，确保审计结论的绝对可信。

<p align="center">
  <img src="Fin-RAG.gif" width="80%" alt="Project Demo">
</p>

*(注：如果视频无法直接播放，请点击仓库根目录的 Fin-RAG.mp4 文件进行预览)*

### 2. 微调前后指令遵循度对比 (A/B Test)

我们在单卡环境下构建了多模对比工作台。左侧为原始 Base 模型（格式发散），右侧为 LoRA 微调后的模型（严格输出 JSON 且无冗余）。

<p align="center">
  <img src="model_comparison.png" alt="Model Comparison" width="80%">
</p>

---

## 🏗️ 核心系统架构 (System Architecture)

本项目并非简单的 API 调用，而是从底层微调到上层应用的完整闭环：

1. **底座模型**：Qwen/Qwen2.5-7B-Instruct (兼顾算力与推理性能)
2. **微调框架**：HuggingFace PEFT + LoRA + bitsandbytes (4-bit 量化)
3. **RAG 引擎 (LlamaIndex)**：
      * **粗筛 (Coarse Retrieval)**：BAAI/bge-small-zh-v1.5 向量检索 (Top-12)
      * **精排 (Re-ranking)**：FlagEmbedding Reranker Base 交叉编码打分 (Top-3)
4. **全链路监控**：Langfuse (Token 追踪、检索质量评估、耗时监控)
5. **前端交互**：Streamlit (动态多文档状态管理与交互式审计)

---

## 🗺️ 项目演进路线 (Evolution Roadmap)

本项目采用敏捷开发模式，历经六个核心阶段实现工程闭环：

### ✅ Phase 1: 环境搭建与算力极限优化 (Environment & Memory Optimization)
* **痛点**：大模型本地部署受限于显存容量（如 RTX 3090/4090 24GB），直接加载 7B 模型极易导致 OOM (Out of Memory)，且开发环境容易出现依赖冲突。
* **实现**：配置隔离环境，利用 `bitsandbytes` 4-bit 双重量化技术，成功将基座模型显存占用压降至 6GB 左右，为后续“单卡双引擎”共存打下坚实基础。

### ✅ Phase 2: 领域指令微调 (Domain-Specific Fine-Tuning)
* **痛点**：通用模型在执行结构化提取任务时，经常夹杂“好的，我已经为您提取...”等废话，导致下游 JSON 解析器崩溃。
* **实现**：构建高质量财务指标提取数据集。采用 LoRA 策略仅训练旁路矩阵（Rank=8, Alpha=16），在保留基座泛化能力的同时注入“无冗余输出”的领域约束。权重采用 Safetensors 格式持久化，实现 Zero-copy 高速加载。

### ✅ Phase 3: 双引擎对比与交互式评测 (Comparative Evaluation UI)
* **痛点**：缺乏直观的手段来验证微调权重是否真的比原始模型好用，反复跑脚本测试效率低下。
* **实现**：利用 Peft 的 `with model.disable_adapter():` 上下文管理器，实现单底座、多 LoRA 权重的零延迟切换。搭建 Gradio A/B Test 工作台，直观展示微调带来的指令遵循度（Instruction Following）跃升。

<img width="1665" height="615" alt="pay" src="https://github.com/user-attachments/assets/cf08828e-8efd-455b-b3ab-29cce23771f5" />


### ✅ Phase 4: 双阶 RAG 架构引入 (Two-Stage RAG Pipeline)
* **痛点**：财报篇幅极长（数百页），且存在大量相似词汇（如“研发费用”与“研发信用额度”），传统单一向量检索极易产生语义漂移，导致审计错误。
* **实现**：弃用单调的 Embedding 检索，引入 **Two-Stage Retrieval** 架构。先通过 `BGE-Small` 扩大召回范围（Top-12），随后引入 `BGE-Reranker-Base` 进行精准交叉编码打分，截取最相关的 Top-3 喂入大模型，将准确率提升至极高水平。

### ✅ Phase 5: 全链路追踪与可观测性 (Full-Trace Observability)
* **痛点**：RAG 系统如同“黑盒”，开发者难以量化评估检索命中的准确率、Token 消耗成本以及大模型各节点的推理延迟。
* **实现**：无缝接入 **Langfuse** 监控平台。对检索器（Retriever）、重排序器（Reranker）以及大模型生成（Generation）每一环进行埋点，实现对耗时、打分和 Token 成本的全景可视化跟踪。

  <img width="1358" height="781" alt="cost" src="https://github.com/user-attachments/assets/b906c8bb-94e4-4e3c-a5c0-1ac7afded285" />
  
<img width="1667" height="557" alt="time" src="https://github.com/user-attachments/assets/2b34676a-2c8d-4d07-8292-0a42d3998e2f" />


### ✅ Phase 6: 工业级 Streamlit 应用闭环交付 (Industrial Application Deployment)
* **痛点**：此前的 Demo 无法处理多文档动态上传，且缺乏审计场景中最核心的“证据展示”环节。
* **实现**：使用 Streamlit 重构前端，支持多文档动态状态管理（Session State）与内存级空间隔离。创新性引入 **“Source Nodes（溯源片段）”** 模块，真正解决金融场景下“结果必须可信、可溯源”的业务诉求，完成产品闭环。

---

## 🛠️ 核心工程采坑笔记 (Engineering Insights)

作为一次完整的 AI 落地实践，本项目解决了一系列棘手的工程痛点：

1. **依赖冲突与环境隔离**：在引入精排机制时，遇到了 `FlagEmbedding` 与最新版 `transformers` (XLMRobertaTokenizer) 的不兼容问题。通过对底层依赖进行版本锁定和精准降级，确保了交叉编码器架构的稳定运行。
2. **长文本表格截断问题**：财务报表包含大量跨行表格。在 LlamaIndex 处理文档时，由于默认 `chunk_size` 过小导致关键财务数值被物理切断。通过将 `chunk_size` 调优至 768 并增加 `chunk_overlap=50`，有效维持了上下文连续性。
3. **状态污染与显存管理**：在 Streamlit 支持多文档动态上传后，早期的向量空间容易发生不同财报之间的数据污染。通过绑定临时文件路径并在重构索引前执行内存清理，实现了严密的多文件隔离。

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备

```bash
git clone [https://github.com/CodeNomad043/FinRAG-Auditor.git](https://github.com/CodeNomad043/FinRAG-Auditor.git)
cd FinRAG-Auditor
pip install -r requirements.txt
# 核心依赖包含: llama-index, streamlit, peft, langfuse, FlagEmbedding 等
```

### 2. 环境变量配置

在根目录创建 `.env` 文件，配置 Langfuse 监控平台密钥：

```env
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_HOST="[https://(us.)cloud.langfuse.com](https://(us.)cloud.langfuse.com)"
```

### 3. 启动审计工作台

先开启加速，确保快速响应：

```bash
source /etc/network_turbo
```

在终端启动vLLm：

```bash
VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN /root/miniconda3/bin/python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/qwen_audit_merged \
    --served-model-name qwen-audit \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --trust-remote-code \
    --enforce-eager \
    --port 8000 \
    --guided-decoding-backend lm-format-enforcer \
    --disable-frontend-multiprocessing
```

确保已下载对应的底座模型与 LoRA 权重，然后新建另一个终端运并行：

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

访问 `http://localhost:8501` 即可体验。通过侧边栏上传任意 PDF 财报文件，点击构建索引后，即可发起带有数据溯源的深度审计查询。
