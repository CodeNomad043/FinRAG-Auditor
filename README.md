# 📑 FinRAG-Auditor: 工业级智能财务审计工作台

## 📌 项目愿景 (Project Vision)
本项目旨在打造一个端到端的生产级 AI 财务审计助手。系统通过对 **Qwen2.5-7B** 进行领域指令微调，结合 **Two-Stage RAG（双阶检索增强生成）** 架构，实现多格式财报的高精度审计。

## 🎬 核心功能演示
- **动态审计**：支持上传 PDF 并实时构建 RAG 索引。
- **双阶检索**：集成 BGE-Reranker 解决财务科目混淆问题。
- **数据溯源**：展示 Source Nodes 片段，确保结果可追溯。

## 🏗️ 核心技术栈
- **LLM**: Qwen2.5-7B (LoRA 微调)
- **RAG**: LlamaIndex + BGE-Reranker
- **监控**: Langfuse
- **交互**: Streamlit
