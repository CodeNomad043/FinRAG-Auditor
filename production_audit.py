import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# --- [加固 1] 绝对路径加载环境变量 ---
env_path = Path('/root/autodl-tmp/.env')
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print("✅ 已通过绝对路径加载 .env 文件")
else:
    load_dotenv()
    print("ℹ️ 尝试加载当前目录下的 .env 文件")

# --- [加固 2] 镜像站锁定 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["no_proxy"] = "localhost,127.0.0.1"

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.llms.openai_like import OpenAILike
# 导入 Reranker 组件
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

class FinancialAuditor:
    def __init__(self):
        print("🔍 正在初始化【双阶检索】增强型审计引擎...")
        
        # 1. 校验 Langfuse
        if not os.getenv("LANGFUSE_PUBLIC_KEY"):
            print("⚠️ 警告: 未能读取到 Langfuse 配置。")

        # 2. 全局参数调优
        Settings.chunk_size = 768 
        Settings.chunk_overlap = 50 

        # 3. 初始化监控
        self.langfuse_callback_handler = LlamaIndexCallbackHandler()
        Settings.callback_manager = CallbackManager([self.langfuse_callback_handler])

        # 4. 加载 Embedding 模型 (基础检索)
        print("📦 正在加载 Embedding 模型 (BGE-Small)...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-zh-v1.5",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 5. [新增] 加载 Reranker 模型 (深度精排)
        print("🎯 正在加载 Reranker 模型 (BGE-Reranker-Base)...")
        self.reranker = FlagEmbeddingReranker(
            model="BAAI/bge-reranker-base",
            top_n=3,  # 精排后只保留最相关的 3 个片段，节省 Token 并提升精度
            # device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 6. 配置 LLM
        Settings.llm = OpenAILike(
            model="qwen-audit",
            api_key="empty",
            api_base="http://localhost:8000/v1",
            context_window=4096,
            is_chat_model=True,
            temperature=0.1
        )
        
        self.index = None
        self.query_engine = None
        self._prepare_index()

    def _prepare_index(self, pdf_path=None):
        """构建支持精排的查询引擎"""
        if pdf_path is None:
            pdf_path = "/root/autodl-tmp/apple_2025_10k.pdf"
            
        if not os.path.exists(pdf_path):
            print(f"❌ 错误：未找到文件 {pdf_path}")
            return

        print(f"📚 正在对文档建立索引: {pdf_path}")
        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        self.index = VectorStoreIndex.from_documents(documents)
        
        # --- [核心修改点] ---
        # similarity_top_k=12: 粗筛阶段多抓取一些候选片段
        # node_postprocessors: 引入精排器对 12 个片段进行重新打分，选出最强的 3 个
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=12, 
            node_postprocessors=[self.reranker],
            response_mode="compact"
        )

    def audit_task(self, query_str):
        """执行审计查询"""
        if not self.query_engine:
            return "错误：查询引擎未就绪。"
        
        try:
            enhanced_query = f"{query_str}。请仔细检索文档中的财务报表部分，如果找到相关数值请直接列出，不要轻易回答未提及。"
            response = self.query_engine.query(enhanced_query)
            self.langfuse_callback_handler.flush()
            return response
        except Exception as e:
            return f"查询过程中出错: {str(e)}"

if __name__ == "__main__":
    auditor = FinancialAuditor()
    print("\n🚀 正在发送深度审计请求 (含 Reranker)...")
    result = auditor.audit_task("请从财务数据表中提取 2024 年的研发费用 (Research and Development) 及其占净销售额的比率。")
    
    print("\n--- 审计结果 ---")
    print(result)
    print("\n✅ 任务完成！")