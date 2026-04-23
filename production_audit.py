import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# --- [加固 1] 绝对路径加载环境变量 (最优先) ---
env_path = Path('/root/autodl-tmp/.env')
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print("✅ 已通过绝对路径加载 .env 文件")
else:
    # 备选：尝试加载当前目录下的 .env
    load_dotenv()
    print("ℹ️ 尝试加载当前目录下的 .env 文件")

# --- [加固 2] 镜像站锁定 (必须在 llama_index 导入前) ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 解决本地代理可能导致的 localhost 连接问题
os.environ["no_proxy"] = "localhost,127.0.0.1"

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.llms.openai_like import OpenAILike

class FinancialAuditor:
    def __init__(self):
        print("🔍 正在初始化增强型审计引擎...")
        
        # 1. 校验 Langfuse 配置
        if not os.getenv("LANGFUSE_PUBLIC_KEY"):
            print("⚠️ 警告: 未能读取到 Langfuse 配置，请检查 .env 文件内容。")

        # 2. 全局 RAG 参数调优 (核心修改点)
        # 增加块大小，确保财务报表中的表格行不会被切断
        Settings.chunk_size = 768 
        # 增加重叠度，保证上下文连续性
        Settings.chunk_overlap = 50 

        # 3. 初始化 Langfuse 处理器
        self.langfuse_callback_handler = LlamaIndexCallbackHandler()
        Settings.callback_manager = CallbackManager([self.langfuse_callback_handler])

        # 4. 配置 Embedding 模型
        print("📦 正在加载 Embedding 模型 (BGE-Small)...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-zh-v1.5",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 5. 配置 LLM (对接本地 vLLM)
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

    def _prepare_index(self):
        """加载数据并构建索引"""
        pdf_path = "/root/autodl-tmp/apple_2025_10k.pdf"
        if not os.path.exists(pdf_path):
            print(f"❌ 错误：未找到文件 {pdf_path}")
            return

        print(f"📚 正在进行深度索引 (ChunkSize: 1024)...")
        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        
        # 构建索引时会自动应用 Settings 中的 chunk_size
        self.index = VectorStoreIndex.from_documents(documents)
        
        # [优化点]：将检索深度从 5 提升到 8，并使用 compact 模式合并文本
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=4,
            response_mode="compact" 
        )

    def audit_task(self, query_str):
        """执行审计查询的任务接口"""
        if not self.query_engine:
            return "错误：查询引擎未就绪。"
        
        try:
            # 在提问前加入强约束，防止模型轻易说“找不到”
            enhanced_query = f"{query_str}。请仔细检索文档中的财务报表部分，如果找到相关数值请直接列出，不要轻易回答未提及。"
            
            response = self.query_engine.query(enhanced_query)
            # 强制刷新数据到 Langfuse 云端
            self.langfuse_callback_handler.flush()
            return response
        except Exception as e:
            return f"查询过程中出错: {str(e)}"

# --- 脚本直接运行逻辑 ---
if __name__ == "__main__":
    auditor = FinancialAuditor()
    print("\n🚀 正在发送深度审计请求...")
    # 尝试一个更具体的测试问题
    result = auditor.audit_task("请从财务数据表中提取 2025 年和 2024 年的研发费用 (Research and Development) 及其增长率。")
    
    print("\n--- 审计结果 ---")
    print(result)
    print("\n✅ 任务完成！请检查 Langfuse 后台。")