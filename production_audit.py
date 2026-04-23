import os
import torch
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.llms.openai_like import OpenAILike

# --- 1. 环境初始化 ---
# 加载 .env 文件。load_dotenv 会自动把文件里的 Key 注入到系统的 os.environ 中
load_dotenv()

# 设置镜像站（必须在加载模型前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class FinancialAuditor:
    def __init__(self):
        print("🔍 正在初始化审计引擎...")
        
        # --- [修复重点] 安全地校验 Langfuse 配置 ---
        # 只要执行了 load_dotenv()，系统环境里就已经有这些变量了，不需要再手动赋值
        # 我们这里做一个简单的安全校验，防止 NoneType 错误
        required_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
        for var in required_vars:
            if not os.getenv(var):
                print(f"⚠️ 警告: 环境变量 {var} 未找到，请检查 .env 文件。")

        # 初始化 Langfuse 处理器
        self.langfuse_callback_handler = LlamaIndexCallbackHandler()
        Settings.callback_manager = CallbackManager([self.langfuse_callback_handler])

        # 配置 Embedding 模型
        print("📦 正在加载 Embedding 模型 (BGE-Small)...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-zh-v1.5",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 配置 LLM (对接本地 vLLM)
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

        print(f"📚 正在索引文档: {pdf_path}")
        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        self.index = VectorStoreIndex.from_documents(documents)
        self.query_engine = self.index.as_query_engine(similarity_top_k=3)

    def audit_task(self, query_str):
        """执行审计查询的任务接口"""
        if not self.query_engine:
            return "错误：查询引擎未就绪。"
        
        try:
            response = self.query_engine.query(query_str)
            # 强制刷新数据到 Langfuse 云端
            self.langfuse_callback_handler.flush()
            return response
        except Exception as e:
            return f"查询过程中出错: {str(e)}"

# --- 2. 脚本直接运行逻辑 ---
if __name__ == "__main__":
    # 如果直接运行这个文件，执行默认审计任务
    auditor = FinancialAuditor()
    print("\n🚀 正在发送审计请求...")
    result = auditor.audit_task("请提取 Apple 2025 年穿戴设备的净销售额 JSON。")
    
    print("\n--- 审计结果 ---")
    print(result)
    print("\n✅ 任务完成！请检查 Langfuse 后台。")