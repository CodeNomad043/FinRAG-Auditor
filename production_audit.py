import os
from dotenv import load_dotenv

# --- [安全增强] 加载 .env 文件中的环境变量 ---
# 这会自动寻找当前目录下的 .env 文件并将内容载入 os.environ
load_dotenv()

# --- [重点] 1. 强制使用镜像加速 (保持不变) ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- [安全增强] 2. 从环境变量读取 Key，不再硬编码 ---
# 如果 .env 文件配置正确，os.getenv 会拿到对应的值
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.llms.openai_like import OpenAILike

# 3. 初始化 Langfuse 处理器
# 此时它会读取上面 os.environ 中被 load_dotenv 加载进来的 Key
langfuse_callback_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([langfuse_callback_handler])

# 4. 配置 Embedding 模型
print("正在通过镜像站加载 Embedding 模型...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 5. 配置 LLM (使用 OpenAILike 绕过官方模型校验)
Settings.llm = OpenAILike(
    model="qwen-audit",
    api_key="empty",
    api_base="http://localhost:8000/v1",
    context_window=4096,
    is_chat_model=True,
    temperature=0.1
)

# 6. 加载数据并构建索引
print("正在构建索引...")
if not os.path.exists("/root/autodl-tmp/apple_2025_10k.pdf"):
    print("❌ 错误：未找到 PDF 文件，请检查路径。")
else:
    documents = SimpleDirectoryReader(
        input_files=["/root/autodl-tmp/apple_2025_10k.pdf"]
    ).load_data()

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)

    # 7. 执行审计
    print("\n🚀 正在向 vLLM 发送请求...")
    response = query_engine.query("请提取 Apple 2025 年穿戴设备的净销售额 JSON。")

    print("\n--- 审计结果 ---")
    print(response)

    # 8. 确保数据上传到 Langfuse
    langfuse_callback_handler.flush()
    print("\n✅ 全部任务完成！请检查 Langfuse 后台。")