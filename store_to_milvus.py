import os
# 必须放在所有 import 之前，解决 HuggingFace 连接超时问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# 1. 基础配置
DATA_PATH = "./data"
MILVUS_URL = "http://localhost:19530"

def main():
    print("🚀 开始处理金融文档...")

    # 2. 加载并切分文档
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误：找不到文件夹 {DATA_PATH}")
        return

    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    
    if not docs:
        print("❌ 警告：data 文件夹里没有找到任何 PDF 内容！")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"✅ 文档切分完成，共 {len(chunks)} 个片段")

    # 3. 初始化 Embedding 模型 (改用更适合中文的 text2vec 模型)
    print("🧠 正在从镜像站下载/加载 Embedding 模型 (约 400MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': 'cpu'}
    )

    # 4. 连接 Milvus 并存入数据
    print("📥 正在存入 Milvus 数据库...")
    try:
        vector_db = Milvus.from_documents(
            documents=chunks,
            embedding=embeddings,
            connection_args={"uri": MILVUS_URL},
            collection_name="Financial_Reports",
            drop_old=True  # 覆盖旧数据
        )
        print("✨ 所有文档已成功向量化并存入 Milvus！")
    except Exception as e:
        print(f"❌ 存入 Milvus 失败: {e}")

if __name__ == "__main__":
    main()