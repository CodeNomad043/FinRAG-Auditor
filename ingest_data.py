import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_community.document_loaders import PDFPlumberLoader # 新增导入

# 1. 配置参数
DATA_PATH = "./data"
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "Financial_Reports"
# 使用一个对中文支持较好的轻量化向量模型
# MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def load_documents():
    print("📂 正在加载文档（使用 PDFPlumber 增强解析）...")
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"❌ 错误：请在 {DATA_PATH} 文件夹中放入 PDF 文件。")
        return None
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PDFPlumberLoader)
    documents = loader.load()
    print(f"✅ 成功加载 {len(documents)} 页文档")
    return documents

def split_text(documents):
    print("✂️ 正在切分文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 已将文档切分为 {len(chunks)} 个文本块")
    return chunks

def save_to_milvus(chunks):
    print("🧠 正在初始化向量模型并写入 Milvus (这可能需要一点时间)...")
    
    # 初始化 Embedding 模型
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    # 存入 Milvus
    vector_db = Milvus.from_documents(
        chunks,
        embeddings,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        collection_name=COLLECTION_NAME,
        drop_old=True,  # 如果表已存在，先删除旧的，确保数据干净
    )
    
    print(f"🚀 写入完成！现在可以在 Attu 或 Chat 脚本中查看数据了。")
    return vector_db

if __name__ == "__main__":
    # 确保目录存在
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📢 已创建 {DATA_PATH} 文件夹，请放入 PDF 后运行。")
    else:
        docs = load_documents()
        if docs:
            chunks = split_text(docs)
            save_to_milvus(chunks)