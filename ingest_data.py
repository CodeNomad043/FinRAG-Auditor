import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

# 1. 指定数据路径
DATA_PATH = "./data"

def load_documents():
    # 这里我们支持加载 PDF 和 TXT
    print("📂 正在加载文档...")
    # 注意：如果 data 文件夹里只有 PDF，请使用 PyPDFLoader
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"✅ 成功加载 {len(documents)} 页文档")
    return documents

def split_text(documents):
    # 2. 切分文档：金融文档逻辑严密，建议 chunk_size 设为 500-800
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100, # 段落重叠，防止语义在切分点断掉
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ 已将文档切分为 {len(chunks)} 个文本块")
    return chunks

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📢 请在 {DATA_PATH} 文件夹中放入一些 PDF 文件后再运行。")
    else:
        docs = load_documents()
        if docs:
            chunks = split_text(docs)
            # 下一节我们将这些 chunks 变成向量存入 Milvus