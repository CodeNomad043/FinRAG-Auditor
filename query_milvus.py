import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# 1. 配置
MILVUS_URL = "http://127.0.0.1:19530"

def main():
    # 2. 初始化同样的 Embedding 模型 (必须和存入时一致！)
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': 'cpu'}
    )

    # 3. 连接已有的 Collection
    vector_db = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": MILVUS_URL},
        collection_name="Financial_Reports",
    )

    # 4. 进行语义搜索
    query = "这份报告中提到的财务风险有哪些？" # 你也可以换成 Apple 相关的其他问题
    print(f"\n🔍 正在检索问题: {query}")
    
    # 搜索前 3 个最相关的片段
    docs = vector_db.similarity_search(query, k=3)

    print("\n✅ 检索结果如下：")
    print("-" * 50)
    for i, doc in enumerate(docs):
        print(f"[结果 {i+1}] 来源: {doc.metadata.get('source', '未知')}")
        print(f"内容摘要: {doc.page_content[:200]}...") # 只打印前200字
        print("-" * 50)

if __name__ == "__main__":
    main()