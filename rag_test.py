import pdfplumber
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. 表格感知解析函数 (Table-Aware Parsing)
# ==========================================
def extract_pdf_content(file_path):
    all_content = []
    print(f"开始解析文档: {file_path}")
    
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # 提取纯文本
            text = page.extract_text() or ""
            
            # 提取表格并转化为 Markdown 风格的字符串
            tables = page.extract_tables()
            table_strings = ""
            for table in tables:
                for row in table:
                    # 过滤掉 None 值并用 | 分隔列
                    processed_row = [str(item).replace('\n', ' ') for item in row if item is not None]
                    table_strings += "| " + " | ".join(processed_row) + " |\n"
            
            # 将页面文本和表格组合
            page_data = f"--- 第 {i+1} 页 ---\n{text}\n\n[表格数据]:\n{table_strings}"
            all_content.append(page_data)
            
            if (i+1) % 10 == 0:
                print(f"已处理 {i+1} 页...")
                
    return "\n\n".join(all_content)

# ==========================================
# 2. 语义切片与向量库构建 (RAG Core)
# ==========================================
def build_vector_db(full_text):
    # 文本切片：金融文档数字多，overlap(重叠度)要大一点，防止上下文断裂
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "!", "?"]
    )
    chunks = text_splitter.split_text(full_text)
    print(f"文本已切分为 {len(chunks)} 个片段")

    # 初始化嵌入模型 (使用轻量级、对中文支持较好的多语言模型)
    print("正在加载 Embedding 模型 (首次运行需下载)...")
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 构建 Chroma 向量库 (持久化在本地目录下)
    print("正在构建向量索引，请稍候...")
    persist_directory = "./chroma_db"
    vector_db = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_db

# ==========================================
# 3. 检索测试
# ==========================================
def run_audit_query(vector_db, query):
    print(f"\n查询问题: {query}")
    # 搜索最相关的 3 个片段
    results = vector_db.similarity_search(query, k=3)
    
    print("\n" + "="*30 + " 检索到的审计线索 " + "="*30)
    for i, doc in enumerate(results):
        print(f"\n[线索 {i+1}]:")
        print(doc.page_content)
        print("-" * 50)

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 请确保该文件就在你的 /root/autodl-tmp/ 目录下
    PDF_PATH = "apple_2025_10k.pdf" 
    
    if not os.path.exists(PDF_PATH):
        print(f"错误：找不到文件 {PDF_PATH}，请确认是否已上传到当前目录。")
    else:
        # 1. 提取
        full_text = extract_pdf_content(PDF_PATH)
        # 2. 向量化
        db = build_vector_db(full_text)
        # 3. 检索
        run_audit_query(db, "Apple 2025 年的 iPhone 收入是多少？")