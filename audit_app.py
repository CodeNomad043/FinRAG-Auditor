import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ==========================================
# 1. 初始化：加载向量库 (延续阶段一的结果)
# ==========================================
print("正在加载向量库和嵌入模型...")
embedding_model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# ==========================================
# 2. 加载 Qwen2.5 大模型 (加载到 3090 显存)
# ==========================================
model_path = "/root/autodl-tmp/qwen2_5_7b_model"
print(f"正在加载大模型到显存: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
# device_map="auto" 会自动利用你的 3090 显卡
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, # 使用 bfloat16 节省显存且保持精度
    device_map="auto"
)

# ==========================================
# 3. 核心审计函数：RAG 闭环
# ==========================================
def smart_audit(query):
    # A. 检索：从向量库找证据
    print(f"\n[用户提问]: {query}")
    docs = vector_db.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    
    # B. 构建 Prompt：告诉 AI 它的身份和规则
    prompt = f"""你是一名专业的资深审计师。请根据提供的【财报原始数据】来回答用户的问题。
要求：
1. 回答必须准确，直接引用数据。
2. 如果数据中没有提到，请直说“根据现有资料无法确认”。
3. 保持职业、简洁。

【财报原始数据】：
{context}

用户问题：{query}
审计结论："""

    # C. 生成：让大脑思考
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.1 # 设置低随机性，保证审计严谨
    )
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # 截取 AI 生成的内容部分
    return response.split("assistant\n")[-1]

# ==========================================
# 4. 运行测试
# ==========================================
if __name__ == "__main__":
    question = "请对比 2024 年和 2025 年 iPad 的业务表现，并分析变化原因。"
    result = smart_audit(question)
    print("\n" + "*"*50)
    print(result)
    print("*"*50)