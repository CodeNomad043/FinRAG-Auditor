import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. 配置路径
MODEL_PATH = "/root/autodl-tmp/qwen2_5_7b_model"
LORA_PATH = "/root/autodl-tmp/qwen_lora_checkpoint"

# 2. 定义 Embedding 模型（使用本地化路径或自动下载）
print("正在加载 Embedding 模型 (BGE-Small)...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5", # 如果报错请改为本地绝对路径
    device="cuda"
)

# 3. 初始化底座模型与 LoRA 插件
print("正在初始化微调版审计引擎...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    quantization_config=bnb_config, 
    device_map="auto"
)

# 关键点：显式加载微调权重
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# 4. 封装符合 LlamaIndex 规范的 LLM 接口
Settings.llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    context_window=32768,
    max_new_tokens=256, # 缩短生成长度，减少复读机会
    system_prompt="你是一名资深财务审计专家。你的任务是提取数据并只输出 JSON 格式。严禁输出任何解释、推导或提示性文字。如果找到数据，直接以 { 开头，以 } 结尾。",
    
    generate_kwargs={
        "temperature": 0.1, 
        "do_sample": True,
        "top_p": 0.8,
        # 移除 stop_strings 避免底层报错
    },
    device_map="auto",
)


# 5. 构建财务知识库索引
print("正在构建财务知识库索引...")
# 使用更健壮的 SimpleDirectoryReader
reader = SimpleDirectoryReader(
    input_dir="./data", 
    recursive=True, 
    required_exts=[".pdf", ".txt"]
)
documents = reader.load_data()
print(f"✅ 成功加载了 {len(documents)} 页文档内容！")

# 将文档切片并向量化
index = VectorStoreIndex.from_documents(documents)

# 6. 创建查询引擎并执行测试
# similarity_top_k=3 表示搜索最相关的 3 个片段
query_engine = index.as_query_engine(similarity_top_k=3)

print("\n🚀 正在执行 RAG 检索提取任务...")
query_str = "请提取 Apple 2025 年穿戴设备的净销售额并以 JSON 格式输出，包含年份、类别、数值和单位。"
response = query_engine.query(query_str)

print("\n" + "="*30)
print("--- RAG 提取结果 ---")
print(response)
print("="*30)