import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_path = "/root/autodl-tmp/qwen2_5_7b_model"
lora_path = "/root/autodl-tmp/qwen_lora_checkpoint"

print("🚀 正在加载原始大脑 (Qwen2.5)...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    trust_remote_code=True
)

print("贴上审计专家贴纸 (LoRA)...")
# 加载微调后的权重并合并
model = PeftModel.from_pretrained(base_model, lora_path)
model = model.eval()

print("\n--- 审计专家测试开始 ---")
prompt = "作为审计助手，请从文本中提取财务指标并以JSON格式输出。文本：Apple 2025年穿戴设备净销售额为 37,015 百万美元，相比去年的 36,990 百万美元基本持平。"
# 使用 Qwen 标准的 ChatML 格式
full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        repetition_penalty=1.1,
        temperature=0.1 # 降低随机性，让输出更稳定
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
# 只截取助手回答的部分
print(result.split("assistant\n")[-1])