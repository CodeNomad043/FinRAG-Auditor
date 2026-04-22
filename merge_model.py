import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "/root/autodl-tmp/qwen2_5_7b_model"
lora_path = "/root/autodl-tmp/qwen_lora_checkpoint"
save_path = "/root/autodl-tmp/qwen_audit_merged"

print("正在合并模型，请稍候...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, 
    device_map="cpu" # 合并建议在 CPU 或 显存充足的情况下进行
)

model = PeftModel.from_pretrained(base_model, lora_path)
merged_model = model.merge_and_unload()

print(f"正在保存合并后的完整模型至 {save_path}...")
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("✅ 合并完成！")