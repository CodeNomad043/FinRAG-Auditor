import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 配置路径
MODEL_PATH = "/root/autodl-tmp/qwen2_5_7b_model"
LORA_PATH = "/root/autodl-tmp/qwen_lora_checkpoint"

# 1. 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print("🚀 正在初始化双引擎（量化加载中...）")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 加载原始模型
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    quantization_config=bnb_config, 
    device_map="auto", 
    trust_remote_code=True
)

# 加载 LoRA 插件
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print("✅ 模型就绪！")

def generate_response(text):
    prompt = f"作为审计助手，请从文本中提取财务指标并以JSON格式输出。文本：{text}"
    full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            temperature=0.1,
            repetition_penalty=1.1
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n")[-1]

def compare_audit(input_text):
    if not input_text.strip():
        return "请输入内容", "请输入内容"

    # 1. 在 disable_adapter 上下文中运行 -> 得到原始模型结果
    with model.disable_adapter():
        print("正在生成原始模型结果...")
        base_res = generate_response(input_text)
    
    # 2. 正常运行 -> 得到微调后的专家模型结果
    print("正在生成审计专家结果...")
    tuned_res = generate_response(input_text)
    
    return base_res, tuned_res

# 构建专业界面
# 修正点：将 theme 放回 Blocks 构造函数中
with gr.Blocks(title="FinRAG-Auditor 审计专家系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📑 FinRAG-Auditor 智能审计工作台 (双模对比)")
    gr.Markdown("> **左侧：** 原始 Qwen2.5 7B (通用回复) | **右侧：** 审计专家 LoRA (精准 JSON 提取)")
    
    with gr.Row():
        input_box = gr.Textbox(label="输入待审计财务文本", placeholder="输入例如：Apple 2025年销售额为...", lines=4)
    
    btn = gr.Button("🔍 执行跨模型对比分析", variant="primary")
    
    with gr.Row():
        with gr.Column():
            out_base = gr.Code(label="原始模型结果", language="markdown")
        with gr.Column():
            out_tuned = gr.Code(label="审计专家结果", language="json")
            
    btn.click(fn=compare_audit, inputs=input_box, outputs=[out_base, out_tuned])

if __name__ == "__main__":
    # 修正点：launch() 括号里只保留服务器配置
    demo.launch(server_name="0.0.0.0", server_port=6006)