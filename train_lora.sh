#!/bin/bash

# 确保网络畅通
source /etc/network_turbo

echo "🚀 开始启动 Qwen2.5 审计专家微调任务..."

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /root/autodl-tmp/qwen2_5_7b_model \
    --dataset audit_apple \
    --dataset_dir /root/autodl-tmp \
    --template qwen \
    --finetuning_type lora \
    --output_dir /root/autodl-tmp/qwen_lora_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 2 \
    --save_steps 50 \
    --learning_rate 5e-5 \
    --num_train_epochs 10.0 \
    --fp16

echo "✅ 微调任务执行完毕！模型权重已保存在 qwen_lora_checkpoint"