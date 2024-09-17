#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
open_clip_dir="$script_dir/../open_clip"
export PYTHONPATH="$script_dir:$open_clip_dir:$PYTHONPATH"

# MODEL_VERSION=vicuna-13b
MODEL_VERSION=llama-2-7b-chat

deepspeed llava/train/train_xformers.py \
    --deepspeed "$script_dir/llava/scripts/zero2.json" \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version plain \
    --data_path ./playground/data/ecg_pretrain.json \
    --ecg_folder path/to/your/mimic-iv-ecg \
    --ecg_tower path/to/your/open_clip/checkpoint \
    --open_clip_config coca_ViT-B-32 \
    --tune_mm_mlp_adapter True \
    --mm_use_ecg_start_end False \
    --mm_use_ecg_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True