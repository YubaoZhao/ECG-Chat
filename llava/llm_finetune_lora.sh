#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
open_clip_dir="$script_dir/../open_clip"
export PYTHONPATH="$script_dir:$open_clip_dir:$PYTHONPATH"

# MODEL_VERSION="vicuna-v1-3-7b"

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path ./playground/data/ecg_instruct_45k.json \
    --ecg_folder path/to/your/mimic-iv-ecg \
    --ecg_tower path/to/your/open_clip/checkpoint \
    --open_clip_config coca_ViT-B-32 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-$MODEL_VERSION-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_use_ecg_start_end False \
    --mm_use_ecg_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-finetune_lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True