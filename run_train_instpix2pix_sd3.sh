#!/usr/bin/env bash

#SBATCH --job-name=magicBrush_sd3
#SBATCH --nodes=1
#SBATCH --time=30:00:00
#SBATCH --signal=SIGINT
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=80G
#SBATCH --partition=long
#SBATCH --output=/tmp/magicBrush_sd3%j.out

source cc_job.sh
pip install transformers -U
pip install sentencepiece
pip install git+https://github.com/huggingface/diffusers
export TRANSFORMERS_CACHE=""
export HF_HOME=""

accelerate launch --mixed_precision="fp16"  /home/mila/z/zichao.li/agent/world/hf_examples/train_ins_p2p_sd3.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
    --dataset_name="osunlp/MagicBrush" \
    --resolution=512 --random_flip \
    --train_batch_size=8 --gradient_accumulation_steps=8 --gradient_checkpointing \
    --original_image_column "source_img" --edited_image_column "target_img" --edit_prompt_column "instruction"\
    --num_train_epochs=50 \
    --checkpointing_steps=500 --checkpoints_total_limit=10 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=100 \
    --conditioning_dropout_prob=0.05 \
    --seed=42\
    --lr_scheduler="constant" \
    --output_dir="./saved_models/magicBrush_sd3_2/"