#!/bin/bash
cd "$(dirname "$0")/.."

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    CUDA_VISIBLE_DEVICES=0 python trainer/rl.py \
                        --model_path ./output/sft/${category} \
                        --train_batch_size 64 \
                        --eval_batch_size 16 \
                        --num_train_epochs 1 \
                        --gradient_accumulation_steps 2 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train False \
                        --eval_step 0.25 \
                        --reward_type ranking \
                        --num_generations 8 \
                        --mask_all_zero False \
                        --dynamic_sampling False \
                        --sync_ref_model True \
                        --beam_search True \
                        --test_during_training False \
                        --temperature 1.0 \
                        --learning_rate 1e-5 \
                        --add_gt False \
                        --beta 1e-3 \
                        --dapo False \
                        --output_dir ./output/rl/${category} \
                        --wandb_run_name rl_${category}_qwen3.5-2b \
                        --sid_index_path ./data/Amazon/index/${category}.index.json \
                        --item_meta_path ./data/Amazon/index/${category}.item.json
done
