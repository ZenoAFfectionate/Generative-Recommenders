cd "$(dirname "$0")/.."

# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}

    CUDA_VISIBLE_DEVICES=0 python trainer/sft.py \
            --base_model Qwen/Qwen3.5-2B \
            --batch_size 128 \
            --micro_batch_size 32 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ./output/sft/${category} \
            --wandb_project MiniOneRec \
            --wandb_run_name sft_${category}_qwen3.5-2b \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/${category}.index.json \
            --item_meta_path ./data/Amazon/index/${category}.item.json \
            --freeze_LLM False
done
