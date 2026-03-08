#!/bin/bash
# Evaluate the original Qwen3.5-2B model (before fine-tuning)
cd "$(dirname "$0")/.."
for category in "Industrial_and_Scientific"
do
    exp_name="Qwen/Qwen3.5-2B"
    echo "Processing category: $category with model: $exp_name"

    test_file=$(ls ./data/Amazon/test/${category}*11.csv 2>/dev/null | head -1)
    train_file=$(ls ./data/Amazon/train/${category}*.csv 2>/dev/null | head -1)
    info_file=$(ls ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1)

    if [[ ! -f "$test_file" ]]; then
        echo "Error: Test file not found for category $category"
        continue
    fi
    if [[ ! -f "$info_file" ]]; then
        echo "Error: Info file not found for category $category"
        continue
    fi

    result_dir="./results/baseline"
    mkdir -p "$result_dir"

    echo "Starting evaluation on single GPU..."
    CUDA_VISIBLE_DEVICES=1 python -u ./evaluate.py \
        --base_model "$exp_name" \
        --info_file "$info_file" \
        --category ${category} \
        --test_data_path "$test_file" \
        --result_json_data "$result_dir/result_${category}.json" \
        --batch_size 8 \
        --num_beams 50 \
        --max_new_tokens 256 \
        --length_penalty 0.0

    echo "Calculating metrics..."
    python ./utils/calc.py \
        --path "$result_dir/result_${category}.json" \
        --item_path "$info_file" \
        --train_path "$train_file"

    echo "Completed processing for category: $category"
    echo "Results saved to: $result_dir/result_${category}.json"
    echo "----------------------------------------"
done

echo "All categories processed!"
