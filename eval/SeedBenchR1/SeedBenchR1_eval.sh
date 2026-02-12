#!/bin/bash

# 设置通用参数（不变的部分）
SAMPLE_MAX_NUM=40480000
API=${API:-"http://localhost:8000/v1"}
API_KEY=${API_KEY:-"EMPTY"}
MODEL_NAME=${MODEL_NAME:-"/path/to/Judge"}
REQ_SIZE=${REQ_SIZE:-64}
BATCH_SIZE=${BATCH_SIZE:-2}
OUT_FREQ=${OUT_FREQ:-128}
MAX_TOKENS=${MAX_TOKENS:-1024}
NUM_WORK=${NUM_WORK:-32}
ROOT_PATH=${ROOT_PATH:-'/path/to/Seed-Bench-R1_dir'}
TOP_K=1
MIN_P=0.0
TEMPERATURE=0.0
NUM_FRAMES=8
START_FRAMES=1
END_FRAMES=1

declare -a DATA_PATHS=(
    "/path/to/SeedBenchR1_results/model_response_unscore.jsonl"
)

#for gpt
declare -a OUT_PATHS=(
    "/path/to/SeedBenchR1_results/model_response_score_gpt.jsonl"
)

if [ ${#DATA_PATHS[@]} -ne ${#OUT_PATHS[@]} ]; then
    echo "Error: DATA_PATHS and OUT_PATHS must have the same number of elements."
    exit 1
fi

TOTAL=${#DATA_PATHS[@]}

# 循环执行
for i in "${!DATA_PATHS[@]}"; do
    IDX=$((i+1))
    DATA="${DATA_PATHS[i]}"
    OUT="${OUT_PATHS[i]}"

    echo "[$IDX/$TOTAL] Starting evaluation: $DATA → $OUT"
    
    python eval/gencot_vlm/gencot_SeedBenchR1_eval.py \
        --sample_max_num $SAMPLE_MAX_NUM \
        --api "$API" \
        --api_key "$API_KEY" \
        --model_name "$MODEL_NAME" \
        --data_path "$DATA" \
        --out_path "$OUT" \
        --req_size $REQ_SIZE \
        --batch_size $BATCH_SIZE \
        --out_freq $OUT_FREQ \
        --max_tokens $MAX_TOKENS \
        --num_work $NUM_WORK \
        --top_k $TOP_K \
        --min_p $MIN_P \
        --temperature $TEMPERATURE \
        --num_frames $NUM_FRAMES \
        --start_frames $START_FRAMES \
        --end_frames $END_FRAMES \
        --root_dir "$ROOT_PATH"

    if [ $? -eq 0 ]; then
        echo "[$IDX/$TOTAL] Successfully completed: $OUT"
    else
        echo "[$IDX/$TOTAL] Failed on: $DATA"
        exit 1
    fi
done

echo "All $TOTAL jobs completed successfully!"