#!/bin/bash

MODEL_DIR=${MODEL_DIR:-'/path/to/model_dir'}

MAX_ROUND=${MAX_ROUND:-10}
WORLD_SIZE=${WORLD_SIZE:-1}
CHECKPOINT_FILE=${CHECKPOINT_FILE:-'model.safetensors'}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-'/path/to/checkpoint_dir'}
OUTPUT_DIR=${OUTPUT_DIR:-'/path/to/output_dir'}
ROOT_PATH=${ROOT_PATH:-'/path/to/Seed-Bench-R1_dir'}
QA_FILE=${QA_FILE:-'/path/to/qa_file'}
SEED=${SEED:-1}
DIS_OPT=${DIS_OPT:-0}
NUM_GPUS=${NUM_GPUS:-1}


PIDS=()


cleanup() {
    echo "[INFO] Received interrupt signal. Killing all child processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
    wait "${PIDS[@]}" 2>/dev/null 
    echo "[INFO] All child processes terminated."
    exit 1
}

trap cleanup SIGINT SIGTERM

for ((rank=0; rank<WORLD_SIZE; rank++)); do
    GPU_ID=$((rank % NUM_GPUS))
    echo "[INFO] Launching rank=$rank on GPU $GPU_ID"


    if [ "$DIS_OPT" -eq 0 ]; then
        echo "[INFO] $rank on GPU $GPU_ID use options!"
        CUDA_VISIBLE_DEVICES=$GPU_ID \
        python inference_SeedBenchR1_TwiFF_mp.py \
            --max_round $MAX_ROUND \
            --model_dir "$MODEL_DIR" \
            --checkpoint_file "$CHECKPOINT_FILE" \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --root_dir "$ROOT_PATH" \
            --QA_file "$QA_FILE" \
            --seed $SEED \
            --visual_gen \
            --rank $rank \
            --world_size $WORLD_SIZE \
            &
    else
        echo "[INFO] $rank on GPU $GPU_ID disable options!"
        CUDA_VISIBLE_DEVICES=$GPU_ID \
        python inference_SeedBenchR1_TwiFF_mp.py \
            --max_round $MAX_ROUND \
            --model_dir "$MODEL_DIR" \
            --checkpoint_file "$CHECKPOINT_FILE" \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --root_dir "$ROOT_PATH" \
            --QA_file "$QA_FILE" \
            --seed $SEED \
            --visual_gen \
            --rank $rank \
            --world_size $WORLD_SIZE \
            --disable_options \
            &
    fi
    PIDS+=($!)
done

wait "${PIDS[@]}"
echo "[INFO] All processes finished."

for i in $(seq 0 $((WORLD_SIZE - 1))); do
    if [[ -f "${OUTPUT_DIR}/model_response_rank${i}.jsonl" ]]; then
        cat "${OUTPUT_DIR}/model_response_rank${i}.jsonl"
    fi
done > "${OUTPUT_DIR}/model_response_unscore.jsonl"

echo "[INFO] All file saved to ${OUTPUT_DIR}/model_response_unscore.jsonl."