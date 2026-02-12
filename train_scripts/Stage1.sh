# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
set -x
GPUS=${GPUS:-1}
NUM_WORKER=${NUM_WORKER:-1}
OUTPUT_DIR=${OUTPUT_DIR:-'./checkpoints/TwiFF'}
MODEL_PATH=${MODEL_PATH:-'/path/to/BAGEL-7B-MoT'}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export MASTER_PORT=34229



if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# replace the variables with your own
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=${GPUS} \
  --master_addr=127.0.0.1 \
  --master_port=${MASTER_PORT} \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/UniCOT.yaml \
  --model_path ${MODEL_PATH} \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from ${MODEL_PATH} \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --use_flex True \
  --timestep_shift 1.0 \
  --ema 0.995 \
  --mse_weight 1 \
  --ce_weight 1 \
  --log_every 1 \
  --lr 2e-5 \
  --min_lr 1e-6 \
  --lr_scheduler cosine \
  --num_worker=${NUM_WORKER} \
  --prefetch_factor 2 \
  --expected_num_tokens 32768 \
  --max_num_tokens 36864 \
  --max_num_tokens_per_sample 20480 \
  --prefer_buffer_before 16384 \
  --max_buffer_size 50 \
  --num_shard=${GPUS} \
  --sharding_strategy="HYBRID_SHARD" \
  --save_every 6000 \
  --save_only_model False \
  --checkpoint_num 1 \
  --warmup_steps 1080 \
  --total_steps 36000 \
  --results_dir ${OUTPUT_DIR}/ \
  --checkpoint_dir ${OUTPUT_DIR}/checkpoints/ \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"