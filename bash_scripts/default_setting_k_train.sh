#!/bin/bash
set -euo pipefail

BASE_NAME="${BASE_NAME:-k_setting}"
K_VALUES="${K_VALUES:-1 2 3 4 5}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
TRAIN_CUDA="${TRAIN_CUDA:-0}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints/}"
DATA_DIR="${DATA_DIR:-/home/108/u108009/dataset/ForenSynths}"

TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
VALIDATE_PYTHON="${VALIDATE_PYTHON:-python3}"

P="${P:-0.7}"
ARCH="${ARCH:-CLIP:ViT-L/14}"
LR="${LR:-0.00005}"
DATA_MODE="${DATA_MODE:-sd1_4}"
SELECT_K="${SELECT_K:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"
TEST_DATA_LIST="${TEST_DATA_LIST:-ForenSynths GenImage}"
RESULT_ROOT="${RESULT_ROOT:-./results}"

for K in ${K_VALUES}; do
    NAME="${BASE_NAME}_k${K}"
    CKPT_PATH="${CHECKPOINTS_DIR%/}/${NAME}/best_model.pth"

    echo "========== Train ${NAME} =========="
    CUDA_VISIBLE_DEVICES="${TRAIN_CUDA}" "${TRAIN_PYTHON}" train.py \
        --p="${P}" \
        --k="${K}" \
        --name="${NAME}" \
        --wang2020_data_path="${DATA_DIR}" \
        --checkpoints_dir="${CHECKPOINTS_DIR}" \
        --data_mode="${DATA_MODE}" \
        --arch="${ARCH}" \
        --lr="${LR}" \
        --fix_backbone \
        --select_k="${SELECT_K}" \
        --batch_size="${BATCH_SIZE}" \
        --save_epoch_freq="${SAVE_EPOCH_FREQ}" \
        --niter="${TOTAL_EPOCHS}"

    if [ ! -f "${CKPT_PATH}" ]; then
        echo "Checkpoint not found: ${CKPT_PATH}" >&2
        exit 1
    fi

    for TEST_DATA in ${TEST_DATA_LIST}; do
        RESULT_FOLDER="${RESULT_ROOT}/${NAME}/${TEST_DATA}"
        mkdir -p "${RESULT_FOLDER}"

        echo "========== Test ${NAME} on ${TEST_DATA} =========="
        CUDA_VISIBLE_DEVICES="${TRAIN_CUDA}" "${VALIDATE_PYTHON}" validate.py \
            --p="${P}" \
            --k="${K}" \
            --test_data="${TEST_DATA}" \
            --arch="${ARCH}" \
            --ckpt="${CKPT_PATH}" \
            --result_folder="${RESULT_FOLDER}" \
            --select_k="${SELECT_K}" \
            --batch_size="${BATCH_SIZE}"
    done
done
