#!/bin/bash
# class_labels = ['tvmonitor', 'diningtable']
NAME="default_setting"
TOTAL_EPOCHS=5
TRAIN_CUDA=0
CHECKPOINTS_DIR="checkpoints/"
DATA_DIR="/home/108/u108009/dataset/ForenSynths"
# DRCT: lr=0.0001, UFD: lr=0.00005
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA}" python train.py \
    --p=0.7 \
    --name="${NAME}" \
    --wang2020_data_path=${DATA_DIR} \
    --checkpoints_dir="${CHECKPOINTS_DIR}" \
    --data_mode="sd1_4" \
    --arch="CLIP:ViT-L/14" \
    --lr=0.00005 \
    --fix_backbone \
    --select_k=5 \
    --batch_size=256 \
    --save_epoch_freq=5 \
    --niter="${TOTAL_EPOCHS}"

CUDA_ID="0"
CKPT_PATH="./checkpoints/default_setting/best_model.pth"
RESULT_FOLDER="./results/default_setting/"

TEST_DATA="UFD"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
    --p=0.7 \
    --test_data="${TEST_DATA}" \
    --arch="CLIP:ViT-L/14" \
    --ckpt="${CKPT_PATH}" \
    --result_folder="${RESULT_FOLDER}" \
    --select_k=5 \
    --batch_size=256

TEST_DATA="GenImage"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
    --p=0.7 \
    --test_data="${TEST_DATA}" \
    --arch="CLIP:ViT-L/14" \
    --ckpt="${CKPT_PATH}" \
    --result_folder="${RESULT_FOLDER}" \
    --select_k=5 \
    --batch_size=256