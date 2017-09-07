#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an Inception Resnet V2 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_resnet_v2_on_flowers.sh
set -e

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/tmp/checkpoints

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
MODEL_NAME=darknet_19

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/flowers-models/${MODEL_NAME}

# Where the dataset is saved to.
DATASET_DIR=/tmp/flowers


# Download the dataset
python download_and_convert_data.py \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 30000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --preprocessing_name=inception \
  --model_name=${MODEL_NAME} \
  --max_number_of_steps=30000 \
  --batch_size=32 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --learning_rate=0.001 \
  --optimizer=adam \

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME}