#!/usr/bin/env bash
export DATASET_BUCKET=pascal-tf-record

export TRAIN_DIR=${DATASET_BUCKET}/train
export EVAL_DIR=${DATASET_BUCKET}/eval
export PIPELINE_CONFIG_PATH=${DATASET_BUCKET}/ssd_mobilenet_v1_pascal_cloud.config

export PATH_TO_LOCAL_YAML_FILE=object_detection/samples/cloud/cloud.yml

gcloud ml-engine jobs submit training object_detection_`date +%s` \
    --job-dir=gs://${TRAIN_DIR} \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region us-east1 \
    --config ${PATH_TO_LOCAL_YAML_FILE} \
    -- \
    --train_dir=gs://${TRAIN_DIR} \
    --pipeline_config_path=gs://${PIPELINE_CONFIG_PATH}