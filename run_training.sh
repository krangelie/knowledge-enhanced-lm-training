#!/bin/bash

model_name=gpt2-medium
data_path=/export/home/kraft/data/kelm/output
kelm_version=kelm_full
log_dir=$data_path/$model_name/$kelm_version
log_file=$log_dir/train_log.out

export NEPTUNE_API_TOKEN="" ## Customize
export NEPTUNE_PROJECT="" ## Customize
export CUDA_DEVICE_ORDER=PCI_BUS_ID


mkdir -p $log_dir
touch $log_file

nohup python main.py model.model_name=$model_name model.checkpoint=$data_path/gpt2-medium/kelm_full/checkpoint-400000 output_dir=$data_path > $log_file 2>&1 &
