#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python -m experiments.ex_dcase24 with \
data_loader.batch_size=16 \
data_loader.batch_size_eval=16 \
audio_features.segment_length=10 \
audio_features.model="dymn20_as(4)" \
sentence_features.model=roberta-large \
rampdown_type=cosine \
max_epochs=20 \
rampdown_stop=15 \
warmup_length=1 \
rampdown_start=1 \
train_on=clothov2 \
seed=409194 \
directories.data_dir=/mnt/SSD1/st5265