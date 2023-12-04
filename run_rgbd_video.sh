#!/bin/bash

DATA_PATH="$(pwd)/viola_sample_data/redwood/loft_short"
LIDAR_PATH="$(pwd)/viola_sample_data/redwood/loft_lidar_dense.mat"
OPEN3D_PATH="/home/selim/code/thirdparty/Open3D"
MASK2FORMER_PATH="$(pwd)/mask2former"

cd preprocess/
python redwood_open3d_m2f.py \
--data_path $DATA_PATH \
--open3d_path $OPEN3D_PATH \
--m2f_path $MASK2FORMER_PATH \
--skip_every_n_frames 15

python run_redwood.py \
--data_path $DATA_PATH \
--lidar_path $LIDAR_PATH