#!/bin/bash

DATA_PATH="$(pwd)/viola_sample_data/posed_rgb"
SCENE_NAME="arcore-dataset-2023-10-27-18-46-17"
LIDAR_PATH="$(pwd)/viola_sample_data/posed_rgb/office.ply"

while getopts n: flag
do
    case "${flag}" in
        n) scene_name=${OPTARG};;
    esac
done

configpath=configs/data/aaa_default.yaml
echo "Scene name:" $SCENE_NAME;
echo "Config file name:" "${configpath/aaa/"$SCENE_NAME"}";

# parse input to simplerecon format
cd ./simplerecon
python parse_simplerecon_input.py --scene_name $SCENE_NAME --data_path $DATA_PATH --path_simplerecon ./

# run simplerecon
python ./data_scripts/generate_test_tuples.py  --num_workers 16 --data_config "${configpath/aaa/"$SCENE_NAME"}"

CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
--output_base_path output \
--config_file configs/models/hero_model.yaml \
--load_weights_from_checkpoint weights/hero_model.ckpt \
--data_config "${configpath/aaa/"$SCENE_NAME"}" \
--num_workers 8 \
--batch_size 2 \
--fast_cost_volume \
--run_fusion \
--depth_fuser open3d \
--fuse_color

echo "Result will be saved to: ./simplerecon/output/HERO_MODEL/viola/default/meshes/0.04_3.0_open3d_color"

# viola

# estimate floor
CUDA_VISIBLE_DEVICES=0 python process_viola.py --seq_name $SCENE_NAME --data_path $DATA_PATH --path_simplerecon ./

# viola matching
cd ..
CUDA_VISIBLE_DEVICES=0 python run_simplerecon.py --scene_name $SCENE_NAME --data_path $DATA_PATH --lidar_path $LIDAR_PATH

