#!/bin/bash

datapath=/home/jj/work/data/our_itw/office/long_sequences
scene_name=arcore-dataset-2023-10-27-18-46-17
lidarpath=/home/jj/work/data/our_itw/office/lidar/office.ply
while getopts n: flag
do
    case "${flag}" in
        n) scene_name=${OPTARG};;
    esac
done

configpath=configs/data/aaa_default.yaml
echo "Scene name:" $scene_name;
echo "config file name:" "${configpath/aaa/"$scene_name"}";

# parse input to simplerecon format
cd ./simplerecon
python parse_simplerecon_input.py --scene_name $scene_name --data_path $datapath --path_simplerecon ./

# run Simplerecon
python ./data_scripts/generate_test_tuples.py  --num_workers 16 --data_config "${configpath/aaa/"$scene_name"}"

CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
            --output_base_path output \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config "${configpath/aaa/"$scene_name"}" \
            --num_workers 8 \
            --batch_size 2 \
            --fast_cost_volume \
            --run_fusion \
            --depth_fuser open3d \
            --fuse_color #--cache_depths --dump_depth_visualization
# result will be saved to: ./simplerecon/output/HERO_MODEL/viola/default/meshes/0.04_3.0_open3d_color

# viola
# estimate floor
CUDA_VISIBLE_DEVICES=0 python process_viola.py --seq_name $scene_name --data_path $datapath --path_simplerecon ./

# viola matching
cd ..

CUDA_VISIBLE_DEVICES=0 python run_simplerecon.py --scene_name $scene_name --data_path $datapath --lidar_path $lidarpath

