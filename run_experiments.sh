#!/bin/bash
# GPU worker for running s-lab experiments. Continues until the parameter list is empty.

if [ -z $1 ]
then
	read -p "Enter the ID of the gpu you want to use: "  gpu
else
	gpu=$1
fi
echo "Developing worker for gpu $gpu."

cm_type=('contextual_vector')
layer_name=('conv3_3')
output_type=('sparse_pool')
project_name=('sheinberg_data_noise_subtracted')
timesteps=(5)

for ((i=0;i<${#cm_type[@]};i++)); do
	CUDA_VISIBLE_DEVICES=$gpu python extract_image_features.py --cm_type="${cm_type[i]}" --layer_name="${layer_name[i]}" --output_type="${output_type[i]}" --project_name="${project_name[i]}" --timesteps="${timesteps[i]}"
done