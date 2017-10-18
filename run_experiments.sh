#!/bin/bash
# GPU worker for running s-lab experiments. Continues until the parameter list is empty.

if [ -z $1 ]
then
	read -p "Enter the ID of the gpu you want to use: "  gpu
else
	gpu=$1
fi
echo "Developing worker for gpu $gpu."

cm_type=('contextual_vector_separable_rand_init' 'contextual_vector' 'contextual_vector_separable' 'none')  #  'contextual_vector' 'none')
layer_name=('conv3_2' 'conv3_2' 'conv3_2' 'conv3_2')  #  'conv3_3' 'conv3_3')
output_type=('sparse_pool' 'sparse_pool' 'sparse_pool' 'sparse_pool')  #  'sparse_pool' 'sparse_pool')
project_name=('sheinberg_data' 'sheinberg_data' 'sheinberg_data' 'sheinberg_data')  #  'sheinberg_data_noise_subtracted' 'sheinberg_data_noise_subtracted')  # ('sheinberg_data')
timesteps=(4 4 4 4)

for ((i=0;i<${#cm_type[@]};i++)); do
	CUDA_VISIBLE_DEVICES=$gpu python fit_model.py --cm_type="${cm_type[i]}" --layer_name="${layer_name[i]}" --output_type="${output_type[i]}" --project_name="${project_name[i]}" --timesteps="${timesteps[i]}"
done
