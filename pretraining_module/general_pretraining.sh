
set -e 
set -u  

BSZ=1024
GPU=0
total_steps=10001
seed=42

# MULTI_TASK_STR="p_link p_recon p_ming p_decor p_minsg"
declare -a TASK_COMBINATIONS=(
    "p_link p_recon p_ming p_decor p_minsg"  # all tasks
    "p_link"
    "p_recon"
    "p_ming"
    "p_decor"
    "p_minsg"
)


for MULTI_TASK_STR in "${TASK_COMBINATIONS[@]}"
do

    name="SSL_v1_noProcAttr"
    pareto_option="--not_use_pareto"
    DATASET="SSL_data_SW_BA_100_500_noProcAttr.pt"
    hid_dim=256
    ROOT_CHECKPOINT="SSL_models"
    echo "Running with tasks: ${MULTI_TASK_STR}"


    CUDA_VISIBLE_DEVICES=${GPU} python pretrain_model.py \
    --world_size 1 \
    --worker 1 \
    --name ${name} \
    --checkpoint_dir "${ROOT_CHECKPOINT}" \
    --dataset ${DATASET} \
    --split random \
    --pretrain_label_dir ../pretrain_labels \
    --total_steps ${total_steps} \
    --warmup_step 100 \
    --per_gpu_batch_size ${BSZ} \
    --batch_size_multiplier_minsg 5 \
    --batch_size_multiplier_ming 5 \
    --khop_ming 3 \
    --khop_minsg 3 \
    --lr 5e-5 \
    --optim adamw \
    --scheduler fixed \
    --weight_decay 1e-5 \
    --temperature_gm 0.2 \
    --temperature_minsg 0.1 \
    --sub_size 256 \
    --decor_size 1500 \
    --decor_lamb 1e-3 \
    --hid_dim ${hid_dim} \
    --predictor_dim 512 \
    --n_layer 2 \
    --dropout 0. \
    --seed ${seed} \
    --hetero_graph_path ../hetero_graphs \
    --tasks ${MULTI_TASK_STR} \
    ${pareto_option} \
    --use_prelu \
    --use_saint 

    echo "Completed task combination: ${MULTI_TASK_STR}"

done