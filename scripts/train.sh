 accelerate launch \
    --config_file "./configs/accelerate_config.yaml" \
    --num_processes 1 \
    --gpu_ids "0"
    trainers/vq.py \
    --config_file "./configs/train-config.json"