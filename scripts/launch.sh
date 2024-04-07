accelerate launch \
    --config_file "./configs/accelerate_config.yaml" \
    trainers/vq.py \
    --config_file "./configs/train_config.json"