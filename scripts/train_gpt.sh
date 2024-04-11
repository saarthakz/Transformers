 accelerate launch \
    --config_file "./configs/accelerate_config.yaml" \
    trainers/vq_gpt.py \
    --config_file "./configs/ae_train_config.json"