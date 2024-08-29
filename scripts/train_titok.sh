 accelerate launch \
    --config_file "./configs/accelerate_config.yaml" \
    trainers/titok.py \
    --config_file "./configs/train-config.json"