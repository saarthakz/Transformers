python trainers/train_vitvqvae.py \
    --from_checkpoint \
    --checkpoint_path './models/vitvqvae/model.pt' \
    --mode 'train_gen' \
    --num_test_images 3 \
    --batch_size 256 \
    --model_dir './models/vitvqvae' \
    --epochs 100 \
    --latent_dim 64 \
    --num_embeddings 512 \
    --image_channels 3 \
    --image_size 32 \
    --patch_size 4 \
    --beta 0.25 \
    --lr 0.001 \
    --num_heads 4 \
    --num_blocks 2 \
    --dropout 0.01 \

