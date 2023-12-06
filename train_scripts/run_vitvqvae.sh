python trainers/train_vitvqvae.py \
  --mode 'train' \
  --batch_size 256 \
  --model_dir './models/vitvqvae' \
  --epochs 150 \
  --latent_dim 128 \
  --num_embeddings 1024 \
  --image_channels 3 \
  --image_size 32 \
  --patch_size 4 \
  --beta 0.25 \
  --lr 0.001 \
  --num_heads 4 \
  --num_blocks 3 \
  --dropout 0.01 \

