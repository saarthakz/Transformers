python trainers/train_vitvqvae.py \
  --mode 'train' \
  --batch_size 256 \
  --model_dir './models/vitvqvae' \
  --epochs 250 \
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

