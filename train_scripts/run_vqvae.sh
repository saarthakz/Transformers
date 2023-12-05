python trainers/train_vqvae.py \
  --mode 'train' \
  --batch_size 256 \
  --model_dir './models/vqvae' \
  --epochs 10 \
  --latent_dim 64 \
  --num_embeddings 512 \
  --image_channels 3 \
  --beta 0.25 \
  --lr 0.001 \
  --res_channels 32 \
  --num_residual_layers 2 \

