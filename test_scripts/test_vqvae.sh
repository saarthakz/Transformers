python ./trainers/train_vqvae.py \
    --mode 'test' \
    --from_checkpoint \
    --checkpoint_path './models/vqvae/model.pt' \
    --num_test_images 10 \
    --latent_dim 64 \
    --num_embeddings 512 \
    --image_channels 3 \
    --beta 0.25 \
    --res_channels 32 \
    --num_residual_layers 2 \