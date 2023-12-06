import torch
import torch.optim as optim
import torch.nn.functional as func
import math
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm
import argparse
import os
import sys
sys.path.append(os.path.abspath("."))
from utils.logger import Logger
from classes.VITVQVAE import ViTVQVAE
from classes.Transformers import Transformer

def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = DEVICE
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs

    print(f'Device: {DEVICE}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size = (32, 32), antialias=True),
    ])

    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE, shuffle=True)
  
    # %%
    model = ViTVQVAE(
        args.image_size,
        args.patch_size,
        args.image_channels,
        args.num_embeddings,
        args.latent_dim,
        args.num_heads,
        args.num_blocks,
        args.dropout,
        args.beta
    )
    model.to(DEVICE)

    # Print the number of parameters in the model
    print(sum(param.numel() for param in model.parameters()) / 1e6, 'M parameters')
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    if args.model_from_checkpoint:
        model.from_pretrained(args.model_checkpoint_path)
        print('Model Loaded from checkpoint')

    if args.mode == 'train':
        if args.model_dir is None:
            Exception("Model Directory is not defined for saving")

        os.makedirs(args.model_dir, exist_ok=True)

        logger = Logger(os.path.join(args.model_dir, 'log.txt'))

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_loader  = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True)
        steps = len(train_loader) * EPOCHS
        progress_bar = tqdm(range(steps))

        for epoch in range(EPOCHS):
            total_loss = 0
            for step, batch in enumerate(train_loader):
                x, y = batch
                x = x.to(DEVICE)

                # evaluate the loss
                recon, codebook_indices, q_loss = model.forward(x = x)
                recon_loss = func.mse_loss(x, recon)
                loss = q_loss + recon_loss
                total_loss += loss.item()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                # scheduler.step()
                progress_bar.update(1)
            
            train_loss_log = f"Epoch: {epoch}, Train loss {total_loss / (step + 1)}"
            tqdm.write(train_loss_log)
            logger.log(train_loss_log)

            # Test function
            val_losses = []
            for step, batch in enumerate(test_loader):
            
                decoded, codebook_indices, q_loss = model.forward(x = x)
                recon_loss = func.mse_loss(x, decoded)
                loss = q_loss + recon_loss
                total_loss += loss.item()
                val_losses.append(loss.item())

            val_loss_log = f"Epoch: {epoch}, Val loss: {torch.tensor(val_losses).mean()}" 
            tqdm.write(val_loss_log)  
            logger.log(val_loss_log)  

        torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pt'))
        print('Model Saved!')
     
    if args.mode == 'test':
        num_samples = test_dataset.__len__()
        os.makedirs('images/vitvqvae/', exist_ok=True)
        indices = torch.randint(
            low = 0,
            high = num_samples,
            size = (args.num_test_images,)
        )
        for idx in indices:
            image, target = test_dataset.__getitem__(index = idx)
            save_image(image, f'images/vitvqvae/{idx}_{target}_true.png')
            image = torch.unsqueeze(image, dim = 0).to(device=DEVICE)
            recon_img,_, loss = model.forward(image)
            save_image(recon_img, f'images/vitvqvae/{idx}_{target}.png')

    if args.mode == 'train_gen':

        logger = Logger(os.path.join(args.model_dir, 'log_gen.txt'))

        num_patches = (args.image_size // args.patch_size) ** 2

        transformer = Transformer(
            num_patches,
            args.latent_dim,
            args.num_embeddings,
            args.num_heads
        ).to(device=DEVICE)

        transformer_optim = optim.AdamW(transformer.parameters(), lr=LR)

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_loader  = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True)
        steps = len(train_loader) * EPOCHS
        progress_bar = tqdm(range(steps))

        for epoch in range(EPOCHS):
            total_loss = 0
            for step, batch in enumerate(train_loader):
                x, y = batch
                x = x.to(DEVICE)

                x_enc = model.encode(x) # Encode using the ViT Encoder
                B, C, D  = x_enc.shape
                num_patch_sqrt = math.sqrt(C).__int__()
                x_enc = x_enc.transpose(-2, -1).view(B, D, num_patch_sqrt, num_patch_sqrt)

                z_q, indices, loss = model.quantize.forward(x_enc) # Quantize and get the codebook entries

                # Indices will be fed to the transformer for prediction
                indices = indices.view(B, C)

                # Base indices are also the target for when predicting from noisy indices
                target = indices 

                # Start token so that the Transformer always has a token when generating
                start_tokens = torch.ones(B, 1, device=x.device, dtype=torch.long) * 0

                mask = torch.bernoulli(args.keep_prob * torch.ones(indices.shape, device=indices.device))
                mask = mask.round().to(dtype=torch.int64)

                random_indices = torch.randint_like(indices, low=0, high=args.num_embeddings)

                noised_indices = mask * indices + (1 - mask) * random_indices
                noised_indices = torch.cat((start_tokens, noised_indices), dim=1)[:, :-1]

                logits, loss = transformer.forward(noised_indices, target)
                total_loss += loss.item()
                transformer_optim.zero_grad(set_to_none=True)
                loss.backward()
                transformer_optim.step()
                # scheduler.step()
                progress_bar.update(1)
            
            train_loss_log = f"Epoch: {epoch}, Train loss {total_loss / (step + 1)}"
            tqdm.write(train_loss_log)
            logger.log(train_loss_log)

        torch.save(transformer.state_dict(), os.path.join(args.model_dir, 'model_gen.pt'))

    if args.mode == 'generate':

        num_patch_sqrt = (args.image_size // args.patch_size).__int__()
        num_patches = num_patch_sqrt ** 2

        transformer = Transformer(
            num_patches,
            args.latent_dim,
            args.num_embeddings,
            args.num_heads
        ).to(device=DEVICE)

        if args.transformer_from_checkpoint:
            transformer.load_state_dict(torch.load(args.transformer_checkpoint_path))
            print('Transformer loaded from checkpoint')

        start_tokens = torch.zeros(size=(args.num_new_images, 1), dtype=torch.long, device=DEVICE)

        indices = transformer.generate(start_tokens, num_patches)[: , 1:]

        z_q = model.quantize.embedding.forward(indices).view(size=(args.num_new_images, num_patch_sqrt, num_patch_sqrt, args.latent_dim)) # (B, H, W, D) 

        z_q = z_q.permute(0, 3, 1, 2) # (B, D, H, W)

        z_q = z_q.view(args.num_new_images, args.latent_dim, num_patches).transpose(-2, -1) # (B, D, C) -> [after transpose] -> (B, C, D)
        recon_imgs: torch.Tensor = model.decode(z_q)

        save_image(recon_imgs, fp='images/vitvqvae/gen.png')

parser = argparse.ArgumentParser()

parser.add_argument(
    '--batch_size',
    type=int,
    help='Batch Size',
    default=512
)
parser.add_argument(
    '--lr',
    type=float,
    help='Learning rate',
    default=1e-4
)
parser.add_argument(
    '--epochs',
    type=int,
    help='# of Epochs',
    default=50
)
parser.add_argument(
    '--print_interval',
    type=int,
    help='Step interval for printing',
    default=50
)
parser.add_argument(
    '--model_from_checkpoint',
    action='store_true',
    help='Load model from a checkpoint'
)
parser.add_argument(
    '--model_checkpoint_path',
    type=str,
    help='Path to model checkpoint path [State Dictionary]'
)
parser.add_argument(
    '--transformer_from_checkpoint',
    action='store_true',
    help='Load transformer from a checkpoint'
)
parser.add_argument(
    '--transformer_checkpoint_path',
    type=str,
    help='Path to transformer checkpoint path [State Dictionary]'
)
parser.add_argument(
    '--model_dir',
    type=str,
    help='Model directory for saving',
)
parser.add_argument(
    '--mode',
    type=str,
    help='Mode: ["train", "generate"]',
    required=True
)
parser.add_argument(
    '--num_test_images',
    type=int,
    help='If test mode, how many images to test on',
    default= 10
)
parser.add_argument(
    '--num_new_images',
    type=int,
    help='If generate mode, how many images to generate',
    default= 10
)
parser.add_argument(
    '--latent_dim',
    type=int,
    help='Latent dim for the codebook entries',
    default=128
)
parser.add_argument(
    '--num_embeddings',
    type=int,
    help='Number of vector embeddings for the codebook',
    default=512
)
parser.add_argument(
    '--beta',
    type=float,
    help='Beta is a weighting factor for the "codebook gradient flowing" loss also called the commitment cost',
    default=0.25
)
parser.add_argument(
    '--dropout',
    type=float,
    help='Dropout ',
    default=0.01
)
parser.add_argument(
    '--keep_prob',
    type=float,
    help='Probability for codebook entries to not get masked for Stage 2 [Transformer] training',
    default=0.5
)
parser.add_argument(
    '--image_channels',
    type=int,
    help='Image channel of the training or the testing data',
    required=True
)
parser.add_argument(
    '--image_size',
    type=int,
    help='Image size/dimension',
    required=True
)
parser.add_argument(
    '--patch_size',
    type=int,
    help='Patch size/dimension',
    required=True
)
parser.add_argument(
    '--num_heads',
    type=int,
    help='Number of heads used in the Transformer',
    default=64
)
parser.add_argument(
    '--num_blocks',
    type=int,
    help='Number of Transformer blocks',
    default=2
)


args = parser.parse_args()
main(args=args)