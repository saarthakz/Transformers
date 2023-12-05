# %%
import torch
import torch.optim as optim
import torch.nn.functional as func
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm
import argparse
import os
import sys
sys.path.append(os.path.abspath("."))
from utils.logger import Logger
from classes.VQVAE import VQVAE

def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = DEVICE
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    args.channels = [32, 64, 128]

    print(f'Device: {DEVICE}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size = (32, 32), antialias=True),
    ])

    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE, shuffle=True)
  
    # %%
    model = VQVAE(
        args.image_channels,
        args.latent_dim,
        args.res_channels,
        args.num_residual_layers,
        args.num_embeddings,
        args.beta,
        args.decay
    )
    model.to(DEVICE)

    # print(model)

    # Print the number of parameters in the model
    print(sum(param.numel() for param in model.parameters()) / 1e6, 'M parameters')
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    if args.from_checkpoint:
        model.load_checkpoint(args.checkpoint_path)
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
                y = y.to(DEVICE)

                # evaluate the loss
                decoded, codebook_indices, q_loss = model.forward(x = x)
                recon_loss = func.mse_loss(x, decoded)
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
        os.makedirs('images/vqvae/', exist_ok=True)
        indices = torch.randint(
            low = 0,
            high = num_samples,
            size = (args.num_test_images,)
        )
        for idx in indices:
            image, target = test_dataset.__getitem__(index = idx)
            save_image(image, f'images/vqvae/{idx}_{target}_true.png')
            image = torch.unsqueeze(image, dim = 0).to(device=DEVICE)
            recon_img,_, loss = model.forward(image)
            save_image(recon_img, f'images/vqvae/{idx}_{target}.png')

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
    '--from_checkpoint',
    action='store_true',
    help='Load from a checkpoint'
)
parser.add_argument(
    '--checkpoint_path',
    type=str,
    help='Path to checkpoint path [State Dictionary]'
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
    '--decay',
    type=float,
    help='Decay factor for Exponentially Moving Average updation variant of Vector Quantization [Codebook]',
    default=0.25
)
parser.add_argument(
    '--image_channels',
    type=int,
    help='Image channel of the training or the testing data',
    required=True
)
parser.add_argument(
    '--res_channels',
    type=int,
    help='Number of channels for the residual layer',
    default=64
)
parser.add_argument(
    '--num_residual_layers',
    type=int,
    help='Number of residual layers',
    default=2
)


args = parser.parse_args()
main(args=args)