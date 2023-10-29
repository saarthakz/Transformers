# %%
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm
import argparse
import os
from logger import Logger
from classes.VAE import VAE

def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs

    print(f'Device: {DEVICE}')

    # %%

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size = (32, 32), antialias=True),
    ])

    test_dataset  = datasets.MNIST(root='./data/mnist-test', train=False, download=True, transform=transform)
    test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE, shuffle=True)
  
    # %%
    model = VAE(
        in_channels=1,
        latent_dim=args.latent_dim
    )
    model.to(DEVICE)
    print(model)

    # Print the number of parameters in the model
    print(sum(param.numel() for param in model.parameters()) / 1e6, 'M parameters')
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    if args.from_checkpoint:
        state_dict = torch.load(args.checkpoint_path)
        model.load_state_dict(state_dict)
        print('Model Loaded from checkpoint')

    if args.mode == 'train':
        if args.model_dir is None:
            Exception("Model Directory is not defined for saving")

        os.makedirs(args.model_dir, exist_ok=True)

        logger = Logger(os.path.join(args.model_dir, 'log.txt'))

        train_dataset = datasets.MNIST(root='./data/mnist-train', train=True, download=True, transform=transform)
        train_loader  = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True)
        steps = len(train_loader) * EPOCHS
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps)
        progress_bar = tqdm(range(steps))

        for epoch in range(EPOCHS):
            total_loss = 0
            for step, batch in enumerate(train_loader):
                x, y = batch
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                # evaluate the loss
                img, loss = model.forward(x = x, kl_weight=args.kl_weight)
                total_loss += loss.item()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                # scheduler.step()
                progress_bar.update(1)

            tqdm.write(f"Epoch: {epoch}, Train loss {total_loss / (step + 1)}")
            logger.log(f"Epoch: {epoch}, Train loss {total_loss / (step + 1)}")

            # Test function
            val_losses = []
            for step, batch in enumerate(test_loader):
            
                x, y = batch
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                img, loss = model.forward(x)
                val_losses.append(loss.item())

            tqdm.write(f"Epoch: {epoch}, Val loss: {torch.tensor(val_losses).mean()}")  
            logger.log(f"Epoch: {epoch}, Val loss: {torch.tensor(val_losses).mean()}")  

        torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pt'))
        print('Model Saved!')

    if args.mode == 'generate':
        normal_dist_samples = torch.randn(size=(args.num_test_images, args.latent_dim)).to(DEVICE)
        dec_inputs = model.latent_to_decoder(normal_dist_samples)
        samples = model.decoder(dec_inputs)

        os.makedirs('images/vae/', exist_ok=True)
        for idx, sample in enumerate(samples):
            save_image(sample, f'images/vae/gen_{idx}.png')
            
    if args.mode == 'test':
        num_samples = test_dataset.__len__()
        os.makedirs('images/vae/', exist_ok=True)
        indices = torch.randint(
            low = 0,
            high = num_samples,
            size = (args.num_test_images,)
        )
        for idx in indices:
            image, target = test_dataset.__getitem__(index = idx)
            save_image(image, f'images/vae/{idx}_{target}_true.png')
            image = torch.unsqueeze(image, dim = 0).to(device=DEVICE)
            recon_img, loss = model.forward(image)
            save_image(recon_img, f'images/vae/{idx}_{target}.png')

parser = argparse.ArgumentParser()

parser.add_argument(
    '--batch_size',
    type=int,
    help='Batch Size',
    default=512
)
parser.add_argument(
    '--lr',
    type=int,
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
    help='If test mode, how many images to test on',
    default=4
)
parser.add_argument(
    '--kl_weight',
    type=float,
    help='If test mode, how many images to test on',
    default=0.5
)

args = parser.parse_args()
main(args=args)