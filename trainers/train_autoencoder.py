# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm
import argparse
from classes.Autoencoder import Autoencoder
import os

def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    model_dir = args.model_dir

    # %%

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size = (32, 32), antialias=True),
        transforms.Normalize((0.5), (0.5)),
    ])

    test_dataset  = datasets.MNIST(root='./data/mnist-test', train=False, download=True, transform=transform)
    test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE, shuffle=True)


    # %%
    model = Autoencoder()
    model.to(device=DEVICE)

    # Print the number of parameters in the model
    print(sum(param.numel() for param in model.parameters()) / 1e6, 'M parameters')
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    if args.from_checkpoint:
        state_dict = torch.load(args.checkpoint_path)
        model.load_state_dict(state_dict)

    if args.mode == 'train':

        if args.model_dir is None:
            Exception("Model Directory is not defined for saving")

        os.makedirs(model_dir, exist_ok=True)


        train_dataset = datasets.MNIST(root='./data/mnist-train', train=True, download=True, transform=transform)
        train_loader  = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True)
        steps = len(train_loader) * EPOCHS
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps)
        progress_bar = tqdm(range(steps))

        for epoch in range(EPOCHS):
            total_loss = 0
            for step, batch in enumerate(train_loader):
                x, y = batch
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                # evaluate the loss
                img, loss = model.forward(x = x)
                total_loss += loss.item()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                progress_bar.update(1)

            tqdm.write(f"Epoch: {epoch}, Train loss {total_loss / (step + 1)}")

            # Test function
            val_losses = []
            for step, batch in enumerate(test_loader):
            
                x, y = batch
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                img, loss = model.forward(x)
                val_losses.append(loss.item())

            tqdm.write(f"Epoch: {epoch}, Val loss: {torch.tensor(val_losses).mean()}")  

        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
        print('Model Saved!')

    if args.mode == 'test':
        num_samples = test_dataset.__len__()
        os.makedirs('images/autoencoder/', exist_ok=True)
        indices = torch.randint(
            low = 0,
            high = num_samples,
            size = (args.num_test_images,)
        )
        for idx in indices:
            image, target = test_dataset.__getitem__(index = idx)
            save_image(image, f'images/autoencoder/{idx}_{target}_true.png')
            image = torch.unsqueeze(image, dim = 0).to(device=DEVICE)
            recon_img, loss = model.forward(image)
            print(loss)
            save_image(recon_img, f'images/autoencoder/{idx}_{target}.png')

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
    help='Batch Size',
    default=1e-5
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
    help='Mode: ["train", "test"]',
    required=True
)
parser.add_argument(
    '--num_test_images',
    type=int,
    help='If test mode, how many images to test on'
)

args = parser.parse_args()
main(args=args)