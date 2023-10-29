# %%
import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm
import os
import sys
import argparse
sys.path.append('.')
from classes.VIT import VisionTransformerAutoencoder

def main(args):
    # %%
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Current device: ', device)

    # Model Parameters
    NUM_BLOCKS = 8
    EMBED_DIMS = 256
    PATCH_SIZE = 4
    # # #

    epochs = args.epochs
    print_interval = args.print_interval
    batch_size = args.batch_size
    from_checkpoint = args.from_checkpoint
    checkpoint_path = args.checkpoint_path
    mode = args.mode
    num_test_images = args.num_test_images
    model_dir = args.model_dir

    # %%
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # %%
    testset = torchvision.datasets.CIFAR10(root='./data/cifar-10-test', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    # %%
      
    # %%
    model = VisionTransformerAutoencoder(
        image_size=32,
        patch_size=PATCH_SIZE,
        num_channels=3,
        embed_dim=EMBED_DIMS,
        num_classes=10,
        NUM_BLOCKS=NUM_BLOCKS
    )

    model.to(device=device)
    # Print the number of parameters in the model
    print(sum(param.numel() for param in model.parameters()) / 1e6, 'M parameters')

    if from_checkpoint:
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)

    if mode == 'train':

        if args.model_dir is None:
            Exception("Model Directory is not defined for saving")
            
        os.makedirs(model_dir, exist_ok=True)

        # Train dataset
        trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10-train', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        # Create a PyTorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Training loop

        steps = len(trainloader) * epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps)
        progress_bar = tqdm(range(steps))

        for epoch in range(epochs):
            total_loss = 0
            for step, batch in enumerate(trainloader):
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                # evaluate the loss
                img, loss = model.forward(x = x, targets =y)
                total_loss += loss.item()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                progress_bar.update(1)

            tqdm.write(f"Epoch: {epoch}, Train loss {total_loss / (step + 1)}")

            # Test function
            val_losses = []
            for step, batch in enumerate(testloader):
            
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                img, loss = model.forward(x)
                val_losses.append(loss.item())

            tqdm.write(f"Epoch: {epoch}, Val loss: {torch.tensor(val_losses).mean()}")  

        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
        print('Model Saved!')

    if mode == 'test':
        num_samples = testset.__len__()
        os.makedirs('images/vit/', exist_ok=True)
        indices = torch.randint(
            low = 0,
            high = num_samples,
            size = (num_test_images,)
        )
        for idx in indices:
            image, target = testset.__getitem__(index = idx)
            save_image(image, f'images/vit/{idx}_{target}_true.png')
            image = torch.unsqueeze(image, dim = 0).to(device=device)
            recon_img, loss = model.forward(image)
            print(loss)
            save_image(recon_img, f'images/vit/{idx}_{target}.png')

parser = argparse.ArgumentParser()

parser.add_argument(
    '--batch_size',
    type=int,
    help='Batch Size',
    default=64
)
parser.add_argument(
    '--epochs',
    type=int,
    help='# of Epochs',
    default=50
)
parser.add_argument(
    '--lr',
    type=int,
    help='Batch Size',
    default=1e-5
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
    help='Mode: [train, test]',
    required=True
)
parser.add_argument(
    '--num_test_images',
    type=int,
    help='If test mode, how many images to test on'
)

args = parser.parse_args()
main(args=args)