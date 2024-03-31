import torch
import torch.nn as nn
from classes.VIT import (
    PatchEmbeddings,
    Block,
    Upscale,
    OverlappingPatchExpander,
    OverlappingPatchMerger,
    OverlappingPatchEmbedding,
    OverlappingPatchUnembedding,
    PoolDownsample,
    Upsample,
)
from classes.VectorQuantizer import VectorQuantizer, VectorQuantizerEMA
from classes.Swin import MultiSwinBlock, PatchExpand, PatchMerge, res_scaler
from utils.logger import Logger
from utils.get_recons import get_recons
import math
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import sys


DEVICE = "cuda"
model_name = "vit+overlapping_patchers_celeba_64"
model_dir = os.path.join(os.getcwd(), "models", model_name)
os.makedirs(name=model_dir, exist_ok=True)
logger = Logger(os.path.join(model_dir, "log.txt"))
epochs = 100
batch_size = 256
is_training = True
pre_run_test = False


def overlapping_res_down(input_dim: int, kernel_size, stride: int):
    return math.floor((input_dim - kernel_size) / stride) + 1


# Uses PatchMerge and PatchExpand
class SwinViT_Patcher_Model(nn.Module):
    def __init__(
        self,
        dim=128,
        input_resolution=(64, 64),
        patch_size=4,
        num_channels=3,
        num_codebook_embeddings=1024,
        codebook_dim=32,
    ):
        super().__init__()

        encoder_layers = 1
        self.patch_embedding = PatchEmbeddings(num_channels, dim, patch_size)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        res = res_scaler(input_resolution, 1 / patch_size)
        self.init_patch_res = res

        for _ in range(encoder_layers):
            self.encoder.append(Block(dim, 4))
            self.encoder.append(
                MultiSwinBlock(dim, res, 2, 4, 4, patch_layer=PatchMerge)
            )

            dim = dim * 2
            res = res_scaler(res, 0.5)

        self.pre_quant = nn.Linear(dim, codebook_dim)
        self.vq = VectorQuantizerEMA(num_codebook_embeddings, codebook_dim, 0.5, 0.99)
        self.post_quant = nn.Linear(codebook_dim, dim)

        decoder_layers = 1

        for _ in range(decoder_layers):
            self.decoder.append(Block(dim, 4))
            self.decoder.append(
                MultiSwinBlock(dim, res, 2, 4, 4, patch_layer=PatchExpand)
            )

            dim = dim // 2
            res = res_scaler(res, 2)

        self.upscale = Upscale(num_channels, dim, patch_size)

    def forward(self, img: torch.Tensor):
        x = self.patch_embedding.forward(img)
        for layer in self.encoder:
            x = layer.forward(x)
            # print("Encoder", x.shape)

        B, C, D = x.shape
        num_patch_sqrt = math.sqrt(C).__int__()
        x = x.transpose(-2, -1).view(B, D, num_patch_sqrt, num_patch_sqrt)
        z_q, indices, loss = self.vq.forward(x)  # Vector Quantizer
        z_q = z_q.view(B, D, C).transpose(-2, -1)

        for layer in self.decoder:
            z_q = layer.forward(z_q)
            # print("Decoder:", z_q.shape)

        z_q = self.upscale.forward(z_q, *self.init_patch_res)
        return z_q, indices, loss


# Uses PatchMerge and PatchExpand
class ViT_Patcher_Model(nn.Module):
    def __init__(
        self,
        dim=128,
        input_resolution: tuple[int] = (64, 64),
        patch_size=4,
        num_channels=3,
        num_codebook_embeddings=1024,
        codebook_dim=32,
    ):
        super().__init__()

        encoder_layers = 2
        self.patch_embedding = PatchEmbeddings(num_channels, dim, patch_size)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        res = res_scaler(input_resolution, 1 / patch_size)
        self.init_patch_res = res

        for _ in range(encoder_layers):
            self.encoder.append(Block(dim, 4))
            self.encoder.append(Block(dim, 4))
            self.encoder.append(PatchMerge(res, dim, dim * 2, 2))

            dim = dim * 2
            res = res_scaler(res, 0.5)

        self.pre_quant = nn.Linear(dim, codebook_dim)
        self.vq = VectorQuantizerEMA(num_codebook_embeddings, codebook_dim, 0.5, 0.99)
        self.post_quant = nn.Linear(codebook_dim, dim)

        decoder_layers = 2

        for _ in range(decoder_layers):
            self.decoder.append(Block(dim, 4))
            self.decoder.append(Block(dim, 4))
            self.decoder.append(PatchExpand(res, dim, dim // 2, 2))

            dim = dim // 2
            res = res_scaler(res, 2)

        self.upscale = Upscale(num_channels, dim, patch_size)

    def forward(self, img: torch.Tensor):
        x = self.patch_embedding.forward(img)
        for idx, layer in enumerate(self.encoder):
            x = layer.forward(x)
            # print("Encoder Layer", idx, ":", x.shape)

        B, C, D = x.shape
        num_patch_sqrt = math.sqrt(C).__int__()
        x = x.transpose(-2, -1).view(B, D, num_patch_sqrt, num_patch_sqrt)
        z_q, indices, loss = self.vq.forward(x)  # Vector Quantizer
        z_q = z_q.view(B, D, C).transpose(-2, -1)

        for idx, layer in enumerate(self.decoder):
            z_q = layer.forward(z_q)
            # print("Decoder Layer", idx, ":", z_q.shape)

        z_q = self.upscale.forward(z_q, *self.init_patch_res)
        return z_q, indices, loss


# Uses PoolDownsample and Upsample
class ViT_PoolDownsample_Upsample_Model(nn.Module):
    def __init__(
        self,
        dim=128,
        input_resolution: tuple[int] = (64, 64),
        patch_size=4,
        num_channels=3,
        num_codebook_embeddings=1024,
        codebook_dim=32,
    ):
        super().__init__()

        encoder_layers = 2
        self.patch_embedding = PatchEmbeddings(num_channels, dim, patch_size)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        res = res_scaler(input_resolution, 1 / patch_size)
        self.init_patch_res = res

        for _ in range(encoder_layers):
            self.encoder.append(Block(dim, 4))
            self.encoder.append(Block(dim, 4))
            self.encoder.append(PoolDownsample(res))
            dim = dim * 2
            res = res_scaler(res, 0.5)

        self.pre_quant = nn.Linear(dim, codebook_dim)
        self.vq = VectorQuantizerEMA(num_codebook_embeddings, codebook_dim, 0.5, 0.99)
        self.post_quant = nn.Linear(codebook_dim, dim)

        decoder_layers = 2

        for _ in range(decoder_layers):
            self.decoder.append(Block(dim, 4))
            self.decoder.append(Block(dim, 4))
            self.decoder.append(Upsample(res, dim))

            dim = dim // 2
            res = res_scaler(res, 2)

        self.upscale = Upscale(num_channels, dim, patch_size)

    def forward(self, img: torch.Tensor):
        x = self.patch_embedding.forward(img)
        for idx, layer in enumerate(self.encoder):
            x = layer.forward(x)
            # print("Encoder Layer", idx, ":", x.shape)

        B, C, D = x.shape
        num_patch_sqrt = math.sqrt(C).__int__()
        x = x.transpose(-2, -1).view(B, D, num_patch_sqrt, num_patch_sqrt)
        z_q, indices, loss = self.vq.forward(x)  # Vector Quantizer
        z_q = z_q.view(B, D, C).transpose(-2, -1)

        for idx, layer in enumerate(self.decoder):
            z_q = layer.forward(z_q)
            # print("Decoder Layer", idx, ":", z_q.shape)

        z_q = self.upscale.forward(z_q, *self.init_patch_res)
        return z_q, indices, loss


# Uses overlapping patch embeddings as well as down and upsamples. Takes too long to train
class OverlappingViTAE(nn.Module):
    def __init__(
        self,
        dim=128,
        input_resolution: tuple[int] = (64, 64),
        patch_size=4,
        num_channels=3,
        num_codebook_embeddings=1024,
        codebook_dim=32,
    ):
        super().__init__()

        encoder_layers = 2
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.patch_size = patch_size

        self.res_list = []

        self.patch_embedding = OverlappingPatchEmbedding(
            input_resolution[0], num_channels, dim, patch_size
        )

        self.patch_unembedding = OverlappingPatchUnembedding(
            input_resolution[0], num_channels, dim, patch_size
        )

        res = overlapping_res_down(input_resolution[0], patch_size, patch_size // 2)
        self.res_list.append(res)

        for _ in range(encoder_layers):
            self.encoder.append(Block(dim, 4))
            self.encoder.append(Block(dim, 4))
            self.encoder.append(OverlappingPatchMerger(res, dim, dim * 2, patch_size))

            dim = dim * 2
            res = overlapping_res_down(res, patch_size, patch_size // 2)
            self.res_list.append(res)

        print(self.encoder)
        print(self.res_list)

        self.pre_quant = nn.Linear(dim, codebook_dim)
        self.vq = VectorQuantizerEMA(num_codebook_embeddings, codebook_dim, 0.5, 0.99)
        self.post_quant = nn.Linear(codebook_dim, dim)

        decoder_layers = 2
        self.res_list.pop()
        self.res_list.reverse()

        print(self.res_list)

        for _ in range(decoder_layers):
            self.decoder.append(
                OverlappingPatchExpander(dim, dim // 2, patch_size, self.res_list[_])
            )
            dim = dim // 2
            self.decoder.append(Block(dim, 4))
            self.decoder.append(Block(dim, 4))

        print(self.decoder)

    def forward(self, img: torch.Tensor):
        x = self.patch_embedding.forward(img)
        for idx, layer in enumerate(self.encoder):
            x = layer.forward(x)
            # print("Encoder Layer", idx, ":", x.shape)

        B, C, D = x.shape
        num_patch_sqrt = math.sqrt(C).__int__()
        x = x.transpose(-2, -1).view(B, D, num_patch_sqrt, num_patch_sqrt)
        z_q, indices, loss = self.vq.forward(x)  # Vector Quantizer
        z_q = z_q.view(B, D, C).transpose(-2, -1)

        for idx, layer in enumerate(self.decoder):
            z_q = layer.forward(z_q)
            # print("Decoder Layer", idx, ":", z_q.shape)

        z_q = self.patch_unembedding.forward(z_q)
        return z_q, indices, loss


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.CenterCrop(size=(178, 178)),
        transforms.Resize(size=(64, 64), antialias=True),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

model = ViT_PoolDownsample_Upsample_Model().to(DEVICE)

print(sum(param.numel() for param in model.parameters()) / 1e6, "M parameters")

if pre_run_test:
    model.forward(torch.randn(size=(1, 3, 64, 64), device=DEVICE))
    sys.exit(0)

if is_training == True:
    train_dataset = datasets.CelebA(
        root="./data", split="train", download=True, transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    optim = torch.optim.Adam(model.parameters())

    progress_bar = tqdm(range(epochs * len(train_loader)))

    for epoch in range(epochs):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            x, y = batch
            x = x.to(DEVICE)

            # evaluate the loss
            recon, codebook_indices, q_loss = model.forward(x)
            recon_loss = nn.functional.mse_loss(x, recon)
            loss: torch.Tensor = q_loss + recon_loss
            total_loss += loss.detach().item()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            progress_bar.update(1)

        train_loss_log = f"Epoch: {epoch}, Train loss {total_loss / (step + 1)}"
        logger.log(train_loss_log)

    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))

if is_training == False:
    test_dataset = datasets.CelebA(
        root="./data", split="test", download=True, transform=transform
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=True)

    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))

    x, y = next(iter(test_loader))
    x = x[:16].to(DEVICE)
    get_recons(
        model=model,
        x=x,
        model_name=model_name,
        std=0.5,
        mean=0.5,
        with_vq=True,
    )
