import torch
from torch import nn
import torch.nn.functional as func

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, latent_dim: int, beta: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim
        self.beta = beta # Beta is a weighting factor for the "codebook gradient flowing" loss also called the commitment cost

        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings) # Uniform data initialization for the codebook vectors

    def forward(self, z: torch.Tensor):
        # z is B, D, H, W

        z = z.permute(0, 2, 3, 1).contiguous() # Contigous method used for memory arrangement

        # z is B, H, W, D now

        z_flat = z.view(-1, self.latent_dim) # (B * H * W, D)

        dis = torch.sum(z_flat**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2*(torch.matmul(z_flat, self.embedding.weight.t())) # Calculating the distances of the encoded vectors to the codebook vectors
        
        min_encoding_indices = torch.argmin(dis, dim=1) # Getting the indices of the min distant vector for each input
        z_q = self.embedding.forward(min_encoding_indices).view(z.shape) # Getting the codebook vectors

        e_loss = func.mse_loss(z_q.detach(), z)
        q_loss = func.mse_loss(z_q, z.detach())
        loss = q_loss + self.beta * e_loss

        z_q = z + (z_q - z).detach() # For preserving the gradients for backprop

        z_q = z_q.permute(0, 3, 1, 2) # (B, D, H, W)

        return z_q, min_encoding_indices, loss
  
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        self.embedding = nn.Embedding(self.numembeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()
        self.commitment_cost = commitment_cost
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))
        self.ema_w.data.normal_()
        
        self.decay = decay
        self.epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        # print(encodings.shape)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)
            
            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = func.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, encoding_indices