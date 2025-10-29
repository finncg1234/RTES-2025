import torch
from torch import nn

class AE(nn.Module):
    def __init__(self, input_dim, use_bfw=True, latent_dim=32, bfw_hidden_dim=16):
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        self.bfw_hidden_dim = bfw_hidden_dim
        self.use_bfw = use_bfw
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # BFW: 2-layer network to compute w from z
        self.W1 = nn.Linear(latent_dim, bfw_hidden_dim)
        self.W2 = nn.Linear(bfw_hidden_dim, latent_dim)

        self.b1 = nn.Parameter(torch.rand(bfw_hidden_dim))
        self.b2 = nn.Parameter(torch.rand(latent_dim))

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encode to latent representation z
        z = self.encoder(x)

        if self.use_bfw:
            inner_bfw = nn.functional.relu(self.W1(z) + self.b1)
            outer_bfw = self.W2(inner_bfw) + self.b2
            batch_mean = outer_bfw.mean(dim=0, keepdim=True)
            w = nn.functional.softmax(batch_mean, dim=1)
            z = z * w
        
        decoded = self.decoder(z)
        return decoded