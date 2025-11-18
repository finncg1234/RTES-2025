import torch
from torch import nn

class AE(nn.Module):
    def __init__(self, input_dim, use_bfw=True, latent_dim=16, bfw_hidden_dim=16):
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        self.bfw_hidden_dim = bfw_hidden_dim
        self.use_bfw = use_bfw
        self.input_dim = input_dim

        if input_dim >= 500:
            encoder_dims = [input_dim, 128, 64, 32, latent_dim]
        elif input_dim >= 200:
            encoder_dims = [input_dim, 64, 32, latent_dim]
        else:
            encoder_dims = [input_dim, 32, latent_dim]
        
        layers = []
        for i in range(len(encoder_dims) - 1):
            layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
        if i < len(encoder_dims) - 2:  # No activation on last layer
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.2))

        self.encoder = nn.Sequential(*layers)
        
        # BFW: 2-layer network to compute w from z
        self.W1 = nn.Linear(latent_dim, bfw_hidden_dim)
        self.W2 = nn.Linear(bfw_hidden_dim, latent_dim)

        self.b1 = nn.Parameter(torch.zeros(bfw_hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(latent_dim))

        # Build decoder (reverse of encoder)
        decoder_dims = encoder_dims[::-1]  # Reverse the list
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            if i < len(decoder_dims) - 2:  # No activation/dropout on last layer
                decoder_layers.append(nn.ReLU())
                # decoder_layers.append(nn.Dropout(0.2))
        self.decoder = nn.Sequential(*decoder_layers)
    
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