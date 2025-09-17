import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=2, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, hidden)
        self.fc_out = nn.Linear(hidden, input_dim)
        self.relu = nn.ReLU()
    def encode(self, x):
        h = self.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        h = self.relu(self.fc_decode(z))
        return self.fc_out(h)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
