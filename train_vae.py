# train_vae.py
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from models.vae import VAE

def vae_loss(recon, x, mu, logvar, kld_scale=1e-3):
    recon_loss = nn.MSELoss()(recon, x)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld / x.size(0)
    return recon_loss + kld_scale * kld

def run_vae(X, y, device, epochs=150, n_generate=None, verbose=True):
    X_min = X[y==1].astype(np.float32)
    X_min_t = torch.tensor(X_min, dtype=torch.float32).to(device)
    latent_dim = 2
    vae = VAE(input_dim=2, latent_dim=latent_dim).to(device)
    opt = optim.Adam(vae.parameters(), lr=1e-3)
    for epoch in range(epochs):
        vae.train()
        opt.zero_grad()
        recon, mu, logvar = vae(X_min_t)
        loss = vae_loss(recon, X_min_t, mu, logvar)
        loss.backward()
        opt.step()
        if verbose and (epoch==0 or (epoch+1) % 50 == 0):
            print(f"[VAE] Epoch {epoch+1}/{epochs} loss: {loss.item():.6f}")
    # get latent means and fit gaussian
    vae.eval()
    with torch.no_grad():
        _, mu_all, _ = vae(X_min_t)
        mu_np = mu_all.cpu().numpy()
    mean_global = mu_np.mean(axis=0)
    cov = np.cov(mu_np.T) + 1e-6 * np.eye(mu_np.shape[1])
    if n_generate is None:
        n_generate = X.shape[0] - X_min.shape[0]
    samples_z = np.random.multivariate_normal(mean_global, cov, size=n_generate).astype(np.float32)
    samples_z_t = torch.tensor(samples_z, dtype=torch.float32).to(device)
    with torch.no_grad():
        synthetic = vae.decode(samples_z_t).cpu().numpy()
    y_synth = np.ones(len(synthetic), dtype=int)
    X_comb = np.vstack([X, synthetic])
    y_comb = np.hstack([y, y_synth])
    return X_comb, y_comb, synthetic
