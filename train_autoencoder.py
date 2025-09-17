# train_autoencoder.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.autoencoder import Autoencoder

def run_autoencoder(X, y, device, epochs=200, noise_scale=0.2, lr=1e-2, verbose=True):
    # X, y are numpy arrays
    X_min = X[y==1].astype(np.float32)
    X_min_t = torch.tensor(X_min, dtype=torch.float32).to(device)
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, _ = model(X_min_t)
        loss = criterion(recon, X_min_t)
        loss.backward()
        optimizer.step()
        if verbose and (epoch==0 or (epoch+1) % 50 == 0):
            print(f"[Autoencoder] Epoch {epoch+1}/{epochs} loss: {loss.item():.6f}")
    # generate synthetic
    with torch.no_grad():
        _, latent = model(X_min_t)
        noise = torch.randn_like(latent) * noise_scale
        synthetic_latent = latent + noise
        synthetic_np = model.decoder(synthetic_latent).cpu().numpy()
    # make labels and return combined arrays
    n_to_generate = X.shape[0] - X_min.shape[0]
    synthetic_np = synthetic_np[:n_to_generate]
    y_synth = np.ones(len(synthetic_np), dtype=int)
    X_comb = np.vstack([X, synthetic_np])
    y_comb = np.hstack([y, y_synth])
    return X_comb, y_comb, synthetic_np
