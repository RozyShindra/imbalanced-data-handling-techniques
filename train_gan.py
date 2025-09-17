# train_gan.py
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from models.gan import Generator, Discriminator

def run_gan(X, y, device, noise_dim=4, n_epochs=500, batch_size=32, verbose=True):
    X_min = X[y==1].astype(np.float32)
    X_min_t = torch.tensor(X_min, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_min_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    G = Generator(noise_dim=noise_dim).to(device)
    D = Discriminator().to(device)
    opt_G = optim.Adam(G.parameters(), lr=1e-3)
    opt_D = optim.Adam(D.parameters(), lr=1e-3)
    bce = nn.BCELoss()
    real_label = 1.
    fake_label = 0.
    for epoch in range(n_epochs):
        for (real_batch,) in loader:
            # train D
            D.zero_grad()
            real_data = real_batch.to(device)
            b_size = real_data.size(0)
            labels_real = torch.full((b_size,1), real_label, dtype=torch.float32).to(device)
            out_real = D(real_data)
            loss_real = bce(out_real, labels_real)
            z = torch.randn(b_size, noise_dim).to(device)
            fake_data = G(z)
            labels_fake = torch.full((b_size,1), fake_label, dtype=torch.float32).to(device)
            out_fake = D(fake_data.detach())
            loss_fake = bce(out_fake, labels_fake)
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            opt_D.step()
            # train G
            G.zero_grad()
            labels_gen = torch.full((b_size,1), real_label, dtype=torch.float32).to(device)
            out_gen = D(fake_data)
            loss_G = bce(out_gen, labels_gen)
            loss_G.backward()
            opt_G.step()
        if verbose and (epoch==0 or (epoch+1) % 100 == 0):
            print(f"[GAN] Epoch {epoch+1}/{n_epochs}, Loss_D: {loss_D.item():.6f}, Loss_G: {loss_G.item():.6f}")
    # generate synthetic
    n_to_generate = X.shape[0] - X_min.shape[0]
    with torch.no_grad():
        z = torch.randn(n_to_generate, noise_dim).to(device)
        synthetic = G(z).cpu().numpy()
    y_synth = np.ones(len(synthetic), dtype=int)
    X_comb = np.vstack([X, synthetic])
    y_comb = np.hstack([y, y_synth])
    return X_comb, y_comb, synthetic
