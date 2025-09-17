# main.py
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
import torch
from utils import ensure_dirs, plot_resampled, save_side_by_side
from resampling_methods import random_oversample, smote_resample, adasyn_resample
from train_autoencoder import run_autoencoder
from train_vae import run_vae
from train_gan import run_gan

def generate_data(seed=42):
    X, y = make_classification(
        n_samples=1000, n_features=2, 
        n_informative=2, n_redundant=0, n_repeated=0,
        n_classes=2, weights=[0.95, 0.05], 
        n_clusters_per_class=1, random_state=seed
    )
    return X, y

def main():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    X, y = generate_data()
    print("Original distribution:", Counter(y))
    # Save original figure (uses original plot_resampled behavior)
    plot_resampled(X, y, X, y, title="Original", savepath="plots/original.png")
    # Classical resampling
    X_ros, y_ros = random_oversample(X, y)
    plot_resampled(X, y, X_ros, y_ros, title="Random Oversampling", savepath="plots/ros.png")
    X_smote, y_smote = smote_resample(X, y)
    plot_resampled(X, y, X_smote, y_smote, title="SMOTE", savepath="plots/smote.png")
    X_adasyn, y_adasyn = adasyn_resample(X, y)
    plot_resampled(X, y, X_adasyn, y_adasyn, title="ADASYN", savepath="plots/adasyn.png")
    # Autoencoder
    X_auto, y_auto, synth_auto = run_autoencoder(X, y, device=device, epochs=200, noise_scale=0.2, lr=1e-2, verbose=True)
    plot_resampled(X, y, X_auto, y_auto, title="Autoencoder Oversampling", savepath="plots/auto.png")
    # VAE
    X_vae, y_vae, synth_vae = run_vae(X, y, device=device, epochs=150, verbose=True)
    plot_resampled(X, y, X_vae, y_vae, title="VAE Oversampling", savepath="plots/vae.png")
    # GAN
    X_gan, y_gan, synth_gan = run_gan(X, y, device=device, n_epochs=500, verbose=True)
    plot_resampled(X, y, X_gan, y_gan, title="GAN Oversampling", savepath="plots/gan.png")
    # Create a combined side-by-side figure (Original, VAE, GAN) and save
    figures = [
        (X, y, X, y),
        (X, y, X_vae, y_vae),
        (X, y, X_gan, y_gan)
    ]
    titles = ["Original", "VAE Oversampling", "GAN Oversampling"]
    save_side_by_side(figures, titles, outpath="plots/comparison_vae_gan.png", figsize=(18,6))
    # Combined grid with all techniques (2x3)
    figures_all = [
        (X, y, X_ros, y_ros),
        (X, y, X_smote, y_smote),
        (X, y, X_adasyn, y_adasyn),
        (X, y, X_auto, y_auto),
        (X, y, X_vae, y_vae),
        (X, y, X_gan, y_gan),
    ]
    titles_all = ["Random Oversampling", "SMOTE", "ADASYN", "Autoencoder", "VAE", "GAN"]
    # Save 2x3 grid by generating a combined image (split into two rows)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    mask_new = len(y)
    for ax, (Xr, yr, Xres, yres), title in zip(axes.ravel(), figures_all, titles_all):
        ax.scatter(Xr[y==0,0], Xr[y==0,1], c='blue', alpha=0.5, label='Class 0')
        ax.scatter(Xr[y==1,0], Xr[y==1,1], c='red', label='Class 1 (orig)')
        if Xres is not None and yres is not None:
            mask = (yres[mask_new:] == 1)
            if mask.sum() > 0:
                ax.scatter(Xres[mask_new:][mask,0], Xres[mask_new:][mask,1], c='green', marker='x', label='Synthetic Class 1')
        ax.set_title(title)
        ax.legend()
    plt.suptitle("All Oversampling Methods")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("plots/all_methods_grid.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Plots saved in ./plots/")

if __name__ == "__main__":
    main()
