# utils.py
import numpy as np
import matplotlib.pyplot as plt
import os

def ensure_dirs():
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)

def plot_resampled(X, y, X_res=None, y_res=None, title="Dataset", savepath=None):
    """
    This preserves your original helper behavior:
    - It creates its own figure
    - It expects X_res to be [original_data; synthetic_data] and uses len(y) to locate synthetic start index
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0 (majority)', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1 (minority)', alpha=0.9)
    
    if X_res is not None and y_res is not None:
        # synthetic samples start after original len(y)
        mask_new = len(y)
        mask = (y_res[mask_new:] == 1)
        if mask.sum() > 0:
            plt.scatter(X_res[mask_new:][mask, 0],
                        X_res[mask_new:][mask, 1],
                        c='green', marker='x', label='Synthetic Class 1')
    plt.title(title)
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=150)
    plt.show()

def save_side_by_side(figures, titles, outpath, figsize=(18,6)):
    """
    figures: list of tuples (X, y, X_res, y_res) or (X, y, None, None) for original
    titles: list of titles
    """
    import matplotlib.pyplot as plt
    n = len(figures)
    cols = n
    fig, axes = plt.subplots(1, cols, figsize=figsize)
    if cols == 1:
        axes = [axes]
    for ax, (X, y, X_res, y_res), title in zip(axes, figures, titles):
        ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.5, label='Class 0')
        ax.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1 (orig)')
        if X_res is not None and y_res is not None:
            mask_new = len(y)
            mask = (y_res[mask_new:] == 1)
            if mask.sum() > 0:
                ax.scatter(X_res[mask_new:][mask,0], X_res[mask_new:][mask,1], c='green', marker='x', label='Synthetic Class 1')
        ax.set_title(title)
        ax.legend()
    plt.suptitle("Comparison of Oversampling Techniques")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.show()
