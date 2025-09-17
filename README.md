# 🧩 Imbalanced Data Handling Techniques

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository demonstrates different **imbalanced data handling techniques** ranging from **traditional resampling** methods to **deep learning–based generative models**.  
It uses a **toy 2D dataset** for visualization so that you can directly observe the effect of each technique.

## Techniques Implemented

### Traditional Methods
- Random Oversampling (ROS)
- Random Undersampling (RUS)
- SMOTE
- ADASYN

### Generative Models
- Autoencoder Oversampling
- Variational Autoencoder (VAE)
- Generative Adversarial Network (GAN)

### Results
<img width="1789" height="955" alt="image" src="https://github.com/user-attachments/assets/aa215681-2060-495e-82fb-128f9b4c6e06" />



## Quickstart

```bash
git clone https://github.com/RozyShindra/imbalanced-data-handling-techniques.git
cd imbalanced-data-handling-techniques
python -m pip install -r requirements.txt
python main.py
