import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the weights
weights = torch.load("C:\\Users\\finnc\\Documents\\RTES-2025\\autoencoder_can\\out\\2016-chevrolet-silverado\\fe-2_b-256_input-size-192_epochs-100_lr-0_001.pth")

# Print all layer names and shapes
print("Model weights:")
for name, param in weights.items():
    print(f"{name}: {param.shape}")

# Print statistics for each layer
print("\nWeight statistics:")
for name, param in weights.items():
    print(f"\n{name}:")
    print(f"  Mean: {param.mean().item():.4f}")
    print(f"  Std: {param.std().item():.4f}")
    print(f"  Min: {param.min().item():.4f}")
    print(f"  Max: {param.max().item():.4f}")