"""
Pure PyTorch implementation of PCA for GPU acceleration.

This module provides a TorchPCA class that converts sklearn's PCA to pure PyTorch
operations, allowing PCA transforms to run efficiently on GPU without numpy conversions.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA


class TorchPCA:
    """
    Pure PyTorch PCA that matches sklearn's interface but runs on GPU/CPU.

    This class converts a fitted sklearn PCA model to pure PyTorch operations,
    eliminating numpy conversions and enabling GPU acceleration.

    Key advantages:
    - Zero numpy conversions (torch-native)
    - Runs on GPU or CPU
    - Compatible with torch.compile
    - Matches sklearn numerical results (within floating point precision)
    """

    def __init__(self, sklearn_pca: PCA, device='cpu', dtype=torch.float32):
        """
        Convert sklearn PCA to pure torch operations.

        Args:
            sklearn_pca: Fitted sklearn PCA model with components_ and mean_ attributes
            device: Device to place tensors on ('cpu', 'cuda', etc.)
            dtype: Data type for tensors (default: torch.float32)
        """
        self.device = device
        self.dtype = dtype

        # Convert sklearn PCA components to torch tensors
        # components_ shape: (n_components, n_features)
        self.components = torch.tensor(
            sklearn_pca.components_,
            device=device,
            dtype=dtype
        )

        # mean_ shape: (n_features,)
        self.mean = torch.tensor(
            sklearn_pca.mean_,
            device=device,
            dtype=dtype
        )

        # Store dimensions for reference
        self.n_components = sklearn_pca.n_components_
        self.n_features = sklearn_pca.n_features_in_

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform data to PCA space (dimensionality reduction).

        Implements: Z = (X - mean) @ components.T

        This is equivalent to sklearn's PCA.transform() but runs in pure PyTorch.

        Args:
            X: Input tensor of shape (..., n_features)
               Works with any leading dimensions (batch, sequence, etc.)

        Returns:
            Reduced tensor of shape (..., n_components)
        """
        # Center the data: subtract mean
        X_centered = X - self.mean  # Broadcasting handles leading dims

        # Project onto principal components
        # X_centered @ components.T reduces from n_features to n_components
        Z = torch.matmul(X_centered, self.components.T)

        return Z

    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Transform data from PCA space back to original space.

        Implements: X = Z @ components + mean

        This is equivalent to sklearn's PCA.inverse_transform() but runs in pure PyTorch.

        Args:
            Z: Reduced tensor of shape (..., n_components)
               Works with any leading dimensions (batch, sequence, etc.)

        Returns:
            Reconstructed tensor of shape (..., n_features)
        """
        # Project back to original space
        # Z @ components expands from n_components to n_features
        X_reconstructed = torch.matmul(Z, self.components) + self.mean

        return X_reconstructed

    def to(self, device):
        """
        Move PCA tensors to a different device.

        Args:
            device: Target device ('cpu', 'cuda', etc.)

        Returns:
            self for method chaining
        """
        self.components = self.components.to(device)
        self.mean = self.mean.to(device)
        self.device = device
        return self

    def __repr__(self):
        return (f"TorchPCA(n_components={self.n_components}, "
                f"n_features={self.n_features}, "
                f"device={self.device}, dtype={self.dtype})")
