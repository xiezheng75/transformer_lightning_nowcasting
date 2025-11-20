"""
Data loading and preprocessing utilities for lightning nowcasting.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class LightningDataset(Dataset):
    """Dataset for lightning nowcasting time series data."""
    
    def __init__(self, data, seq_len=24, pred_len=12):
        """
        Initialize the dataset.
        
        Args:
            data: Input data array of shape (n_samples, n_features)
            seq_len: Length of input sequence
            pred_len: Length of prediction sequence
        """
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Normalize data
        self.mean = self.data.mean()
        self.std = self.data.std()
        self.data = (self.data - self.mean) / (self.std + 1e-8)
    
    def __len__(self):
        """Return the number of samples."""
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            src: Source sequence (seq_len, n_features)
            tgt: Target sequence for teacher forcing (pred_len, n_features)
            tgt_y: Target sequence for prediction (pred_len, n_features)
        """
        # Source sequence
        src = self.data[idx:idx + self.seq_len]
        
        # Target sequence (for teacher forcing, starts with last value of src)
        tgt_start = self.data[idx + self.seq_len - 1:idx + self.seq_len + self.pred_len - 1]
        
        # Target output (actual values to predict)
        tgt_y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        
        return src, tgt_start, tgt_y
    
    def denormalize(self, data):
        """Denormalize data back to original scale."""
        return data * self.std + self.mean


def create_synthetic_lightning_data(n_samples=10000, n_features=1, seed=42):
    """
    Create synthetic lightning data for demonstration purposes.
    
    This generates time series data with periodic patterns and noise
    to simulate lightning occurrence patterns.
    
    Args:
        n_samples: Number of time steps to generate
        n_features: Number of features (default 1 for lightning intensity)
        seed: Random seed for reproducibility
        
    Returns:
        data: Synthetic lightning data array
    """
    np.random.seed(seed)
    
    # Create time array
    t = np.linspace(0, 100, n_samples)
    
    # Generate data with multiple components
    # 1. Seasonal pattern (daily cycle)
    seasonal = np.sin(2 * np.pi * t / 24) * 0.3
    
    # 2. Trend component
    trend = 0.0001 * t
    
    # 3. Random noise
    noise = np.random.normal(0, 0.2, n_samples)
    
    # 4. Occasional bursts (simulating storm activity)
    bursts = np.zeros(n_samples)
    burst_locations = np.random.choice(n_samples, size=n_samples // 100, replace=False)
    for loc in burst_locations:
        if loc + 10 < n_samples:
            bursts[loc:loc + 10] = np.random.uniform(0.5, 1.5, 10)
    
    # Combine all components
    data = seasonal + trend + noise + bursts
    
    # Ensure non-negative (lightning intensity can't be negative)
    data = np.maximum(data, 0)
    
    # Reshape to (n_samples, n_features)
    if n_features > 1:
        # For multiple features, create variations
        data = np.column_stack([data * (1 + 0.1 * i) for i in range(n_features)])
    else:
        data = data.reshape(-1, 1)
    
    return data


def get_data_loaders(data, seq_len=24, pred_len=12, batch_size=32, 
                     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data: Input data array
        seq_len: Length of input sequence
        pred_len: Length of prediction sequence
        batch_size: Batch size for data loaders
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train, val, and test ratios must sum to 1"
    
    # Calculate split indices
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Split data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Create datasets
    train_dataset = LightningDataset(train_data, seq_len, pred_len)
    val_dataset = LightningDataset(val_data, seq_len, pred_len)
    test_dataset = LightningDataset(test_data, seq_len, pred_len)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset.mean, train_dataset.std
