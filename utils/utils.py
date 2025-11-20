"""
Utility functions for training and evaluation.
"""

import torch
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        path: Path to checkpoint
        device: Device to load onto
        
    Returns:
        epoch: Epoch number from checkpoint
        loss: Loss from checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    return epoch, loss


def plot_predictions(true_values, predictions, save_path=None):
    """
    Plot true values vs predictions.
    
    Args:
        true_values: Ground truth values
        predictions: Model predictions
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    # Convert to numpy if tensors
    if torch.is_tensor(true_values):
        true_values = true_values.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    # Flatten if needed
    if len(true_values.shape) > 1:
        true_values = true_values.flatten()
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()
    
    plt.plot(true_values, label='True Values', linewidth=2, alpha=0.7)
    plt.plot(predictions, label='Predictions', linewidth=2, alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Lightning Intensity')
    plt.title('Lightning Nowcasting: True vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Convert to numpy if tensors
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # Calculate R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return metrics


def print_metrics(metrics, prefix=''):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for printing
    """
    print(f"\n{prefix}Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
