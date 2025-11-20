"""
Inference script for transformer lightning nowcasting model.
"""

import torch
import argparse
import numpy as np
import os

from models import TransformerLightningNowcasting
from data import create_synthetic_lightning_data, LightningDataset
from utils import load_config, load_checkpoint, plot_predictions, calculate_metrics, print_metrics


def predict(model, data, seq_len, pred_len, device='cpu'):
    """
    Make predictions on data.
    
    Args:
        model: Trained model
        data: Input data
        seq_len: Input sequence length
        pred_len: Prediction length
        device: Device to run on
        
    Returns:
        predictions: Model predictions
        true_values: Ground truth values
    """
    model.eval()
    
    # Create dataset
    dataset = LightningDataset(data, seq_len, pred_len)
    
    all_predictions = []
    all_true_values = []
    
    with torch.no_grad():
        # Use the last sequence for prediction
        src, tgt, tgt_y = dataset[-1]
        src = src.unsqueeze(0).to(device)  # Add batch dimension
        
        # Generate predictions
        predictions = model.predict(src, pred_len, device)
        
        # Denormalize
        predictions = dataset.denormalize(predictions)
        tgt_y = dataset.denormalize(tgt_y.unsqueeze(0))
        
        all_predictions.append(predictions.cpu().numpy())
        all_true_values.append(tgt_y.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    true_values = np.concatenate(all_true_values, axis=0)
    
    return predictions, true_values


def main(args):
    """Main inference function."""
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = TransformerLightningNowcasting(
        input_dim=config['model']['input_dim'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        max_seq_len=config['model']['max_seq_len']
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(config['save_dir'], 'best_model.pt')
    
    print(f"Loading checkpoint from {checkpoint_path}")
    load_checkpoint(model, None, checkpoint_path, device)
    
    # Create test data
    print("Creating test data...")
    data = create_synthetic_lightning_data(
        n_samples=config['data']['n_samples'],
        n_features=config['data']['n_features'],
        seed=config['seed']
    )
    
    # Use last portion for testing
    test_start = int(len(data) * (config['data']['train_ratio'] + config['data']['val_ratio']))
    test_data = data[test_start:]
    
    # Make predictions
    print("Generating predictions...")
    predictions, true_values = predict(
        model,
        test_data,
        config['data']['seq_len'],
        config['data']['pred_len'],
        device
    )
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, true_values)
    print_metrics(metrics, prefix='Test ')
    
    # Plot predictions
    print("Plotting predictions...")
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, 'predictions.png')
    plot_predictions(
        true_values[0, :, 0],  # First batch, all time steps, first feature
        predictions[0, :, 0],
        save_path=plot_path
    )
    
    # Save predictions
    output_file = os.path.join(output_dir, 'predictions.npz')
    np.savez(output_file, predictions=predictions, true_values=true_values)
    print(f"Predictions saved to {output_file}")
    
    print("\nInference completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for transformer lightning nowcasting model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    args = parser.parse_args()
    
    main(args)
