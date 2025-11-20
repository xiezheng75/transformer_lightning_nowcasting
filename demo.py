"""
Demo script to showcase the transformer lightning nowcasting model.
This script demonstrates the complete pipeline: data generation, model training, and inference.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models import TransformerLightningNowcasting, LightningNowcastingLoss
from data import create_synthetic_lightning_data, get_data_loaders
from utils import set_seed, calculate_metrics, print_metrics


def quick_demo():
    """Run a quick demonstration of the model."""
    print("="*60)
    print("Transformer Lightning Nowcasting - Quick Demo")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Data parameters
    seq_len = 24
    pred_len = 12
    n_samples = 2000
    batch_size = 16
    
    # Model parameters
    config = {
        'input_dim': 1,
        'd_model': 128,
        'nhead': 4,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'max_seq_len': 100
    }
    
    # Create synthetic data
    print("\n1. Creating synthetic lightning data...")
    data = create_synthetic_lightning_data(n_samples=n_samples, n_features=1, seed=42)
    print(f"   Data shape: {data.shape}")
    
    # Create data loaders
    print("\n2. Creating data loaders...")
    train_loader, val_loader, test_loader, mean, std = get_data_loaders(
        data, seq_len=seq_len, pred_len=pred_len, batch_size=batch_size,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Create model
    print("\n3. Creating transformer model...")
    model = TransformerLightningNowcasting(**config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {n_params:,}")
    
    # Create loss and optimizer
    criterion = LightningNowcastingLoss(alpha=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training (just a few epochs for demo)
    print("\n4. Training model (quick demo with 5 epochs)...")
    num_epochs = 5
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        for src, tgt, tgt_y in train_loader:
            src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output, tgt_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt, tgt_y in val_loader:
                src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
                output = model(src, tgt)
                loss = criterion(output, tgt_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"   Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Testing
    print("\n5. Testing model on test set...")
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for src, tgt, tgt_y in test_loader:
            src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
            output = model(src, tgt)
            all_predictions.append(output.cpu())
            all_targets.append(tgt_y.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    metrics = calculate_metrics(predictions, targets)
    print_metrics(metrics, prefix='   Test ')
    
    # Visualization
    print("\n6. Creating visualization...")
    
    # Take first sample from test set
    sample_idx = 0
    true_seq = targets[sample_idx, :, 0].numpy()
    pred_seq = predictions[sample_idx, :, 0].numpy()
    
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(len(true_seq))
    plt.plot(time_steps, true_seq, 'b-', label='True Values', linewidth=2, marker='o')
    plt.plot(time_steps, pred_seq, 'r--', label='Predictions', linewidth=2, marker='s')
    plt.xlabel('Time Step (hours ahead)', fontsize=12)
    plt.ylabel('Lightning Intensity (normalized)', fontsize=12)
    plt.title('Lightning Nowcasting: Prediction vs Ground Truth', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / 'demo_predictions.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Visualization saved to {plot_path}")
    plt.close()
    
    # Auto-regressive prediction demo
    print("\n7. Demonstrating auto-regressive prediction...")
    with torch.no_grad():
        # Get one sample
        src, _, _ = next(iter(test_loader))
        src = src[:1].to(device)  # Take first sample only
        
        # Generate future predictions
        future_steps = pred_len
        predictions = model.predict(src, future_steps, device)
        
        print(f"   Generated {future_steps} future time steps")
        print(f"   Prediction shape: {predictions.shape}")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)
    print("\nThe transformer model has been trained and tested.")
    print("Check the 'outputs/' directory for visualizations.")
    print("\nTo train a full model, run: python train.py")
    print("To perform inference, run: python inference.py")


if __name__ == '__main__':
    quick_demo()
