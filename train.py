"""
Training script for transformer lightning nowcasting model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm

from models import TransformerLightningNowcasting, LightningNowcastingLoss
from data import create_synthetic_lightning_data, get_data_loaders
from utils import (
    load_config, set_seed, save_checkpoint, 
    calculate_metrics, print_metrics
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for src, tgt, tgt_y in pbar:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_y = tgt_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt)
        
        # Calculate loss
        loss = criterion(output, tgt_y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for src, tgt, tgt_y in tqdm(val_loader, desc='[Validation]'):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_y = tgt_y.to(device)
            
            # Forward pass
            output = model(src, tgt)
            
            # Calculate loss
            loss = criterion(output, tgt_y)
            total_loss += loss.item()
            
            # Store predictions and targets
            all_predictions.append(output.cpu())
            all_targets.append(tgt_y.cpu())
    
    avg_loss = total_loss / len(val_loader)
    
    # Calculate metrics
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(predictions, targets)
    
    return avg_loss, metrics


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['seed'])
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic data
    print("Creating synthetic lightning data...")
    data = create_synthetic_lightning_data(
        n_samples=config['data']['n_samples'],
        n_features=config['data']['n_features'],
        seed=config['seed']
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, mean, std = get_data_loaders(
        data,
        seq_len=config['data']['seq_len'],
        pred_len=config['data']['pred_len'],
        batch_size=config['training']['batch_size'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
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
    
    # Print model size
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Create loss function
    criterion = LightningNowcastingLoss(alpha=config['training']['loss_alpha'])
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['optimizer']['betas']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['scheduler']['step_size'],
        gamma=config['scheduler']['gamma']
    )
    
    # Create tensorboard writer
    writer = SummaryWriter(config['log_dir'])
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'Metrics/{key}', value, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print_metrics(val_metrics, prefix='  Val ')
        
        # Save checkpoint
        if epoch % config['save_every'] == 0 or val_loss < best_val_loss:
            checkpoint_path = os.path.join(
                config['save_dir'],
                f'checkpoint_epoch_{epoch}.pt'
            )
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(config['save_dir'], 'best_model.pt')
                save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
                print(f"  New best model saved!")
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final evaluation on test set...")
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")
    print_metrics(test_metrics, prefix='Test ')
    
    writer.close()
    print("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train transformer lightning nowcasting model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    main(args)
