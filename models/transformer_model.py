"""
Transformer model for lightning nowcasting.
This module implements a transformer-based architecture for predicting
future lightning occurrences based on historical data.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerLightningNowcasting(nn.Module):
    """
    Transformer model for lightning nowcasting.
    
    This model takes a sequence of historical lightning data and predicts
    future lightning occurrences using a transformer architecture.
    """
    
    def __init__(self, 
                 input_dim=1,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 max_seq_len=100):
        """
        Initialize the transformer model.
        
        Args:
            input_dim: Dimension of input features
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super(TransformerLightningNowcasting, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, input_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass of the model.
        
        Args:
            src: Source sequence (batch_size, src_seq_len, input_dim)
            tgt: Target sequence (batch_size, tgt_seq_len, input_dim)
            src_mask: Source mask
            tgt_mask: Target mask
            
        Returns:
            output: Predicted sequence (batch_size, tgt_seq_len, input_dim)
        """
        # Embed inputs
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        tgt = self.input_embedding(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Generate masks if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer forward pass
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        
        # Output projection
        output = self.output_layer(output)
        
        return output
    
    def predict(self, src, future_steps, device='cpu'):
        """
        Generate predictions for future steps.
        
        Args:
            src: Source sequence (batch_size, src_seq_len, input_dim)
            future_steps: Number of future steps to predict
            device: Device to run on
            
        Returns:
            predictions: Predicted sequence (batch_size, future_steps, input_dim)
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            
            # Start with the last value of src as the first target
            tgt = src[:, -1:, :]
            predictions = []
            
            for _ in range(future_steps):
                # Forward pass
                output = self.forward(src, tgt)
                
                # Get the last prediction
                next_pred = output[:, -1:, :]
                predictions.append(next_pred)
                
                # Append to target for next iteration
                tgt = torch.cat([tgt, next_pred], dim=1)
            
            # Concatenate all predictions
            predictions = torch.cat(predictions, dim=1)
            
        return predictions


class LightningNowcastingLoss(nn.Module):
    """Custom loss function for lightning nowcasting."""
    
    def __init__(self, alpha=0.5):
        """
        Initialize the loss function.
        
        Args:
            alpha: Weight for MSE vs MAE (0.5 means equal weight)
        """
        super(LightningNowcastingLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, predictions, targets):
        """
        Calculate combined loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            loss: Combined loss value
        """
        mse_loss = self.mse(predictions, targets)
        mae_loss = self.mae(predictions, targets)
        return self.alpha * mse_loss + (1 - self.alpha) * mae_loss
