# Transformer Lightning Nowcasting

A transformer-based deep learning model for lightning nowcasting. This repository implements a state-of-the-art transformer architecture to predict future lightning occurrences based on historical time series data.

## Features

- **Transformer Architecture**: Leverages self-attention mechanisms for temporal pattern recognition
- **Flexible Configuration**: Easy-to-use YAML configuration files
- **Synthetic Data Generation**: Built-in synthetic data generator for testing and demonstrations
- **Complete Pipeline**: End-to-end workflow from data loading to model inference
- **Visualization Tools**: Utilities for plotting predictions and analyzing results
- **Checkpointing**: Save and load model checkpoints during training
- **TensorBoard Support**: Monitor training progress with TensorBoard

## Installation

1. Clone the repository:
```bash
git clone https://github.com/xiezheng75/transformer_lightning_nowcasting.git
cd transformer_lightning_nowcasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the demo script to see the model in action:

```bash
python demo.py
```

This will:
- Generate synthetic lightning data
- Train a small transformer model (5 epochs for demo)
- Evaluate the model on test data
- Create visualization plots in the `outputs/` directory

## Project Structure

```
transformer_lightning_nowcasting/
├── models/                 # Model architectures
│   ├── __init__.py
│   └── transformer_model.py  # Transformer model implementation
├── data/                   # Data loading utilities
│   ├── __init__.py
│   └── data_loader.py      # Dataset and data loader classes
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── utils.py            # Helper functions for training/evaluation
├── configs/                # Configuration files
│   └── config.yaml         # Default configuration
├── train.py                # Training script
├── inference.py            # Inference script
├── demo.py                 # Quick demonstration script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Usage

### Training

Train the model using the default configuration:

```bash
python train.py
```

Or specify a custom configuration file:

```bash
python train.py --config path/to/config.yaml
```

Training outputs:
- Model checkpoints saved in `checkpoints/`
- TensorBoard logs saved in `logs/`
- Best model saved as `checkpoints/best_model.pt`

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs
```

### Inference

Run inference on test data using a trained model:

```bash
python inference.py
```

Or specify custom paths:

```bash
python inference.py --config path/to/config.yaml --checkpoint path/to/checkpoint.pt
```

Inference outputs:
- Prediction plots saved in `outputs/`
- Predictions saved as `outputs/predictions.npz`

### Configuration

Edit `configs/config.yaml` to customize model and training parameters:

**Model Parameters:**
- `d_model`: Dimension of the model (default: 256)
- `nhead`: Number of attention heads (default: 8)
- `num_encoder_layers`: Number of encoder layers (default: 6)
- `num_decoder_layers`: Number of decoder layers (default: 6)
- `dim_feedforward`: Dimension of feedforward network (default: 1024)
- `dropout`: Dropout rate (default: 0.1)

**Data Parameters:**
- `seq_len`: Input sequence length (default: 24)
- `pred_len`: Prediction length (default: 12)
- `n_samples`: Number of samples for synthetic data (default: 10000)

**Training Parameters:**
- `batch_size`: Batch size (default: 32)
- `num_epochs`: Number of training epochs (default: 50)
- `learning_rate`: Learning rate (default: 0.0001)

## Model Architecture

The model uses a standard transformer architecture with:

1. **Input Embedding**: Linear projection of input features to model dimension
2. **Positional Encoding**: Sinusoidal positional encodings for temporal information
3. **Transformer Encoder-Decoder**: Multi-head self-attention and feedforward layers
4. **Output Layer**: Linear projection to output dimension

The model is trained using a combined loss function (MSE + MAE) and supports auto-regressive prediction for multi-step forecasting.

## Data Format

The model expects time series data in the shape `(n_samples, n_features)` where:
- `n_samples`: Number of time steps
- `n_features`: Number of features (e.g., lightning intensity)

For custom data, implement a custom data loader following the structure in `data/data_loader.py`.

## Example Output

After running the demo or training, you'll see:
- Training/validation loss curves
- Performance metrics (MSE, RMSE, MAE, R²)
- Visualization comparing predictions vs ground truth

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- scikit-learn >= 1.3.0
- tqdm >= 4.65.0
- PyYAML >= 6.0
- TensorBoard >= 2.13.0

## Citation

If you use this code in your research, please cite:

```bibtex
@software{transformer_lightning_nowcasting,
  title = {Transformer Lightning Nowcasting},
  author = {xiezheng75},
  year = {2025},
  url = {https://github.com/xiezheng75/transformer_lightning_nowcasting}
}
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This implementation is based on the transformer architecture introduced in "Attention Is All You Need" (Vaswani et al., 2017) and adapted for time series forecasting in the context of lightning nowcasting.
