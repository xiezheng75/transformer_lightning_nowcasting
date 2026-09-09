#!/usr/bin/env python3
import os
import sys
import torch

from torch.cuda.amp import GradScaler, autocast
# print(f"Python executable: {sys.executable}")
# print(f"PyTorch version: {torch.__version__ if 'torch' in locals() else 'Not imported yet'}")
# print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
# print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
# print(f"RANK: {os.environ.get('RANK', 'Not set')}")
# print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
import gc
import math
import multiprocessing
# Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues
def init_process():
    """Initialize process settings"""
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import functools
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import signal
import atexit

# `python scripts/test_pytorch_own_data2.py` puts scripts/ on sys.path, not the
# project root, so c4dllightning and scripts.* would be unimportable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Where trained checkpoints live. Defaults to <project root>/models, which is where
# training writes them; override when the weights are kept outside the working tree.
MODELS_DIR = os.environ.get("MSPDVIT_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))

from c4dllightning.features.batch import RadarBatchGenerator, RadarSampleDataset
from c4dllightning.ml.models.models import init_model, compile_model, train_model
from c4dllightning.ml.models.blocks import ConvBlock, ResBlock
from c4dllightning.analysis import evaluation
from c4dllightning.analysis.calibration import calibration_curve_models
from scripts.plots_lightning import plot_examples, calibration_by_loss, plot_multiple_examples, plot_best_examples
from c4dllightning.ml.models.pure_transformer import SpatioTemporalTransformer, MultiPathTransformer
from c4dllightning.ml.models.enhanced_transformer import HybridLightningTransformer  # Add this import
# from c4dllightning.ml.models.models import RNNModel
# # Global variables for DDP
local_rank = -1
world_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
is_distributed = False
model = None
optimizer = None
loss_fn = None
metric_fns = None
batch_gen = None
model_save_dir = None
results_dir = None

# Logging controls
LOG_DEVICE_INFO = False
LOG_GPU_MEMORY = False

# Helper function for distributed printing
def dist_print(*args, **kwargs):
    """Print only from rank 0 in distributed mode, or always in single-process mode"""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
        return

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank == -1 or local_rank == 0:
        print(*args, **kwargs)


def check_ranks_in_step(step, tag=""):
    """Fail fast if the ranks are not all on the same iteration.

    NCCL matches collectives by the order they are issued, not by what they mean.
    Once one rank issues one collective more or less than the others, every later
    collective pairs up with the wrong one and the job deadlocks somewhere
    unrelated, typically only at the end of the epoch and only after the watchdog
    timeout has elapsed. This turns that into an immediate error naming the rank.
    Must be called from every rank at the same iteration.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return

    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    step = float(step)
    t = torch.tensor([step, -step], dtype=torch.float64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    highest, lowest = t[0].item(), -t[1].item()
    if highest != step or lowest != step:
        raise RuntimeError(
            f"Rank {dist.get_rank()} is out of step at {tag}: local step {step:.0f}, "
            f"ranks span {lowest:.0f}..{highest:.0f}"
        )


def save_prediction_results(outputs, targets, batch_idx, save_dir, prefix="valid"):
    """
    Save prediction results as PNG images and numpy arrays.

    Args:
        outputs: Model outputs (probabilities) [B, T, H, W, 1] or [B, T, H, W]
        targets: Ground truth [B, T, H, W, 1] or [B, T, H, W]
        batch_idx: Batch index
        save_dir: Directory to save results
        prefix: Filename prefix
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Ensure numpy
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Squeeze last dim if present
    if outputs.ndim == 5 and outputs.shape[-1] == 1:
        outputs = outputs.squeeze(-1)
    if targets.ndim == 5 and targets.shape[-1] == 1:
        targets = targets.squeeze(-1)

    # Save first sample in batch
    for b in range(min(outputs.shape[0], 1)):  # Only save first sample to avoid too many files
        for t in range(outputs.shape[1]):
            # Save probability map as PNG (Jet colormap for probability)
            plt.figure(figsize=(8, 7))  # 800x700 approx
            # Remove axes and margins
            plt.axis('off')
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            # Plot
            plt.imshow(outputs[b, t], cmap='jet', vmin=0, vmax=1)
            # No colorbar/title for raw dashboard data, but maybe useful for debug.
            # For dashboard use, we want pure data.
            plt.savefig(os.path.join(save_dir, f"{prefix}_batch{batch_idx}_sample{b}_t{t}_pred.png"),
                        bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()

            # Save Ground Truth
            if targets is not None:
                plt.figure(figsize=(8, 7))
                plt.axis('off')
                plt.margins(0, 0)
                plt.imshow(targets[b, t], cmap='gray', vmin=0, vmax=1)
                plt.savefig(os.path.join(save_dir, f"{prefix}_batch{batch_idx}_sample{b}_t{t}_gt.png"),
                            bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()

            # Save raw numpy array for precise data handling
            np.save(os.path.join(save_dir, f"{prefix}_batch{batch_idx}_sample{b}_t{t}_pred.npy"), outputs[b, t])

def cleanup_handler(signum=None, frame=None):
    """Signal handler for cleanup on exit"""
    import traceback

    # Map signal numbers to names
    signal_names = {
        0: "NORMAL_EXIT",
        2: "SIGINT (Ctrl+C)",
        15: "SIGTERM (Termination)",
        9: "SIGKILL (Force kill)"
    }

    signal_name = signal_names.get(signum, f"Unknown signal {signum}")
    dist_print(f"\nReceived signal: {signal_name}")

    # Print stack trace to understand where the signal was received
    if frame is not None:
        dist_print("\nStack trace at signal:")
        traceback.print_stack(frame)

    # Clear matplotlib figures
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except:
        pass

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Destroy distributed process group if initialized
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    # Force garbage collection
    gc.collect()

    # Exit with appropriate code
    exit_code = 0 if signum == 0 else 1
    sys.exit(exit_code)

# Register cleanup handlers
signal.signal(signal.SIGINT, cleanup_handler)
signal.signal(signal.SIGTERM, cleanup_handler)
atexit.register(lambda: cleanup_handler(0, None))

def setup_ddp():
    """Setup distributed training environment"""
    dist_print("Setting up distributed training environment...")

    # Get rank and world size from environment variables set by launcher
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))

    # Check if this is a distributed run
    if local_rank == -1 or world_size == -1:
        if LOG_DEVICE_INFO:
            print("Not running in distributed mode")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, device, False

    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # Verify correct GPU mapping
    actual_gpu_id = torch.cuda.current_device()
    if LOG_DEVICE_INFO:
        dist_print(
            f"Process {local_rank}: PyTorch device cuda:{local_rank} -> Physical GPU "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[local_rank]}"
        )

    # Choose backend based on what's available and working
    backend = 'nccl'  # Use 'gloo' for local multi-GPU, 'nccl' for cluster

    # Backend-specific settings
    if backend == 'gloo':
        os.environ['GLOO_SOCKET_IFNAME'] = 'lo'  # Use loopback interface
    elif backend == 'nccl':
        # CRITICAL FIX: Remove fixed NCCL_COMM_ID - this causes conflicts!
        # os.environ['NCCL_COMM_ID'] = '127.0.0.1:23456'  # REMOVED

        # Debug settings for troubleshooting
        os.environ['NCCL_DEBUG'] = 'INFO'  # Changed to INFO for debugging
        os.environ['NCCL_DEBUG_SUBSYS'] = 'INIT,GRAPH,ENV'

        # Network settings - disable IB if not available
        os.environ['NCCL_IB_DISABLE'] = '0'  # InfiniBand
        os.environ['NCCL_P2P_DISABLE'] = '0'  # Enable P2P for V100s (important!)
        os.environ['NCCL_TREE_THRESHOLD'] = '0'

        # Timeout settings - reduce for faster failure detection
        os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "600"  # 10 minutes
        os.environ['NCCL_TIMEOUT'] = '600'  # 10 minutes - reduced from 7200
        os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
        os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "1"  # Enable monitoring

        # Performance tuning
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
        os.environ['NCCL_SOCKET_IFNAME'] = 'ens12f0'  # Use only primary interface
        os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'  # Reduced from 4
        os.environ['NCCL_SOCKET_NTHREADS'] = '4'
        os.environ['NCCL_MIN_NCHANNELS'] = '4'  # Reduced from 4
        os.environ['NCCL_MAX_NCHANNELS'] = '16'  # V100可以处理更多通道
        os.environ['NCCL_BUFFSIZE'] = '2097152'  # 2MB缓冲区

        # Thread control
        os.environ['OMP_NUM_THREADS'] = '1'  # Prevent thread explosion
        os.environ['MKL_NUM_THREADS'] = '1'



    # Initialize the process group
    try:
        # Try to use the correct init method based on environment
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
        if 'MASTER_PORT' not in os.environ:
            # Use a random port to avoid conflicts
            import random
            os.environ['MASTER_PORT'] = str(random.randint(29500, 29999))

        torch.distributed.init_process_group(
            backend=backend,
            # Lower DDP_TIMEOUT_MIN when debugging a hang: a desynchronised
            # collective otherwise stalls the job for the full timeout.
            timeout=timedelta(minutes=int(os.environ.get("DDP_TIMEOUT_MIN", "60"))),
            init_method='env://',  # Explicitly use environment variables
            world_size=world_size,
            rank=rank
        )
        print(f"Initialized process group: rank {rank}/{world_size} (local_rank: {local_rank})")
        return local_rank, world_size, device, True

    except Exception as e:
        print(f"Failed to initialize process group: {e}")
        print("Falling back to single GPU")
        torch.cuda.set_device(0)
        return 0, 1, torch.device('cuda:0'), False

# 性能监控装饰器
def monitor_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        dist_print(f"\n=== Performance monitoring for {func_name} ===")

        # Before execution
        start_time = time.time()
        dist_print("Before execution:")
        print_gpu_memory()

        # Execute function
        result = func(*args, **kwargs)

        # After execution
        end_time = time.time()
        dist_print("After execution:")
        print_gpu_memory()
        dist_print(f"{func_name} execution time: {end_time - start_time:.2f} seconds")

        return result

    return wrapper

# GPU显存监控函数
def print_gpu_memory():
    if not LOG_GPU_MEMORY:
        return
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
            max_alloc = torch.cuda.max_memory_allocated(i) / 1024 ** 3
            dist_print(f"GPU {i}: Alloc={alloc:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_alloc:.2f}GB")
    else:
        if LOG_DEVICE_INFO and local_rank == 0:
            dist_print("No GPU available")


# 释放GPU内存
def free_gpu_memory():
    """Free GPU memory with safer approach for distributed training"""
    try:
        # Don't synchronize in distributed mode to avoid potential deadlocks
        if not is_distributed:
            torch.cuda.synchronize()

        # Clear Python's garbage first
        gc.collect()

        # Empty CUDA cache
        if torch.cuda.is_available():
            current_allocated = torch.cuda.memory_allocated()
            max_allocated = torch.cuda.max_memory_allocated()

            # Only clear if using more than 90% of peak memory
            if current_allocated > 0.9 * max_allocated:
                torch.cuda.empty_cache()

                # Force collection again only if still high
                if torch.cuda.memory_allocated() > 0.85 * max_allocated:
                    gc.collect()
                    torch.cuda.empty_cache()

    except Exception as e:
        print(f"Warning: Error during memory cleanup: {e}")
        # Continue execution even if cleanup fails

# # 设置路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # repository root (same as PROJECT_ROOT above)
data_dir = os.path.expanduser("~/Weather")  # 更改为用户根目录下的Weather
lightning_csv = os.path.join(data_dir, "Thunder/flash.csv")  # 闪电数据路径

# Define model configurations to train
MODEL_CONFIGS = {
    "radar_baseline": {
        "loss": "weighted_focal_loss",
        "wfc_gamma": 2.0,
        "optimizer": "adamw",
        "lr": 0.000001,
        "weight_decay": 1e-5,
        "dropout": 0.2,
        # "use_class_weight": True,
        "description": "baseline_wfl2",
        "resume_from": os.path.join(project_root, "models_CNNGRU/radar_baseline_wfl_gamma2.0_adamw_lr1e-06_dropout0.2_wd1e-05_baseline_wfl2_checkpoint.pth"),  # 可以设置为模型文件路径来恢复训练
    },
    "radar_wfl_gamma1": {
        "loss": "weighted_focal_loss",
        "wfc_gamma": 1.0,
        "optimizer": "adamw",
        "lr": 0.00001,
        "weight_decay": 1e-5,
        "dropout": 0.2,
        # "use_class_weight": True,
        "description": "wfl_gamma1"
    },
    "radar_bce": {
        "loss": "binary_crossentropy",
        "optimizer": "adamw",
        "lr": 0.00001,
        "weight_decay": 1e-5,
        "dropout": 0.2,
        # "use_class_weight": False,
        "description": "bce"
    },
    "radar_wce": {
        "loss": "weighted_crossentropy",
        "optimizer": "adamw",
        "lr": 0.00001,
        "weight_decay": 1e-5,
        "dropout": 0.2,
        # "use_class_weight": True,
        "description": "wce"
    },
    "radar_iou": {
        "loss": "iou_loss",
        "optimizer": "adamw",
        "lr": 0.00001,
        "weight_decay": 1e-5,
        "dropout": 0.2,
        # "use_class_weight": False,
        "description": "iou"
    },
    "radar_dropout_noweight": {
        "loss": "weighted_focal_loss",
        "wfc_gamma": 2.0,
        "optimizer": "adamw",
        "lr": 0.00001,
        "weight_decay": 1e-5,
        "dropout": 0.3,
        # "use_class_weight": False,
        "description": "dropout_weightdecay_noclassweight"
    },
    "radar_nodropout": {
        "loss": "weighted_focal_loss",
        "wfc_gamma": 2.0,
        "optimizer": "adamw",
        "lr": 0.00001,
        "weight_decay": 0,
        "dropout": 0.0,
        # "use_class_weight": True,
        "description": "nodropout_noweightdecay"
    },
    "radar_higherdropout": {
        "loss": "weighted_focal_loss",
        "wfc_gamma": 2.0,
        "optimizer": "adamw",
        "lr": 0.00001,
        "weight_decay": 1e-4,
        "dropout": 0.5,
        # "use_class_weight": False,
        "description": "highdropout_noclassweight"
    },
    "pure_transformer": {
        "loss": "weighted_focal_loss",
        "wfc_gamma": 1.0,
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "loss_kwargs": {"alpha_multiplier": 1.0, "pm_alpha": 0.0},
        "eval_threshold": 0.5,
        "plot_threshold": 0.3,
        "calibration_file": None,
        "use_calibrated_threshold": False,
        "description": "multipath_vit_p12p50",
        "use_transformer": True,
        "transformer_type": "multipath",
        "patch_size": 12,
        "use_multiscale": False,
        "patch_sizes": [12, 50],
        # Proposed model: HARD uniform fusion w=[0.5,0.5] (unambiguous, no gating
        # parameters). The "strong-regularized dynamic gate converges to uniform"
        # result is reported separately as an explanatory ablation, not as the
        # definition of the main model.
        "fusion_mode": "uniform",
        "gating_entropy_lambda": 0.0,
        "gating_balance_lambda": 0.0,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 4,
        "attn_dropout": 0.1,
        "learnable_pos_emb": True,
        "temporal_encoding": "learned",
        "use_checkpoint": True,
        "use_static_data": False,
        "static_channel_dropout": 0.0,
        "batch_size": 8,
        "only_validate": False,
        "final_model_path": None,
        "val_rich_only": True,
        "max_val_batches": None,
        "val_rich_min_events": 1000,
        "val_rich_time_window": 6,
        "val_compare_full": True,
        "val_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
        "eval_train_rich_only": True,
        "train_rich_min_events": 1000,
        "train_rich_time_window": 6,
        "calibration_batch_size": 2,
        "calibration_max_samples": 200,
        "min_improvement": 0.015,
        # Cap training: loss plateaus by ~25-30 epochs; early stopping (patience 7)
        # usually stops earlier. The old 189-epoch run was unnecessary -- a ~40-epoch
        # budget reaches the same skill (confirmed by the low-epoch run).
        "max_epochs": 40,
        "patience": 7,
    },
    "pure_transformer_singlepatch_cls_train": {
        # Train-from-scratch Single-patch (p=12) CLS-ONLY ViT baseline under the
        # leakage-free split. legacy_prediction_head=True => predictions are decoded
        # from the CLS token only (no patch-wise decoder). Single branch.
        "loss": "weighted_focal_loss",
        "wfc_gamma": 1.0,
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "loss_kwargs": {"alpha_multiplier": 1.0, "pm_alpha": 0.0},
        "eval_threshold": 0.5,
        "plot_threshold": 0.3,
        "calibration_file": None,
        "use_calibrated_threshold": False,
        "description": "singlepatch_vit_p12_clsonly_train",
        "use_transformer": True,
        "transformer_type": "pure",          # single branch
        "patch_size": 12,
        "use_multiscale": False,
        "legacy_prediction_head": True,      # CLS-only decoding head
        # Train from scratch WITH the CLS-to-patch attention path. Without it the
        # CLS token never sees the radar input and this baseline can only learn a
        # climatological constant field, which is what the pre-2026-07 run did.
        "cls_attends_patches": True,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 4,
        "attn_dropout": 0.1,
        "learnable_pos_emb": True,
        "temporal_encoding": "learned",
        "use_checkpoint": True,
        "use_static_data": False,
        "static_channel_dropout": 0.0,
        "batch_size": 8,
        "only_validate": False,
        "final_model_path": None,
        "val_rich_only": True,
        "max_val_batches": None,
        "val_rich_min_events": 1000,
        "val_rich_time_window": 6,
        "val_compare_full": True,
        "val_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        "eval_train_rich_only": True,
        "train_rich_min_events": 1000,
        "train_rich_time_window": 6,
        "calibration_batch_size": 2,
        "calibration_max_samples": 200,
        "min_improvement": 0.015,
        "max_epochs": 40,
        # Reduced from 7. In the post-fix single-patch retrain the validation loss
        # reached its optimum at epoch 13 and had not improved by epoch 16, so a
        # patience of 3 stops at the same checkpoint and saves ~2 h per epoch.
        "patience": 3,
    },
    "pure_transformer_multipath_cls_train": {
        # Train-from-scratch MultiPath (p=12 & p=50) CLS-ONLY ViT baseline under the
        # leakage-free split. Two branches fused by dynamic gating, but each branch
        # decodes from its CLS token only (legacy_prediction_head=True). This is the
        # multi-scale-but-CLS-only baseline that isolates the decoder bottleneck.
        "loss": "weighted_focal_loss",
        "wfc_gamma": 1.0,
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "loss_kwargs": {"alpha_multiplier": 1.0, "pm_alpha": 0.0},
        "eval_threshold": 0.5,
        "plot_threshold": 0.3,
        "calibration_file": None,
        "use_calibrated_threshold": False,
        "description": "multipath_vit_p12p50_clsonly_train",
        "use_transformer": True,
        "transformer_type": "multipath",
        "patch_size": 12,
        "use_multiscale": False,
        "patch_sizes": [12, 50],
        "legacy_prediction_head": True,      # CLS-only decoding head in each branch
        # See the note in pure_transformer_singlepatch_cls_train.
        "cls_attends_patches": True,
        "fusion_mode": "dynamic",
        "gating_entropy_lambda": 0.1,
        "gating_balance_lambda": 0.2,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 4,
        "attn_dropout": 0.1,
        "learnable_pos_emb": True,
        "temporal_encoding": "learned",
        "use_checkpoint": True,
        "use_static_data": False,
        "static_channel_dropout": 0.0,
        "batch_size": 8,
        "only_validate": False,
        "final_model_path": None,
        "val_rich_only": True,
        "max_val_batches": None,
        "val_rich_min_events": 1000,
        "val_rich_time_window": 6,
        "val_compare_full": True,
        "val_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        "eval_train_rich_only": True,
        "train_rich_min_events": 1000,
        "train_rich_time_window": 6,
        "calibration_batch_size": 2,
        "calibration_max_samples": 200,
        "min_improvement": 0.015,
        "max_epochs": 40,
        # Reduced from 7, matching pure_transformer_singlepatch_cls_train.
        "patience": 3,
    },
    "pure_transformer_dynamic_strongreg": {
        # EXPLANATORY ablation (optional, recommended for journal submission):
        # dynamic gating with STRONG entropy + balance regularization. It converges
        # to near-uniform weights (~[0.5, 0.5]) and is expected to match the hard
        # uniform-fusion proposed model. This demonstrates that uniform fusion is the
        # attractor a well-regularized data-driven gate finds -- i.e. uniform is the
        # optimum, not a failure to tune the gate (addresses the "did you train the
        # gate properly?" critique). Same backbone as pure_transformer; only the
        # fusion differs (dynamic+strong-reg here vs hard uniform there).
        "loss": "weighted_focal_loss",
        "wfc_gamma": 1.0,
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "loss_kwargs": {"alpha_multiplier": 1.0, "pm_alpha": 0.0},
        "eval_threshold": 0.5,
        "plot_threshold": 0.3,
        "calibration_file": None,
        "use_calibrated_threshold": False,
        "description": "multipath_vit_p12p50_dynamic_strongreg",
        "use_transformer": True,
        "transformer_type": "multipath",
        "patch_size": 12,
        "use_multiscale": False,
        "patch_sizes": [12, 50],
        "fusion_mode": "dynamic",
        "gating_entropy_lambda": 0.1,
        "gating_balance_lambda": 0.2,
        "gating_confidence_lambda": 0.0,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 4,
        "attn_dropout": 0.1,
        "learnable_pos_emb": True,
        "temporal_encoding": "learned",
        "use_checkpoint": True,
        "use_static_data": False,
        "static_channel_dropout": 0.0,
        "batch_size": 8,
        "only_validate": False,
        "final_model_path": None,
        "resume_from": None,
        "val_rich_only": True,
        "max_val_batches": None,
        "val_rich_min_events": 1000,
        "val_rich_time_window": 6,
        "val_compare_full": True,
        "val_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        "eval_train_rich_only": True,
        "train_rich_min_events": 1000,
        "train_rich_time_window": 6,
        "calibration_batch_size": 2,
        "calibration_max_samples": 200,
        "min_improvement": 0.015,
        "max_epochs": 40,
        "patience": 7,
    },
    "pure_transformer_learnable_fusion": {
        # MultiPath ViT + patch-wise decoder + LEARNABLE FIXED fusion weights.
        # Fusion is NOT input-dependent: a single shared parameter vector of length
        # num_branches is softmax-normalized to produce branch weights, applied to
        # every sample. This is a controlled mid-ground between hard uniform fusion
        # ([0.5, 0.5]) and dynamic per-sample gating, and is the standard "shared
        # mixture weight" setup widely used in mixture-of-experts ablations.
        #
        # Regularization here only uses entropy_lambda as a mild anti-saturation
        # prior so neither branch is fully discarded early in training.
        # batch-level balance and per-sample confidence regs are degenerate for a
        # single shared weight vector and are ignored automatically by the model.
        "loss": "weighted_focal_loss",
        "wfc_gamma": 1.0,
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "loss_kwargs": {"alpha_multiplier": 1.0, "pm_alpha": 0.0},
        "eval_threshold": 0.5,
        "plot_threshold": 0.5,
        "calibration_file": None,
        "use_calibrated_threshold": False,
        "description": "multipath_vit_p12p50_learnablefusion",
        "use_transformer": True,
        "transformer_type": "multipath",
        "patch_size": 12,
        "use_multiscale": False,
        "patch_sizes": [12, 50],
        # Fusion configuration
        "fusion_mode": "learnable_fixed",
        "fusion_temperature": 1.0,
        # Mild anti-saturation prior (kept small so the model can still favor one scale)
        "gating_entropy_lambda": 0.02,
        "gating_balance_lambda": 0.0,
        "gating_confidence_lambda": 0.0,
        # Backbone
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 4,
        "attn_dropout": 0.1,
        "learnable_pos_emb": True,
        "temporal_encoding": "learned",
        "use_checkpoint": True,
        "use_static_data": False,
        "static_channel_dropout": 0.0,
        "batch_size": 8,
        "only_validate": False,
        "final_model_path": None,
        # CONTINUE the leakage-free learnable-fusion run from its best checkpoint to test
        # whether the fusion weights keep drifting toward the fine (p=12) branch under a
        # longer budget (patience=15). Resuming loads model/optimizer/scheduler/scaler and
        # no_improve=0 (best checkpoints store 0), so early-stopping restarts its count.
        "resume_from": os.path.join(
            project_root, "models",
            "spatiotemporaltransformer_puretransformerlearnablefusion_mult_p12_e256_d6_h4_wfl1p0_0702_1247_best_checkpoint.pth"),
        # Eval
        "val_rich_only": True,
        "max_val_batches": None,
        "val_rich_min_events": 1000,
        "val_rich_time_window": 6,
        "val_compare_full": True,
        "val_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        "eval_train_rich_only": True,
        "train_rich_min_events": 1000,
        "train_rich_time_window": 6,
        "calibration_batch_size": 2,
        "calibration_max_samples": 200,
        "min_improvement": 0.015,
        # Extended budget so patience (not max_epochs) decides the stop point when
        # continuing from the 0702_1247 checkpoint. Resumes near epoch ~38 (val loss
        # already near its floor), so with patience=15 the run typically stops ~15
        # epochs later; 80 is a safe cap. If you want to FORCE a longer run toward full
        # collapse, raise patience (e.g. 30) and/or lower min_improvement (e.g. 0.002).
        "max_epochs": 40,
        "patience": 15,
    },
    "pure_transformer_dynamic_gating": {
        # Short retraining run for testing truly dynamic scale gating.
        # Key differences from pure_transformer:
        # - remove per-sample entropy maximization (it forced uniform [0.5, 0.5])
        # - keep a weak batch-level balance term so both scales remain useful
        # - add a weak confidence term so each sample can choose a scale more decisively
        # - feed branch prediction stats to the gate so it can respond to scale-specific confidence/coverage
        "loss": "weighted_focal_loss",
        "wfc_gamma": 1.0,
        "optimizer": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "loss_kwargs": {"alpha_multiplier": 1.0, "pm_alpha": 0.0},
        "eval_threshold": 0.5,
        "plot_threshold": 0.8,
        "calibration_file": None,
        "use_calibrated_threshold": False,
        "description": "multipath_vit_p12p50_dynamicgate",
        "use_transformer": True,
        "transformer_type": "multipath",
        "patch_size": 12,
        "use_multiscale": False,
        "patch_sizes": [12, 50],
        "gating_entropy_lambda": 0.0,
        "gating_balance_lambda": 0.02,
        "gating_confidence_lambda": 0.01,
        "gating_temperature": 0.7,
        "gating_expert_drop_prob": 0.05,
        "gating_use_prediction_stats": True,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 4,
        "attn_dropout": 0.1,
        "learnable_pos_emb": True,
        "temporal_encoding": "learned",
        "use_checkpoint": True,
        "use_static_data": False,
        "static_channel_dropout": 0.0,
        "batch_size": 8,
        "only_validate": False,
        "final_model_path": None,
        "val_rich_only": True,
        "max_val_batches": None,
        "val_rich_min_events": 1000,
        "val_rich_time_window": 6,
        "val_compare_full": True,
        "val_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        "eval_train_rich_only": True,
        "train_rich_min_events": 1000,
        "train_rich_time_window": 6,
        "calibration_batch_size": 2,
        "calibration_max_samples": 200,
        "min_improvement": 0.015,
        "max_epochs": 40,
        "patience": 7,
    },
    "pure_transformer_testeval": {
        # Evaluate the already-trained MultiPath+Patch-wise decoder model on TEST set.
        # Architecture must match training config exactly so the checkpoint loads.
        "loss": "weighted_focal_loss",
        "wfc_gamma": 1.0,
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "loss_kwargs": {"alpha_multiplier": 1.0, "pm_alpha": 0.0},
        "eval_threshold": 0.5,
        "plot_threshold": 0.3,
        "calibration_file": None,
        "use_calibrated_threshold": False,
        "description": "multipath_vit_p12p50_testeval",
        "use_transformer": True,
        "transformer_type": "multipath",
        "patch_size": 12,
        "use_multiscale": False,
        "patch_sizes": [12, 50],
        # Proposed model was trained with HARD uniform fusion; eval architecture
        # must match (no gating network in the checkpoint).
        "fusion_mode": "uniform",
        "gating_entropy_lambda": 0.0,
        "gating_balance_lambda": 0.0,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 4,
        "attn_dropout": 0.1,
        "learnable_pos_emb": True,
        "temporal_encoding": "learned",
        "use_checkpoint": True,
        "use_static_data": False,
        "static_channel_dropout": 0.0,
        "batch_size": 8,
        # --- Test evaluation switches ---
        "only_validate": True,
        "eval_split": "test",
        "final_model_path": os.path.join(MODELS_DIR, "spatiotemporaltransformer_puretransformer_mult_p12_e256_d6_h4_wfl1p0_0622_2350_final.pth"),
        # --- Rich-subset definitions (reuse same criteria as valid) ---
        "val_rich_only": True,
        "max_val_batches": None,
        "val_rich_min_events": 1000,
        "val_rich_time_window": 6,
        "test_rich_min_events": 1000,
        "test_rich_time_window": 6,
        "val_compare_full": True,
        # Extended threshold range: val showed CSI still rising at 0.7, so probe up to 0.9
        "val_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        # Skip train_rich on test eval (irrelevant and time-consuming)
        "eval_train_rich_only": False,
        "calibration_batch_size": 2,
        "calibration_max_samples": 200,
        "min_improvement": 0.015,
    },
    "pure_transformer_lowepoch": {
        "loss": "weighted_focal_loss",
        "wfc_gamma": 1.0,
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "loss_kwargs": {"alpha_multiplier": 1.0, "pm_alpha": 0.0},
        "eval_threshold": 0.5,
        "plot_threshold": 0.3,
        "calibration_file": None,
        "use_calibrated_threshold": False,
        "description": "multipath_vit_p12p50_lowepoch",
        "use_transformer": True,
        "transformer_type": "multipath",
        "patch_size": 12,
        "use_multiscale": False,
        "patch_sizes": [12, 50],
        "gating_entropy_lambda": 0.1,
        "gating_balance_lambda": 0.2,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 4,
        "attn_dropout": 0.1,
        "learnable_pos_emb": True,
        "temporal_encoding": "learned",
        "use_checkpoint": True,
        "use_static_data": False,
        "static_channel_dropout": 0.0,
        "batch_size": 8,
        "only_validate": False,
        "final_model_path": None,
        "val_rich_only": True,
        "max_val_batches": None,
        "val_rich_min_events": 1000,
        "val_rich_time_window": 6,
        "val_compare_full": True,
        "val_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
        "eval_train_rich_only": True,
        "train_rich_min_events": 1000,
        "train_rich_time_window": 6,
        "calibration_batch_size": 2,
        "calibration_max_samples": 200,
        "min_improvement": 0.015,
        "max_epochs": 40,
        "patience": 7,
    },
    "pure_transformer_singlepatch12": {
        "loss": "weighted_focal_loss",
        "wfc_gamma": 1.0,
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "loss_kwargs": {"alpha_multiplier": 1.0, "pm_alpha": 0.0},
        "eval_threshold": 0.5,
        "plot_threshold": 0.3,
        "calibration_file": None,
        "use_calibrated_threshold": False,
        "description": "singlepatch_vit_p12_compare",
        "use_transformer": True,
        "transformer_type": "pure",
        "patch_size": 12,
        "gating_entropy_lambda": 0.0,
        "gating_balance_lambda": 0.0,
        "use_multiscale": False,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 4,
        "attn_dropout": 0.1,
        "learnable_pos_emb": True,
        "temporal_encoding": "learned",
        "use_checkpoint": True,
        "use_static_data": False,
        "static_channel_dropout": 0.0,
        "batch_size": 8,
        "only_validate": False,
        "final_model_path": None,
        "val_rich_only": True,
        "max_val_batches": None,
        "val_rich_min_events": 1000,
        "val_rich_time_window": 6,
        "val_compare_full": True,
        "val_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
        "eval_train_rich_only": True,
        "train_rich_min_events": 1000,
        "train_rich_time_window": 6,
        "calibration_batch_size": 2,
        "calibration_max_samples": 200,
    },
    "pure_transformer_multipath_cls_testeval": {
        # Evaluate the already-trained MultiPath CLS-only ViT on TEST set.
        # This model has TWO branches (patch=12, patch=50), each using legacy CLS-only
        # prediction_head (no patch-wise decoder). Trained around 2026-03-16, BEFORE
        # the patch-wise decoder refactor, so legacy_prediction_head=True is needed.
        "loss": "weighted_focal_loss",
        "wfc_gamma": 1.0,
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "loss_kwargs": {"alpha_multiplier": 1.0, "pm_alpha": 0.0},
        "eval_threshold": 0.5,
        "plot_threshold": 0.3,
        "calibration_file": None,
        "use_calibrated_threshold": False,
        "description": "multipath_vit_p12p50_clsonly_testeval",
        "use_transformer": True,
        "transformer_type": "multipath",
        "patch_size": 12,
        "use_multiscale": False,
        "patch_sizes": [12, 50],
        # Gating regularization values used during training (kept here for reference;
        # not used in eval since gating is frozen at the trained values)
        "gating_entropy_lambda": 0.1,
        "gating_balance_lambda": 0.2,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 4,
        "attn_dropout": 0.1,
        "learnable_pos_emb": True,
        "temporal_encoding": "learned",
        "use_checkpoint": True,
        "use_static_data": False,
        "static_channel_dropout": 0.0,
        "batch_size": 8,
        # --- Architecture flags (must match the run that produced the checkpoint) ---
        "legacy_prediction_head": True,
        # See the note in pure_transformer_singlepatch_testeval. Keep True for the
        # post-fix retrain; the pre-fix 0626 checkpoint requires False.
        "cls_attends_patches": True,
        # --- Test evaluation switches ---
        "only_validate": True,
        "eval_split": "test",
        # Retrained 2026-07-31 with cls_attends_patches=True; early-stopped at epoch 16
        # (val loss 0.1195, against 0.1859 for the pre-fix run, and training loss 0.197
        # against 0.473 -- the CLS token now receives the radar input). The *_final.pth
        # holds the best-validation weights, which are reloaded before it is written.
        # The superseded pre-fix checkpoint was
        # ..._mult_p12_e256_d6_h4_wfl1p0_0626_1428_final.pth and requires
        # cls_attends_patches=False.
        "final_model_path": os.path.join(MODELS_DIR, "spatiotemporaltransformer_puretransformermultipathclstrain_mult_p12_e256_d6_h4_wfl1p0_0731_1933_final.pth"),
        # --- Rich-subset definitions ---
        "val_rich_only": True,
        "max_val_batches": None,
        "val_rich_min_events": 1000,
        "val_rich_time_window": 6,
        "test_rich_min_events": 1000,
        "test_rich_time_window": 6,
        "val_compare_full": True,
        # Same threshold list as other test_eval configs for fair comparison
        "val_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        "eval_train_rich_only": False,
        "calibration_batch_size": 2,
        "calibration_max_samples": 200,
    },
    "pure_transformer_singlepatch_testeval": {
        # Evaluate the already-trained Single-patch (patch=12) CLS-only ViT on TEST set.
        # This checkpoint was trained BEFORE the patch-wise decoder refactor,
        # so the architecture uses legacy CLS-only prediction_head.
        "loss": "weighted_focal_loss",
        "wfc_gamma": 1.0,
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "loss_kwargs": {"alpha_multiplier": 1.0, "pm_alpha": 0.0},
        "eval_threshold": 0.5,
        "plot_threshold": 0.3,
        "calibration_file": None,
        "use_calibrated_threshold": False,
        "description": "singlepatch_vit_p12_testeval",
        "use_transformer": True,
        "transformer_type": "pure",  # single-path, not multipath
        "patch_size": 12,
        "gating_entropy_lambda": 0.0,
        "gating_balance_lambda": 0.0,
        "use_multiscale": False,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 4,
        "attn_dropout": 0.1,
        "learnable_pos_emb": True,
        "temporal_encoding": "learned",
        "use_checkpoint": True,
        "use_static_data": False,
        "static_channel_dropout": 0.0,
        "batch_size": 8,
        # --- Architecture flags (must match the run that produced the checkpoint) ---
        "legacy_prediction_head": True,
        # The 0729 retrain was performed with the CLS-to-patch attention path enabled,
        # so its checkpoint contains cls_attn.* / norm_cls.* weights. This MUST stay
        # True for that checkpoint; set it False only to evaluate a pre-2026-07
        # checkpoint, whose CLS token never saw the radar input.
        "cls_attends_patches": True,
        # --- Test evaluation switches ---
        "only_validate": True,
        "eval_split": "test",
        # Retrained 2026-07-29 with cls_attends_patches=True. Best-validation
        # checkpoint (val loss 0.1230 at epoch 13, against 0.1789 for the pre-fix run;
        # training loss fell from 0.424 to 0.141, confirming the CLS token now
        # receives input). Point this at the *_final.pth once the run completes.
        "final_model_path": os.path.join(MODELS_DIR, "spatiotemporaltransformer_puretransformersinglepatchclstrain_purt_p12_e256_d6_h4_wfl1p0_0729_0859_best_checkpoint.pth"),
        # --- Rich-subset definitions ---
        "val_rich_only": True,
        "max_val_batches": None,
        "val_rich_min_events": 1000,
        "val_rich_time_window": 6,
        "test_rich_min_events": 1000,
        "test_rich_time_window": 6,
        "val_compare_full": True,
        # Same threshold list as MultiPath+PWD for fair comparison
        "val_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        "eval_train_rich_only": False,
        "calibration_batch_size": 2,
        "calibration_max_samples": 200,
    },
    "pure_transformer_learnable_fusion_testeval": {
        # Evaluate the trained MultiPath+PWD model with LEARNABLE FIXED fusion
        # weights on the TEST set. Architecture mirrors pure_transformer_learnable_fusion
        # so the saved checkpoint loads cleanly. Observed: fusion weights converged
        # to roughly [0.95, 0.05] (patch=12 dominant), so this config also serves as
        # a near-proxy for "Single-patch p=12 + PWD".
        "loss": "weighted_focal_loss",
        "wfc_gamma": 1.0,
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "loss_kwargs": {"alpha_multiplier": 1.0, "pm_alpha": 0.0},
        "eval_threshold": 0.5,
        "plot_threshold": 0.5,
        "calibration_file": None,
        "use_calibrated_threshold": False,
        "description": "multipath_vit_p12p50_learnablefusion_testeval",
        "use_transformer": True,
        "transformer_type": "multipath",
        "patch_size": 12,
        "use_multiscale": False,
        "patch_sizes": [12, 50],
        "fusion_mode": "learnable_fixed",
        "fusion_temperature": 1.0,
        "gating_entropy_lambda": 0.02,
        "gating_balance_lambda": 0.0,
        "gating_confidence_lambda": 0.0,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 4,
        "attn_dropout": 0.1,
        "learnable_pos_emb": True,
        "temporal_encoding": "learned",
        "use_checkpoint": True,
        "use_static_data": False,
        "static_channel_dropout": 0.0,
        "batch_size": 8,
        "only_validate": True,
        "eval_split": "test",
        "final_model_path": os.path.join(MODELS_DIR, "spatiotemporaltransformer_puretransformerlearnablefusion_mult_p12_e256_d6_h4_wfl1p0_0710_1741_final.pth"),
        "val_rich_only": True,
        "max_val_batches": None,
        "val_rich_min_events": 1000,
        "val_rich_time_window": 6,
        "test_rich_min_events": 1000,
        "test_rich_time_window": 6,
        "val_compare_full": True,
        "val_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        "eval_train_rich_only": False,
        "calibration_batch_size": 2,
        "calibration_max_samples": 200,
    },
    "pure_transformer_dynamic_gating_testeval": {
        # Evaluate the trained MultiPath+PWD model with DYNAMIC (input-dependent)
        # gating on the TEST set. Architecture mirrors pure_transformer_dynamic_gating
        # so the saved checkpoint loads cleanly. Observed: gating collapsed to
        # patch=50 dominance during training, making this a near-proxy for
        # "Single-patch p=50 + PWD".
        "loss": "weighted_focal_loss",
        "wfc_gamma": 1.0,
        "optimizer": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "loss_kwargs": {"alpha_multiplier": 1.0, "pm_alpha": 0.0},
        "eval_threshold": 0.5,
        "plot_threshold": 0.8,
        "calibration_file": None,
        "use_calibrated_threshold": False,
        "description": "multipath_vit_p12p50_dynamicgate_testeval",
        "use_transformer": True,
        "transformer_type": "multipath",
        "patch_size": 12,
        "use_multiscale": False,
        "patch_sizes": [12, 50],
        "fusion_mode": "dynamic",
        "gating_entropy_lambda": 0.0,
        "gating_balance_lambda": 0.02,
        "gating_confidence_lambda": 0.01,
        "gating_temperature": 0.7,
        "gating_expert_drop_prob": 0.05,
        "gating_use_prediction_stats": True,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 4,
        "attn_dropout": 0.1,
        "learnable_pos_emb": True,
        "temporal_encoding": "learned",
        "use_checkpoint": True,
        "use_static_data": False,
        "static_channel_dropout": 0.0,
        "batch_size": 8,
        "only_validate": True,
        "eval_split": "test",
        "final_model_path": os.path.join(MODELS_DIR, "spatiotemporaltransformer_puretransformerdynamicgating_mult_p12_e256_d6_h4_wfl1p0_0629_1440_final.pth"),
        "val_rich_only": True,
        "max_val_batches": None,
        "val_rich_min_events": 1000,
        "val_rich_time_window": 6,
        "test_rich_min_events": 1000,
        "test_rich_time_window": 6,
        "val_compare_full": True,
        "val_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        "eval_train_rich_only": False,
        "calibration_batch_size": 2,
        "calibration_max_samples": 200,
    },

    "lightweight_transformer": {
        "loss": "weighted_focal_loss",
        "wfc_gamma": 2.0,
        "optimizer": "adamw",
        "lr": 0.0001,
        "weight_decay": 1e-4,
        "dropout": 0.15,
        "description": "lightweight_transformer",
        "use_transformer": True,
        "transformer_type": "lightweight",
        # 不需要其他参数，LightweightPureTransformer 会使用默认值
    },
    "large_pure_transformer": {
        "loss": "weighted_focal_loss",
        "wfc_gamma": 2.0,
        "optimizer": "adamw",
        "lr": 0.00005,  # 更小的学习率
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "description": "large_vit",
        "use_transformer": True,
        "transformer_type": "pure",
        "patch_size": 14,  # 更小的patch获得更多token
        "embed_dim": 1024,  # 更大的embedding
        "depth": 24,  # 更深的网络
        "num_heads": 16,
        "mlp_ratio": 4,
        "attn_dropout": 0.1,
        "temporal_encoding": "sinusoidal"  # 使用固定的正弦编码
    },
    "hybrid_transformer": {
            "loss": "weighted_focal_loss",
            "wfc_gamma": 2.0,
            "optimizer": "adamw",
            "lr": 0.0001,
            "weight_decay": 1e-4,
            "dropout": 0.1,
            "description": "hybrid_transformer",
            "use_transformer": True,
            "transformer_type": "hybrid",
            "cnn_channels": 256,
            "embed_dim": 512,
            "spatial_depth": 4,
            "temporal_depth": 4,
            "num_heads": 8,
            "window_size": 7,
            "mlp_ratio": 4,
            "use_checkpoint": True  # Enable gradient checkpointing for memory efficiency
        },
    "efficient_transformer": {
        "loss": "weighted_focal_loss",
        "wfc_gamma": 2.0,
        "optimizer": "adamw",
        "lr": 0.00005,
        "weight_decay": 1e-4,
        "dropout": 0.15,
        "description": "efficient_transformer",
        "use_transformer": True,
        "transformer_type": "hybrid",
        "cnn_channels": 128,  # Reduced for efficiency
        "embed_dim": 256,  # Reduced for efficiency
        "spatial_depth": 3,
        "temporal_depth": 3,
        "num_heads": 4,
        "window_size": 14,  # Larger window, fewer windows
        "mlp_ratio": 2,  # Reduced MLP size
        "use_checkpoint": True
    }

}


def generate_model_filename(config_name, config_dict, model_name="SpatioTemporalTransformer"):
    """Generate a compact model filename (short on purpose)."""
    parts = [model_name.lower()]

    # Add configuration name (compact)
    parts.append(config_name.replace("_", ""))

    # Add architecture details for transformer models (compact)
    if config_dict.get("use_transformer", False):
        transformer_type = config_dict.get("transformer_type", "pure")
        parts.append(f"{transformer_type[:3]}t")

        patch_size = config_dict.get("patch_size", 32)
        embed_dim = config_dict.get("embed_dim", 384)
        depth = config_dict.get("depth", 6)
        num_heads = config_dict.get("num_heads", 6)
        parts.append(f"p{patch_size}_e{embed_dim}_d{depth}_h{num_heads}")

    # Add loss function info (compact)
    loss = config_dict.get("loss", "unknown")
    if loss == "weighted_focal_loss":
        gamma = config_dict.get("wfc_gamma", 2.0)
        parts.append(f"wfl{gamma:.1f}".replace(".", "p"))
    elif loss == "binary_crossentropy":
        parts.append("bce")
    elif loss == "weighted_crossentropy":
        parts.append("wce")
    elif loss == "iou_loss":
        parts.append("iou")
    else:
        parts.append(loss.replace("_", "")[:6])

    # Add static data flags if enabled (compact)
    if config_dict.get("use_static_data", False):
        static_parts = []
        if config_dict.get("use_dem", False):
            static_parts.append("dem")
        if config_dict.get("use_landcover", False):
            static_parts.append("lc")
        if static_parts:
            parts.append("s" + "".join(static_parts))

    # Add timestamp for uniqueness (short format)
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    parts.append(timestamp)

    # Join parts and add extension
    filename = "_".join(parts) + ".pth"

    # Ensure filename isn't too long (max 255 chars for most filesystems)
    if len(filename) > 200:
        # Truncate middle parts while keeping model name, config, and timestamp
        essential_parts = [parts[0], parts[1], parts[-1]]
        param_parts = parts[2:-1][:5]  # Keep only first 5 parameter parts
        filename = "_".join(essential_parts[:2] + param_parts + essential_parts[-1:]) + ".pth"

    return filename

# Configure static data sources
static_config = {
    'target_shape': (700, 800),
    'lon_range': (109.505, 117.495),
    'lat_range': (19.0519, 26.0419),
    'land_cover': {
        'enabled': True,
        'data_root': os.path.join(project_root, "data/Owned_data/Land_Cover_Type"),
        # Update to point to the generated cache in project data directory
        'cache_file': os.path.join(project_root, "data/land_cover_cache.npy")
    },
    'dem': {
        'enabled': True,  # Set to True when you're ready to use DEM data
        'data_root': os.path.join(project_root, "data/Owned_data/DEM"),
        # Update to point to the generated cache in project data directory
        'cache_file': os.path.join(project_root, "data/dem_cache.npy"),
        'normalize': True
    }
}

# 1. 数据加载测试
@monitor_performance
def test_data_loading():

    dist_print("\n=== Testing Data Loading ===")

    # 动态调整静态数据配置
    current_config_dict = MODEL_CONFIGS.get(CURRENT_CONFIG, {})
    if not current_config_dict.get("use_static_data", False):
        # 禁用所有静态数据
        for key in static_config:
            if isinstance(static_config[key], dict) and 'enabled' in static_config[key]:
                static_config[key]['enabled'] = False
        dist_print("Disabled static data loading based on model configuration")

    # 优化数据加载：添加缓存索引功能
    cache_file = os.path.join(data_dir, "radar_time_index_cache.pkl")
    dist_print(f"Using cache file: {cache_file}")

    # 初始化数据生成器
    # 增加RAM缓存大小以利用Node 1的大内存(340GB+可用)
    # 计算每个worker的安全内存限额：目标总缓存128GB / (4 GPUs * 4 workers) = 8GB per worker
    # 这样可以防止30+个进程同时申请大内存导致OOM
    safe_cache_gb_per_worker = 128.0 / (4 * 4)  # ~8.0 GB
    safe_cache_size_per_worker = int(10000 / (4 * 4)) # ~625 samples

    dist_print(f"Configuring per-worker RAM cache: {safe_cache_gb_per_worker} GB, {safe_cache_size_per_worker} items")

    aug_config = {
        'enabled': True,
        'augment_prob': 0.5,
        'spatial_expansion': {
            'enabled': True,
            'iterations': 1,
            'confidence': 0.9
        },
        'temporal_propagation': {
            'enabled': True,
            'forward_prob': 0.5,
            'backward_prob': 0.5,
            'decay_factor': 0.6
        },
        'convection_based': {
            'enabled': False,
            'prob': 0.0,
            'dbz_threshold': 0.65,
            'min_pixels': 50,
            'max_centers': 5,
            'radius_range': [5, 20],
            'confidence': 0.7,
            'max_duration': 4
        }
    }

    batch_gen = RadarBatchGenerator(
        data_root=data_dir,
        img_size=(700, 800),
        timesteps=(6, 6),  # 过去和未来时间步
        batch_size=None,
        lightning_csv=lightning_csv,  # 添加闪电数据
        cache_index=cache_file,  # 添加索引缓存路径
        parallel_workers=1,  # 使用多进程构建索引
        max_files_per_product=241000,  # 限制每个产品处理的最大文件数1533600，241000
        prefetch_factor=1,  # 预取因子
        lightning_sample_prob=0.8, # Increase lightning sampling probability to 80%
        augmentation_config=aug_config,
        static_data_config=static_config,  # Pass the static config here!
        # static_data_config=static_config  # Add static data configuration
        ram_cache_size=safe_cache_size_per_worker,  # 限制每个worker的缓存条目
        ram_cache_memory_gb=safe_cache_gb_per_worker,  # 限制每个worker的内存占用
        disk_cache_dir=None, # 显式禁用磁盘缓存，仅使用RAM缓存以避免网络存储瓶颈
        results_dir=results_dir  # 传入正确的结果保存目录
    )

    # 计算事件发生率（使用训练集的一部分）
    global event_stats  # Make it global so it's accessible in other functions
    event_stats = batch_gen.calculate_event_occurrence(
        num_batches=5,  # 采样10个批次来估计
        dataset="train"
    )

    # Only rank 0 does visualization
    if is_distributed and local_rank != 0:
        dist_print(f"Process {local_rank}: Skipping data visualization")
        return batch_gen

    dist_print(f"Batch generator initialized")

    # Create a dataset and dataloader for testing
    test_dataset = RadarSampleDataset(batch_gen, dataset="train")
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,  # Set desired batch size
        shuffle=False,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )

    # Find a batch with lightning events
    valid_batch_found = False
    selected_batch = 0
    selected_sample = 0
    max_batches_to_check = 10  # Add this variable definition

    # Try multiple batches to find samples with valid data
    dist_print(f"\nSearching for batches with lightning events...")
    valid_batch_found = False
    # Get an iterator from the dataloader
    data_iter = iter(test_loader)
    for batch_idx in range(max_batches_to_check):  # 尝试first 个批次
        try:
            # 获取样本数据
            sample_inputs, sample_target = next(data_iter)

            dist_print(f"\n=== Data Shape Validation for Transformer ===")
            # 验证输入形状
            radar_input = sample_inputs['radar_past']
            B, T, H, W, C = radar_input.shape

            # 打印输入信息
            dist_print(f"\n--- 分析批次 {batch_idx} ---")
            dist_print("Input keys:", list(sample_inputs.keys()))
            dist_print(f"Input shape: {sample_inputs['radar_past'].shape}")
            dist_print(f"Target shape: {sample_target.shape}")

            # 计算patch相关信息
            for patch_size in [14, 16, 32, 50]:  # 常用的patch sizes
                num_patches = (H // patch_size) * (W // patch_size)
                dist_print(f"  With patch_size={patch_size}: {num_patches} patches per image")
                dist_print(f"    Total sequence length: {num_patches * T}")

            # Check for lightning events in target
            lightning_pixels = (sample_target > 0).sum().item()
            total_pixels = sample_target.numel()

            if lightning_pixels > 0:
                dist_print(f"✓ Found batch {batch_idx} with lightning events!")
                dist_print(f"  Lightning pixels: {lightning_pixels}/{total_pixels} ({100*lightning_pixels/total_pixels:.2f}%)")

                # Check each sample in the batch
                for sample_idx in range(sample_target.shape[0]):
                    sample_lightning = (sample_target[sample_idx] > 0).sum().item()
                    if sample_lightning > 0:
                        dist_print(f"  Sample {sample_idx} has {sample_lightning} lightning pixels")
                        selected_sample = sample_idx

                selected_batch = batch_idx
                valid_batch_found = True

                # Also check radar data variation
                radar_data = sample_inputs["radar_past"][selected_sample].cpu().numpy()
                data_range = np.nanmax(radar_data) - np.nanmin(radar_data)
                dist_print(f"  Radar data range: {data_range:.4f}")

                break
            else:
                if batch_idx % 10 == 0:
                    dist_print(f"  Batch {batch_idx}: No lightning events")
        except Exception as e:
            dist_print(f"Error loading batch {batch_idx}: {e}")
            continue


    if not valid_batch_found:
        dist_print(f"⚠️ No batches with lightning found in first {max_batches_to_check} batches")
        dist_print("Using first batch for visualization anyway...")
        selected_batch = 0
        selected_sample = 0
        sample_inputs, sample_target = next(data_iter)
    else:
        # Re-load the selected batch
        test_dataset_single = RadarSampleDataset(batch_gen, dataset="train")
        test_loader_single = DataLoader(test_dataset_single, batch_size=2, shuffle=False)
        sample_inputs, sample_target = next(iter(test_loader_single))

    # 使用选定的样本
    radar_data = sample_inputs["radar_past"][selected_sample].cpu().numpy()
    dist_print(f"\n--- 可视化批次 {selected_batch}, 样本 {selected_sample} ---")
    dist_print(f"Radar data shape: {radar_data.shape}")
    dist_print(f"Radar data type: {radar_data.dtype}")
    dist_print(f"Radar data range: min={np.nanmin(radar_data)}, max={np.nanmax(radar_data)}")

    # 为每个时间步和通道打印详细统计信息
    for t in range(min(3, radar_data.shape[0])):
        for c in range(min(3, radar_data.shape[-1])):
            channel_data = radar_data[t, :, :, c]
            valid_data = ~np.isnan(channel_data)
            if np.any(valid_data):
                valid_values = channel_data[valid_data]
                dist_print(f"T-{t}, Channel {c}:")
                dist_print(f"  范围: min={np.min(valid_values):.2f}, max={np.max(valid_values):.2f}")
                dist_print(f"  均值: {np.mean(valid_values):.2f}, 中位数: {np.median(valid_values):.2f}")
                dist_print(f"  标准差: {np.std(valid_values):.2f}")
                dist_print(f"  非零值数量: {np.sum(valid_values != 0)} / {valid_values.size}")

    return batch_gen

# 2. 模型初始化测试
@monitor_performance
def test_model_initialization():
    dist_print("\n=== Testing Model Initialization ===")

    try:
        # Get current configuration
        config = MODEL_CONFIGS[CURRENT_CONFIG]
        globals()['config'] = config

        # 检查是否使用transformer
        use_transformer = config.get("use_transformer", False)
        transformer_type = config.get("transformer_type", "pure")

        # 初始化模型
        model, optimizer, loss_fn, metric_fns = init_model(
            batch_gen,
            model_class=SpatioTemporalTransformer,
            compile=True,
            dropout=config["dropout"],
            distributed=is_distributed,  # Pass distributed flag
            use_transformer= use_transformer,
            transformer_type=transformer_type,
            patch_size=config.get("patch_size", 32),
            embed_dim=config.get("embed_dim", 384),
            depth=config.get("depth", 6),
            num_heads=config.get("num_heads", 6),
            mlp_ratio=config.get("mlp_ratio", 2),
            attn_dropout=config.get("attn_dropout", 0.1),
            learnable_pos_emb=config.get("learnable_pos_emb", True),
            temporal_encoding=config.get("temporal_encoding", "learned"),
            use_multiscale=config.get("use_multiscale", False),  # Add this
            patch_sizes=config.get("patch_sizes", [8, 16, 32]),  # Add this
            use_static_data=config.get("use_static_data", False), # Pass static data flag to model init
            static_channel_dropout=config.get("static_channel_dropout", 0.0),  # Pass dropout rate
            legacy_prediction_head=config.get("legacy_prediction_head", False),  # For loading old CLS-only checkpoints
            # Whether the CLS token attends over the patch tokens. Pre-2026-07 runs
            # had no such path, so their CLS output was input-independent. Configs
            # that LOAD a pre-fix checkpoint must set this False, otherwise the new
            # cls_attn / norm_cls parameters are absent from the state dict and get
            # randomly initialised, silently changing the model. Set True for any
            # config that trains from scratch.
            cls_attends_patches=config.get("cls_attends_patches", False),
            gating_temperature=config.get("gating_temperature", 1.0),
            gating_expert_drop_prob=config.get("gating_expert_drop_prob", 0.15),
            gating_use_prediction_stats=config.get("gating_use_prediction_stats", False),
            fusion_mode=config.get("fusion_mode", "dynamic"),
            fusion_temperature=config.get("fusion_temperature", 1.0),
            compile_kwargs={
                'loss': config["loss"],
                'wfc_gamma': config.get("wfc_gamma", 2.0),
                'event_occurrence': event_stats['overall_occurrence'],
                'optimizer': config["optimizer"],
                'opt_kwargs': {
                    'lr': config["lr"],
                    'weight_decay': config["weight_decay"]
                },
                'loss_kwargs': config.get('loss_kwargs', {'alpha_multiplier': 1.0, 'pm_alpha': 0.0}),
                'metrics': ['binary_accuracy', "iou_metric", "dice_metric"],
                # 'use_class_weight': config.get("use_class_weight", True)
            }
        )

        # Ensure model is created on all ranks
        if model is None:
            raise RuntimeError(f"Model initialization failed on rank {local_rank}")

        # Debug: Check if parameters require gradients
        trainable_params = sum(p.requires_grad for p in model.parameters())
        total_params = sum(1 for _ in model.parameters())
        dist_print(f"Rank {local_rank}: Model has {trainable_params}/{total_params} trainable parameters")

        if trainable_params == 0:
            dist_print("WARNING: No trainable parameters found! Setting requires_grad=True for all parameters")
            for param in model.parameters():
                param.requires_grad = True
            # Verify fix worked
            trainable_params = sum(p.requires_grad for p in model.parameters())
            dist_print(f"After fix: {trainable_params}/{total_params} trainable parameters")

        # Enable gradient checkpointing to save memory
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        # Move model to the correct device for this process
        if LOG_DEVICE_INFO:
            dist_print(f"Rank {local_rank}: Moving model to device: {device}")
        model.to(device)

        # Ensure model is in training mode
        model.train()

        # Set up model for distributed training if needed
        if is_distributed or torch.cuda.device_count() > 1:
            if LOG_DEVICE_INFO:
                dist_print(f"Available GPUs: {torch.cuda.device_count()}")
            if is_distributed:
                # Ensure all ranks have the same model before wrapping with DDP
                # Count parameters on each rank
                param_count = sum(p.numel() for p in model.parameters())
                param_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]

                # Gather parameter counts from all ranks
                if world_size > 1:
                    torch.distributed.all_gather(param_counts,
                                                 torch.tensor([param_count], dtype=torch.long, device=device))
                    param_counts_list = [p.item() for p in param_counts]

                    if local_rank == 0:
                        dist_print(f"Parameter counts across ranks: {param_counts_list}")

                    # Check if all ranks have the same number of parameters
                    if len(set(param_counts_list)) > 1:
                        raise RuntimeError(f"Model parameter mismatch across ranks: {param_counts_list}")

                # For DDP, wrap after moving to device
                # static_graph pins the reduction order after the first iteration
                # instead of letting each rank rebuild its buckets from the order it
                # happened to observe gradients in. It is also what makes activation
                # checkpointing safe under DDP, which this model enables. Set
                # DDP_STATIC_GRAPH=0 to fall back if the graph ever stops being fixed.
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False,  # Better performance when all parameters are used
                    broadcast_buffers=True,  # Enablebufferbroadcasting
                    gradient_as_bucket_view=True,  # Memoryoptimization
                    static_graph=os.environ.get("DDP_STATIC_GRAPH", "1") == "1",
                )
                dist_print(f"Rank {local_rank}: Model wrapped with DistributedDataParallel")
            else:
                print("Multiple GPUs available but DDP not initialized")
        else:
            print("Single GPU being used")

        # 测试前向传播
        dist_print("Testing forward pass...")
        # Create a small test dataset instead of using batch_gen.batch()
        test_dataset = RadarSampleDataset(batch_gen, dataset="train")
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        sample_inputs, sample_target = next(iter(test_loader))

        dist_print(f"Sample input keys: {list(sample_inputs.keys())}")
        for k, v in sample_inputs.items():
            dist_print(f"  {k} shape: {v.shape}")

        sample_inputs = {k: v.to(device) for k, v in sample_inputs.items()}

        # Manually concatenate static data for the test forward pass, just like in the training loop
        if config.get("use_static_data", False) and ('static_dem' in sample_inputs or 'static_land_cover' in sample_inputs):
            radar_past = sample_inputs['radar_past'] # [B, T, H, W, C]
            B, T, H, W, C = radar_past.shape
            tensors_to_cat = [radar_past]
            
            if 'static_dem' in sample_inputs:
                dem = sample_inputs['static_dem']
                dem_expanded = dem.unsqueeze(-1).expand(B, T, H, W, 1)
                tensors_to_cat.append(dem_expanded)
                
            if 'static_land_cover' in sample_inputs:
                lc = sample_inputs['static_land_cover']
                # Normalize just like in training loop
                lc = lc.float() / 20.0
                lc_expanded = lc.unsqueeze(-1).expand(B, T, H, W, 1)
                tensors_to_cat.append(lc_expanded)
                
            combined_input = torch.cat(tensors_to_cat, dim=-1)
            sample_inputs['radar_past'] = combined_input
            dist_print(f"Concatenated static data for test. New shape: {combined_input.shape}")

        with torch.no_grad():
            dist_print("Running model forward pass...")
            output = model(sample_inputs)
            dist_print(f"Model output shape: {output.shape}")

        print_gpu_memory()
        return model, optimizer, loss_fn, metric_fns

    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()

        # Return None values to indicate failure
        return None, None, None, None

# 3. 训练流程测试
@monitor_performance
def test_training_loop():
    dist_print("\n=== Testing Training Loop ===")
    # Access global variables
    global model, optimizer, loss_fn, metric_fns, is_distributed, local_rank, world_size

    # Import the train_model function
    try:
        from c4dllightning.ml.models import train_model
        use_builtin_train = True
    except ImportError:
        dist_print("Could not import train_model, using custom training loop")
        use_builtin_train = False

    if use_builtin_train and not is_distributed:
        # For single GPU, use the original train_model with all its features
        dist_print("Using train_model from models.py")

        model_filename = generate_model_filename(CURRENT_CONFIG, MODEL_CONFIGS[CURRENT_CONFIG])
        weight_fn = model_filename

        dist_print(f"Training model with configuration: {CURRENT_CONFIG}")
        dist_print(f"Model will be saved as: {weight_fn}")

        trained_model = train_model(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            batch_gen=batch_gen,
            weight_fn=weight_fn,
            monitor="val_loss",
            max_epochs=50,
            patience=6,
            lr_patience=3,
            lr_factor=0.2,
            max_batches_per_epoch=10000,  # Limit for testing
            max_val_batches=100
        )

        # Save the final model
        final_filename = weight_fn.replace('.pth', '_final.pth')
        model_path = os.path.join(model_save_dir, final_filename)
        torch.save(trained_model.state_dict(), model_path)
        dist_print(f"Final model saved to {model_path}")

    else:
        # For distributed training or when train_model is not available
        dist_print("Using custom distributed training loop")
        train_model_distributed()


def worker_init_fn(worker_id):
    """Initialize worker process with necessary globals"""
    import numpy as np
    import torch
    import os
    import signal
    import random

    # Ignore SIGTERM in workers to prevent premature termination
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Set random seeds for reproducibility
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # Force CPU for DataLoader workers to avoid CUDA context issues
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPUs from workers

    # Disable CUDA completely for workers
    torch.cuda.is_available = lambda: False

    # Set CPU as default device for tensor creation
    torch.set_default_dtype(torch.float32)

    # Set CPU affinity for better performance
    # if hasattr(os, 'sched_setaffinity'):
    #     cpu_count = os.cpu_count()
    #     if cpu_count:
    #         # Distribute workers across CPUs
    #         cpu_id = worker_id % cpu_count
    #         os.sched_setaffinity(0, {cpu_id})

    # DO NOT re-initialize batch_gen, data_dir, or any other globals here
    # The Dataset object is already pickled and passed to workers

@monitor_performance
def train_model_distributed():
    """Distributed version of training loop with features from train_model"""

    # Access global variables
    global batch_gen, model, optimizer, loss_fn, metric_fns, is_distributed, local_rank, world_size, config

    # Add lists to track losses for plotting
    train_losses = []
    val_losses = []
    epochs_list = []

    dist_print("Setting up distributed training...")

    # Get current configuration for resume settings
    config = MODEL_CONFIGS[CURRENT_CONFIG]
    resume_from = config.get("resume_from", None)
    start_epoch = 0
    best_metric = float("inf")
    best_state_dict = None
    no_improve = 0

    # Ensure batch_gen is available in current scope
    global batch_gen
    # Create datasets
    train_dataset = RadarSampleDataset(batch_gen, dataset="train")
    valid_dataset = RadarSampleDataset(batch_gen, dataset="valid")
    dist_print(f"Length of training dataset: {len(train_dataset)}")
    dist_print(f"Length of validation dataset: {len(valid_dataset)}")

    # Use DistributedSampler for DDP
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,  # DistributedSampler handles shuffling
            drop_last=True,  # Changed: Allow uneven distribution
            seed=1234  # Add fixed seed for reproducibility
        )
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False,
            drop_last=True,
            seed=1234
        )
        shuffle = False  # Don't shuffle when using DistributedSampler
    else:
        train_sampler = None
        valid_sampler = None
        shuffle = True

    accumulation_steps = 1
    batch_size = 8

    # A sample is ~54 MB of float32 (6 x 700 x 800 x (3 in + 1 out)), so every
    # queued batch costs ~430 MB of shared memory and each rank keeps
    # num_workers * prefetch_factor of them in flight. Across 8 ranks that is tens
    # of GB in /dev/shm; when it runs out, a worker blocks forever and its rank
    # silently stops taking part in the collectives.
    prefetch_factor = int(os.environ.get("DATALOADER_PREFETCH", "2"))
    # Without a timeout a stuck worker hangs its rank for good, and the failure
    # only surfaces on the other ranks as an unrelated NCCL watchdog timeout.
    loader_timeout = int(os.environ.get("DATALOADER_TIMEOUT", "600"))

    # Create data loaders with proper batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,  # Keep False as in original
        persistent_workers=True,
        drop_last=True,  # Important for distributed training
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        prefetch_factor=prefetch_factor,  # Add prefetching
        timeout=loader_timeout if num_workers > 0 else 0,
        multiprocessing_context='spawn' if num_workers > 0 else None,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle validation
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False, # Changed: Allow last incomplete batch
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        prefetch_factor=prefetch_factor,
        timeout=loader_timeout if num_workers > 0 else 0,
        multiprocessing_context='spawn' if num_workers > 0 else None,
    )

    dist_print(f"Length of training loader: {len(train_loader)}")
    dist_print(f"Length of validation loader: {len(valid_loader)}")
    dist_print(f"DataLoader: {num_workers} workers, prefetch {prefetch_factor}, "
               f"timeout {loader_timeout}s")

    # Training components from train_model
    scaler = torch.cuda.amp.GradScaler(
        init_scale=2 ** 10,  # 1024 - 更保守的初始scale (V100不需要太高)
        growth_factor=2.0,  # 标准增长因子
        backoff_factor=0.5,  # 标准回退因子
        growth_interval=2000,  # 更保守的增长间隔
        enabled=torch.cuda.is_available())

    transformer_type = config.get("transformer_type", "pure")
    # Learning rate scheduler with warmup
    is_transformer_cfg = bool(config.get("use_transformer", False))
    warmup_epochs = 10 if is_transformer_cfg else 5  # More warmup for transformers
    base_lr = config.get("lr", 5e-4 if is_transformer_cfg else 1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True
    )

    # Manual warmup
    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Training parameters (configurable per-config, with safe defaults)
    max_epochs = config.get("max_epochs", 40)
    patience = config.get("patience", 15)
    monitor = config.get("monitor", "val_loss")
    model_filename = generate_model_filename(CURRENT_CONFIG, MODEL_CONFIGS[CURRENT_CONFIG])
    best_metric = float("inf")
    no_improve = 0
    min_improvement = config.get("min_improvement", 0.005)

    # --- Checkpoint resume (after all training components are initialized) ---
    if not resume_from:
        auto_checkpoint = os.path.join(model_save_dir,
            model_filename.replace('.pth', '_best_checkpoint.pth'))
        if os.path.exists(auto_checkpoint):
            resume_from = auto_checkpoint
            dist_print(f"Auto-detected checkpoint: {resume_from}")

    if resume_from and os.path.exists(resume_from):
        dist_print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            dist_print("Loading full checkpoint...")

            state_dict = checkpoint['model_state_dict']
            clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            if hasattr(model, 'module'):
                model.module.load_state_dict(clean_state_dict)
            else:
                model.load_state_dict(clean_state_dict)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            start_epoch = checkpoint.get('epoch', 0) + 1
            best_metric = checkpoint.get('best_metric', float("inf"))
            no_improve = checkpoint.get('no_improve', 0)
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            epochs_list = checkpoint.get('epochs_list', [])

            best_state_dict = {k: v.cpu().clone() for k, v in clean_state_dict.items()}

            dist_print(f"Resumed from epoch {start_epoch}, best {monitor}: {best_metric:.6f}, "
                       f"no_improve: {no_improve}/{patience}")
            if 'last_metrics' in checkpoint:
                dist_print(f"  Last metrics: {checkpoint['last_metrics']}")
        else:
            dist_print("Loading model weights only (no training state)...")
            state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint
            clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            if hasattr(model, 'module'):
                model.module.load_state_dict(clean_state_dict)
            else:
                model.load_state_dict(clean_state_dict)
            dist_print("Loaded model weights, training from epoch 0")
    elif resume_from:
        dist_print(f"WARNING: Checkpoint not found at {resume_from}, training from scratch")

    # Add minimum loss threshold to detect collapse
    min_valid_loss = 1e-8
    loss_stuck_count = 0
    max_loss_stuck = 5
    # How often the ranks verify they are on the same iteration. 0 disables it.
    sync_check_every = int(os.environ.get("DDP_SYNC_CHECK_EVERY", "200"))
    gating_entropy_lambda = float(config.get("gating_entropy_lambda", 0.0))
    gating_balance_lambda = float(config.get("gating_balance_lambda", 0.0))
    gating_confidence_lambda = float(config.get("gating_confidence_lambda", 0.0))

    # Performance optimization settings for Nvidia Tesla V100 32GB
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # V100-specific: Enable TensorCore operations
    torch.set_float32_matmul_precision('high')  # Use TensorCores

    # 添加CUDNN调优
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)  # 使用更多GPU内存

    # V100 supports larger batch accumulation
    if torch.cuda.get_device_properties(0).total_memory > 30 * 1024 ** 3:  # If V100 32GB
        dist_print(f"Detected V100 32GB, using accumulation_steps={accumulation_steps}")

    # Enable CUDA graphs for better GPU utilization
    if torch.cuda.is_available() and hasattr(torch.cuda, 'graphs'):
        use_cuda_graphs = True
    else:
        use_cuda_graphs = False

    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        dist_print("Gradient checkpointing enabled")
    elif hasattr(model, 'module') and hasattr(model.module, 'gradient_checkpointing_enable'):
        model.module.gradient_checkpointing_enable()
        dist_print("Gradient checkpointing enabled on module")

    # For transformer models, enable flash attention if available
    if is_transformer_cfg and torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            dist_print("Flash attention and memory-efficient SDP enabled")
        except:
            pass



    # # Ensure all GPUs are properly synchronized before training
    if is_distributed:
        # Add explicit synchronization with timeout
        try:
            torch.distributed.barrier()
            dist_print(f"Rank {local_rank}: All processes synchronized before training")
        except RuntimeError as e:
            dist_print(f"Rank {local_rank}: Synchronization failed: {e}")
            # Continue anyway, but log the issue

    # Training loop
    # Force clear cache before training starts to release reserved memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if LOG_GPU_MEMORY:
            dist_print(f"Cleared GPU cache before training. Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")

    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.time()

        # Learning rate warmup
        if epoch < warmup_epochs:
            # Exponential warmup for better early training
            warmup_lr = base_lr * ((epoch + 1) / warmup_epochs) ** 2
            set_lr(optimizer, warmup_lr)
            current_lr = warmup_lr
            dist_print(f"Warmup epoch {epoch + 1}/{warmup_epochs}, LR: {current_lr:.6f}")
        elif epoch == warmup_epochs:
            # Set to full learning rate after warmup
            set_lr(optimizer, base_lr * 2)  # Boost initial learning
            current_lr = base_lr * 2
            dist_print(f"Post-warmup boost, LR: {current_lr:.6f}")

        # Set epoch for DistributedSampler
        if is_distributed and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Training phase
        model.train()

        # Add debugging for first few batches
        debug_first_batches = (epoch == 0)  # Debug on first epoch

        train_loss = 0.0
        train_batches = 0
        max_grad = 0.0  # Initialize max_grad here

        # Pre-allocate CUDA streams for async transfers
        transfer_streams = []
        compute_stream = None

        # Pin memory for faster transfers
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        # Simplify data loading: Use standard DataLoader loop
        # The manual stream handling below was complex and potentially buggy in DDP

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Model handles static data concatenation internally in forward()
            # Just move all inputs to device
            model_inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            targets = targets.to(device, non_blocking=True)

            targets = targets.to(device, non_blocking=True)

            # Debug: check if all ranks are processing
            if batch_idx % 1000 == 0:
                print(f"Rank {local_rank}: Processing batch {batch_idx}")

            if sync_check_every and batch_idx % sync_check_every == 0:
                check_ranks_in_step(epoch * len(train_loader) + batch_idx,
                                    f"epoch {epoch + 1} batch {batch_idx}")

            # Zero gradients at start of accumulation cycle
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            try:
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(model_inputs)

                    # Check for model collapse (Optimized: check less frequently to avoid CPU-GPU sync)
                    if batch_idx % 2000 == 0:
                        output_std = outputs.std().item()
                        if output_std < 1e-4:  # More sensitive threshold
                            dist_print(f"⚠️ Model collapse detected! Output std: {output_std:.2e}")

                    # Check for NaN in outputs (Optimized: check only on error or less frequently)
                    # Logged, not acted on: the batch still goes through backward so
                    # that every rank issues the same collectives, and GradScaler
                    # skips the optimizer step once the NaN reaches the gradients.
                    if batch_idx % 2000 == 0 and torch.isnan(outputs).any():
                        print(f"Rank {local_rank}: NaN in model outputs at batch {batch_idx}")

                    # Check target imbalance
                    positive_ratio = (targets > 0.5).float().mean().item()
                    if batch_idx % 2000 == 0:
                        dist_print(f"Batch {batch_idx} - Positive target ratio: {positive_ratio:.4f}")

                    # Calculate loss - model should output logits, loss_fn expects logits
                    loss = loss_fn(outputs, targets)
                    # Multipath fusion regularization (only meaningful for dynamic / learnable_fixed)
                    if transformer_type == "multipath" and (
                        gating_entropy_lambda > 0.0
                        or gating_balance_lambda > 0.0
                        or gating_confidence_lambda > 0.0
                    ):
                        model_root = model.module if hasattr(model, "module") else model
                        if hasattr(model_root, "get_gating_regularization"):
                            gating_reg = model_root.get_gating_regularization(
                                entropy_lambda=gating_entropy_lambda,
                                balance_lambda=gating_balance_lambda,
                                confidence_lambda=gating_confidence_lambda,
                            )
                            loss = loss + gating_reg
                            if batch_idx % 500 == 0 and local_rank == 0:
                                fusion_mode = getattr(model_root, "fusion_mode", "dynamic")
                                dist_print(
                                    f"Fusion regularization ({fusion_mode}): {gating_reg.item():.6f} "
                                    f"(entropy={gating_entropy_lambda}, "
                                    f"balance={gating_balance_lambda}, "
                                    f"confidence={gating_confidence_lambda})"
                                )

                    # Periodic fusion-weight stats. Print regardless of regularization
                    # so we can monitor learnable_fixed / uniform / dynamic gate behaviour
                    # during training.
                    if (
                        transformer_type == "multipath"
                        and batch_idx % 500 == 0
                        and local_rank == 0
                    ):
                        model_root = model.module if hasattr(model, "module") else model
                        gate_w = getattr(model_root, "last_gating_weights_tensor", None)
                        if gate_w is not None:
                            with torch.no_grad():
                                gw = gate_w.detach().float()
                                entropy = -(gw * torch.log(gw + 1e-8)).sum(dim=-1)
                                entropy = entropy / math.log(gw.shape[-1])
                                fusion_mode = getattr(model_root, "fusion_mode", "dynamic")
                                dist_print(
                                    f"  Fusion stats [{fusion_mode}]: "
                                    f"mean={gw.mean(dim=0).cpu().numpy().round(4).tolist()}, "
                                    f"std={gw.std(dim=0).cpu().numpy().round(4).tolist()}, "
                                    f"entropy={entropy.mean().item():.4f}"
                                )

                    # Check if loss is too small (indicating collapse)
                    if loss.item() < min_valid_loss:
                        loss_stuck_count += 1
                        if loss_stuck_count >= max_loss_stuck:
                            dist_print(f"Loss stuck at {loss.item():.2e}, reinitializing last layer")
                            # Reinitialize with larger variance
                            if hasattr(model, 'module'):
                                if hasattr(model.module, 'head'):
                                    torch.nn.init.xavier_normal_(model.module.head.weight, gain=1.0)  # Increased gain
                                    # Initialize bias with small random values instead of zeros
                                    torch.nn.init.uniform_(model.module.head.bias, -0.01, 0.01)
                            else:
                                if hasattr(model, 'head'):
                                    torch.nn.init.xavier_normal_(model.head.weight, gain=1.0)
                                    torch.nn.init.uniform_(model.head.bias, -0.01, 0.01)

                            # More aggressive learning rate increase
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = max(param_group['lr'] * 50, 5e-4)
                            dist_print(f"Increased learning rate to {optimizer.param_groups[0]['lr']}")
                            loss_stuck_count = 0
                    else:
                        loss_stuck_count = 0

                        # Scale loss for gradient accumulation
                        loss = loss / accumulation_steps

                    # Additional check for vanishing outputs
                    if outputs.abs().max() < 1e-10:
                        dist_print(f"⚠️ CRITICAL: Outputs near zero (max abs: {outputs.abs().max().item():.2e})")
                        dist_print(f"  This indicates dead neurons or severe vanishing gradient problem")

                    # For first few batches, check layer-wise activations if possible
                    if debug_first_batches and batch_idx < 3 and hasattr(model, 'get_intermediate_outputs'):
                        intermediates = model.get_intermediate_outputs(inputs)
                        for layer_name, activation in intermediates.items():
                            act_mean = activation.mean().item()
                            act_std = activation.std().item()
                            dist_print(f"  {layer_name}: mean={act_mean:.6f}, std={act_std:.6f}")

                    # Debug: Check output statistics
                    if batch_idx % 500 == 0:
                        with torch.no_grad():
                            # Check raw logits statistics
                            out_min = outputs.min().item()
                            out_max = outputs.max().item()
                            out_mean = outputs.mean().item()
                            out_std = outputs.std().item()

                            # Convert to probabilities to check actual predictions
                            probs = torch.sigmoid(outputs)
                            prob_mean = probs.mean().item()
                            prob_std = probs.std().item()
                            prob_min = probs.min().item()
                            prob_max = probs.max().item()

                            # Check target distribution
                            target_mean = targets.mean().item()
                            target_std = targets.std().item()
                            target_positive = (targets > 0.5).sum().item()
                            target_total = targets.numel()

                            # Calculate binary predictions using a stricter threshold
                            # Use 0. to match visualization logic and reduce false alarms
                            eval_threshold = config.get("eval_threshold", 0.6)
                            predictions_binary = (probs > eval_threshold).float()
                            pred_positive = predictions_binary.sum().item()

                            # Simple accuracy check
                            correct = (predictions_binary == targets).float().sum().item()
                            batch_accuracy = correct / target_total

                            dist_print(f"Batch {batch_idx} Statistics (Threshold {eval_threshold}):")
                            dist_print(
                                f"  Logits - mean: {out_mean:.6f}, std: {out_std:.6f}, range: [{out_min:.6f}, {out_max:.6f}]")
                            dist_print(
                                f"  Probs  - mean: {prob_mean:.6f}, std: {prob_std:.6f}, range: [{prob_min:.6f}, {prob_max:.6f}]")
                            dist_print(
                                f"  Target - positive ratio: {target_positive / target_total:.6f} ({target_positive}/{target_total})")
                            dist_print(f"  Predictions - positive ratio: {pred_positive / target_total:.6f}")
                            dist_print(f"  Batch Accuracy: {batch_accuracy:.4f}")

                            # Warning if outputs are collapsing
                            if out_std < 0.1:
                                dist_print(
                                    f"⚠️ WARNING: Low output variance (std={out_std:.6f}), increasing regularization")
                                # Dynamically adjust L2 regularization
                                if is_transformer_cfg:
                                    l2_lambda = 0.00001  # Reduce L2 to allow more variance

                            if abs(prob_mean - 0.5) < 0.05 and prob_std < 0.05:
                                dist_print(f"⚠️ CRITICAL: Model predicting constant values around 0.5!")
                                dist_print(
                                    f"   Consider: 1) Increasing learning rate, 2) Adjusting pos_weight, 3) Using focal loss")

                            # Check if model is learning anything
                            if epoch > 5 and out_std < 0.001:
                                dist_print(f"CRITICAL: Model outputs collapsed after {epoch} epochs!")
                                # Consider adjusting learning rate or reinitializing

                    # Check output magnitude
                    output_max = outputs.abs().max().item()
                    if output_max > 1e6:
                        print(f"Rank {local_rank}: large output at batch {batch_idx}: {output_max}")

            except RuntimeError as e:
                # Deliberately fatal. Recovering on one rank only (skipping the
                # batch, retrying after empty_cache) makes that rank issue a
                # different number of collectives from the others, and the process
                # group then deadlocks somewhere unrelated much later. Dying here
                # with the rank and batch named is far cheaper to debug.
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    gc.collect()
                raise RuntimeError(f"Rank {local_rank} failed at batch {batch_idx}: {e}") from e

            # Add attention regularization for transformer models
            if is_transformer_cfg and hasattr(model, 'get_attention_weights'):
                try:
                    attention_weights = model.get_attention_weights()
                    if attention_weights is not None:
                        # Add entropy regularization to prevent attention collapse
                        entropy_reg = -torch.mean(torch.sum(attention_weights * torch.log(attention_weights + 1e-10), dim=-1))
                        loss = loss + 0.01 * entropy_reg  # Small regularization weight
                except:
                    pass  # If attention weights not available, continue without regularization

            # # Check for NaN in loss
            # if torch.isnan(loss) or torch.isinf(loss):
            #     dist_print(f"WARNING: Invalid loss detected at batch {batch_idx}: {loss.item()}")
            #     optimizer.zero_grad()
            #     # Skip gradient accumulation for this step
            #     if (batch_idx + 1) % accumulation_steps == 0:
            #         scaler.update()
            #     continue  # Skip this batch

            # Only clamp minimum to prevent negative loss
            loss = torch.clamp_min(loss, 0.0)

            # Backward pass with gradient noise for early epochs
            if epoch < 5:  # Add noise in early training
                # Add small noise to gradients to prevent collapse
                scaler.scale(loss).backward()
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            noise = torch.randn_like(param.grad) * 0.001
                            param.grad.add_(noise)
            else:
                scaler.scale(loss).backward()

            # Check gradients after backward
            # has_nan = False
            # for name, param in model.named_parameters():
            #     if param.grad is not None and torch.isnan(param.grad).any():
            #         has_nan = True
            #         dist_print(f"NaN gradient detected in {name}")
            #         break

            # if has_nan:
            #     dist_print(f"Skipping optimization step due to NaN gradients at batch {batch_idx}")
            #     optimizer.zero_grad()
            #     if (batch_idx + 1) % accumulation_steps == 0:
            #         scaler.update()
            #     continue

            # Update weights after accumulation
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Sync gradients across all processes before optimizer step (for DDP)
                # DDP handles this automatically during backward(), explicit sync is usually not needed unless using no_sync()
                # if is_distributed and hasattr(model, 'module'):
                #     torch.cuda.synchronize()

                scaler.unscale_(optimizer)

                # # Monitor gradient norms before clipping
                # total_norm = 0
                # param_norms = []
                # grad_norms = []
                # for name, p in model.named_parameters():
                #     if p.grad is not None:
                #         param_norm = p.data.norm(2).item()
                #         grad_norm = p.grad.data.norm(2).item()
                #         param_norms.append(param_norm)
                #         grad_norms.append(grad_norm)
                #         total_norm += grad_norm ** 2
                # total_norm = total_norm ** 0.5

                clip_value = 5.0 if is_transformer_cfg else 1.0
                # Use return value of clip_grad_norm_ which is the total norm
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                if batch_idx % 2000 == 0 and local_rank == 0:
                    if isinstance(total_norm, torch.Tensor):
                        total_norm_val = total_norm.item()
                    else:
                        total_norm_val = total_norm

                    dist_print(f"Gradient norm after clipping: {total_norm_val:.4f}")

                    # Check for vanishing/exploding gradients
                    if total_norm_val < 1e-7:
                        dist_print(f"⚠️ VANISHING GRADIENTS: norm={total_norm_val:.2e}")
                    elif total_norm_val > 100:
                        dist_print(f"⚠️ EXPLODING GRADIENTS: norm={total_norm_val:.2f}")

                # # Check for NaN gradients before optimizer step
                # has_nan_grad = False
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         if torch.isnan(param.grad).any():
                #             has_nan_grad = True
                #             if local_rank == 0:
                #                 dist_print(f"NaN gradient in {name}")
                #                 dist_print(f"Gradient stats - min: {param.grad.min()}, max: {param.grad.max()}")

                # if has_nan_grad:
                #     if local_rank == 0:
                #         dist_print(f"Skipping batch {batch_idx} due to NaN gradients")
                #     optimizer.zero_grad()
                #     scaler.update()  # Important: update scaler even when skipping
                #     # Reset scaler if too many NaN encountered
                #     if hasattr(scaler, '_scale') and scaler._scale < 1e-4:
                #         scaler._scale = 1.0
                #         dist_print("Reset gradient scaler scale to 1.0")
                #     continue

                scaler.step(optimizer)
                scaler.update()

            train_loss += loss.item() * accumulation_steps
            train_batches += 1

            # Progress logging
            if batch_idx % 1000 == 0 and local_rank == 0:
                mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB

                if LOG_GPU_MEMORY:
                    dist_print(
                        f"Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {loss.item() * accumulation_steps:.4f} | "
                        f"GPU Mem: {mem_allocated:.2f}/{mem_reserved:.2f} GB"
                    )

            # Memory clean up
            if batch_idx % 5000 == 0:
                gc.collect()
                # Only clear cache if really needed
                torch.cuda.empty_cache()

            # # Limit batches for testing (from train_model max_batches_per_epoch)
            # if batch_idx >= 10000:
            #     break

        # Calculate average training loss
        train_loss /= max(train_batches, 1)
        train_losses.append(train_loss)  # Store training loss

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        # Validation runs without a single collective, so any speed difference
        # between ranks accumulates over its whole length and is paid in one go by
        # the fast ranks at the all-reduce below. Shorten it if they time out there.
        max_val_batches = int(os.environ.get("MAX_VAL_BATCHES", "501"))
        metric_results = {fn.__name__: 0.0 for fn in metric_fns} if metric_fns else {}

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                # Model handles static data concatenation internally
                model_inputs = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                                for k, v in inputs.items()}
                targets = targets.to(device, non_blocking=True)
                
                # Simple validation for NaNs
                for k, v in model_inputs.items():
                    if isinstance(v, torch.Tensor) and (torch.isnan(v).any() or torch.isinf(v).any()):
                        # dist_print(f"WARNING: Invalid values in input {k} at validation batch {batch_idx}")
                        model_inputs[k] = torch.nan_to_num(v, 0.0)

                # Forward pass
                outputs = model(model_inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
                val_batches += 1

                # Calculate metrics
                if metric_fns:
                    # Convert logits to probabilities for metrics
                    outputs_prob = torch.sigmoid(outputs)
                    for fn in metric_fns:
                        # Use probabilities for metric calculation
                        metric_value = fn(targets, outputs_prob)
                        # Handle batch-wise metrics - take mean if tensor has multiple elements
                        if metric_value.numel() > 1:
                            metric_value = metric_value.mean()
                        metric_results[fn.__name__] += metric_value.item()
                        # Log if metric is zero
                        if metric_value.item() == 0 and batch_idx == 0:
                            dist_print(f"WARNING: {fn.__name__} is 0 at first validation batch")

                # Limit validation batches
                if batch_idx + 1 >= max_val_batches:
                    break

        # Average validation loss and metrics
        val_loss /= max(val_batches, 1)
        if metric_fns:
            for k in metric_results:
                metric_results[k] /= max(val_batches, 1)

        # Gather metrics across all processes if distributed. One fused all-reduce
        # rather than one per quantity: every separate collective is another place
        # where the ranks can fall out of step. The batch count rides along so a
        # rank that ran a different number of iterations is caught here.
        if is_distributed:
            metric_keys = list(metric_results.keys())
            packed = torch.tensor(
                [val_loss] + [metric_results[k] for k in metric_keys] + [float(train_batches)],
                dtype=torch.float64, device=device,
            )
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)

            val_loss = packed[0].item() / world_size
            for i, k in enumerate(metric_keys):
                metric_results[k] = packed[1 + i].item() / world_size

            total_batches = packed[-1].item()
            if abs(total_batches - train_batches * world_size) > 0.5:
                raise RuntimeError(
                    f"Rank {local_rank} ran {train_batches} training batches in epoch "
                    f"{epoch + 1} but the ranks total {total_batches:.0f} "
                    f"(expected {train_batches * world_size})"
                )

        # Update learning rate
        if epoch >= warmup_epochs:
            # Use scheduler after warmup
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Prevent learning rate from going too low
            if current_lr < 1e-6:
                set_lr(optimizer, 1e-5)
                current_lr = 1e-5
                dist_print(f"Learning rate too low, reset to {current_lr}")
        else:
            # During warmup, current_lr is already set above
            pass

        # Store validation loss
        val_losses.append(val_loss)
        epochs_list.append(epoch + 1)

        # Epoch timing
        epoch_time = time.time() - epoch_start_time

        # Logging (only on rank 0)
        if local_rank == 0:
            dist_print(f"\n{'=' * 50}")
            dist_print(f"Epoch {epoch + 1}/{max_epochs} completed in {epoch_time:.2f}s")
            dist_print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if metric_fns:
                dist_print("Metrics: " + " | ".join(f"{k}: {v:.4f}" for k, v in metric_results.items()))
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0][
                'lr']
            dist_print(f"Learning Rate: {current_lr:.6f}")

        # Best tracking with checkpoint saving
        current_metric = val_loss if monitor == "val_loss" else metric_results.get(monitor, val_loss)

        if epoch >= warmup_epochs:
            improvement_threshold = best_metric * (1.0 - min_improvement)
            if current_metric < improvement_threshold:
                best_metric = current_metric
                model_to_save = model.module if hasattr(model, 'module') else model
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model_to_save.state_dict().items()}
                if local_rank == 0:
                    dist_print(f"New best {monitor}: {current_metric:.4f}")

                    # Save checkpoint immediately to disk
                    checkpoint_path = os.path.join(model_save_dir, model_filename.replace('.pth', '_best_checkpoint.pth'))
                    checkpoint_data = {
                        'epoch': epoch,
                        'model_state_dict': {k: v.clone() for k, v in best_state_dict.items()},
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_metric': best_metric,
                        'no_improve': 0,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'epochs_list': epochs_list,
                        'config_name': CURRENT_CONFIG,
                    }
                    if metric_fns:
                        checkpoint_data['last_metrics'] = {k: v for k, v in metric_results.items()}
                    torch.save(checkpoint_data, checkpoint_path)
                    dist_print(f"  Checkpoint saved to {checkpoint_path}")

                no_improve = 0
            else:
                no_improve += 1
                if local_rank == 0:
                    dist_print(f"No improvement for {no_improve} epochs (patience: {patience})")

        # No collective needed to agree on early stopping: val_loss and
        # metric_results are bitwise identical on every rank after the fused
        # all-reduce above, so no_improve is derived identically everywhere.

        # Early stopping check
        if no_improve >= patience:
            dist_print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best model (kept in memory)
    if best_state_dict is not None:
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_state_dict)
        else:
            model.load_state_dict(best_state_dict)

    # Save final model (only on rank 0)
    final_filename = model_filename.replace('.pth', '_final.pth')
    final_model_path = os.path.join(model_save_dir, final_filename)
    globals()['final_model_path'] = final_model_path

    if local_rank == 0:
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), final_model_path)
        dist_print(f"Final model saved to {final_model_path}")

        plot_training_curves(epochs_list, train_losses, val_losses,
                             model_filename=model_filename)

def plot_training_curves(epochs, train_losses, val_losses, model_filename=None):
    """Plot training and validation loss curves"""
    if local_rank == 0:  # Only plot on rank 0
        plt.figure(figsize=(10, 6))

        # Plot losses
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

        # Add grid and labels
        plt.grid(True, alpha=0.3)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Curves', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)

        # Add annotations for min validation loss
        min_val_loss = min(val_losses)
        min_val_epoch = epochs[val_losses.index(min_val_loss)]
        plt.annotate(f'Min Val Loss: {min_val_loss:.4f}',
                     xy=(min_val_epoch, min_val_loss),
                     xytext=(min_val_epoch + 2, min_val_loss + 0.1),
                     arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

        # Save the plot
        plt.tight_layout()
        # Build a model-specific filename so different runs don't overwrite each other.
        # Falls back to the legacy generic name if model_filename is not provided.
        if model_filename:
            stem = os.path.splitext(os.path.basename(model_filename))[0]
            fig_name = f"training_loss_curves_{stem}.png"
        else:
            fig_name = "training_loss_curves.png"
        loss_plot_path = os.path.join(results_dir, 'calibration', fig_name)
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        dist_print(f"Training loss curves saved to {loss_plot_path}")

# 4. 验证与指标测试
@monitor_performance
def test_validation_and_metrics(model=None, batch_gen=None, loss_fn=None):
    dist_print("\n=== Testing Validation & Metrics (GPU Accelerated) ===")

    global is_distributed, local_rank, world_size

    if model is None:
        model = globals().get('model')
    if batch_gen is None:
        batch_gen = globals().get('batch_gen')
    if loss_fn is None:
        loss_fn = globals().get('loss_fn')

    if model is None:
        raise ValueError("Model is not available")
    if batch_gen is None:
        raise ValueError("Batch generator is not available")
    if loss_fn is None:
        raise ValueError("Loss function is not available")

    config = MODEL_CONFIGS.get(CURRENT_CONFIG, {})
    val_rich_only = config.get("val_rich_only", False)
    max_val_batches = config.get("max_val_batches", None)
    thresholds = config.get("val_thresholds") or [0.3, 0.5, 0.7, 0.9]
    val_compare_full = config.get("val_compare_full", False)
    val_rich_min_events = config.get("val_rich_min_events", None)
    val_rich_time_window = config.get("val_rich_time_window", 12)
    eval_train_rich_only = config.get("eval_train_rich_only", False)
    train_rich_min_events = config.get("train_rich_min_events", val_rich_min_events)
    train_rich_time_window = config.get("train_rich_time_window", val_rich_time_window)
    # Test-split eval: set eval_split="test" in config to run on test data instead of valid
    eval_split = config.get("eval_split", "valid")
    test_rich_min_events = config.get("test_rich_min_events", val_rich_min_events)
    test_rich_time_window = config.get("test_rich_time_window", val_rich_time_window)

    def _get_dataset(split="valid", subset_label="full"):
        dataset = RadarSampleDataset(batch_gen, dataset=split)
        if subset_label != "rich":
            return dataset

        rich_indices = []
        if split == "valid":
            min_events = val_rich_min_events
            time_window = val_rich_time_window
            times = batch_gen.valid_times
        elif split == "test":
            min_events = test_rich_min_events
            time_window = test_rich_time_window
            times = batch_gen.test_times
        else:  # train or fallback
            min_events = train_rich_min_events
            time_window = train_rich_time_window
            times = batch_gen.train_times

        if min_events is not None and hasattr(batch_gen, 'lightning_processor'):
            try:
                rich_indices = batch_gen.lightning_processor.find_lightning_rich_indices(
                    times,
                    time_window_minutes=time_window,
                    min_events=min_events
                )
            except Exception as e:
                dist_print(f"Rich subset scan failed, falling back to cached indices: {e}")

        if not rich_indices and hasattr(batch_gen, 'lightning_rich_samples'):
            rich_indices = batch_gen.lightning_rich_samples.get(split, [])

        if rich_indices:
            dist_print(f"Using rich-only {split} subset: {len(rich_indices)} samples")
            return Subset(dataset, rich_indices)

        dist_print(f"No lightning-rich {split} samples found; using full dataset.")
        return dataset

    def _run_validation(valid_dataset, subset_label="full"):
        dist_print(f"Length of dataset ({subset_label}): {len(valid_dataset)}")

        if is_distributed:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_dataset,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=False
            )
        else:
            valid_sampler = None

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=8,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            prefetch_factor=2,
            multiprocessing_context='spawn' if num_workers > 0 else None
        )

        dist_print(f"Number of batches in loader ({subset_label}): {len(valid_loader)}")
        free_gpu_memory()

        device_param = next(model.parameters())
        device = device_param.device

        if LOG_DEVICE_INFO:
            if is_distributed:
                dist_print(f"Model device check - DDP wrapper on: {device}")
            else:
                dist_print(f"Model device check - Parameters on: {device}")

        model.eval()

        val_loss = 0.0
        val_batches = 0
        global_stats = {th: {'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0} for th in thresholds}
        last_outputs = None
        last_targets = None
        gating_stats = None  # Filled only for multipath models
        # --- Threshold-independent / calibration metrics (Brier, PR-AUC) ---
        # Memory-bounded accumulation over all pixels: Brier via running sum of
        # squared errors; PR-AUC (average precision) via per-class probability
        # histograms. Lazily initialized on the first batch (device-agnostic).
        prob_nbins = 1000
        prob_hist_pos = None   # histogram of predicted prob for target==1 pixels
        prob_hist_neg = None   # histogram of predicted prob for target==0 pixels
        brier_sum = None
        brier_count = None

        try:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(valid_loader):
                    try:
                        if batch_idx == 0:
                            dist_print(
                                f"Batch shapes_validation_dataset ({subset_label}): "
                                f"{[(k, v.shape) for k, v in inputs.items()]}"
                            )
                            dist_print(f"Target shape_validation_dataset ({subset_label}): {targets.shape}")

                        if any(len(v.shape) > 4 and v.shape[0] == 1 for v in inputs.values()):
                            inputs = {k: v.squeeze(0) if v.shape[0] == 1 else v for k, v in inputs.items()}
                        if len(targets.shape) > 4 and targets.shape[0] == 1:
                            targets = targets.squeeze(0)

                        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                        targets = targets.to(device, non_blocking=True)

                        if len(targets.shape) == 6 and targets.shape[0] == 1:
                            targets = targets.squeeze(0)

                        if targets.max() > 1.0 or targets.min() < 0.0 or torch.isnan(targets).any():
                            targets = torch.nan_to_num(targets, nan=0.0)
                            targets = torch.clamp(targets, 0.0, 1.0)

                        outputs = model(inputs)
                        # Collect multipath gating statistics for analysis.
                        model_root = model.module if hasattr(model, "module") else model
                        gating_tensor = getattr(model_root, "last_gating_weights_tensor", None)
                        if gating_tensor is not None and gating_tensor.ndim == 2:
                            gating_tensor = gating_tensor.detach()
                            num_experts = gating_tensor.shape[1]
                            if gating_stats is None:
                                gating_stats = {
                                    "sum": torch.zeros(num_experts, device=device, dtype=torch.float64),
                                    "sum_sq": torch.zeros(num_experts, device=device, dtype=torch.float64),
                                    "top_counts": torch.zeros(num_experts, device=device, dtype=torch.float64),
                                    "count": torch.zeros(1, device=device, dtype=torch.float64),
                                    "entropy_sum": torch.zeros(1, device=device, dtype=torch.float64),
                                }
                            gw = gating_tensor.to(torch.float64)
                            gating_stats["sum"] += gw.sum(dim=0)
                            gating_stats["sum_sq"] += (gw ** 2).sum(dim=0)
                            gating_stats["count"] += gw.shape[0]
                            entropy = -(gw * torch.log(gw + 1e-8)).sum(dim=-1) / float(np.log(num_experts))
                            gating_stats["entropy_sum"] += entropy.sum()
                            top_idx = gw.argmax(dim=-1)
                            gating_stats["top_counts"] += torch.bincount(
                                top_idx, minlength=num_experts
                            ).to(device=device, dtype=torch.float64)

                        # Handle missing channel dimension in outputs/targets
                        if outputs.ndim == 4 and targets.ndim == 5 and outputs.shape[:4] == targets.shape[:4]:
                            outputs = outputs.unsqueeze(-1)
                        if targets.ndim == 4 and outputs.ndim == 5 and targets.shape[:4] == outputs.shape[:4]:
                            targets = targets.unsqueeze(-1)

                        if outputs.ndim != 5 or targets.ndim != 5:
                            dist_print(
                                f"Skipping batch {batch_idx} ({subset_label}) due to unexpected shapes: "
                                f"outputs {outputs.shape}, targets {targets.shape}"
                            )
                            continue

                        if outputs.shape[2:4] != targets.shape[2:4]:
                            b, t, h, w, c = outputs.shape
                            target_h, target_w = targets.shape[2:4]
                            outputs_reshaped = outputs.reshape(b * t, c, h, w).permute(0, 3, 1, 2)
                            outputs_reshaped = F.interpolate(
                                outputs_reshaped, size=(target_h, target_w),
                                mode='bilinear', align_corners=False
                            )
                            outputs = outputs_reshaped.permute(0, 2, 3, 1).reshape(b, t, target_h, target_w, c)

                        loss = loss_fn(outputs, targets)
                        val_loss += loss.item()
                        val_batches += 1

                        if batch_idx % 20 == 0:
                            dist_print(f"[{subset_label}] Batch {batch_idx} loss: {loss.item():.4f}")
                            probs = torch.sigmoid(outputs)
                            dist_print(
                                f"  [{subset_label}] Val Probs: min={probs.min().item():.4f}, "
                                f"max={probs.max().item():.4f}, mean={probs.mean().item():.4f}"
                            )
                            dist_print(f"  [{subset_label}] Val Targets: positive ratio={targets.mean().item():.4f}")

                        outputs_prob = torch.sigmoid(outputs)
                        # --- Threshold-independent accumulation (Brier + PR-AUC histograms) ---
                        dev = outputs_prob.device
                        if brier_sum is None:
                            prob_hist_pos = torch.zeros(prob_nbins, dtype=torch.float64, device=dev)
                            prob_hist_neg = torch.zeros(prob_nbins, dtype=torch.float64, device=dev)
                            brier_sum = torch.zeros(1, dtype=torch.float64, device=dev)
                            brier_count = torch.zeros(1, dtype=torch.float64, device=dev)
                        p_flat = outputs_prob.reshape(-1).to(torch.float64)
                        t_flat = (targets > 0.5).reshape(-1).to(torch.float64)
                        brier_sum += ((p_flat - t_flat) ** 2).sum()
                        brier_count += float(p_flat.numel())
                        bins = torch.clamp((p_flat * prob_nbins).long(), 0, prob_nbins - 1)
                        pos_mask = t_flat > 0.5
                        prob_hist_pos += torch.bincount(bins[pos_mask], minlength=prob_nbins).to(torch.float64)
                        prob_hist_neg += torch.bincount(bins[~pos_mask], minlength=prob_nbins).to(torch.float64)

                        for th in thresholds:
                            pred_binary = (outputs_prob > th).float()
                            target_binary = (targets > 0.5).float()
                            tp = (pred_binary * target_binary).sum().item()
                            fp = (pred_binary * (1 - target_binary)).sum().item()
                            fn = ((1 - pred_binary) * target_binary).sum().item()
                            tn = ((1 - pred_binary) * (1 - target_binary)).sum().item()
                            global_stats[th]['tp'] += tp
                            global_stats[th]['fp'] += fp
                            global_stats[th]['fn'] += fn
                            global_stats[th]['tn'] += tn

                        if batch_idx < 5:
                            save_dir = os.path.join(project_root, "radar_results", "dashboard_data")
                            save_prediction_results(outputs_prob, targets, batch_idx, save_dir, prefix=subset_label)

                        last_outputs = outputs_prob.cpu().numpy()
                        last_targets = targets.cpu().numpy()

                        if max_val_batches is not None and batch_idx >= max_val_batches:
                            break

                    except Exception as e:
                        dist_print(f"Error processing batch {batch_idx} ({subset_label}): {e}")
                        continue
        except Exception as e:
            dist_print(f"Error in validation loop ({subset_label}): {e}")

        if val_batches == 0:
            dist_print(f"No batches were successfully processed ({subset_label}).")
            return np.array([]), np.array([]), {}

        val_loss /= val_batches
        dist_print(f"[{subset_label}] Validation Loss: {val_loss:.4f}")

        if is_distributed:
            dist_print(f"[{subset_label}] Aggregating metrics across processes...")
            val_loss_tensor = torch.tensor(val_loss).to(device)
            torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            val_loss = val_loss_tensor.item() / world_size

            for th in thresholds:
                stats_vec = torch.tensor([
                    global_stats[th]['tp'],
                    global_stats[th]['fp'],
                    global_stats[th]['fn'],
                    global_stats[th]['tn']
                ], device=device, dtype=torch.float64)
                torch.distributed.all_reduce(stats_vec, op=torch.distributed.ReduceOp.SUM)
                global_stats[th]['tp'] = stats_vec[0].item()
                global_stats[th]['fp'] = stats_vec[1].item()
                global_stats[th]['fn'] = stats_vec[2].item()
                global_stats[th]['tn'] = stats_vec[3].item()

            # Reduce threshold-independent accumulators across ranks
            if brier_sum is not None:
                for _t in (brier_sum, brier_count, prob_hist_pos, prob_hist_neg):
                    torch.distributed.all_reduce(_t, op=torch.distributed.ReduceOp.SUM)

            if gating_stats is not None:
                for key in ("sum", "sum_sq", "top_counts", "count", "entropy_sum"):
                    torch.distributed.all_reduce(gating_stats[key], op=torch.distributed.ReduceOp.SUM)

        metric_funcs = {
            "CSI": evaluation.intersection_over_union,
            "HSS": evaluation.heidke_skill_score,
            "PSS": evaluation.peirce_skill_score,
            "ETS": evaluation.equitable_threat_score,
            "POD": evaluation.recall,            # POD == recall == TP/(TP+FN)
            "FAR": evaluation.false_alarm_ratio,  # FAR == 1 - precision
            "Precision": evaluation.precision,    # TP/(TP+FP) == 1 - FAR
            "Recall": evaluation.recall,          # explicit alias of POD for ML readers
            "F1": evaluation.f1_score,            # 2TP/(2TP+FP+FN); = 2*CSI/(1+CSI)
            "Accuracy": evaluation.accuracy       # near-useless for rare events (TN-dominated)
        }

        metrics_results = {}
        for th in thresholds:
            metrics_results[f"Threshold_{th}"] = {}
            tp = global_stats[th]['tp']
            fp = global_stats[th]['fp']
            fn = global_stats[th]['fn']
            tn = global_stats[th]['tn']
            conf_matrix = np.array([[tp, fn], [fp, tn]])
            for metric_name, metric_func in metric_funcs.items():
                try:
                    score = metric_func(conf_matrix)
                    metrics_results[f"Threshold_{th}"][metric_name] = score
                except Exception:
                    metrics_results[f"Threshold_{th}"][metric_name] = None
            metrics_results[f"Threshold_{th}"]["ROC AUC"] = "N/A (Skipped for Speed)"

        # ---- Threshold-independent / calibration metrics: Brier score + PR-AUC ----
        # These do not depend on a decision threshold and are the appropriate way to
        # support the "probabilistic forecast" framing (cf. Leinonen et al., 2022).
        if brier_sum is not None:
            try:
                prob_indep = {}
                bc = float(brier_count.item())
                if bc > 0:
                    prob_indep["Brier"] = float(brier_sum.item() / bc)
                pos = prob_hist_pos.detach().cpu().numpy()
                neg = prob_hist_neg.detach().cpu().numpy()
                total_pos = float(pos.sum())
                total_neg = float(neg.sum())
                if total_pos > 0:
                    # "Predict positive if prob >= bin b": cumulative counts from the top.
                    tp_cum = np.cumsum(pos[::-1])[::-1]
                    fp_cum = np.cumsum(neg[::-1])[::-1]
                    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
                    recall = tp_cum / total_pos
                    # Average precision = sum (R_b - R_{b+1}) * P_b  (recall decreasing in b),
                    # plus the highest-threshold point contribution R_last * P_last.
                    dr = recall[:-1] - recall[1:]
                    ap = float((dr * precision[:-1]).sum() + recall[-1] * precision[-1])
                    prob_indep["PR_AUC"] = ap
                    prob_indep["positive_base_rate"] = float(total_pos / (total_pos + total_neg))
                    # Persist the probability histograms. They are the sufficient
                    # statistic for the reliability diagram, the expected calibration
                    # error, and the Murphy (reliability/resolution/uncertainty)
                    # decomposition of the Brier score, so saving them here means all
                    # three can be produced offline for every checkpoint without a
                    # second inference pass -- and on exactly the predictions and
                    # targets that produced the CSI, Brier and AP above.
                    prob_indep["prob_hist_nbins"] = int(prob_nbins)
                    prob_indep["prob_hist_pos"] = [int(v) for v in pos]
                    prob_indep["prob_hist_neg"] = [int(v) for v in neg]
                if prob_indep:
                    metrics_results["Threshold_independent"] = prob_indep
            except Exception as e:
                dist_print(f"Threshold-independent metric computation failed: {e}")

        # File prefix: "validation_" for valid split (backward compat), "test_" for test split
        file_prefix = "test" if eval_split == "test" else "validation"
        log_header = "TEST METRICS" if eval_split == "test" else "VALIDATION METRICS"

        # Model-specific filename suffix prevents one model's metrics from
        # overwriting another's when multiple test evals are run sequentially.
        # Priority:
        #   1) explicit "final_model_path" in the config (set in *_testeval configs)
        #   2) globals()['final_model_path'] (set after a training run completes)
        #   3) the CURRENT_CONFIG name as a last-ditch identifier
        cfg_final_path = config.get("final_model_path")
        runtime_final_path = globals().get("final_model_path") if cfg_final_path is None else None
        model_id_source = cfg_final_path or runtime_final_path
        if model_id_source:
            model_id = os.path.splitext(os.path.basename(str(model_id_source)))[0]
        else:
            model_id = str(CURRENT_CONFIG)
        # Sanitize: avoid path-separator characters showing up by accident.
        model_id = model_id.replace("/", "_").replace("\\", "_")

        dist_print("\n" + "=" * 60)
        dist_print(f"{log_header} ({subset_label})  model_id={model_id}")
        dist_print("=" * 60)
        for threshold_key, metrics in metrics_results.items():
            dist_print(f"\n{threshold_key}:")
            for metric_name, score in metrics.items():
                if isinstance(score, (int, float)):
                    dist_print(f"  {metric_name:15s}: {score:.4f}")
                else:
                    dist_print(f"  {metric_name:15s}: {score}")

        if local_rank == 0:
            metrics_filename = f'{file_prefix}_metrics_{subset_label}__{model_id}.json'
            metrics_file = os.path.join(results_dir, f'calibration/{metrics_filename}')
            import json
            with open(metrics_file, 'w') as f:
                json.dump(metrics_results, f, indent=2, default=str)
            dist_print(f"\nMetrics saved to {metrics_file}")

            # Also write a model-agnostic copy for backward compatibility with
            # existing downstream tools that read `{prefix}_metrics_{subset}.json`.
            legacy_metrics_file = os.path.join(
                results_dir, f'calibration/{file_prefix}_metrics_{subset_label}.json'
            )
            with open(legacy_metrics_file, 'w') as f:
                json.dump(metrics_results, f, indent=2, default=str)
            dist_print(f"Legacy (overwriting) copy: {legacy_metrics_file}")

            if gating_stats is not None and gating_stats["count"].item() > 0:
                count = gating_stats["count"].item()
                mean_w = (gating_stats["sum"] / count).cpu().numpy()
                var_w = (gating_stats["sum_sq"] / count - (gating_stats["sum"] / count) ** 2).cpu().numpy()
                top_ratio = (gating_stats["top_counts"] / count).cpu().numpy()
                entropy_mean = (gating_stats["entropy_sum"] / count).item()

                gating_report = {
                    "num_samples": int(count),
                    "num_experts": int(mean_w.shape[0]),
                    "mean_weights": mean_w.tolist(),
                    "var_weights": var_w.tolist(),
                    "top1_ratio": top_ratio.tolist(),
                    "normalized_entropy_mean": float(entropy_mean),
                    "model_id": model_id,
                }
                gating_file = os.path.join(
                    results_dir,
                    f'calibration/{file_prefix}_gating_report_{subset_label}__{model_id}.json',
                )
                with open(gating_file, "w") as f:
                    json.dump(gating_report, f, indent=2)
                # Legacy copy
                legacy_gating_file = os.path.join(
                    results_dir, f'calibration/{file_prefix}_gating_report_{subset_label}.json'
                )
                with open(legacy_gating_file, "w") as f:
                    json.dump(gating_report, f, indent=2)
                dist_print(f"Gating report saved to {gating_file}")
                dist_print(
                    f"[{subset_label}] Gating mean={np.round(mean_w, 4).tolist()}, "
                    f"top1_ratio={np.round(top_ratio, 4).tolist()}, entropy={entropy_mean:.4f}"
                )

        return last_outputs, last_targets, metrics_results

    # Run evaluation on the configured split ("valid" by default, "test" for final paper results)
    dist_print(f"\n>>> Running evaluation on split: '{eval_split}' <<<")
    full_outputs, full_targets, full_metrics = _run_validation(_get_dataset(eval_split, "full"), "full")
    rich_outputs, rich_targets, rich_metrics = (np.array([]), np.array([]), {})
    train_rich_outputs, train_rich_targets, train_rich_metrics = (np.array([]), np.array([]), {})

    if val_rich_only:
        rich_outputs, rich_targets, rich_metrics = _run_validation(_get_dataset(eval_split, "rich"), "rich")

    # Only run train_rich when evaluating on valid (skip on test to save time)
    if eval_train_rich_only and eval_split == "valid":
        train_rich_outputs, train_rich_targets, train_rich_metrics = _run_validation(
            _get_dataset("train", "rich"),
            "train_rich"
        )

    if val_rich_only and val_compare_full and local_rank == 0 and full_metrics and rich_metrics:
        def _best_threshold(metrics, metric_name="CSI"):
            best_th = None
            best_score = -1.0
            for key, vals in metrics.items():
                # Skip non-threshold entries such as "Threshold_independent"
                # (which holds threshold-free metrics like Brier / PR-AUC).
                if not key.startswith("Threshold_"):
                    continue
                try:
                    th = float(key.replace("Threshold_", ""))
                except ValueError:
                    continue
                if not isinstance(vals, dict):
                    continue
                score = vals.get(metric_name, None)
                if isinstance(score, (int, float)) and score > best_score:
                    best_score = score
                    best_th = th
            return best_th, best_score

        full_best = _best_threshold(full_metrics)
        rich_best = _best_threshold(rich_metrics)
        train_best = _best_threshold(train_rich_metrics) if train_rich_metrics else (None, None)

        compare_report = {
            "full_best_csi": {"threshold": full_best[0], "score": full_best[1]},
            "rich_best_csi": {"threshold": rich_best[0], "score": rich_best[1]},
            "train_rich_best_csi": {"threshold": train_best[0], "score": train_best[1]},
            "thresholds": thresholds,
            "note": "Rich subsets use *_rich_min_events/time_window; full uses all validation samples."
        }

        import json
        compare_file = os.path.join(results_dir, 'calibration/validation_compare.json')
        with open(compare_file, 'w') as f:
            json.dump(compare_report, f, indent=2, default=str)
        dist_print(f"Comparison report saved to {compare_file}")

        try:
            metrics_to_plot = ["CSI", "POD", "FAR"]
            plt.figure(figsize=(8, 10))
            for i, metric_name in enumerate(metrics_to_plot, 1):
                plt.subplot(len(metrics_to_plot), 1, i)
                full_vals = [full_metrics[f"Threshold_{th}"][metric_name] for th in thresholds]
                rich_vals = [rich_metrics[f"Threshold_{th}"][metric_name] for th in thresholds]
                plt.plot(thresholds, full_vals, label="full", color="#1f77b4")
                plt.plot(thresholds, rich_vals, label="rich", color="#ff7f0e")
                plt.ylabel(metric_name)
                if i == 1:
                    plt.legend()
                if i == len(metrics_to_plot):
                    plt.xlabel("Threshold")
            plt.tight_layout()
            curve_file = os.path.join(results_dir, 'calibration/validation_threshold_curves_compare.png')
            plt.savefig(curve_file, dpi=200, bbox_inches='tight')
            plt.close()
            dist_print(f"Threshold curves saved to {curve_file}")
        except Exception as e:
            dist_print(f"Failed to plot threshold curves: {e}")

    if val_rich_only and rich_outputs.size:
        return rich_outputs, rich_targets
    return full_outputs, full_targets

# 5. 校准曲线生成
@monitor_performance
def generate_calibration_data():
    dist_print("\n=== Generating Calibration Data ===")

    # In distributed runs, only rank 0 should run calibration
    if is_distributed and local_rank != 0:
        return

    # 创建输出目录
    calibration_dir = os.path.join(results_dir, 'calibration')
    os.makedirs(calibration_dir, exist_ok=True)

    config = MODEL_CONFIGS[CURRENT_CONFIG]
    calibration_batch_size = config.get("calibration_batch_size", 1)
    calibration_max_samples = config.get("calibration_max_samples", None)

    # Priority order for choosing model file:
    # 1. config["final_model_path"] (explicit, used in only_validate mode)
    # 2. globals()['final_model_path'] (set after training)
    # 3. Fallback: scan directory (legacy behavior, only safe when architectures match)
    config_model_path = config.get("final_model_path")
    saved_path = globals().get('final_model_path')

    if config_model_path and os.path.exists(config_model_path):
        model_files = [config_model_path]
        dist_print(f"Using config final_model_path: {config_model_path}")
    elif saved_path and os.path.exists(saved_path):
        model_files = [saved_path]
        dist_print(f"Using just-trained model: {saved_path}")
    else:
        # Safety guard: if only_validate is True but no path specified, don't scan dir
        # (would try loading incompatible checkpoints)
        if config.get("only_validate", False):
            dist_print("only_validate=True but no final_model_path specified; skipping calibration.")
            return
        model_files = sorted([
            os.path.join(model_save_dir, f)
            for f in os.listdir(model_save_dir)
            if f.endswith('_final.pth')
        ], key=os.path.getmtime, reverse=True)

    if not model_files:
        dist_print("No saved model files found, running calibration with current model state")

    dist_print(f"Calibration will use: {model_files if model_files else 'current model in memory'}")


    # Ensure model is properly moved to the correct device
    model_device = next(model.parameters()).device
    # dist_print(f"Current model device: {model_device}")

    # In distributed mode, ensure inputs match model device
    def process_batch_for_calibration(inputs, targets):
        # Move all inputs to the same device as model
        inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}
        targets = targets.to(model_device, non_blocking=True)
        return inputs, targets

    process_fn = process_batch_for_calibration if torch.cuda.is_available() else None
    save = (not is_distributed or local_rank == 0)

    if model_files:
        calibration_curve_models(
            model=model,
            batch_gen=batch_gen,
            weight_files=model_files,
            out_dir=calibration_dir,
            dataset='valid',
            process_batch_fn=process_fn,
            save_results=save,
            calibration_batch_size=calibration_batch_size,
            calibration_max_samples=calibration_max_samples,
        )
    else:
        from c4dllightning.analysis.calibration import calibration_curve
        dist_print("Running calibration directly with current model state...")
        p, occurrence_rate = calibration_curve(
            model, batch_gen, 'valid',
            process_batch_fn=process_fn,
            batch_size=calibration_batch_size,
            max_samples=calibration_max_samples,
        )
        if save:
            import numpy as np
            out_path = os.path.join(calibration_dir, 'calibration-current_model.npy')
            np.save(out_path, occurrence_rate)
            dist_print(f"Calibration data saved to {out_path}")

    dist_print(f"Calibration complete. Output dir: {calibration_dir}")

# 6. 可视化测试
@monitor_performance
def test_visualizations():
    dist_print("\n=== Testing Visualizations ===")

    # In distributed runs, only rank 0 should run visualizations
    if is_distributed and local_rank != 0:
        return
    # 使用plot_multiple_examples函数
    # 需要所有进程都参与模型推理么? 但只有 rank 0 保存图片
    try:
        # # 修改 plot_multiple_examples 调用
        # result = plot_multiple_examples(
        #     batch_gen=batch_gen,
        #     model=model,
        #     out_dir=results_dir if (not is_distributed or local_rank == 0) else None,  # 只有 rank 0 设置输出目录,
        #     shown_inputs=("radar",),
        #     input_names=("Radar Composite",),
        #     skip_missing_inputs=True,
        #     debug=True if (not is_distributed or local_rank == 0) else False,  # 只有 rank 0 打印调试信息
        #     min_lightning_percentage=0.00001,
        #     max_examples=500,
        #     max_attempts=1000,  # 尝试次数是目标样本数的 倍
        #     random_seed=1234
        # )

        # Use plot_best_examples which internally calls find_lightning_batches and plot_examples
        result = plot_best_examples(
            batch_gen=batch_gen,
            model=model,
            out_dir= os.path.join(results_dir, "best_examples") if (not is_distributed or local_rank == 0) else None,
            num_examples=100,
            debug=True if (not is_distributed or local_rank == 0) else False,
            shown_inputs=("radar",),
            input_names=("Radar Composite",),
            min_lightning_percentage=0.0001,  # 设置为0以显示所有样本，包括没有闪电的
            max_check=100,  # Scan more batches to find good examples
            plot_kwargs={
                "threshold": config.get("plot_threshold", 0.7),
                "calibration_file": config.get("calibration_file"),
                "use_calibrated_threshold": config.get("use_calibrated_threshold", False)
            }
        )

        if not is_distributed or local_rank == 0:
            dist_print(f"Generated {len(result)} visualization examples")

        free_gpu_memory()
    except Exception as e:
        dist_print(f"Error in plot_examples: {e}")

    # Use calibration_by_loss function
    # 只有 rank 0 生成校准曲线
    if not is_distributed or local_rank == 0:
        try:
            calibration_by_loss(
                out_file=os.path.join(results_dir, 'calibration/calibration_curve_transformer.png'),
                dataset='calibration',  # 此时dataset参数实际上是指向包含.npy文件的子目录名
                results_dir=results_dir,  # base results directory
            )
            dist_print(f"Calibration curve plotted.")
        except Exception as e:
            dist_print(f"Error plotting calibration curve: {e}")

    # Custom visualization logic using last_outputs from validation (if available)
    # Note: 'val_outputs' and 'val_targets' are assumed to be available globally if validation ran
    if 'val_outputs' in globals() and 'val_targets' in globals() and len(val_outputs) > 0:
        try:
            # 修改自定义可视化部分
            plt.figure(figsize=(10, 8))

            # 打印形状信息以便调试
            # dist_print(f"Outputs shape: {val_outputs.shape}, Targets shape: {val_targets.shape}")

            # 处理输出和目标形状不匹配的情况
            processed_targets = val_targets
            processed_outputs = val_outputs

            # 检查具体维度和重塑数据
            if val_targets.ndim == 5:  # Shape: [B, T, H, W, 1]
                # Take first sample
                processed_targets = val_targets[0]
                processed_outputs = val_outputs[0]
            elif val_targets.ndim == 6:  # Shape: [B, M, T, H, W, 1]
                # 取第一个批次中的第一个样本
                b_idx = 0
                member_idx = 0
                processed_targets = val_targets[b_idx, member_idx]

                # 从输出中选择对应的样本
                if val_outputs.shape[0] == val_targets.shape[0] * val_targets.shape[1]:
                    output_idx = b_idx * val_targets.shape[1] + member_idx
                    processed_outputs = val_outputs[output_idx]
                else:
                    processed_outputs = val_outputs[0]

            # 检查维度是否匹配
            if processed_outputs.ndim >= 3 and processed_targets.ndim >= 3:
                # 选择第一个时间步
                t_idx = 0

                # 根据维度选择合适的切片
                if processed_outputs.ndim == 4:  # [T, H, W, C]
                    pred_slice = processed_outputs[t_idx, :, :, 0]
                else:
                    pred_slice = processed_outputs[0, :, :, 0]  # 默认取第一个通道

                if processed_targets.ndim == 4:  # [T, H, W, C]
                    target_slice = processed_targets[t_idx, :, :, 0]
                else:
                    target_slice = processed_targets[0, :, :, 0]  # 默认取第一个通道

                # 显示预测结果
                plt.subplot(2, 2, 1)
                plt.imshow(pred_slice, cmap='viridis')
                plt.title("Prediction")
                plt.colorbar()

                plt.subplot(2, 2, 2)
                plt.imshow(target_slice, cmap='viridis')
                plt.title("Target")
                plt.colorbar()

                plt.subplot(2, 2, 3)
                error = np.abs(pred_slice - target_slice)
                plt.imshow(error, cmap='hot')
                plt.title("Absolute Error")
                plt.colorbar()

                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'prediction_comparison.png'))
                plt.close()
            else:
                dist_print(
                    f"Shape mismatch after processing: outputs {processed_outputs.shape}, targets {processed_targets.shape}")
        except Exception as e:
            dist_print(f"Error in custom visualization: {e}")


# 全局内存和进程清理函数
def cleanup_all_resources():
    """Thoroughly clean memory and terminate all processes"""
    # Check if we're already cleaning up
    if hasattr(cleanup_all_resources, '_cleaning'):
        return
    cleanup_all_resources._cleaning = True

    dist_print("Performing complete cleanup...")

    # First handle all model and tensor cleanup
    # Move model to CPU and delete if it exists
    global model, batch_gen
    if 'model' in globals() and globals()['model'] is not None:
        try:
            # Get base model if it's wrapped with any parallel wrapper
            if hasattr(globals()['model'], 'module'):
                base_model = globals()['model'].module
                base_model.cpu()
                del base_model
            else:
                globals()['model'].cpu()
            # Don't delete the global variable, just set to None to avoid NameError
            globals()['model'] = None
            dist_print("Model moved to CPU and cleared")
        except Exception as e:
            dist_print(f"Error while cleaning up model: {e}")

    # Delete all tensors and other PyTorch objects
    for name in list(globals().keys()):
        if isinstance(globals()[name], torch.Tensor):
            del globals()[name]

    # Delete batch generator
    if 'batch_gen' in globals() and globals()['batch_gen'] is not None:
        try:
            # If batch_gen has any cleanup method, call it
            if hasattr(globals()['batch_gen'], 'cleanup'):
                globals()['batch_gen'].cleanup()
            globals()['batch_gen'] = None
            dist_print("Batch generator cleared")
        except Exception as e:
            dist_print(f"Error cleaning batch generator: {e}")

    # Force collection cycles
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(i)
            except:
                pass

    # IMPORTANT: Clean up distributed process group with timeout
    if is_distributed and torch.distributed.is_initialized():
        try:
            # Don't use barrier here - it can cause deadlock during cleanup
            dist_print(f"Rank {local_rank}: Destroying process group")

            # Set a timeout for destroy_process_group
            import threading
            def destroy_pg():
                torch.distributed.destroy_process_group()

            thread = threading.Thread(target=destroy_pg)
            thread.start()
            thread.join(timeout=15.0)  #  second timeout

            if thread.is_alive():
                dist_print(f"Rank {local_rank}: Process group destruction timed out")
            else:
                dist_print(f"Rank {local_rank}: Process group destroyed successfully")
        except Exception as e:
            dist_print(f"Rank {local_rank}: Error destroying process group: {e}")

    dist_print("Cleanup completed")


# Register cleanup to run on normal exit
import atexit

# Don't use atexit for distributed training
if not is_distributed:
    atexit.register(cleanup_all_resources)

# Also register signal handlers for abnormal termination
import signal


def signal_handler(sig, frame):
    # Don't handle signals in worker processes
    if 'worker_init_fn' in str(frame.f_code.co_name):
        return

    dist_print(f"Received signal {sig}, cleaning up...")
    # Only cleanup in main process
    if is_distributed and local_rank != 0:
        sys.exit(0)
    cleanup_all_resources()
    sys.exit(0)


# Register for common signals (Unix-like systems)
try:
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, signal_handler)
except (ValueError, OSError):
    # Signal handling may not work on Windows for some signals
    pass

if __name__ == '__main__':
    # # Parse command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='radar_baseline',
    #                     choices=list(MODEL_CONFIGS.keys()),
    #                     help='Configuration to train')
    # parser.add_argument('--resume', type=str, default=None,
    #                     help='Path to checkpoint to resume training from')
    # args = parser.parse_args()
    #
    # # Use the configuration from command line
    # CURRENT_CONFIG = args.config
    # if args.resume:
    #     MODEL_CONFIGS[CURRENT_CONFIG]["resume_from"] = args.resume

    # Configuration settings (override via env: MODEL_CONFIG=pure_transformer_lowepoch)
    CURRENT_CONFIG = os.environ.get("MODEL_CONFIG", "pure_transformer")
    num_workers = int(os.environ.get("DATALOADER_WORKERS", "4"))
    torch.set_num_threads(1)  # Limit CPU threads for PyTorch

    # Make these accessible globally
    globals()['CURRENT_CONFIG'] = CURRENT_CONFIG
    globals()['num_workers'] = num_workers

    # Initialize multiprocessing
    import resource

    # Enable file descriptor pooling for PyTorch
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    try:
        # Increase file descriptor limit more aggressively
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_limit = min(65536, hard)  # Try a much higher limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))
        dist_print(f"File descriptor limit set to {new_limit}")
    except Exception as e:
        dist_print(f"Could not set file descriptor limit: {e}")
        pass

    init_process()

    # 整体执行时间跟踪
    start = time.time()

    # Initialize device and DDP variables
    if torch.cuda.is_available():
        # [CHANGED] Remove hardcoded CUDA_VISIBLE_DEVICES to allow flexible multi-card launching
        # If you need to restrict GPUs, do it in the terminal: CUDA_VISIBLE_DEVICES=4,5,6,7 python ...
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            if LOG_DEVICE_INFO:
                dist_print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

        if LOG_DEVICE_INFO:
            dist_print(f"CUDA available: {torch.cuda.device_count()} GPUs")
    else:
        if LOG_DEVICE_INFO:
            dist_print("CUDA not available, using CPU")

    model_save_dir = os.path.join(project_root, 'models')
    os.makedirs(model_save_dir, exist_ok=True)
    results_dir = os.path.join(project_root, 'radar_results')
    os.makedirs(results_dir, exist_ok=True)

    # 这些变量也需要在全局作用域中
    globals()['model_save_dir'] = model_save_dir
    globals()['results_dir'] = results_dir

    # 导入自定义模块
    sys.path.append(project_root)

    # 3. 创建数据加载器（在CUDA初始化之前）
    try:
        # 执行所有测试
        batch_gen = test_data_loading()
        
        # verify_data_alignment
        from scripts.verify_alignment import verify_data_alignment
        if local_rank == 0 or not is_distributed:
            dist_print("Verifying data alignment...")
            # batch_gen is available from test_data_loading
            verify_data_alignment(batch_gen, os.path.join(results_dir, "alignment_check"))

        # 【在这里插入截断代码】
        # 强制只使用前 1000 个样本进行全流程测试
        # dist_print("！！！测试模式：仅使用前 1000 个样本！！！")
        # batch_gen.train_times = batch_gen.train_times[:1000]
        # batch_gen.valid_times = batch_gen.valid_times[:200]  # 验证集也相应减少
        # batch_gen.test_times = batch_gen.test_times[:200]

        globals()['batch_gen'] = batch_gen  # Add this line to update global variable
        dist_print("Data loading test completed")

        # 4. 现在初始化DDP和CUDA
        local_rank, world_size, device, is_distributed = setup_ddp()

        # 这些变量需要在全局作用域中定义，以便其他函数可以访问
        globals()['local_rank'] = local_rank
        globals()['world_size'] = world_size
        globals()['device'] = device
        globals()['is_distributed'] = is_distributed

        if LOG_DEVICE_INFO:
            dist_print(f"Using device: {device}, Distributed: {is_distributed}")

        # Ensure all processes use the same device for model parameters
        if is_distributed:
            # Make sure everyone knows which device they should use
            torch.cuda.set_device(local_rank)
            if local_rank == 0:
                dist_print(f"Process {local_rank}: Set CUDA device to local rank")

        # Print information about DDP setup
        if LOG_DEVICE_INFO and (not is_distributed and torch.cuda.device_count() > 1):
            dist_print("Multiple GPUs available but not using DDP. To use DDP, launch with:")
            dist_print("python -m torch.distributed.launch --nproc_per_node=NUM_GPUS script.py")

        # 5. 启用混合精度训练（在DataLoader创建之后）
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            # Set memory allocator configuration to minimize fragmentation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.8'
            if LOG_DEVICE_INFO:
                dist_print("Memory optimization settings enabled")

        if not is_distributed or local_rank == 0:
            print_gpu_memory()

        model, optimizer, loss_fn, metric_fns = test_model_initialization()

        # 将这些变量设置为全局可访问
        globals()['model'] = model
        globals()['optimizer'] = optimizer
        globals()['loss_fn'] = loss_fn
        globals()['metric_fns'] = metric_fns

        # 检查模型是否成功初始化
        if model is None:
            print("Model initialization failed. Exiting...")
            if is_distributed:
                torch.distributed.destroy_process_group()
            sys.exit(1)

        dist_print("Model initialization test completed")

        # Optionally skip training and only run validation/evaluation
        config = MODEL_CONFIGS[CURRENT_CONFIG]
        only_validate = config.get("only_validate", False)

        if only_validate:
            expected_model = generate_model_filename(CURRENT_CONFIG, config)
            expected_final = os.path.join(model_save_dir, expected_model.replace('.pth', '_final.pth'))
            override_path = config.get("final_model_path")
            model_path = override_path or expected_final
            if os.path.exists(model_path):
                dist_print(f"Loading existing model for validation: {model_path}")
                state_dict = torch.load(model_path, map_location=device, weights_only=False)

                # Two on-disk formats exist. A *_final.pth holds a bare state dict;
                # a *_best_checkpoint.pth wraps the weights in a dict alongside the
                # epoch and optimizer state. Accept either -- passing the wrapper
                # straight to load_state_dict would match nothing and, with
                # strict=False, would silently evaluate randomly initialised weights.
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    dist_print(
                        f"Unwrapping training checkpoint "
                        f"(epoch={state_dict.get('epoch')}, "
                        f"val_loss={state_dict.get('best_val_loss', state_dict.get('val_loss'))})."
                    )
                    state_dict = state_dict['model_state_dict']

                # We always load into the inner model (no 'module.' prefix in state_dict expected).
                # If checkpoint was saved from DDP-wrapped model, its keys have 'module.' prefix.
                # Strip the prefix so keys align with the inner model's parameter names.
                target = model.module if hasattr(model, 'module') else model
                if all(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
                    dist_print("Stripped 'module.' prefix from checkpoint keys.")

                # Use strict=False to tolerate legacy-only keys (e.g., 'temperature')
                # that exist in old checkpoints but not in current code
                missing, unexpected = target.load_state_dict(state_dict, strict=False)
                if missing:
                    dist_print(f"Missing keys ({len(missing)}): {list(missing)[:5]}"
                               + (" ..." if len(missing) > 5 else ""))
                if unexpected:
                    dist_print(f"Unexpected keys ({len(unexpected)}): {list(unexpected)[:5]}"
                               + (" ..." if len(unexpected) > 5 else ""))
                if not missing and not unexpected:
                    dist_print("All keys matched perfectly.")

                # Guard against evaluating a model that did not actually receive its
                # weights. The usual cause is an architecture/config mismatch, e.g.
                # cls_attends_patches or legacy_prediction_head set differently from
                # the run that produced the checkpoint.
                n_expected = len(target.state_dict())
                if len(missing) > 0.05 * max(n_expected, 1):
                    raise RuntimeError(
                        f"{len(missing)} of {n_expected} parameters were not found in "
                        f"{model_path}. The checkpoint does not match this model "
                        f"configuration; check cls_attends_patches and "
                        f"legacy_prediction_head before trusting any metric. "
                        f"First missing keys: {list(missing)[:10]}"
                    )
            else:
                dist_print(f"Expected final model not found: {model_path}")
                dist_print("Falling back to current initialized model weights.")
        else:
            dist_print("Starting training loop test...")
            test_training_loop()
            dist_print("Training test completed successfully")

        # 只在训练完成后运行验证（可选）
        if model is not None and batch_gen is not None and loss_fn is not None:
            dist_print("\nRunning post-training validation...")
            val_outputs, val_targets = test_validation_and_metrics(model, batch_gen, loss_fn)

            # 将验证结果保存为全局变量，供可视化使用
            globals()['val_outputs'] = val_outputs
            globals()['val_targets'] = val_targets
            dist_print("Post-training validation completed")

        # Calibration still needs the model on GPU for inference
        if (not is_distributed or local_rank == 0) and 'model' in globals() and model is not None and 'batch_gen' in globals() and batch_gen is not None:
            try:
                generate_calibration_data()
            except Exception as e:
                dist_print(f"Error generating calibration data: {e}")

        # Release GPU before CPU-only plotting so GPUs are free for others
        if 'model' in globals() and model is not None:
            model_cpu = (model.module if hasattr(model, 'module') else model).cpu()
            dist_print("Model moved to CPU for plotting phase")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            dist_print(f"GPU memory released. Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        if (not is_distributed or local_rank == 0) and 'batch_gen' in globals() and batch_gen is not None:
            try:
                test_visualizations()
            except Exception as e:
                dist_print(f"Error in visualization test: {e}")

        # 计算总运行时间
        end_time = time.time()
        dist_print(f"\nTotal execution time: {end_time - start:.2f} seconds")

    except KeyboardInterrupt:
        dist_print("\nTraining interrupted by user")
        cleanup_handler(signal.SIGINT, None)
    except Exception as e:
        dist_print(f"Test failed with error: {e}")
        dist_print("Exception details:", str(e))
        import traceback

        traceback.print_exc()
        # Ensure cleanup happens on error
        if is_distributed:
            torch.distributed.destroy_process_group()
        sys.exit(1)

    finally:
        # Clean up
        dist_print("Cleaning up...")

        # Clear matplotlib
        try:
            plt.close('all')
        except:
            pass

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all CUDA operations complete

        # Destroy process group with error handling
        if is_distributed and torch.distributed.is_initialized():
            try:
                # Give some time for other processes to finish
                time.sleep(2)
                torch.distributed.destroy_process_group()
            except Exception as e:
                print(f"Error in final cleanup: {e}")





# # Run cleanup before exit
# if __name__ == '__main__':
#     try:
#         # 不要在这里立即调用清理函数，让程序正常完成
#         # cleanup_all_resources() 将由 atexit 自动调用
#         pass
#     except Exception as e:
#         print(f"Error during execution: {e}")
#         print("Continuing with script termination despite error")

# 完成
# dist_print(f"Whole test_radar_data.py executed in {time.time() - start:.2f}s")

