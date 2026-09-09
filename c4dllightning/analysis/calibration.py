import os
import gc
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d

from ..features import batch  # 使用之前转换的PyTorch版本batch.py

def dist_print(*args, **kwargs):
    """
    Print only from rank 0 in distributed training to avoid duplicate output
    when running with multiple GPUs
    """
    import os
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
        return

    # Fallback to LOCAL_RANK if distributed not initialized
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank <= 0:
        print(*args, **kwargs)

def calibration_curve(
    model,
    batch_gen,
    dataset='valid',
    nbins=100,
    process_batch_fn=None,
    log_device_info=False,
    batch_size=1,
    max_samples=None
):
    """计算模型预测的校准曲线 (兼容雷达图像和.nc数据)

    Args:
        model (nn.Module): 训练好的PyTorch模型
        batch_gen (BatchGenerator): 数据批次生成器
        dataset (str): 数据集类型 ('train', 'valid', 'test')
        nbins (int): 分箱数量

    Returns:
        tuple: (分箱中点概率, 实际观测频率)
    """
    model_device = next(model.parameters()).device
    dist_print(f"=== Generating Calibration Data ===")
    if log_device_info:
        dist_print(f"Current model device: {model_device}")

    # 检查batch_gen类型
    is_radar_generator = hasattr(batch_gen, 'train_times')  # 雷达数据生成器标识

    if is_radar_generator:
        # 创建适配雷达数据的Dataset，并传递模型设备
        dataset_obj = RadarCalibrationDataset(batch_gen, dataset=dataset)
    else:
        # 原始.nc数据的Dataset，并传递模型设备
        dataset_obj = BatchDataset(batch_gen, dataset=dataset)


    # 获取数据集
    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False)

    # 初始化分箱统计，确保在模型设备上
    bin_counts = torch.zeros(nbins, dtype=torch.int64, device=model_device)
    bin_occurrences = torch.zeros(nbins, dtype=torch.int64, device=model_device)

    model.eval()

    with torch.no_grad():
        for batch_idx, (X_batched, Y_batched) in enumerate(loader):
            try:
                if max_samples is not None and batch_idx >= max_samples:
                    break

                # Normalize batch dimensions (support DataLoader batch_size > 1)
                def _merge_batch(tensor):
                    if tensor.dim() >= 6:
                        # Merge DataLoader batch dim and inner batch dim
                        return tensor.reshape(-1, *tensor.shape[2:])
                    if tensor.dim() >= 1 and tensor.shape[0] == 1:
                        return tensor.squeeze(0)
                    return tensor

                if isinstance(X_batched, dict):
                    X = {k: _merge_batch(v) for k, v in X_batched.items()}
                else:
                    X = [_merge_batch(x) for x in X_batched]
                Y = _merge_batch(Y_batched)

                # Record input shapes for debugging (only first batch)
                if batch_idx == 0:
                    if isinstance(X, dict):
                        dist_print(
                            f"Batch {batch_idx} - Input shapes: {{{', '.join([f'{k}: {v.shape}' for k, v in X.items()])}}}")
                    else:
                        dist_print(f"Batch {batch_idx} - Input shapes: {[x.shape for x in X]}")
                    dist_print(f"Target shape: {Y.shape}")

                # Check device and print for debugging (only first batch)
                if log_device_info and batch_idx == 0:
                    if isinstance(X, dict):
                        input_device = next(iter(X.values())).device
                    else:
                        input_device = X[0].device if isinstance(X, list) else X.device
                    dist_print(
                        f"Process {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}: "
                        f"Input device: {input_device}, Model device: {model_device}"
                    )

                # Fix device mismatch
                if process_batch_fn is not None:
                    X, Y = process_batch_fn(X, Y)
                else:
                    # Default device mapping - explicitly move all data to model device
                    if isinstance(X, dict):
                        # Radar data format: {"radar_past": tensor}
                        X = {k: v.to(model_device) for k, v in X.items()}
                    else:
                        # Original .nc data format: tensor tuple
                        X = [x.to(model_device) for x in X]
                        X = torch.cat(X, dim=1) if len(X) > 1 else X[0]
                    Y = Y.to(model_device)

                # 获取模型预测
                Y_pred = model(X)
                # 如果输出是logits，需要应用sigmoid转换为概率
                Y_pred = torch.sigmoid(Y_pred)

                # 确保Y_pred和Y有相同的形状
                if Y_pred.shape != Y.shape:
                    # 将两个张量展平为1D进行比较
                    Y_pred_flat = Y_pred.view(-1)
                    Y_flat = Y.view(-1)

                    # 确保它们有相同的长度
                    min_len = min(Y_pred_flat.size(0), Y_flat.size(0))
                    Y_pred_flat = Y_pred_flat[:min_len]
                    Y_flat = Y_flat[:min_len]
                else:
                    Y_pred_flat = Y_pred.view(-1)
                    Y_flat = Y.view(-1)

                # 分箱统计
                bin_indices = torch.clamp((Y_pred_flat * nbins).long(), 0, nbins - 1)

                for bin_idx in range(nbins):
                    mask = (bin_indices == bin_idx)
                    bin_counts[bin_idx] += mask.sum().item()
                    bin_occurrences[bin_idx] += (Y_flat[mask] > 0).sum().item() # Use > 0 instead of > 0.5 for rare events

                # 清理内存
                del X, Y, Y_pred, Y_pred_flat, Y_flat, bin_indices, mask
                torch.cuda.empty_cache()

                # Memory logging disabled by default to reduce noise

            except Exception as e:
                dist_print(f"Error processing batch {batch_idx}: {e}")
                dist_print("Skipping this batch and continuing...")
                # 清理内存
                torch.cuda.empty_cache()
                continue

    # 计算概率
    p = torch.linspace(0, 1, nbins + 1, device=model_device)[:-1] + 0.5 / nbins  # 箱中点
    occurrence_rate = bin_occurrences.float() / (bin_counts.float() + 1e-10)  # 避免除以零

    # 筛选出有样本的箱
    valid_bins = bin_counts > 0

    # 在索引前移动到CPU以确保设备一致性
    valid_bins = valid_bins.cpu()
    p = p.cpu()
    occurrence_rate = occurrence_rate.cpu()

    p = p[valid_bins]
    occurrence_rate = occurrence_rate[valid_bins]

    # 正确转换张量为numpy数组
    p_numpy = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p
    occurrence_rate_numpy = occurrence_rate.detach().cpu().numpy() if isinstance(occurrence_rate,
                                                                                 torch.Tensor) else occurrence_rate

    return p_numpy, occurrence_rate_numpy


class RadarCalibrationDataset(torch.utils.data.Dataset):
    """雷达图像数据的校准数据集适配器"""

    def __init__(self, batch_gen, dataset='valid', device=None):
        self.batch_gen = batch_gen
        self.dataset = dataset
        self.device = device  # 存储设备信息

        # 根据数据集类型确定时间点
        if dataset == "train":
            self.times = batch_gen.train_times
        elif dataset == "valid":
            self.times = batch_gen.valid_times
        elif dataset == "test":
            self.times = batch_gen.test_times
        else:
            raise ValueError(f"未知数据集类型: {dataset}")

        # Determine batch size, default to 1 if not set in generator
        self.batch_size = batch_gen.batch_size if batch_gen.batch_size is not None else 1

        # Calculate length
        self.length = max(1, len(self.times) // self.batch_size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 获取批次数据
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.times))
        time_batch = self.times[start:end]

        # 初始化输入和目标张量
        inputs = torch.zeros(
            len(time_batch),
            self.batch_gen.past_timesteps,
            *self.batch_gen.img_size,
            3
        )

        targets = torch.zeros(
            len(time_batch),
            self.batch_gen.future_timesteps,
            self.batch_gen.grid_info['height'],
            self.batch_gen.grid_info['width'],
            1
        )

        # 加载每个时间序列
        for i, t in enumerate(time_batch):
            try:
                # 加载过去时间步的雷达图像
                past = self.batch_gen.load_radar_sequence(t)
                
                # 防御性处理：确保past是有效的numpy数组
                if past is None or not isinstance(past, np.ndarray):
                    # dist_print(f"Warning: Failed to load radar sequence for {t}, using zeros")
                    past = np.zeros((self.batch_gen.past_timesteps, *self.batch_gen.img_size, 3), dtype=np.float32)
                elif np.isnan(past).any() or np.isinf(past).any():
                    past = np.nan_to_num(past, nan=0.0, posinf=0.0, neginf=0.0)
                
                inputs[i] = torch.from_numpy(past.astype(np.float32))
            except Exception as e:
                # 如果任何异常发生，使用全零数据
                dist_print(f"Warning: Exception loading radar for {t}: {e}")
                past = np.zeros((self.batch_gen.past_timesteps, *self.batch_gen.img_size, 3), dtype=np.float32)
                inputs[i] = torch.from_numpy(past)

            # 加载未来时间步的闪电数据
            if self.batch_gen.lightning_processor:
                for j in range(self.batch_gen.future_timesteps):
                    future_time = t + timedelta(minutes=6 * (self.batch_gen.past_timesteps + j))
                    lightning_grid = self.batch_gen.lightning_processor.create_lightning_grid(future_time)
                    targets[i, j, :, :, 0] = lightning_grid

        # Handle static data concatenation if available
        if hasattr(self.batch_gen, 'static_data') and self.batch_gen.static_data:
            tensors_to_cat = [inputs]
            B, T, H, W, _ = inputs.shape

            # Check for DEM
            if 'dem' in self.batch_gen.static_data:
                dem_data = self.batch_gen.static_data['dem']
                dem_tensor = torch.from_numpy(dem_data).float()
                # Expand [H, W] -> [B, T, H, W, 1]
                dem_expanded = dem_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, T, H, W, 1)
                tensors_to_cat.append(dem_expanded)

            # Check for Land Cover
            if 'land_cover' in self.batch_gen.static_data:
                lc_data = self.batch_gen.static_data['land_cover']
                lc_tensor = torch.from_numpy(lc_data).float()
                # Normalize logic (consistent with training script)
                lc_tensor = lc_tensor / 20.0
                # Expand [H, W] -> [B, T, H, W, 1]
                lc_expanded = lc_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, T, H, W, 1)
                tensors_to_cat.append(lc_expanded)

            if len(tensors_to_cat) > 1:
                inputs = torch.cat(tensors_to_cat, dim=-1)

        return {"radar_past": inputs}, targets


def calibration_curve_models(
    model,
    batch_gen,
    weight_files,
    out_dir,
    out_file=None,
    dataset='valid',
    process_batch_fn=None,
    save_results=True,
    calibration_batch_size=1,
    calibration_max_samples=None,
    **kwargs
):
    """计算多个模型的校准曲线 (PyTorch版本)

    Args:
        model (nn.Module): 模型实例
        batch_gen (BatchGenerator): 数据批次生成器
        weight_files (list): 模型权重文件路径列表
        out_dir (str): 输出目录
        dataset (str): 数据集类型
        process_batch_fn (callable): 批次处理函数
        save_results (bool): 是否保存结果到文件
    """
    if save_results:
        os.makedirs(out_dir, exist_ok=True)

    # Filter for .pth files only
    pth_files = [fn for fn in weight_files if isinstance(fn, str) and fn.endswith('.pth')]
    if len(pth_files) < len(weight_files):
        dist_print(f"Warning: Filtered out {len(weight_files) - len(pth_files)} non-pth files from weight_files")

    if len(pth_files) == 0:
        dist_print("No valid .pth files found to process")
        return

    # 记录模型的原始设备
    model_device = next(model.parameters()).device
    dist_print(f"Original model device: {model_device}")

    for fn in pth_files:
        try:
            # Check if file exists
            if not os.path.isfile(fn):
                dist_print(f"Warning: File {fn} does not exist. Skipping.")
                continue

            # 加载模型权重
            dist_print(f"Loading model weights from {fn}...")
            state_dict = torch.load(fn, map_location=model_device)  # 确保加载到正确的设备

            # 检查模型和权重结构
            model_state_dict = model.state_dict()
            model_has_module = any(k.startswith('module.') for k in model_state_dict.keys())
            weights_has_module = any(k.startswith('module.') for k in state_dict.keys())

            dist_print(f"Model has 'module.' prefix: {model_has_module}")
            dist_print(f"Weights has 'module.' prefix: {weights_has_module}")

            # 处理前缀不匹配情况
            if model_has_module and not weights_has_module:
                dist_print("Adding 'module.' prefix to weights for DDP model...")
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            elif not model_has_module and weights_has_module:
                dist_print("Removing 'module.' prefix from weights...")
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # 检查2D/3D channel adapter不匹配的情况
            model_keys = set(model_state_dict.keys())
            weight_keys = set(state_dict.keys())

            # 检测模型是否使用2D而权重使用3D (或反之)
            model_has_2d = any('2d' in k for k in model_keys)
            model_has_3d = any('3d' in k for k in model_keys)
            weights_has_2d = any('2d' in k for k in weight_keys)
            weights_has_3d = any('3d' in k for k in weight_keys)

            if (model_has_2d and weights_has_3d) or (model_has_3d and weights_has_2d):
                dist_print(f"Warning: Dimension mismatch - Model has 2D: {model_has_2d}, 3D: {model_has_3d}; "
                      f"Weights has 2D: {weights_has_2d}, 3D: {weights_has_3d}")
                dist_print("Attempting to adapt keys...")

                # 创建新的state_dict，将3D替换为2D或2D替换为3D
                adapted_state_dict = {}
                for k, v in state_dict.items():
                    if model_has_2d and '3d' in k:
                        new_key = k.replace('3d', '2d')
                        if new_key in model_keys:
                            dist_print(f"Replacing key: {k} -> {new_key}")
                            adapted_state_dict[new_key] = v
                        else:
                            dist_print(f"Warning: Adapted key {new_key} not found in model")
                    elif model_has_3d and '2d' in k:
                        new_key = k.replace('2d', '3d')
                        if new_key in model_keys:
                            dist_print(f"Replacing key: {k} -> {new_key}")
                            adapted_state_dict[new_key] = v
                        else:
                            dist_print(f"Warning: Adapted key {new_key} not found in model")
                    else:
                        adapted_state_dict[k] = v

                # 使用适配后的权重
                state_dict = adapted_state_dict

            # 加载权重
            try:
                # 尝试严格加载
                model.load_state_dict(state_dict, strict=False)
                dist_print(f"Successfully loaded weights from {fn}")

                # 检查是否有未加载的键
                missing_keys = model_keys - set(state_dict.keys())
                unexpected_keys = set(state_dict.keys()) - model_keys
                if missing_keys or unexpected_keys:
                    dist_print(
                        f"Note: Loaded with non-strict matching. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                    if missing_keys:
                        dist_print(f"Sample missing keys: {list(missing_keys)[:5]}...")
                    if unexpected_keys:
                        dist_print(f"Sample unexpected keys: {list(unexpected_keys)[:5]}...")
            except Exception as e:
                dist_print(f"Error loading weights: {e}")
                # 打印模型和权重的键差异，帮助调试
                missing_keys = model_keys - weight_keys
                unexpected_keys = weight_keys - model_keys
                if missing_keys:
                    print(f"Missing keys: {list(missing_keys)[:5]}...")
                if unexpected_keys:
                    print(f"Unexpected keys: {list(unexpected_keys)[:5]}...")
                continue  # Skip this weight file

            # 计算校准曲线，传递批次处理函数
            p, occurrence_rate = calibration_curve(
                model,
                batch_gen,
                dataset,
                process_batch_fn=process_batch_fn,
                batch_size=calibration_batch_size,
                max_samples=calibration_max_samples
            )

            # 确保结果是numpy数组 - 强制转换
            if isinstance(p, torch.Tensor):
                p = p.detach().cpu().numpy()
            if isinstance(occurrence_rate, torch.Tensor):
                occurrence_rate = occurrence_rate.detach().cpu().numpy()

            dist_print(f"Converted calibration data: type(p)={type(p)}, type(occurrence_rate)={type(occurrence_rate)}")

            # 确保数据是连续的C-style数组 (np.save需要)
            occurrence_rate = np.ascontiguousarray(occurrence_rate)

            # 保存结果 - 只在需要时保存
            if save_results:
                fn_root = os.path.splitext(os.path.basename(fn))[0]
                out_file = os.path.join(out_dir, f"calibration-{fn_root}.npy")
                np.save(out_file, occurrence_rate)
                dist_print(f"Successfully saved calibration data to {out_file}")
            else:
                dist_print(f"Calibration computed for {fn} (not saved due to distributed mode)")
        except Exception as e:
            dist_print(f"Error processing {fn}: {e}")
            import traceback
            traceback.print_exc()  # 打印完整错误堆栈
        finally:
            # 清理GPU内存
            torch.cuda.empty_cache()
            gc.collect()
            dist_print("Memory cleaned up")


class CalibratedModel(torch.nn.Module):
    """校准模型包装器 (PyTorch版本)"""

    def __init__(self, model, p, occurrence_rate):
        super().__init__()
        self.model = model
        self.calibrator = interp1d(
            p, occurrence_rate,
            kind='linear',
            bounds_error=False,
            fill_value=(0, occurrence_rate[-1]) # Force very low probabilities for out-of-bounds values
        )

    def forward(self, inputs):
        # 获取原始预测
        with torch.no_grad():
            raw_pred = self.model(inputs)

        # 转换为numpy进行校准
        if isinstance(raw_pred, torch.Tensor):
            raw_pred_np = raw_pred.cpu().numpy()
        else:
            raw_pred_np = raw_pred

        # 应用校准
        calibrated_np = self.calibrator(raw_pred_np)

        # 转换回tensor
        calibrated = torch.from_numpy(calibrated_np).to(raw_pred.device)

        # 确保值在[0,1]范围内
        return torch.clamp(calibrated, 0.0, 1.0)


def calibrated_model(model, p, occurrence_rate):
    """创建校准模型 (兼容接口)"""
    return CalibratedModel(model, p, occurrence_rate)


# 辅助函数
class BatchDataset(torch.utils.data.Dataset):
    """PyTorch Dataset适配器"""

    def __init__(self, batch_gen, dataset='train', device=None):
        self.batch_gen = batch_gen
        self.dataset = dataset
        self.device = device  # 存储设备信息
        self.length = len(batch_gen.time_coords[dataset]) // batch_gen.batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pred_batch, target_batch = self.batch_gen.batch(idx, self.dataset)

        # Convert to dictionary format instead of trying to concatenate tensors of different sizes
        X = {}
        for i, x in enumerate(pred_batch):
            if isinstance(x, torch.Tensor):
                tensor_x = x
            else:
                tensor_x = torch.from_numpy(x)

            # Use the same keys as in the model input
            key = f"input_{i}"
            X[key] = tensor_x

        # Handle target - convert to tensor if needed
        if isinstance(target_batch[0], torch.Tensor):
            Y = target_batch[0]
        else:
            Y = torch.from_numpy(target_batch[0])

        return X, Y
