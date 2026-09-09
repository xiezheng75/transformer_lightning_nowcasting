import os
import random
from datetime import datetime, timedelta
from itertools import chain
import numpy as np
import torch
from torch.utils.data import Dataset
import numba
from numba import njit, prange
from collections import defaultdict
from .image_loader import RadarImageGenerator  # 添加导入
from .lightning_data import LightningDataProcessor
import torch.nn.functional as F
import threading
from collections import OrderedDict


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

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank <= 0:
        print(*args, **kwargs)

# class BatchDataset(Dataset):
#     def __init__(self, batch_gen, dataset="train", use_cache=True):
#         self.batch_gen = batch_gen
#         self.dataset = dataset
#         self.use_cache = use_cache
#         # Cache length to avoid repeated calculation
#         self._length = self._calculate_length()
#         # Pre-generated batch cache for frequently accessed batches
#         self.batch_cache = {}
#         self.cache_hits = 0
#         self.cache_misses = 0
#
#     def _calculate_length(self):
#         # Check if batch_gen is None (happens in worker processes)
#         if self.batch_gen is None:
#             return 0
#
#         # Handle different batch generator types
#         if hasattr(self.batch_gen, 'time_coords'):
#             # Original BatchGenerator
#             return len(self.batch_gen.time_coords[self.dataset]) // self.batch_gen.batch_size
#         else:
#             # RadarBatchGenerator
#             if self.dataset == "train":
#                 return max(1, len(getattr(self.batch_gen, 'train_times', [])) // self.batch_gen.batch_size)
#             elif self.dataset == "valid":
#                 return max(1, len(getattr(self.batch_gen, 'valid_times', [])) // self.batch_gen.batch_size)
#             elif self.dataset == "test":
#                 return max(1, len(getattr(self.batch_gen, 'test_times', [])) // self.batch_gen.batch_size)
#             else:
#                 return 0
#
#     def __len__(self):
#         return self._length
#
#     def __getitem__(self, idx):
#         # Check if batch_gen is available
#         if self.batch_gen is None:
#             raise RuntimeError("batch_gen is None. This may happen in worker processes. "
#                              "Consider using num_workers=0 or implementing a worker_init_fn.")
#
#         # Try batch-level cache first
#         if self.use_cache and idx in self.batch_cache:
#             self.cache_hits += 1
#             if self.cache_hits % 100 == 0:
#                 hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
#                 dist_print(f"BatchDataset cache hit rate: {hit_rate:.3f}")
#             return self.batch_cache[idx]
#
#         self.cache_misses += 1
#
#         pred_batch, target_batch = self.batch_gen.batch(idx, dataset=self.dataset)
#
#         # Debug information
#         if idx % 100 == 0:  # 每100个batch才输出一次
#             if isinstance(pred_batch, dict):
#                 dist_print(f"Batch {idx} - Pred keys: {list(pred_batch.keys())}, Target shape: {target_batch.shape}")
#             else:
#                 dist_print(f"Batch {idx} - Pred type: {type(pred_batch)}, Target type: {type(target_batch)}")
#
#         # If batch generator is RadarBatchGenerator, it already returns dict and tensor
#         if isinstance(pred_batch, dict) and torch.is_tensor(target_batch):
#             # Save original shape for debugging
#             original_shape = target_batch.shape
#
#             # Make sure target has 5 dimensions [B, T, H, W, C]
#             if len(target_batch.shape) == 6:  # [1, B, T, H, W, C] - remove extra batch dim
#                 target_batch = target_batch.squeeze(0)
#             elif len(target_batch.shape) < 5:
#                 if len(target_batch.shape) == 3:  # [B, H, W]
#                     target_batch = target_batch.unsqueeze(1).unsqueeze(-1)  # [B, 1, H, W, 1]
#                 elif len(target_batch.shape) == 4:
#                     if target_batch.shape[1] <= 32:  # Likely [B, T, H, W]
#                         target_batch = target_batch.unsqueeze(-1)  # [B, T, H, W, 1]
#                     else:  # Likely [B, H, W, C]
#                         target_batch = target_batch.unsqueeze(1)  # [B, 1, H, W, C]
#
#             # Ensure values are between 0 and 1 for BCE loss
#             target_batch = torch.clamp(target_batch, 0.0, 1.0)
#
#             # Ensure model input has consistent dimensions with target
#             for key in pred_batch:
#                 input_tensor = pred_batch[key]
#                 if len(input_tensor.shape) == 5:  # [B, T, H, W, C]
#                     # Reshape target to match height and width if needed
#                     if input_tensor.shape[2:4] != target_batch.shape[2:4]:
#                         if idx % 100 == 0:  # 减少日志输出
#                             dist_print(
#                                 f"Warning: Input shape {input_tensor.shape} doesn't match target shape {target_batch.shape}")
#                         # We could resize target here if needed
#
#             if idx % 100 == 0:  # 减少日志输出
#                 dist_print(f"Target shape transformed from {original_shape} to {target_batch.shape}")
#         return pred_batch, target_batch
#
#     def on_epoch_end(self):
#         self.batch_gen.rng.shuffle(self.batch_gen.time_coords["train"])

# 添加数据缓存类
class DataCache:
    """Simple cache to store processed data in memory to avoid repeated disk I/O"""

    def __init__(self, max_size=10 , max_memory_gb=1.0):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.current_memory = 0
        self.access_count = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.Lock()

    def __getstate__(self):
        """Custom pickling handling to exclude lock"""
        state = self.__dict__.copy()
        # Remove the lock from the state since it can't be pickled
        state.pop('lock', None)
        return state

    def __setstate__(self, state):
        """Custom unpickling handling to recreate lock"""
        self.__dict__.update(state)
        self.lock = threading.Lock()

    def _estimate_size(self, data):
        """Estimate memory size of data"""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, torch.Tensor):
            return data.element_size() * data.nelement()
        else:
            return len(str(data))

    def get(self, key):
        # Handle case where lock might not exist (e.g., after unpickling)
        if not hasattr(self, 'lock'):
            self.lock = threading.Lock()

        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.access_count[key] = self.access_count.get(key, 0) + 1
                self.hit_count += 1
                return self.cache[key]
            self.miss_count += 1
            return None

    def put(self, key, value):
        # Handle case where lock might not exist (e.g., after unpickling)
        if not hasattr(self, 'lock'):
            self.lock = threading.Lock()

        with self.lock:
            value_size = self._estimate_size(value)
            # Remove items if exceeding memory limit
            while self.current_memory + value_size > self.max_memory_bytes and self.cache:
                oldest_key = next(iter(self.cache))
                oldest_value = self.cache.pop(oldest_key)
                self.current_memory -= self._estimate_size(oldest_value)
                self.access_count.pop(oldest_key, None)

            # Also check max_size limit
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = next(iter(self.cache))
                oldest_value = self.cache.pop(oldest_key)
                self.current_memory -= self._estimate_size(oldest_value)
                self.access_count.pop(oldest_key, None)

            self.cache[key] = value
            self.current_memory += value_size
            self.access_count[key] = 0

    def get_stats(self):
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(total_requests, 1) if total_requests > 0 else 0
        return {
            'hit_rate': hit_rate,
            'size_gb': self.current_memory / (1024 ** 3),
            'num_items': len(self.cache),
            'hits': self.hit_count,
            'misses': self.miss_count
        }
class RadarSampleDataset(Dataset):
    """Dataset that returns individual samples instead of batches"""

    def __init__(self, batch_gen, dataset="train"):
        self.batch_gen = batch_gen
        self.dataset = dataset
        self.static_data = batch_gen.static_data if hasattr(batch_gen, 'static_data') else {}

        # Get all time indices for this dataset
        if dataset == "train":
            self.times = batch_gen.train_times
        elif dataset == "valid":
            self.times = batch_gen.valid_times
        else:
            self.times = batch_gen.test_times

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        """Return a single sample instead of a batch"""
        start_time = self.times[idx]

        # For training, implement lightning-rich sampling
        if self.dataset == "train" and hasattr(self.batch_gen, 'lightning_rich_samples'):
            if idx in self.batch_gen.lightning_rich_samples['train']:
                pass
            elif self.batch_gen.rng.random() < self.batch_gen.lightning_sample_prob:
                rich_samples = self.batch_gen.lightning_rich_samples['train']
                if rich_samples:
                    rich_idx = self.batch_gen.rng.choice(rich_samples)
                    start_time = self.times[rich_idx]

        # Load single radar sequence
        radar_data = self.batch_gen.load_radar_sequence(start_time)
        if radar_data is None:
            radar_data = np.zeros((self.batch_gen.past_timesteps, *self.batch_gen.img_size, 3), dtype=np.float32)

        # Convert to tensor
        inputs = torch.from_numpy(radar_data.astype(np.float32))

        # Build input dictionary
        input_dict = {"radar_past": inputs}

        # Add static data if available
        if self.static_data:
            for name, data in self.static_data.items():
                # Static data should have shape [H, W], expand to [1, H, W] for single sample
                static_tensor = torch.tensor(data, dtype=torch.float32)
                if static_tensor.dim() == 2:  # [H, W]
                    static_tensor = static_tensor.unsqueeze(0)  # [1, H, W]
                input_dict[f"static_{name}"] = static_tensor

        # Load lightning targets for this single sample
        targets = torch.zeros(self.batch_gen.future_timesteps, *self.batch_gen.img_size, dtype=torch.float32)

        lightning_time_window = 6
        lightning_spatial_radius = 8
        lightning_field = 'FLASH'
        lightning_flash_type = None
        lightning_binary = True

        if self.batch_gen.lightning_processor:
            for j in range(self.batch_gen.future_timesteps):
                future_time = start_time + timedelta(minutes=6 * (self.batch_gen.past_timesteps + j))
                lightning_grid = self.batch_gen.lightning_processor.create_lightning_grid(
                    future_time,
                    field=lightning_field,
                    flash_type=lightning_flash_type,
                    time_window_minutes=lightning_time_window,
                    spatial_radius_km=lightning_spatial_radius,
                    binary=lightning_binary
                )
                if lightning_grid is not None:
                    # Binary conversion if needed
                    if lightning_binary:
                        lightning_grid = (lightning_grid > 0).to(torch.uint8)

                    # Clamp values
                    lightning_grid = torch.clamp(lightning_grid, 0.0, 1.0)

                    # Resize to match radar dimensions
                    targets[j] = F.interpolate(
                        lightning_grid.unsqueeze(0).unsqueeze(0).float(),
                        size=self.batch_gen.img_size,
                        mode='nearest'
                    )[0, 0]

        # Apply data augmentation for training
        if self.dataset == "train" and self.batch_gen.augmentation_config.get('enabled', False):
            # Apply augmentation to single sample
            targets = self.batch_gen._augment_single_lightning_target(targets, inputs)

        # # Validate tensors before returning
        # inputs = self._validate_single_tensor(inputs, "inputs")
        # targets = self._validate_single_tensor(targets, "targets")

        # Return single sample with target shape [T, H, W, 1]
        return input_dict, targets.unsqueeze(-1)

    def _validate_single_tensor(self, tensor, name):
        """Validate single tensor data"""
        if torch.isnan(tensor).any():
            tensor = torch.nan_to_num(tensor, 0.0)
        if torch.isinf(tensor).any():
            tensor = torch.clamp(tensor, -1e6, 1e6)
        return tensor

class RadarBatchGenerator:
    """雷达图像批生成器 (独立实现)"""

    def __init__(self, data_root, img_size, timesteps, batch_size=None,
                 lightning_csv=None,  # 添加闪电数据路径
                 cache_index=None, # Add cache file path parameter
                 parallel_workers=None,  # Add parallel workers parameter
                 prefetch_factor=1,  # 预取因子
                 cache_ttl=2592000, # Cache time-to-live in seconds (default: 30 day)
                 max_files_per_product=None,  # Limit files processed per product
                 static_data_config=None,
                 valid_frac=0.1, test_frac=0.1, random_seed=42,
                 lightning_sample_prob=0.5,# 添加闪电样本采样概率):
                 augmentation_config=None, # 添加数据增强配置
                 results_dir=None,  # 结果保存根目录
                 ram_cache_size=5000,  # RAM缓存项目数
                 ram_cache_memory_gb=64.0,  # RAM缓存内存限制(GB)
                 disk_cache_dir=None,  # 磁盘挂载
                 split_mode="chronological",  # 'chronological' (leakage-free) or 'random' (legacy, leaky)
                 split_gap=None  # buffer frames dropped between splits; default = past+future window length
                 ):
        """
        data_root: 雷达数据根目录
        img_size: 图像尺寸 (height, width),700*800
        timesteps: 时间步数 (past, future)
        batch_size: 批大小 (can be None when using RadarSampleDataset)
        cache_index: 缓存索引文件路径
        parallel_workers: 并行工作进程数
        cache_ttl: 缓存有效期（秒）
        max_files_per_product: 每种产品最大处理文件数
        valid_frac: 验证集比例
        test_frac: 测试集比例
        random_seed: 随机种子
        lightning_sample_prob: 闪电样本采样概率
        results_dir: 结果保存根目录
        ram_cache_size: 内存缓存的最大样本数
        ram_cache_memory_gb: 内存缓存的最大占用(GB)
        disk_cache_dir: 磁盘挂载缓存目录。设为None可完全禁用磁盘读写。
        """
        self.data_root = data_root
        self.img_size = img_size # 700*800尺寸
        self.past_timesteps, self.future_timesteps = timesteps
        self.timesteps = timesteps
        self.batch_size = batch_size # Can be None when using RadarSampleDataset
        self.cache_index = cache_index
        self.parallel_workers = parallel_workers if parallel_workers else os.cpu_count()
        self.cache_ttl = cache_ttl
        self.max_files_per_product = max_files_per_product
        self.prefetch_factor = prefetch_factor
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.lightning_sample_prob = lightning_sample_prob
        # Data-split protocol. 'chronological' performs a leakage-free split:
        # contiguous train | gap | valid | gap | test in time order, with a buffer
        # gap so that no sample window is shared across splits. 'random' reproduces
        # the legacy (leaky) behaviour. See _split_datasets.
        self.split_mode = split_mode
        self.split_gap = split_gap

        # 设置结果目录
        if results_dir is None:
            results_dir = os.path.join(os.path.dirname(data_root), "radar_results")
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # 数据增强配置
        self.augmentation_config = augmentation_config or {
            'enabled': True,
            'augment_prob': 0.3,  # 总体增强概率
            'spatial_expansion': {
                'enabled': True,
                'iterations': 2,
                'confidence': 0.8
            },
            'temporal_propagation': {
                'enabled': True,
                'forward_prob': 0.5,
                'backward_prob': 0.5,
                'decay_factor': 0.5
            },
            'convection_based': {
                'enabled': True,
                'prob': 0.3,
                'dbz_threshold': 0.7,  # 归一化后的DBZ阈值
                'min_pixels': 100,
                'max_centers': 3,
                'radius_range': [5, 15],
                'confidence': 0.6,
                'max_duration': 3
            }
        }

        # 增强统计
        self.augmentation_stats = {
            'total_augmented': 0,
            'batches_augmented': 0
        }

        # Initialize data cache with user configured limits
        # Using configured limits to leverage available RAM (Node 1 has ~340GB free)
        dist_print(f"Initializing RAM cache: {ram_cache_size} items, limit {ram_cache_memory_gb:.1f} GB")
        self.data_cache = DataCache(max_size=ram_cache_size, max_memory_gb=ram_cache_memory_gb)

        # 添加与BatchGenerator兼容的属性
        self.pred_names_past = ["radar"]
        self.pred_names_future = []

        # Initialize static predictor names from config if available
        self.pred_names_static = []
        if static_data_config:
            # Extract expected static data fields from config
            if static_data_config.get('dem', {}).get('enabled', False):
                self.pred_names_static.append('dem')
            if static_data_config.get('land_cover', {}).get('enabled', False):
                self.pred_names_static.append('land_cover')
            # Log expected static fields
            if self.pred_names_static:
                dist_print(f"Expecting static data fields: {self.pred_names_static}")
        # Initialize input specs with past timeframe
        self.input_specs = [{"timeframe": "past", "shape_divisor": 1}]
        # Add static data timeframe to input specs if we expect static data
        if static_data_config and self.pred_names_static:
            self.input_specs.append({"timeframe": "static", "shape_divisor": 1})

        # 初始化雷达图像加载器
        self.radar_loader = RadarImageGenerator(
            data_root=data_root,
            products=['dbz', 'dbzh', 'vil'],
            img_size=img_size, # 700*800尺寸
            timesteps=timesteps,
            parallel_workers=parallel_workers,  # Pass parallel workers parameter
            prefetch_factor=self.prefetch_factor,  # Pass prefetch factor parameter
            max_files_per_product=max_files_per_product,  # Pass max files parameter
            cache_dir=disk_cache_dir,  # Pass disk cache directory (None to disable)
            filename_pattern=[
                # For dbz, laid out as <data_root>/CR/YYYY/YYYYMMDD/ref_all_YYYYMMDDHHMM_14.ref
                r'ref_all_(\d{12})_14\.ref',
                r'dbzh_(\d{12})\.png',  # For dbzh: dbzh_202104090000.png
                r'vil_(\d{12})\.png'  # For vil: vil_202104090000.png
            ]
        )
        dist_print(f"Initialized RadarImageGenerator with {self.radar_loader.parallel_workers} workers")

        # Optional one-off sanity checks on the reader; each writes diagnostic
        # figures to the directory it is given, e.g. `self.results_dir`:
        #   self.radar_loader.validate_data_loading(output_dir=...)
        #   self.radar_loader.validate_colorbar_extraction(output_dir=...)
        #   self.radar_loader.validate_background_masking(output_dir=...)

        # 如果有缓存文件，尝试加载
        self._load_cache_index()

        # 建立时间索引
        self.all_times = sorted(self.radar_loader.time_index.keys())
        if not self.all_times:
            raise ValueError("No valid time indices found. Check data directory and file naming pattern.")

        # 划分训练/验证/测试集
        self._split_datasets(valid_frac, test_frac)

        # 保存缓存文件
        self._save_cache_index()

        # Store static data config
        self.static_data_config = static_data_config
        self.static_data = {}

        # Load static data if configured
        if self.static_data_config:
            self._load_static_data()

        # 闪电数据处理
        self.lightning_processor = None
        self.lightning_rich_samples = {'train': [], 'valid': [], 'test': []}  # 初始化

        if lightning_csv:
            # 定义网格信息 (与雷达图像匹配)
            grid_info = {
                'lon_min': 109.505,
                'lon_max': 117.495,
                'lat_min': 19.0519,
                'lat_max': 26.0419,
                'resolution': 0.01,
                'width': 800,
                'height': 700
            }
            # Determine cache directory for lightning data
            lightning_cache_dir = None
            if self.cache_index:
                 lightning_cache_dir = os.path.dirname(os.path.abspath(self.cache_index))

            # 初始化闪电数据处理器
            self.lightning_processor = LightningDataProcessor(lightning_csv, grid_info, cache_dir=lightning_cache_dir)
            self.lightning_processor.load_and_preprocess()

            # 扫描含有闪电的样本
            self.lightning_rich_samples = self._find_lightning_rich_samples()

            # Visualize only ground strikes (CGFLASH=0)
            analysis_dir = os.path.join(self.results_dir, "lightning_analysis")
            self.lightning_processor.visualize_lightning_distribution(output_dir=analysis_dir, field="FLASH")

            # 验证闪电数据时间覆盖
            dist_print("\n=== Lightning Data Coverage ===")
            sample_times = random.sample(self.all_times, min(10, len(self.all_times)))
            lightning_found = 0
            for t in sample_times:
                test_grid = self.lightning_processor.create_lightning_grid(
                    t,
                    field='FLASH',
                    flash_type=None,
                    time_window_minutes=6,
                    spatial_radius_km=8
                )
                if torch.sum(test_grid > 0) > 0:
                    lightning_found += 1
            dist_print(f"Lightning data found in {lightning_found}/{len(sample_times)} test samples")

            self.grid_info = grid_info

        # 更新目标名称
        self.target_names = ["lightning"]  # 使用闪电作为目标

    def _find_lightning_rich_samples(self):
        """预先识别含有丰富闪电数据的时间段（使用快速筛选）"""
        lightning_rich = {'train': [], 'valid': [], 'test': []}

        if self.lightning_processor is None:
            return lightning_rich

        dist_print("Scanning for lightning-rich time periods (Fast Mode)...")

        time_window = 6
        min_events = 1000

        for dataset in ['train', 'valid', 'test']:
            times = getattr(self, f'{dataset}_times', [])
            if not times:
                continue

            # 使用 LightningDataProcessor 的快速筛选
            # 直接传入时间列表，返回满足条件的索引
            rich_indices = self.lightning_processor.find_lightning_rich_indices(
                times,
                time_window_minutes=time_window,
                min_events=min_events
            )

            lightning_rich[dataset] = rich_indices

        dist_print(f"Found lightning-rich samples - Train: {len(lightning_rich['train'])}, "
                   f"Valid: {len(lightning_rich['valid'])}, Test: {len(lightning_rich['test'])}")
        return lightning_rich

    def _load_static_data(self):
        """Load all configured static data sources"""
        try:

            from c4dllightning.features.static_data_loader import load_static_data
            # Create results directory for static data visualizations
            static_viz_dir = os.path.join(self.results_dir, 'static_data')
            os.makedirs(static_viz_dir, exist_ok=True)

            # Load static data with visualization
            self.static_data = load_static_data(self.static_data_config, results_dir=static_viz_dir)

            # Log success
            dist_print(f"Loaded static data: {list(self.static_data.keys())}")

            # Print data shape and statistics for each static data source
            for name, data in self.static_data.items():
                dist_print(f"  {name} shape: {data.shape}, range: [{np.min(data):.2f}, {np.max(data):.2f}]")

            # Update pred_names_static with the loaded data sources
            self.pred_names_static = list(self.static_data.keys())

        except ImportError:
            dist_print("static_data_loader module not found. Static data will not be available.")
            self.static_data = {}
        except Exception as e:
            dist_print(f"Error loading static data: {e}")
            import traceback
            traceback.print_exc()
            self.static_data = {}

    def _load_cache_index(self):
        """加载时间索引缓存"""
        if not self.cache_index:
            return

        if os.path.exists(self.cache_index):
            try:
                import pickle
                from datetime import datetime

                with open(self.cache_index, 'rb') as f:
                    cache_data = pickle.load(f)

                # 验证缓存是否过期
                cache_time = cache_data.get('timestamp', 0)
                current_time = datetime.now().timestamp()

                if current_time - cache_time <= self.cache_ttl:
                    dist_print(f"Loading index cache from {self.cache_index}")
                    self.radar_loader.time_index = cache_data.get('time_index', {})
                    dist_print(f"Loaded {len(self.radar_loader.time_index)} time indices from cache")
                else:
                    dist_print(f"Cache expired (age: {(current_time - cache_time) // 3600:.1f} hours)")
            except Exception as e:
                dist_print(f"Error loading cache: {e}")

    def _save_cache_index(self):
        """保存时间索引缓存"""
        # Add rank check to prevent race conditions
        import os
        if int(os.environ.get('LOCAL_RANK', -1)) > 0:
            return

        if not self.cache_index or not self.radar_loader.time_index:
            return

        try:
            import pickle
            from datetime import datetime

            os.makedirs(os.path.dirname(os.path.abspath(self.cache_index)), exist_ok=True)

            # 创建包含时间戳的缓存数据
            cache_data = {
                'timestamp': datetime.now().timestamp(),
                'time_index': self.radar_loader.time_index
            }

            with open(self.cache_index, 'wb') as f:
                pickle.dump(cache_data, f)

            dist_print(f"Saved {len(self.radar_loader.time_index)} time indices to {self.cache_index}")
        except Exception as e:
            dist_print(f"Error saving cache: {e}")

    def _split_datasets(self, valid_frac, test_frac):
        """Split the (chronologically sorted) time index into train/valid/test.

        Each sample that starts at time index k uses radar frames spanning the
        window [k, k + past + future - 1] (input frames 0..past-1 and target
        frames past..past+future-1). With the legacy 'random' split, neighbouring
        windows that overlap by ~11/12 frames can be scattered into different
        splits, so the *same* radar frame and lightning label appear in both train
        and test (temporal data leakage), inflating skill scores.

        The default 'chronological' mode performs a leakage-free split: the sorted
        times are partitioned into three contiguous blocks in time order
        (train | gap | valid | gap | test). A buffer gap of `split_gap` frames is
        discarded at each boundary so that no window in one split shares any frame
        with another split. A gap >= (past + future) guarantees this.
        """
        n = len(self.all_times)
        n_valid = int(n * valid_frac)
        n_test = int(n * test_frac)

        mode = getattr(self, "split_mode", "chronological")

        if mode == "random":
            # ---- Legacy (leaky) behaviour: random timestamp shuffle ----
            n_train = n - n_valid - n_test
            indices = np.arange(n)
            self.rng.shuffle(indices)
            self.train_times = [self.all_times[i] for i in indices[:n_train]]
            self.valid_times = [self.all_times[i] for i in indices[n_train:n_train + n_valid]]
            self.test_times = [self.all_times[i] for i in indices[n_train + n_valid:]]
            dist_print(
                "WARNING: using legacy 'random' split (temporally leaky). "
                "Set split_mode='chronological' for a leakage-free split."
            )
            self._log_split_summary()
            return

        # ---- Leakage-free chronological split with buffer gaps ----
        window = int(self.past_timesteps + self.future_timesteps)  # frames per sample
        gap = self.split_gap if getattr(self, "split_gap", None) else window
        gap = int(gap)

        n_train = n - n_valid - n_test - 2 * gap
        if n_train <= 0:
            # Dataset too small for the requested gap; shrink the gap to fit.
            gap = max(0, (n - n_valid - n_test) // 2 - 1)
            n_train = n - n_valid - n_test - 2 * gap
            dist_print(f"WARNING: split gap reduced to {gap} frames (dataset has only {n} times).")

        train_end = n_train
        valid_start = train_end + gap
        valid_end = valid_start + n_valid
        test_start = valid_end + gap
        test_end = n

        # all_times is sorted ascending; slices are contiguous in time
        self.train_times = list(self.all_times[:train_end])
        self.valid_times = list(self.all_times[valid_start:valid_end])
        self.test_times = list(self.all_times[test_start:test_end])

        dist_print(
            f"Chronological split: train={len(self.train_times)} | gap={gap} | "
            f"valid={len(self.valid_times)} | gap={gap} | test={len(self.test_times)} "
            f"(window={window} frames)"
        )
        self._log_split_summary()

    def _log_split_summary(self):
        """Print the time span of each split for verification."""
        def _span(times):
            if not times:
                return "empty"
            try:
                lo, hi = min(times), max(times)
                return f"{lo} -> {hi}"
            except Exception:
                return f"{len(times)} samples"
        dist_print(f"  train span: {_span(self.train_times)}")
        dist_print(f"  valid span: {_span(self.valid_times)}")
        dist_print(f"  test  span: {_span(self.test_times)}")

        # # 添加时间点列表 (用于兼容性)
        # self.time_coords = {
        #     "train": self.train_times,
        #     "valid": self.valid_times,
        #     "test": self.test_times
        # }

    def _get_time_batch(self, idx, dataset="train"):
        """
        This method is deprecated when using RadarSampleDataset.
        Kept for backward compatibility only.
        """
        if self.batch_size is None:
            raise ValueError("_get_time_batch should not be called when using RadarSampleDataset")

        # Original implementation for backward compatibility
        if dataset == "train":
            times = self.train_times
            # 训练时以一定概率强制选择含闪电的样本
            if self.lightning_rich_samples['train'] and self.rng.random() < self.lightning_sample_prob:
                # 选择含闪电的批次
                rich_indices = self.lightning_rich_samples['train']
                if rich_indices:
                    # 随机选择一些含闪电的索引
                    n_lightning = max(1, int(self.batch_size * 0.5))  # 至少50%含闪电
                    selected_rich = self.rng.choice(rich_indices, min(n_lightning, len(rich_indices)), replace=False)

                    # 从选中的索引获取时间
                    batch = [times[i] for i in selected_rich]

                    # 如果不够batch_size，补充其他样本
                    if len(batch) < self.batch_size:
                        remaining = self.batch_size - len(batch)
                        other_indices = [i for i in range(len(times)) if i not in selected_rich]
                        if other_indices:
                            extra_indices = self.rng.choice(other_indices, min(remaining, len(other_indices)),
                                                            replace=False)
                            batch.extend([times[i] for i in extra_indices])

                    dist_print(f"Batch {idx}: Forced {len(selected_rich)} lightning-rich samples")
                    return batch[:self.batch_size]

        elif dataset == "valid":
            times = self.valid_times
        elif dataset == "test":
            times = self.test_times
        else:
            raise ValueError(f"Invalid dataset: {dataset}")

        # 防止索引超出范围 - 使用模运算循环使用数据
        if len(times) == 0:
            return []

        # 使用循环索引确保所有GPU都有数据
        start = (idx * self.batch_size) % len(times)
        end = min(start + self.batch_size, len(times))

        # 如果到达末尾但批次不足，从头补充
        batch = times[start:end]
        if len(batch) < self.batch_size and len(times) > 0:
            remaining = self.batch_size - len(batch)
            batch.extend(times[:remaining])

        return batch

    def load_radar_sequence(self, start_time):
        """
        加载单个雷达时间序列
        start_time: 起始时间 (datetime)
        返回: past_data (numpy array)

        Note: This method is designed to be called by DataLoader workers,
        so it should NOT spawn additional parallel processes.
        """
        try:
            # Check if data is in cache first
            cache_key = f"radar_seq_{start_time.strftime('%Y%m%d%H%M%S')}"
            cached_data = self.data_cache.get(cache_key)
            if cached_data is not None:
                # 验证缓存数据
                if not np.all(cached_data == 0):
                    return cached_data
                else:
                    # 清除无效缓存
                    self.data_cache.cache.pop(cache_key, None)

            # Load from disk (sequential loading - no parallel spawning)
            past_data = self.radar_loader.load_time_series(
                start_time=start_time,
                time_steps=self.past_timesteps
            )

            # 验证数据有效性
            if past_data is None:
                return None

            if isinstance(past_data, np.ndarray):
                if np.all(past_data == 0) or np.any(np.isnan(past_data)) or np.any(np.isinf(past_data)):
                    return None

            # Store in cache for future use
            self.data_cache.put(cache_key, past_data)
            return past_data

        except Exception as e:
            dist_print(f"Error loading radar sequence at {start_time}: {e}")
            return None

    # 实现与BatchGenerator相同的接口方法
    def __len__(self):
        """数据集长度 (训练集)"""
        # When using RadarSampleDataset, return number of samples not batches
        # This should return the length of the training set only
        # Used by DataLoader to determine dataset size
        return len(self.train_times)

    def get_num_batches(self, dataset="train", batch_size=32):
        """Get number of batches for a given dataset and batch size

        Args:
            dataset: "train", "valid", or "test"
            batch_size: batch size for calculation

        Returns:
            Number of batches (rounded up)
        """
        if dataset == "train":
            num_samples = len(self.train_times)
        elif dataset == "valid":
            num_samples = len(self.valid_times)
        else:
            num_samples = len(self.test_times)

        # Use ceiling division to ensure all samples are covered
        return (num_samples + batch_size - 1) // batch_size

    def batch(self, idx, dataset="train"):
        """[DEPRECATED] Use RadarSampleDataset with DataLoader instead"""
        raise NotImplementedError(
            "batch() method is deprecated. Use RadarSampleDataset with DataLoader instead. "
            "DataLoader will automatically handle batching."
        )
        """生成批次数据"""
        # 获取时间批次
        time_batch = self._get_time_batch(idx, dataset)

        if not time_batch:
            raise ValueError(f"Empty time batch for index {idx} in {dataset} dataset")

        # 智能设备选择
        if torch.cuda.is_available() and not torch.distributed.is_initialized():
            # 单GPU训练时直接在GPU上创建张量
            device = torch.device('cuda:0')
        else:
            # CPU运行或分布式训练时使用CPU
            device = torch.device('cpu')

        # 检查缓存
        cache_key = f"{dataset}_batch_{idx}"
        cached_batch = self.data_cache.get(cache_key)
        if cached_batch is not None and idx % 100 == 0:
            stats = self.data_cache.get_stats()
            dist_print(f"Cache hit for batch {idx}! Stats: hit_rate={stats['hit_rate']:.2f}, size={stats['size_gb']:.1f}GB")
            return cached_batch

        # 收集有效的时间样本
        valid_samples = []
        valid_times = []
        max_retries = len(time_batch) * 2  # 最大重试次数
        retry_count = 0

        # 并行加载所有时间点的数据
        parallel_workers = min(8, max(2, len(time_batch) // 2))  # 动态调整并行度
        all_radar_data = self.load_radar_sequences_parallel(time_batch, max_workers=parallel_workers)

        # 验证并收集有效数据
        for t, radar_data in zip(time_batch, all_radar_data):
            if radar_data is not None:
                # 额外验证数据有效性
                if isinstance(radar_data, np.ndarray):
                    if not np.all(radar_data == 0) and not np.any(np.isnan(radar_data)):
                        valid_samples.append(radar_data)
                        valid_times.append(t)
                    else:
                        dist_print(f"Skipping time {t}: invalid data (all zeros or contains NaN)")
                elif isinstance(radar_data, torch.Tensor):
                    if not torch.all(radar_data == 0) and not torch.any(torch.isnan(radar_data)):
                        valid_samples.append(radar_data)
                        valid_times.append(t)
                    else:
                        dist_print(f"Skipping time {t}: invalid tensor data")
            else:
                if retry_count < 5:  # 只打印前5次警告
                    dist_print(f"Skipping time {t} due to loading failure")
                retry_count += 1

        # 如果没有足够的有效样本，尝试使用备用时间点
        if len(valid_samples) < self.batch_size:
            dist_print(f"WARNING: Only {len(valid_samples)} valid samples in batch {idx}, trying backup times")

            # 根据数据集选择备用时间列表
            if dataset == "train":
                backup_times = self.train_times[:]
            elif dataset == "valid":
                backup_times = self.valid_times[:]
            else:
                backup_times = self.test_times[:]

            # 打乱备用时间列表以避免总是使用相同的备用数据
            import random
            random.shuffle(backup_times)

            # 尝试从备用时间中加载数据
            backup_candidates = [t for t in backup_times[:max_retries]
                                 if t not in time_batch and t not in valid_times]

            if backup_candidates:
                # 并行加载备用数据
                backup_data = self.load_radar_sequences_parallel(
                    backup_candidates,
                    max_workers=min(4, len(backup_candidates))
                )

                for t, radar_data in zip(backup_candidates, backup_data):
                    if radar_data is not None:
                        if isinstance(radar_data, np.ndarray):
                            if not np.all(radar_data == 0) and not np.any(np.isnan(radar_data)):
                                valid_samples.append(radar_data)
                                valid_times.append(t)
                                if len(valid_samples) >= self.batch_size:
                                    break
                        elif isinstance(radar_data, torch.Tensor):
                            if not torch.all(radar_data == 0) and not torch.any(torch.isnan(radar_data)):
                                valid_samples.append(radar_data)
                                valid_times.append(t)
                                if len(valid_samples) >= self.batch_size:
                                    break

        # 如果仍然没有任何有效样本，使用随机噪声作为最后手段
        if len(valid_samples) == 0:
            dist_print(f"ERROR: Cannot find any valid samples for batch {idx}, using random noise")
            # 创建带有少量随机噪声的批次（避免全零导致梯度消失）
            noise_data = np.random.randn(self.past_timesteps, *self.img_size, 3) * 0.01
            valid_samples = [noise_data]
            valid_times = [self.train_times[0] if self.train_times else datetime.now()]

        # 填充到批次大小
        while len(valid_samples) < self.batch_size:
            # 循环使用有效样本
            idx_to_repeat = len(valid_samples) % len(valid_samples) if valid_samples else 0
            valid_samples.append(valid_samples[idx_to_repeat])
            valid_times.append(valid_times[idx_to_repeat])

        # 确保批次大小正确
        valid_samples = valid_samples[:self.batch_size]
        valid_times = valid_times[:self.batch_size]

        actual_batch_size = len(valid_samples)
        if actual_batch_size < self.batch_size:
            dist_print(f"Batch {idx}: {actual_batch_size} valid samples out of {self.batch_size} requested")

        # 定期清理缓存以防止内存累积
        if idx % 1000 == 0:
            if hasattr(self, 'data_cache'):
                # 清理缓存
                if hasattr(self.data_cache, 'cache'):
                    cache_size = len(self.data_cache.cache)
                    if cache_size > 100:
                        # 只清理一半缓存，保留最近使用的
                        keys_to_remove = list(self.data_cache.cache.keys())[:cache_size//2]
                        for key in keys_to_remove:
                            self.data_cache.cache.pop(key, None)
                        if hasattr(self.data_cache, 'access_count'):
                            for key in keys_to_remove:
                                self.data_cache.access_count.pop(key, None)
                        dist_print(f"Partially cleared cache at batch {idx}")
            # 清理Python垃圾回收
            import gc
            gc.collect()

        # # 确定正确的设备
        # # DataLoader worker进程应该始终使用CPU
        # if torch.utils.data.get_worker_info() is not None:
        #     # 在DataLoader worker进程中，始终使用CPU
        #     device = torch.device('cpu')
        # else:
        #     # 只有在主进程中才使用GPU
        #     device_id = int(os.environ.get('LOCAL_RANK', 0))
        #     device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

        # 初始化输入和目标张量
        inputs = torch.zeros(
            len(time_batch), self.past_timesteps, *self.img_size, 3,
            device='cpu', dtype=torch.float32  # Use float16 to reduce memory
        )

        # Initialize static data tensors if available
        static_tensors = {}
        if self.static_data:
            for name, data in self.static_data.items():
                # Create tensors for each static data source with correct shape and device
                static_tensors[name] = torch.tensor(data, dtype=torch.float32).to(device)
                # Log static data being added
                # dist_print(f"Added static data '{name}' with shape {static_tensors[name].shape}")

        # 目标改为闪电网格- 使用uint8因为是二值化数据
        targets = torch.zeros(
            len(time_batch), self.future_timesteps, *self.img_size,
            device='cpu', dtype=torch.uint8  # Binary data only needs uint8
        )

        # 用于时间对齐检查的列表
        batch_time_info = []

        # 控制可视化的变量
        min_lightning_strikes = 250  # 至少需要多少闪电点才进行可视化
        visualize_max_samples = 4  # 每个批次最多可视化的样本数

        lightning_time_window = 6
        lightning_spatial_radius = 8
        lightning_smoothing = False  # 是否应用平滑
        lightning_field = 'FLASH'  # 使用的闪电字段 ('FLASH'或'CGFLASH')
        lightning_flash_type = None  # 闪电类型 (0=地闪, 1=云闪)
        lightning_binary = True      #将闪电数据二值化

        # 加载每个有效时间序列
        for i, (past, t) in enumerate(zip(valid_samples, valid_times)):
            # 记录每个样本的时间信息
            sample_times = {
                "start_time": t,
                "input_times": [],
                "target_times": [],
                "has_lightning": False
            }

            try:
                # 将雷达数据转换为张量 - 保持在CPU上
                if isinstance(past, np.ndarray):
                    inputs[i] = torch.from_numpy(past.astype(np.float16)).to('cpu')
                elif isinstance(past, torch.Tensor):
                    inputs[i] = past.to(dtype=torch.float16).to('cpu')
                else:
                    # 尝试转换未知类型
                    inputs[i] = torch.tensor(past, dtype=torch.float16, device='cpu')

                # 记录输入时间戳
                for j in range(self.past_timesteps):
                    input_time = t + timedelta(minutes=6 * j)
                    sample_times["input_times"].append(input_time)

                # 加载未来时间步的闪电数据
                if self.lightning_processor:
                    try:
                        lightning_found_in_batch = False  # 添加批次级别的闪电检测
                        for j in range(self.future_timesteps):
                            future_time = t + timedelta(minutes=6 * (self.past_timesteps + j))
                            sample_times["target_times"].append(future_time)

                            # 使用增强的闪电网格生成（带空间扩展）
                            lightning_grid = self.lightning_processor.create_lightning_grid(
                                future_time,
                                field=lightning_field,
                                flash_type=lightning_flash_type,
                                time_window_minutes=lightning_time_window,
                                smoothing=lightning_smoothing,
                                spatial_radius_km=lightning_spatial_radius,  # 添加空间扩展
                                binary = lightning_binary
                            )
                            # 保持在CPU上，使用uint8存储二值化闪电数据
                            if lightning_binary:
                                lightning_grid = (lightning_grid > 0).to(torch.uint8)

                            # 检查是否有闪电数据并打印详细信息
                            non_zero_count = torch.sum(lightning_grid > 0).item()
                            if non_zero_count > 0:
                                sample_times["has_lightning"] = True
                                lightning_found_in_batch = True

                                # 计算覆盖率
                                coverage = non_zero_count / (self.img_size[0] * self.img_size[1]) * 100

                                # 每个批次打印一次增强后的闪电统计
                                if idx % 10 == 0 and j == 0 and i == 0:
                                    dist_print(f"\n=== Enhanced Lightning Data ===")
                                    dist_print(f"✓ Time: {future_time}, Spatial radius: {lightning_spatial_radius}km")
                                    dist_print(f"  Coverage: {coverage:.2f}% ({non_zero_count} pixels)")
                                    dist_print(f"  Time window: {lightning_time_window} minutes")
                                    non_zero_values = lightning_grid[lightning_grid > 0]
                                    dist_print(
                                        f"  Value range: [{non_zero_values.min().item():.4f}, {non_zero_values.max().item():.4f}]")
                                # dist_print(f"Found lightning data at {future_time}: {non_zero_count} points")
                                # dist_print(
                                #     f"  Values: min={non_zero_values.min().item():.4f}, max={non_zero_values.max().item():.4f}")
                                # dist_print(f"  First 10 non-zero positions: {torch.nonzero(lightning_grid)[:10]}")
                                # dist_print(f"  First 10 non-zero values: {non_zero_values[:10]}")

                            # Adjust lightning grid values to be between 0 and 1
                            lightning_grid = torch.clamp(lightning_grid, 0.0, 1.0)

                            # Resize lightning grid to match radar image dimensions
                            lightning_grid = F.interpolate(
                                lightning_grid.unsqueeze(0).unsqueeze(0),
                                size=self.img_size,
                                mode='nearest',
                                align_corners=None
                            )[0, 0]

                            targets[i, j] = lightning_grid

                            # 可视化闪电网格 - 修改为根据闪电数据存在与否决定是否可视化
                            non_zero_count = torch.sum(lightning_grid > 0).item()
                            if non_zero_count >= min_lightning_strikes:
                                # 转到CPU进行可视化，避免GPU内存问题
                                cpu_lightning_grid = lightning_grid.cpu()
                                cpu_radar_img = inputs[i, -1].cpu().numpy()

                                # self.visualize_lightning_grid(
                                #     lightning_grid=cpu_lightning_grid,
                                #     radar_img=cpu_radar_img,  # 最后一帧雷达图像
                                #     future_time=future_time,
                                #     dataset=dataset,
                                #     idx=idx,
                                #     i=i,  # Pass sample index
                                #     j=j,  # Pass timestep index
                                #     field=lightning_field,
                                #     flash_type=lightning_flash_type,
                                #     time_window=lightning_time_window,
                                #     smoothing=lightning_smoothing
                                # )
                                # dist_print(
                                #     f"Visualized sample with {non_zero_count} lightning strikes: batch={idx}, sample={i}, timestep={j}")
                    except Exception as e:
                        if idx % 100 == 0:  # 减少错误输出频率
                            dist_print(f"Error processing lightning data for time {t}: {e}")
                        # 如果闪电数据处理失败，使用零值
                        targets[i] = torch.zeros(self.future_timesteps, *self.img_size, device=device)
            except Exception as e:
                dist_print(f"Error processing sample {i} at time {t}: {e}")
                # 如果处理失败，使用零值
                inputs[i] = torch.zeros(self.past_timesteps, *self.img_size, 3, device=device)
                targets[i] = torch.zeros(self.future_timesteps, *self.img_size, device=device)

            batch_time_info.append(sample_times)

            # 数据增强：仅在训练时对闪电目标进行增强
            if dataset == "train" and self.lightning_processor and self.augmentation_config.get('enabled', True):
                targets = self._augment_lightning_targets(targets, inputs)

            # 添加闪电数据统计
            lightning_samples = sum(1 for time_info in batch_time_info if time_info.get("has_lightning", False))
            total_lightning_points = torch.sum(targets > 0).item()
            if idx % 10 == 0:  # 每10个批次打印一次
                dist_print(f"\n=== Batch {idx} Lightning Statistics ===")
                dist_print(f"Samples with lightning: {lightning_samples}/{len(batch_time_info)}")
                dist_print(f"Total lightning points: {total_lightning_points}")
                dist_print(
                    f"Target tensor stats: min={targets.min():.6f}, max={targets.max():.6f}, mean={targets.mean():.6f}")
                if total_lightning_points == 0:
                    dist_print("⚠️ WARNING: No lightning data in this batch!")
                    # 打印时间范围帮助调试
                    for i, time_info in enumerate(batch_time_info[:2]):  # 只打印前2个样本
                        dist_print(
                            f"  Sample {i}: {time_info['start_time']} -> {time_info['target_times'][-1] if time_info['target_times'] else 'N/A'}")

        # 打印时间对齐信息 - 减少输出频率
        if idx % 500 == 0:  # 每500个批次才输出一次
            valid_sample_count = sum(1 for time_info in batch_time_info if time_info.get("has_lightning", False))
            dist_print(f"\nBatch {idx} ({dataset}): {valid_sample_count}/{actual_batch_size} samples with lightning")

            for i, time_info in enumerate(batch_time_info):
                start = time_info["start_time"]
                end = time_info["target_times"][-1] if time_info["target_times"] else start
                time_span = end - start

                dist_print(f"Sample {i} time range: {start} to {end} (span: {time_span})")
                dist_print(f"  Input times: {[t.strftime('%Y-%m-%d %H:%M') for t in time_info['input_times']]}")
                dist_print(f"  Has lightning: {time_info.get('has_lightning', False)}")

                if time_info["target_times"]:
                    # dist_print(f"  Target times: {[t.strftime('%Y-%m-%d %H:%M') for t in time_info['target_times']]}")

                    # 检查时间间隔是否一致
                    intervals = [(time_info["target_times"][j] - time_info["target_times"][j - 1]).total_seconds() / 60
                                 for j in range(1, len(time_info["target_times"]))]
                    if len(set(intervals)) > 1:
                        dist_print(f"  WARNING: Inconsistent time intervals between targets: {intervals} minutes")

                # 检查输入和目标之间的时间连续性
                if time_info["input_times"] and time_info["target_times"]:
                    gap = (time_info["target_times"][0] - time_info["input_times"][-1]).total_seconds() / 60
                    if gap != 6:  # 假设预期间隔为6分钟
                        dist_print(f"  WARNING: Unexpected gap between last input and first target: {gap} minutes")

        # 添加数据验证和统计
        def validate_tensor(tensor, name, epoch=0):
            """
            验证张量数据的有效性
            在训练初期进行详细验证，后期简化验证以提高性能
            """
            # 获取当前epoch（如果可用）
            current_epoch = getattr(self, 'current_epoch', 0)

            if current_epoch < 3:  # 前3个epoch进行详细验证
                # 检查NaN
                if torch.isnan(tensor).any():
                    nan_count = torch.isnan(tensor).sum().item()
                    dist_print(f"Invalid data in {name}: {nan_count} NaN values detected, replacing with 0")
                    tensor = torch.nan_to_num(tensor, 0.0)

                # 检查Inf
                if torch.isinf(tensor).any():
                    inf_count = torch.isinf(tensor).sum().item()
                    dist_print(f"Invalid data in {name}: {inf_count} Inf values detected, clamping")
                    tensor = torch.clamp(tensor, -1e6, 1e6)

                # 详细统计（仅在特定批次）
                if idx % 100 == 0:
                    non_zero = torch.sum(tensor > 0.01).item()
                    total = tensor.numel()
                    if non_zero > 0:
                        dist_print(
                            f"{name} contains {non_zero}/{total} non-zero values ({non_zero / total * 100:.2f}%)")

            elif current_epoch < 10:  # 第3-10个epoch进行基本验证
                # 快速NaN/Inf检查（不计数）
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    tensor = torch.nan_to_num(tensor, 0.0)
                    tensor = torch.clamp(tensor, -1e6, 1e6)
                    if idx % 500 == 0:  # 减少日志频率
                        dist_print(f"Fixed invalid values in {name}")

            else:  # 10个epoch后最小化验证
                # 仅在检测到问题时进行修复，不输出日志
                if torch.isnan(tensor).any():
                    tensor = torch.nan_to_num(tensor, 0.0)
                if torch.isinf(tensor).any():
                    tensor = torch.clamp(tensor, -1e6, 1e6)

            return tensor

        inputs = validate_tensor(inputs, "inputs")
        targets = validate_tensor(targets, "targets")

        # 构建输入字典
        input_dict = {
            "radar_past": inputs  # 雷达数据（时间序列）
        }

        # Add static data to input dictionary if available
        if static_tensors:
            # For each static data source, add it to inputs with appropriate expansion
            for name, tensor in static_tensors.items():
                # Static data should have shape [H, W], expand to [B, 1, H, W] for model input
                if tensor.dim() == 2:  # [H, W]
                    expanded_tensor = tensor.unsqueeze(0).unsqueeze(0).expand(len(time_batch), 1, *self.img_size)
                elif tensor.dim() == 3:  # [1, H, W]
                    expanded_tensor = tensor.unsqueeze(0).expand(len(time_batch), 1, *self.img_size)
                else:
                    expanded_tensor = tensor
                input_dict[f"static_{name}"] = expanded_tensor
                dist_print(f"Added {name} to input dict with shape {expanded_tensor.shape}")

        # Update input specs to include static data
        if self.static_data and "static" not in [spec["timeframe"] for spec in self.input_specs]:
            self.input_specs.append({"timeframe": "static", "shape_divisor": 1})

        # 调整目标维度 [B, T, H, W] -> [B, T, H, W, 1]
        targets = targets.unsqueeze(-1)

        # # 记录设备信息用于调试
        # if idx % 100 == 0:
        #     dist_print(f"Batch {idx} returning tensors on device {inputs.device}")

        return input_dict, targets

    def _augment_lightning_targets(self, targets, inputs):
        """
        对闪电目标进行数据增强，基于雷达反射率增加合理的正样本

        Args:
            targets: [B, T, H, W] 闪电目标张量
            inputs: [B, T, H, W, C] 雷达输入张量
            augment_prob: 增强概率
        """
        config = self.augmentation_config
        augment_prob = config.get('augment_prob', 0.3)

        # 只在有闪电的批次中进行增强
        if torch.rand(1).item() > augment_prob or targets.sum() == 0:
            return targets

        device = targets.device
        augmented = targets.clone()

        # 策略1: 时空扩展现有闪电区域（物理合理性：闪电具有时空连续性）
        spatial_config = config.get('spatial_expansion', {})
        if spatial_config.get('enabled', True) and targets.sum() > 0:
            # 空间扩展：使用形态学膨胀
            from scipy import ndimage
            for b in range(targets.shape[0]):
                for t in range(targets.shape[1]):
                    if targets[b, t].sum() > 0:
                        # 转到CPU进行形态学操作
                        lightning_map = targets[b, t].cpu().numpy()

                        # 使用不同大小的结构元素进行膨胀
                        struct = ndimage.generate_binary_structure(2, 2)
                        iterations = spatial_config.get('iterations', 2)
                        dilated = ndimage.binary_dilation(lightning_map > 0, struct, iterations=iterations)

                        # 时间连续性：如果当前时刻有闪电，相邻时刻也可能有
                        confidence = spatial_config.get('confidence', 0.8)
                        augmented[b, t] = torch.maximum(
                            augmented[b, t],
                            torch.tensor(dilated.astype(float), device=device) * confidence  # 略低的置信度
                        )

                        temporal_config = config.get('temporal_propagation', {})
                        if temporal_config.get('enabled', True):
                            decay = temporal_config.get('decay_factor', 0.5)

                        # 向前传播（概率递减）
                        if t > 0 and torch.rand(1).item() < temporal_config.get('forward_prob', 0.5):
                            augmented[b, t - 1] = torch.maximum(
                                augmented[b, t - 1],
                                augmented[b, t] * decay
                            )

                        # 向后传播（概率递减）
                        if t < targets.shape[1] - 1 and torch.rand(1).item() < temporal_config.get('backward_prob',
                                                                                                   0.5):
                            augmented[b, t + 1] = torch.maximum(
                                augmented[b, t + 1],
                                augmented[b, t] * decay
                            )

        # 策略2: 基于高反射率区域添加潜在闪电（物理合理性：强对流区域易产生闪电）
        convection_config = config.get('convection_based', {})
        if convection_config.get('enabled', True) and torch.rand(1).item() < convection_config.get('prob', 0.3):
            # 提取DBZ通道（假设是第一个通道）
            dbz = inputs[:, -1, :, :, 0]  # 使用最后一个时刻的DBZ

            # 寻找强对流区域（DBZ > 45 dBZ）
            dbz_threshold = convection_config.get('dbz_threshold', 0.7)
            strong_convection = (dbz > dbz_threshold)  # 假设已归一化

            for b in range(targets.shape[0]):
                min_pixels = convection_config.get('min_pixels', 100)
                if strong_convection[b].sum() > min_pixels:  # 至少有指定数量的强对流像素
                    # 在强对流区域中随机选择位置
                    convection_coords = torch.nonzero(strong_convection[b])
                    if len(convection_coords) > 0:
                        # 随机选择1-3个中心点
                        max_centers = convection_config.get('max_centers', 3)
                        n_centers = min(max_centers, torch.randint(1, max_centers + 1, (1,)).item())
                        indices = torch.randperm(len(convection_coords))[:n_centers]

                        for idx in indices:
                            y, x = convection_coords[idx]

                            # 创建小范围的闪电区域
                            radius_range = convection_config.get('radius_range', [5, 15])
                            radius = torch.randint(radius_range[0], radius_range[1], (1,)).item()
                            yy, xx = torch.meshgrid(
                                torch.arange(self.img_size[0], device=device),
                                torch.arange(self.img_size[1], device=device),
                                indexing='ij'
                            )

                            # 高斯分布的闪电强度
                            dist_sq = (yy - y) ** 2 + (xx - x) ** 2
                            gaussian_kernel = torch.exp(-dist_sq / (2 * radius ** 2))
                            gaussian_kernel[dist_sq > radius ** 2] = 0

                            # 随机选择时间步
                            t_start = torch.randint(0, targets.shape[1], (1,)).item()
                            max_duration = convection_config.get('max_duration', 3)
                            duration = min(max_duration, torch.randint(1, max_duration + 1, (1,)).item())

                            for dt in range(duration):
                                t = t_start + dt
                                if 0 <= t < targets.shape[1]:
                                    # 添加合成闪电（强度随时间衰减）
                                    decay = 1.0 / (dt + 1)
                                    confidence = convection_config.get('confidence', 0.6)
                                    augmented[b, t] = torch.maximum(
                                        augmented[b, t],
                                        gaussian_kernel * confidence * decay  # 合成闪电置信度较低
                                    )

        # 策略3: 基于历史模式的数据增强（如果某个区域历史上常有闪电）
        # 这里可以加载历史统计数据进行增强

        # 确保值在[0, 1]范围内
        augmented = torch.clamp(augmented, 0.0, 1.0)

        # 记录增强效果
        original_points = (targets > 0).sum().item()
        augmented_points = (augmented > 0).sum().item()
        if augmented_points > original_points:
            increase_pct = (augmented_points / max(original_points, 1) - 1) * 100
            # dist_print(f"Lightning augmentation: {original_points} -> {augmented_points} points "
            #            f"(+{increase_pct:.1f}%)")

            # 记录各策略的贡献（可选）
            if hasattr(self, 'augmentation_stats'):
                self.augmentation_stats['total_augmented'] += augmented_points - original_points
                self.augmentation_stats['batches_augmented'] += 1

        return augmented

    def _augment_single_lightning_target(self, target, input_data):
        """
        对单个闪电目标进行数据增强

        Args:
            target: [T, H, W] 闪电目标张量
            input_data: [T, H, W, C] 雷达输入张量
        """
        config = self.augmentation_config
        augment_prob = config.get('augment_prob', 0.3)

        # Only augment with probability
        if torch.rand(1).item() > augment_prob or target.sum() == 0:
            return target

        # Add batch dimension temporarily for compatibility
        target_batched = target.unsqueeze(0)  # [1, T, H, W]
        input_batched = input_data.unsqueeze(0)  # [1, T, H, W, C]

        # Call existing augmentation method
        augmented_batched = self._augment_lightning_targets(target_batched, input_batched)

        # Remove batch dimension
        return augmented_batched[0]  # [T, H, W]

    def calculate_event_occurrence(self, num_batches=5, dataset="train", batch_size=8):
        """
        计算数据集中闪电事件的实际比例

        Args:
            num_batches: 采样的批次数量
            dataset: 数据集类型 ("train", "valid", "test")
            batch_size: 批大小

        Returns:
            dict: 包含详细统计信息的字典
        """
        import time
        from torch.utils.data import DataLoader

        start_time = time.time()

        # Create dataset and dataloader
        radar_dataset = RadarSampleDataset(self, dataset=dataset)
        dataloader = DataLoader(
            radar_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Use single worker for statistics calculation
            pin_memory=False
        )

        # 统计变量
        total_pixels = 0
        positive_pixels = 0
        batch_occurrences = []
        sample_occurrences = []
        timestep_occurrences = [0] * self.future_timesteps

        dist_print(f"\n=== Calculating Event Occurrence from {dataset} dataset ===")
        dist_print(f"Sampling up to {num_batches} batches...")

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            try:
                # targets shape: [B, T, H, W, 1]
                targets = targets.squeeze(-1)  # Remove last dimension

                # 批次级统计
                batch_total = targets.numel()
                batch_positive = (targets > 0.5).sum().item()
                total_pixels += batch_total
                positive_pixels += batch_positive

                if batch_total > 0:
                    batch_occurrence = batch_positive / batch_total
                    batch_occurrences.append(batch_occurrence)

                # 样本级统计
                for b in range(targets.shape[0]):
                    sample_data = targets[b]  # [T, H, W]
                    sample_positive = (sample_data > 0.5).sum().item()
                    sample_total = sample_data.numel()
                    if sample_total > 0:
                        sample_occurrences.append(sample_positive / sample_total)

                # 时间步级统计
                for t in range(self.future_timesteps):
                    timestep_data = targets[:, t, :, :]  # [B, H, W]
                    timestep_positive = (timestep_data > 0.5).sum().item()
                    timestep_occurrences[t] += timestep_positive

                # 进度报告
                if (batch_idx + 1) % 10 == 0:
                    current_occurrence = positive_pixels / max(total_pixels, 1)
                    dist_print(f"  Processed {batch_idx + 1}/{min(num_batches, len(dataloader))} batches, "
                               f"current occurrence: {current_occurrence:.6f}")

            except Exception as e:
                dist_print(f"  Error in batch {batch_idx}: {e}")
                continue

        # 计算最终统计
        overall_occurrence = positive_pixels / max(total_pixels, 1)

        # 计算时间步平均
        timestep_avg_occurrences = []
        pixels_per_timestep = (total_pixels / self.future_timesteps) if self.future_timesteps > 0 else 1
        for t_count in timestep_occurrences:
            timestep_avg_occurrences.append(t_count / max(pixels_per_timestep, 1))

        # 计算统计摘要
        import numpy as np
        batch_occurrences_np = np.array(batch_occurrences) if batch_occurrences else np.array([0])
        sample_occurrences_np = np.array(sample_occurrences) if sample_occurrences else np.array([0])

        stats = {
            'overall_occurrence': overall_occurrence,
            'total_pixels': total_pixels,
            'positive_pixels': positive_pixels,
            'num_batches': num_batches,
            'batch_occurrence_mean': np.mean(batch_occurrences_np),
            'batch_occurrence_std': np.std(batch_occurrences_np),
            'batch_occurrence_max': np.max(batch_occurrences_np),
            'sample_occurrence_mean': np.mean(sample_occurrences_np),
            'sample_occurrence_std': np.std(sample_occurrences_np),
            'samples_with_lightning': np.sum(sample_occurrences_np > 0),
            'total_samples': len(sample_occurrences),
            'timestep_occurrences': timestep_avg_occurrences,
            'computation_time': time.time() - start_time
        }

        # 打印详细统计
        dist_print(f"\n=== Event Occurrence Statistics ({dataset}) ===")
        dist_print(f"Overall occurrence: {stats['overall_occurrence']:.6f} ({stats['overall_occurrence'] * 100:.4f}%)")
        dist_print(f"Total pixels analyzed: {stats['total_pixels']:,}")
        dist_print(f"Positive pixels found: {stats['positive_pixels']:,}")
        dist_print(f"Batches processed: {stats['num_batches']}")
        dist_print(f"\nBatch-level statistics:")
        dist_print(f"  Mean: {stats['batch_occurrence_mean']:.6f}")
        dist_print(f"  Std: {stats['batch_occurrence_std']:.6f}")
        dist_print(f"  Max: {stats['batch_occurrence_max']:.6f}")
        dist_print(f"\nSample-level statistics:")
        dist_print(f"  Samples with lightning: {stats['samples_with_lightning']}/{stats['total_samples']} "
                   f"({stats['samples_with_lightning'] / max(stats['total_samples'], 1) * 100:.2f}%)")
        dist_print(f"  Mean occurrence: {stats['sample_occurrence_mean']:.6f}")
        dist_print(f"  Std: {stats['sample_occurrence_std']:.6f}")
        dist_print(f"\nTimestep occurrences:")
        for t, occ in enumerate(stats['timestep_occurrences']):
            dist_print(f"  Timestep {t + 1}: {occ:.6f}")
        dist_print(f"\nComputation time: {stats['computation_time']:.2f} seconds")

        # 建议的类权重（用于不平衡数据）
        if stats['overall_occurrence'] > 0:
            suggested_pos_weight = (1 - stats['overall_occurrence']) / stats['overall_occurrence']
            dist_print(f"\nSuggested positive class weight: {suggested_pos_weight:.2f}")
            stats['suggested_pos_weight'] = suggested_pos_weight

        return stats

    def visualize_lightning_grid(self, lightning_grid, radar_img, future_time, dataset, idx, i=0, j=0,
                                 field='CGFLASH', flash_type=0, time_window=6, smoothing=False):

        """
        Visualize lightning grid data with optional radar overlay

        Args:
            lightning_grid: The lightning grid tensor
            radar_img: Optional radar image for overlay
            future_time: The timestamp of the lightning data
            dataset: Current dataset name (train/valid/test)
            idx: Batch index
            i: Sample index within batch
            j: Time step index
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            import matplotlib.patches as mpatches

            # Create output directory
            vis_output_dir = os.path.join(self.results_dir, 'lightning_vis')
            os.makedirs(vis_output_dir, exist_ok=True)

            # Get lightning strike locations
            lightning_data = lightning_grid.cpu().numpy()
            strike_locations = np.where(lightning_data > 0)

            # Create coordinate grids for reference
            height, width = lightning_data.shape
            # Create lat/lon grid based on the grid_info from lightning processor
            if hasattr(self, 'grid_info'):
                lons = np.linspace(self.grid_info['lon_min'], self.grid_info['lon_max'], width)
                # IMPORTANT: array/image coordinates are top-down (y=0 is the top row).
                # Our lightning rasterization maps lat_max -> y=0, lat_min -> y=H-1.
                # So for visualization, latitude labels must decrease from top to bottom.
                lats = np.linspace(self.grid_info['lat_max'], self.grid_info['lat_min'], height)
                lon_ticks = np.linspace(0, width - 1, 5).astype(int)
                lat_ticks = np.linspace(0, height - 1, 5).astype(int)
                lon_labels = [f"{lons[i]:.1f}°E" for i in lon_ticks]
                lat_labels = [f"{lats[i]:.1f}°N" for i in lat_ticks]
            else:
                # Fallback if grid_info is not available
                lon_ticks = np.linspace(0, width - 1, 5).astype(int)
                lat_ticks = np.linspace(0, height - 1, 5).astype(int)
                lon_labels = lon_ticks
                lat_labels = lat_ticks

            # Save original lightning grid
            plt.figure(figsize=(12, 10))

            # Remove the intensity visualization and only show lightning strikes
            # Create a white background for better visibility
            plt.imshow(np.ones_like(lightning_data), cmap='gray', vmin=0, vmax=1)
            # plt.colorbar(im, label='Lightning Intensity')  # No colorbar needed

            # Mark actual lightning strike locations with '*' symbols
            if len(strike_locations[0]) > 0:
                plt.scatter(strike_locations[1], strike_locations[0],
                            marker='*', s=80, color='yellow', edgecolor='black', linewidth=1.5,
                            label=f'Lightning Strikes ({len(strike_locations[0])})')
                plt.legend(loc='upper right')

            # Add coordinate grid and labels
            plt.xticks(lon_ticks, lon_labels)
            plt.yticks(lat_ticks, lat_labels)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.5)

            # Enhanced title with more metadata
            timestamp_str = future_time.strftime('%Y-%m-%d %H:%M')
            flash_type_str = "Ground Flash" if field == 'CGFLASH' and flash_type == 0 else "Cloud Flash" if field == 'CGFLASH' and flash_type == 1 else "All Flashes"
            plt.title(f"Lightning Activity ({flash_type_str}) - {timestamp_str}\n"
                      f"{dataset.upper()} Dataset (Batch {idx}) | Window: {time_window}min | Smoothing: {'On' if smoothing else 'Off'}",
                      fontsize=14, fontweight='bold')

            # Add detailed statistics as text
            stats_text = (
                f"Total Lightning Strikes: {np.count_nonzero(lightning_data)}\n"
                f"Strikes Coverage: {np.count_nonzero(lightning_data) / (width * height) * 100:.6f}%"
            )
            plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=10,
                        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

            # Save the enhanced lightning visualization
            lightning_file = f"{vis_output_dir}/lightning_grid_{dataset}_{idx}_{future_time.strftime('%Y%m%d%H%M')}.png"
            plt.savefig(lightning_file, dpi=150, bbox_inches='tight')

            # If radar image is provided, create overlay visualization
            if radar_img is not None:
                plt.figure(figsize=(12, 10))

                # Plot radar data first
                radar_data = radar_img[:, :, 0]  # DBZ channel
                radar_im = plt.imshow(radar_data, cmap='Blues', alpha=0.7)

                # Create a separate colorbar for radar
                radar_cbar = plt.colorbar(radar_im, location='left', label='Radar Reflectivity (dBZ)')

                # Don't show lightning intensity, only the strikes
                # lightning_im = plt.imshow(lightning_data, cmap='hot', alpha=0.6, norm=norm)
                # lightning_cbar = plt.colorbar(lightning_im, label='Lightning Intensity')

                # Mark actual lightning strike locations
                if len(strike_locations[0]) > 0:
                    plt.scatter(strike_locations[1], strike_locations[0],
                                marker='*', s=100, color='yellow', edgecolor='black', linewidth=1.5,
                                label=f'Lightning Strikes ({len(strike_locations[0])})')

                # Add coordinate grid and labels
                plt.xticks(lon_ticks, lon_labels)
                plt.yticks(lat_ticks, lat_labels)
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)

                # Create legend for the overlay
                radar_patch = mpatches.Patch(color='blue', alpha=0.5, label='Radar Reflectivity')
                lightning_patch = mpatches.Patch(color='yellow', alpha=0.7,
                                                 label=f'Lightning Strikes ({len(strike_locations[0])})')
                plt.legend(handles=[radar_patch, lightning_patch], loc='upper left')

                # Enhanced title
                plt.title(f"Radar + Lightning Overlay - {timestamp_str}\n{dataset.upper()} Dataset (Batch {idx})",
                          fontsize=14, fontweight='bold')

                # Add correlation information between radar and lightning
                try:
                    # Check if there's sufficient variance to calculate correlation
                    if np.var(lightning_data) > 0 and np.var(radar_data) > 0:
                        correlation = np.corrcoef(radar_data.flatten(), lightning_data.flatten())[0, 1]
                        correlation_text = f"Radar-Lightning Correlation: {correlation:.4f}"
                    else:
                        correlation_text = "Radar-Lightning Correlation: N/A (insufficient variance)"
                except Exception as e:
                    correlation_text = f"Correlation calculation error: {str(e)}"

                overlay_stats = (
                    f"{correlation_text}\n"
                    f"High Reflectivity Areas (>30 dBZ): {np.sum(radar_data > 30) / (width * height) * 100:.2f}%"
                )
                plt.figtext(0.5, 0.01, overlay_stats, ha="center", fontsize=10,
                            bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

                # Save the enhanced overlay visualization
                overlay_file = f"{vis_output_dir}/radar_lightning_{dataset}_{idx}_{future_time.strftime('%Y%m%d%H%M')}.png"
                plt.savefig(overlay_file, dpi=150, bbox_inches='tight')

            plt.close('all')
            dist_print(f"Saved enhanced lightning visualizations to {vis_output_dir}")
        except Exception as e:
            dist_print(f"Error visualizing lightning grid: {e}")
            import traceback
            traceback.print_exc()

# # Old batch generator for old dataset
# class BatchGenerator:
#     def __init__(self,  predictors, targets, raw, coords_by_time, primary_raw_var, data_root,
#                  box_size=(8, 8),  img_size=(256, 256), timesteps=(12, 12), batch_size=24, interval=timedelta(minutes=5),
#                  random_seed=1234, valid_frac=0.1, test_frac=0.1,  dataset_block_size=12 * 24, **kwargs):
#
#
#         self.img_size = img_size
#         self.batch_size = batch_size
#         self.interval = interval
#         self.timesteps = kwargs.get('timesteps', (12, 12))
#         self.primary_raw_var = primary_raw_var
#         self.rng = np.random.RandomState(seed=random_seed)
#
#         # Initialize other attributes
#         self._setup_predictors_targets(predictors, targets)
#         self._setup_raw_indices(raw, box_size)
#         self._setup_coords(coords_by_time, valid_frac, test_frac, dataset_block_size)
#
#
#
#
#     def get_radar_batch(self, t_batch):
#         """
#         加载雷达数据批次
#         t_batch: 批次时间列表 (datetime)
#         返回: [B, T, H, W, C]
#         """
#         batch_size = len(t_batch)
#         # 初始化输出张量 [B, T, H, W, C]
#         radar_data = torch.zeros(
#             batch_size, self.timesteps[0], *self.img_size, 3
#         )
#
#         for b, t in enumerate(t_batch):
#             # 加载过去12个时间步
#             radar_data[b] = self.image_loader.load_time_series(
#                 start_time=t,
#                 time_steps=self.timesteps[0]
#             )
#
#         return radar_data
#
#     def _setup_predictors_targets(self, predictors, targets):
#         self.pred_sources = {}
#         self.pred_transform = {}
#         pred_timeframe = {}
#
#         for pred_name, pred_data in predictors.items():
#             self.pred_sources[pred_name] = pred_data["source_vars"]
#             self.pred_transform[pred_name] = pred_data["transform"]
#             pred_timeframe[pred_name] = pred_data.get("timeframe", "past")
#
#         self.pred_names_past = [v for v in predictors.keys() if pred_timeframe[v] == "past"]
#         self.pred_names_future = [v for v in predictors.keys() if pred_timeframe[v] == "future"]
#         self.pred_names_static = [v for v in predictors.keys() if pred_timeframe[v] == "static"]
#
#         self.target_sources = {}
#         self.target_transform = {}
#         for target_name, target_data in targets.items():
#             self.target_sources[target_name] = target_data["source_vars"]
#             self.target_transform[target_name] = target_data["transform"]
#
#         self.target_names = list(targets.keys())
#
#     def _setup_raw_indices(self, raw, box_size):
#         self.raw_batch_index = {}
#         self.setup_batch_index(self.primary_raw_var, raw[self.primary_raw_var], box_size)
#         index_limits = self.raw_batch_index[self.primary_raw_var].index_limits
#
#         for raw_name, raw_data in raw.items():
#             if raw_name == self.primary_raw_var:
#                 continue
#             self.setup_batch_index(raw_name, raw_data, box_size, index_limits=index_limits)
#
#     def _setup_coords(self, coords_by_time, valid_frac, test_frac, block_size):
#         self.coords_by_time = {t: np.array(coords) for t, coords in coords_by_time.items()}
#         self.time_coords = self.train_valid_test_split(valid_frac, test_frac, block_size)
#
#         self.fixed_coord = {}
#         fixed_coord_times = chain(self.time_coords["valid"], self.time_coords["test"])
#         for t in fixed_coord_times:
#             coords = self.coords_by_time[t]
#             self.fixed_coord[t] = self.rng.randint(len(coords))
#
#     def setup_batch_index(self, raw_name, raw_data, box_size, index_limits=None):
#         time_dim = 0
#
#         # source (raw) variables used for past and future timeframes
#         sources_past = set(chain(*[self.pred_sources[v] for v in self.pred_names_past]))
#         sources_future = set(chain(*[self.pred_sources[v] for v in self.pred_names_future])) | \
#                          set(chain(*[self.target_sources[v] for v in self.target_names]))
#         sources_static = set(chain(*[self.pred_sources[v] for v in self.pred_names_static]))
#
#         # select longest time dimension needed
#         if (raw_name in sources_past) or (raw_name == self.primary_raw_var):
#             time_dim = self.timesteps[0]
#         if raw_name in sources_future:
#             time_dim = max(time_dim, self.timesteps[1])
#         if raw_name in sources_static:
#             time_dim = max(time_dim, 1)
#         if time_dim == 0:  # this source is not used so we don't need to index it
#             return
#
#         interp = raw_data.get("interpolation", None)
#         zero_value = raw_data.get("zero_value", 0)
#         missing_value = raw_data.get("missing_value", zero_value)
#
#         if interp is None:
#             self.raw_batch_index[raw_name] = PatchIndex(
#                 raw_data["patches"],
#                 raw_data["patch_coords"],
#                 raw_data["patch_times"],
#                 raw_data["zero_patch_coords"],
#                 raw_data["zero_patch_times"],
#                 zero_value=zero_value,
#                 missing_value=missing_value,
#                 interval=self.interval,
#                 box_size=(time_dim,) + box_size,
#                 index_limits=index_limits,
#                 static=raw_data.get("static", False)
#             )
#         else:
#             self.raw_batch_index[raw_name] = InterpolatingPatchIndex(
#                 raw_data["patches"],
#                 raw_data["patch_coords"],
#                 raw_data["patch_times"],
#                 raw_data["zero_patch_coords"],
#                 raw_data["zero_patch_times"],
#                 zero_value=zero_value,
#                 missing_value=missing_value,
#                 interval=self.interval,
#                 box_size=(time_dim,) + box_size,
#                 index_limits=index_limits,
#                 method=interp,
#                 stride=raw_data["stride"]
#             )
#
#     def train_valid_test_split(self, valid_frac=None,
#                                test_frac=None, block_size=None):
#
#         times = np.array(sorted(self.coords_by_time))
#         n = len(times)
#
#         times_valid = []
#         if valid_frac is not None:
#             n_valid = int(valid_frac * n)
#             while len(times_valid) < n_valid:
#                 t0 = times[self.rng.randint(len(times))]
#                 t1 = t0 + block_size
#                 selection = (t0 <= times) & (times < t1)
#                 times_valid.extend(list(times[selection]))
#                 times = times[~selection]
#         times_valid = np.array(times_valid)
#
#         times_test = []
#         if test_frac is not None:
#             n_test = int(test_frac * n)
#             while len(times_test) < n_test:
#                 t0 = times[self.rng.randint(len(times))]
#                 t1 = t0 + block_size
#                 selection = (t0 <= times) & (times < t1)
#                 times_test.extend(list(times[selection]))
#                 times = times[~selection]
#         times_test = np.array(times_test)
#
#         self.rng.shuffle(times)
#         self.rng.shuffle(times_valid)
#         self.rng.shuffle(times_test)
#
#         return {
#             "train": times,
#             "valid": times_valid,
#             "test": times_test
#         }
#
#     def frame_spatial_coordinates(self, t_batch, dataset="train"):
#         i_batch = []
#         j_batch = []
#         for t in t_batch:
#             coords = self.coords_by_time[t]
#             if dataset == "train":
#                 coord_ind = self.rng.randint(len(coords))
#             else:
#                 coord_ind = self.fixed_coord[t]
#             (i, j) = coords[coord_ind, :]
#             i_batch.append(i)
#             j_batch.append(j)
#
#         return (np.array(i_batch), np.array(j_batch))
#
#     def random_augments(self):
#         transpose = bool(self.rng.randint(2))
#         flipud = bool(self.rng.randint(2))
#         fliplr = bool(self.rng.randint(2))
#         return (transpose, flipud, fliplr)
#
#     def augment(self, batch, augments):
#         (transpose, flipud, fliplr) = augments
#
#         if transpose:
#             batch = batch.permute(0, 1, 3, 2, 4)  # PyTorch equivalent of transpose
#         if flipud:
#             batch = torch.flip(batch, [2])  # Flip height dimension
#         if fliplr:
#             batch = torch.flip(batch, [3])  # Flip width dimension
#         return batch
#
#     def get_batch(self, t, i, j, var_names, var_sources, transform, num_timesteps):
#         raw_data = {}
#         for var_name in var_names:
#             sources = var_sources[var_name]
#             for raw_var in sources:
#                 if raw_var not in raw_data:
#                     raw_data[raw_var] = self.raw_batch_index[raw_var](
#                         t, i, j, num_timesteps=num_timesteps)
#                     # Convert numpy array to torch tensor
#                     raw_data[raw_var] = torch.from_numpy(raw_data[raw_var])
#
#         batch_data = []
#         for var_name in var_names:
#             raw_vars = [raw_data[raw_var][:, :num_timesteps, ...]
#                         for raw_var in var_sources[var_name]]
#             transformed_vars = transform[var_name](*raw_vars)
#
#             # Ensure 5D tensor (batch, time, h, w, channels)
#             if len(transformed_vars.shape) == 4:
#                 transformed_vars = transformed_vars.unsqueeze(-1)
#             if len(transformed_vars.shape) != 5:
#                 raise ValueError(
#                     f"Transformed variable {var_name} has invalid shape: {transformed_vars.shape}"
#                 )
#             batch_data.append(transformed_vars)
#
#         return batch_data
#
#     def batch(self, idx, dataset="train"):
#         # # 获取时间批次
#         # t_batch = self._get_time_batch(idx, dataset)
#         #
#         # # 加载雷达数据 - 作为主要输入
#         # radar_batch = self.get_radar_batch(t_batch)  # [B, T, H, W, 3]
#         #
#         # # 创建虚拟目标 (暂时使用雷达数据代替闪电)
#         # # TODO: 替换为真实闪电数据
#         # target_batch = radar_batch[:, :, :, :, 0].unsqueeze(-1)  # 使用dbz作为目标
#         #
#         # # 构建输入字典 (适配模型期望格式)
#         # inputs = {
#         #     "radar_past_1": radar_batch  # 键格式: {timeframe}_{divisor}
#         # }
#         #
#         # return inputs, target_batch
#
#         t_pred = self.time_coords[dataset][
#                  idx * self.batch_size:(idx + 1) * self.batch_size
#                  ]
#         t_target = t_pred + self.timesteps[0]
#         (i, j) = self.frame_spatial_coordinates(t_pred, dataset=dataset)
#
#         pred_batch_past = self.get_batch(t_pred, i, j,
#                                          self.pred_names_past, self.pred_sources, self.pred_transform,
#                                          self.timesteps[0]
#                                          )
#         pred_batch_future = self.get_batch(t_target, i, j,
#                                            self.pred_names_future, self.pred_sources, self.pred_transform,
#                                            self.timesteps[1]
#                                            )
#         pred_batch_static = self.get_batch(t_pred, i, j,
#                                            self.pred_names_static, self.pred_sources, self.pred_transform, 1
#                                            )
#         pred_batch = pred_batch_past + pred_batch_future + pred_batch_static
#         target_batch = self.get_batch(t_target, i, j,
#                                       self.target_names, self.target_sources, self.target_transform,
#                                       self.timesteps[1]
#                                       )
#
#         if dataset == "train":
#             augments = self.random_augments()
#             pred_batch = [self.augment(b, augments) for b in pred_batch]
#             target_batch = [self.augment(b, augments) for b in target_batch]
#
#         # Convert lists to tuples for PyTorch compatibility
#         return (tuple(pred_batch), tuple(target_batch))


### 原来BatchDataset位置

        # For BatchGenerator (tuple input), convert to expected format
        # else:
        #     # === 重建TensorFlow的分组逻辑 ===
        #     grouped_inputs = defaultdict(list)
        #
        #     for i, (tensor, spec) in enumerate(zip(pred_batch, self.batch_gen.input_specs)):
        #         key = f"{spec['timeframe']}_{spec['shape_divisor']}"
        #         tensor = torch.as_tensor(tensor).float()
        #
        #         # 确保静态变量时间维扩展（模仿TF的tf.repeat）
        #         if spec['timeframe'] == 'static':
        #             tensor = tensor.repeat(1, self.batch_gen.timesteps[0], 1, 1, 1)
        #
        #         grouped_inputs[key].append(tensor)
        #
        #     # 合并同组输入（沿通道维）
        #     processed_inputs = {}
        #     for key, tensors in grouped_inputs.items():
        #         processed_inputs[key] = torch.cat(tensors, dim=-1)  # [B,T,H,W,C]
        #
        #     # 处理target（确保正确的维度顺序）
        #     target = torch.as_tensor(target_batch[0]).float()
        #     # Ensure target has correct shape with channels last [B, T, H, W, C]
        #     if len(target.shape) == 4:
        #         target = target.unsqueeze(-1)

            # return processed_inputs, target


class PatchIndex:
    IDX_ZERO = -1
    IDX_MISSING = -2

    def __init__(
            self, patch_data, patch_coords, patch_times,
            zero_patch_coords, zero_patch_times,
            interval=timedelta(minutes=5),
            box_size=(12, 8, 8), zero_value=0,
            missing_value=0,
            index_limits=None, static=False
    ):
        # Convert inputs to numpy arrays if they are torch tensors
        if torch.is_tensor(patch_data):
            patch_data = patch_data.numpy()
        if torch.is_tensor(patch_coords):
            patch_coords = patch_coords.numpy()
        if torch.is_tensor(patch_times):
            patch_times = patch_times.numpy()
        if torch.is_tensor(zero_patch_coords):
            zero_patch_coords = zero_patch_coords.numpy()
        if torch.is_tensor(zero_patch_times):
            zero_patch_times = zero_patch_times.numpy()

        # Ensure all inputs are numpy arrays
        patch_coords = np.asarray(patch_coords, dtype=np.int32)
        patch_times = np.asarray(patch_times)
        zero_patch_coords = np.asarray(zero_patch_coords, dtype=np.int32)
        zero_patch_times = np.asarray(zero_patch_times)

        if (index_limits is None) or static:
            t0 = patch_times.min()
            t1 = patch_times.max()
            # Convert patch_coords to a compatible data type before computing max
            patch_coords_array = np.asarray(patch_coords, dtype=np.int32)
            (i1, j1) = patch_coords_array.max(axis=0)
        else:
            (t0, t1, i1, j1) = index_limits
        self.index_limits = (t0, t1, i1, j1)
        (self.t0, self.t1, self.i1, self.j1) = self.index_limits

        self.dt = int(round(interval.total_seconds()))
        self.box_size = box_size
        self.zero_value = zero_value
        self.missing_value = missing_value
        self.patch_data = patch_data
        self.sample_shape = (
            box_size[0],
            self.patch_data.shape[1] * box_size[1],
            self.patch_data.shape[2] * box_size[2]
        )
        self.static = static

        self.patch_index = np.full(
            ((t1 - t0) // self.dt + 1, i1 + 1, j1 + 1),
            PatchIndex.IDX_MISSING,
            dtype=np.int32
        )
        init_patch_index(self.patch_index, patch_coords,
                         patch_times, self.t0, self.dt)
        init_patch_index_zero(self.patch_index, zero_patch_coords,
                              zero_patch_times, self.t0, self.dt, PatchIndex.IDX_ZERO)

        self._batch = None

    def _alloc_batch(self, n):
        if (self._batch is None) or (self._batch.shape[0] < n):
            del self._batch
            self._batch = np.zeros((n,) + self.sample_shape, self.patch_data.dtype)
        return self._batch

    def __call__(self, t0_all, i0_all, j0_all, num_timesteps=None):
        # Convert inputs to numpy arrays if they are torch tensors
        if torch.is_tensor(t0_all):
            t0_all = t0_all.numpy()
        if torch.is_tensor(i0_all):
            i0_all = i0_all.numpy()
        if torch.is_tensor(j0_all):
            j0_all = j0_all.numpy()

        # Ensure arrays are numpy arrays
        t0_all = np.asarray(t0_all)
        i0_all = np.asarray(i0_all)
        j0_all = np.asarray(j0_all)

        n = len(t0_all)
        batch = self._alloc_batch(n)
        if num_timesteps is None:
            num_timesteps = self.box_size[0]

        if self.static:  # override time coordinate
            t0_all = np.zeros_like(t0_all)
        t1_all = t0_all + num_timesteps
        i1_all = i0_all + self.box_size[1]
        j1_all = j0_all + self.box_size[2]
        bi_size = self.patch_data.shape[1]
        bj_size = self.patch_data.shape[2]

        build_batch(batch, self.patch_data, self.patch_index,
                    t0_all, t1_all, i0_all, i1_all, j0_all, j1_all,
                    bi_size, bj_size, self.zero_value,
                    self.missing_value, static=self.static)

        return batch


@njit(parallel=True)
def init_patch_index(patch_index, patch_coords, patch_times, t0, dt):
    """Initialize patch index with coordinates and times.

    All inputs must be NumPy arrays.
    """
    for k in prange(patch_coords.shape[0]):
        t = (patch_times[k] - t0) // dt
        if (t < 0) or (t >= patch_index.shape[0]):
            continue
        i = patch_coords[k, 0]
        j = patch_coords[k, 1]
        patch_index[t, i, j] = k


@njit(parallel=True)
def init_patch_index_zero(patch_index, zero_patch_coords,
                          zero_patch_times, t0, dt, idx_zero):
    for k in prange(zero_patch_coords.shape[0]):
        t = (zero_patch_times[k] - t0) // dt
        if (t < 0) or (t >= patch_index.shape[0]):
            continue
        i = zero_patch_coords[k, 0]
        j = zero_patch_coords[k, 1]
        patch_index[t, i, j] = idx_zero


class InterpolatingPatchIndex(PatchIndex):
    def __init__(self, *args, stride=12, method='linear', **kwargs):
        super().__init__(*args, **kwargs)
        self.stride = stride
        self.method = method

        times_with_data = (self.patch_index >= 0).any(axis=(1, 2))
        self.first_valid_step = np.nonzero(times_with_data)[0][0]
        self._batches = None

    def _alloc_batches(self, n_batches, n_samples):
        shape = (n_samples, n_batches)
        if (self._batches is None) or (self._batches.shape[:2] != shape):
            del self._batches
            self._batches = np.zeros(shape + self.sample_shape[1:],
                                     self.patch_data.dtype)
        return self._batches

    def __call__(self, t0_all, i0_all, j0_all, num_timesteps=None):
        # Convert inputs to numpy arrays if they are torch tensors
        if torch.is_tensor(t0_all):
            t0_all = t0_all.numpy()
        if torch.is_tensor(i0_all):
            i0_all = i0_all.numpy()
        if torch.is_tensor(j0_all):
            j0_all = j0_all.numpy()

        # Ensure arrays are numpy arrays
        t0_all = np.asarray(t0_all)
        i0_all = np.asarray(i0_all)
        j0_all = np.asarray(j0_all)

        n_samples = len(t0_all)
        if num_timesteps is None:
            num_timesteps = self.box_size[0]

        # find where the last valid time step is for each batch member
        dt0 = (t0_all - self.first_valid_step) % self.stride
        t0_mod = t0_all - dt0
        t0_mod.clip(0, self.patch_index.shape[0] - 1, out=t0_mod)

        # retrieve valid time steps overlapping the search period
        t_steps = range(0, num_timesteps + self.stride + 1, self.stride)
        num_steps = len(t_steps)
        batches = self._alloc_batches(num_steps, n_samples)
        t_step = t0_mod.copy()  # Make a copy to avoid modifying original

        for i in range(num_steps):
            b = super().__call__(t_step, i0_all, j0_all, num_timesteps=1)
            batches[:, i, ...] = b[:, 0, ...]
            valid_ind = (t_step + self.stride) < self.patch_index.shape[0]
            t_step[valid_ind] += self.stride

        # compute returned batch using interpolation
        batch_ip = self._alloc_batch(n_samples)
        interp_batch(batch_ip, batches, t0_all, dt0, self.stride, num_timesteps,
                     ip_linear=(self.method == 'linear'))

        return batch_ip


# numba can't find these values from PatchIndex
IDX_ZERO = PatchIndex.IDX_ZERO
IDX_MISSING = PatchIndex.IDX_MISSING


@njit(parallel=True)
def build_batch(
        batch, patch_data, patch_index,
        t0_all, t1_all, i0_all, i1_all, j0_all, j1_all,
        bi_size, bj_size, zero_value, missing_value, static=False
):
    for k in prange(t0_all.shape[0]):
        t0 = t0_all[k]
        t1 = t1_all[k]
        i0 = i0_all[k]
        i1 = i1_all[k]
        j0 = j0_all[k]
        j1 = j1_all[k]

        for t in range(t0, t1):
            bt = t - t0
            tt = t0 if static else t
            for i in range(i0, i1):
                bi0 = (i - i0) * bi_size
                bi1 = bi0 + bi_size
                for j in range(j0, j1):
                    ind = int(patch_index[tt, i, j])
                    bj0 = (j - j0) * bj_size
                    bj1 = bj0 + bj_size
                    if ind >= 0:
                        batch[k, bt, bi0:bi1, bj0:bj1] = patch_data[ind]
                    elif ind == IDX_ZERO:
                        batch[k, bt, bi0:bi1, bj0:bj1] = zero_value
                    elif ind == IDX_MISSING:
                        batch[k, bt, bi0:bi1, bj0:bj1] = missing_value


@njit(parallel=True)
def interp_batch(batch_ip, batches, t0_all, dt0, stride, num_timesteps,
                 ip_linear=True):
    n = len(t0_all)
    for k in prange(n):
        t0 = t0_all[k]
        t1 = t0 + num_timesteps
        dt = dt0[k]
        prev_batch_index = 0
        for t in range(t0, t1):
            bt = t - t0
            prev_batch = batches[:, prev_batch_index, ...]
            next_batch = batches[:, prev_batch_index + 1, ...]

            if ip_linear:
                w_next = dt / stride
                w_prev = 1 - w_next
                batch_ip[k, bt, :, :] = w_prev * prev_batch[k, :, :] + \
                                        w_next * next_batch[k, :, :]
            else:
                batch_ip[k, bt, :, :] = prev_batch[k, :, :] if \
                    dt < stride / 2 else next_batch[k, :, :]

            dt += 1
            if dt >= stride:
                dt = 0
                prev_batch_index += 1

# class ImageBatchGenerator(BatchGenerator):
#     def __init__(self, data_dirs, **kwargs):
#         """
#         data_dirs: 包含dbz/dbzh/vil子目录的根目录
#         """
#         self.data_dirs = {
#             'dbz': os.path.join(data_dirs, 'newCAPPI'),
#             'dbzh': os.path.join(data_dirs, 'Thunder/PONDSImage/dbzh'),
#             'vil': os.path.join(data_dirs, 'Thunder/PONDSImage/vil')
#         }
#         # 初始化时间序列索引
#         self.time_index = self._build_time_index()
#
#     def _build_time_index(self):
#         """建立时间戳到文件路径的映射"""
#         # 实现按时间戳组织文件路径的逻辑
#         return defaultdict(dict)