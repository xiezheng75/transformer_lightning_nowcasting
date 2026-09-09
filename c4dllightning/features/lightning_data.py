# lightning_data.py
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import torch
from scipy.interpolate import griddata
import torch.nn.functional as F

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

class LightningDataProcessor:
    def __init__(self, csv_path, grid_info, cache_dir=None):
        """
        闪电数据处理类

        参数:
        csv_path: 闪电数据CSV文件路径
        grid_info: 网格信息字典，包含:
            'lon_min', 'lon_max': 经度范围
            'lat_min', 'lat_max': 纬度范围
            'resolution': 网格分辨率(度)
            'width', 'height': 网格尺寸
            cache_dir: 缓存目录路径 (可选)
        """
        self.csv_path = csv_path
        self.grid_info = grid_info
        self.cache_dir = cache_dir
        self.flash_data = None
        self.grid_cache = {}

        if self.cache_dir:
            self.processed_cache_path = os.path.join(self.cache_dir, 'lightning_processed_cache.pkl')
        else:
            self.processed_cache_path = None

        # 计算网格坐标
        self.lon_grid = np.linspace(
            grid_info['lon_min'],
            grid_info['lon_max'],
            grid_info['width']
        )
        self.lat_grid = np.linspace(
            grid_info['lat_min'],
            grid_info['lat_max'],
            grid_info['height']
        )
        self.lon_mesh, self.lat_mesh = np.meshgrid(self.lon_grid, self.lat_grid)

        # 网格点坐标扁平化
        self.grid_points = np.column_stack([
            self.lon_mesh.ravel(),
            self.lat_mesh.ravel()
        ])

    def load_and_preprocess(self):

        # 尝试加载缓存
        if self.processed_cache_path and os.path.exists(self.processed_cache_path):
            try:
                # 检查CSV文件修改时间
                csv_mtime = os.path.getmtime(self.csv_path)

                with open(self.processed_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)

                if cache_data.get('csv_mtime') == csv_mtime:
                    self.flash_data = cache_data['flash_data']
                    self.time_buckets = cache_data['time_buckets']
                    dist_print(f"Loaded preprocessed lightning data from cache: {self.processed_cache_path}")
                    return self
                else:
                    dist_print(f"Cache expired (CSV modified). Reloading from source.")
            except Exception as e:
                dist_print(f"Failed to load lightning cache: {e}")

        """加载并预处理闪电数据"""
        dist_print(f"Loading lightning data from {self.csv_path}")
        # 读取CSV文件
        df = pd.read_csv(self.csv_path)

        # 转换时间格式
        df['datetime'] = pd.to_datetime(df['DDATETIME'], format='mixed')

        # 创建6分钟时间桶
        df['time_bucket'] = df['datetime'].dt.floor('6min')

        # 筛选有效数据点
        valid_mask = (
                (df['LONGITUDE'] >= self.grid_info['lon_min']) &
                (df['LONGITUDE'] <= self.grid_info['lon_max']) &
                (df['LATITUDE'] >= self.grid_info['lat_min']) &
                (df['LATITUDE'] <= self.grid_info['lat_max'])
        )
        df = df[valid_mask].copy()

        # 聚合闪电数据
        self.flash_data = df.groupby('time_bucket').agg({
            'LONGITUDE': list,
            'LATITUDE': list,
            'FLASH': list,
            'CGFLASH': list
        }).reset_index()

        # Ensure data is sorted by time for binary search
        self.flash_data = self.flash_data.sort_values('time_bucket').reset_index(drop=True)
        # Pre-convert time buckets to numpy array for fast search
        self.time_buckets = self.flash_data['time_bucket'].values

        # 保存缓存
        # 仅在非分布式模式或分布式模式下的主进程(Rank 0)保存缓存
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        if self.processed_cache_path and (local_rank == -1 or local_rank == 0):
            try:
                os.makedirs(os.path.dirname(self.processed_cache_path), exist_ok=True)

                # 原子写入：先写临时文件，再重命名
                temp_path = self.processed_cache_path + f".tmp.{os.getpid()}"
                cache_data = {
                    'flash_data': self.flash_data,
                    'time_buckets': self.time_buckets,
                    'csv_mtime': os.path.getmtime(self.csv_path),
                    'timestamp': datetime.now().timestamp()
                }

                with open(temp_path, 'wb') as f:
                    pickle.dump(cache_data, f)

                # 原子重命名（Windows兼容）
                if os.path.exists(self.processed_cache_path):
                    os.remove(self.processed_cache_path)
                os.rename(temp_path, self.processed_cache_path)

                dist_print(f"Saved preprocessed lightning data to cache: {self.processed_cache_path}")
            except Exception as e:
                dist_print(f"Failed to save lightning cache: {e}")
                # 清理临时文件
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass

        dist_print(f"Loaded {len(self.flash_data)} time buckets of lightning data")
        return self

    def find_lightning_rich_indices(self, timestamps, time_window_minutes=60, min_events=1000):
        """
        快速筛选闪电丰富的样本索引（纯Pandas统计，零图像生成）

        Args:
            timestamps: 所有待检查的时间戳列表
            time_window_minutes: 预测时间窗口（分钟）
            min_events: 最小闪电事件数阈值

        Returns:
            list: 满足条件的样本索引列表
        """
        """优化版：使用numpy向量化加速"""
        import numpy as np

        dist_print(f"Fast scanning {len(timestamps)} samples...")

        if self.flash_data is None or len(self.flash_data) == 0:
            return []

        # 1. 构建时间->计数的字典（与当前版本相同）
        flash_counts = {}
        for _, row in self.flash_data.iterrows():
            flash_counts[row['time_bucket']] = len(row['FLASH'])

        # 2. 批量处理（减少Python循环开销）
        rich_indices = []
        batch_size = 1000

        for batch_start in range(0, len(timestamps), batch_size):
            batch_end = min(batch_start + batch_size, len(timestamps))
            batch_ts = timestamps[batch_start:batch_end]

            # 并行检查批次内的样本
            for local_idx, ts in enumerate(batch_ts):
                total_flashes = 0
                ts_pd = pd.Timestamp(ts)

                # 使用列表推导式加速
                check_times = [ts_pd + pd.Timedelta(minutes=m) for m in range(0, time_window_minutes + 1, 6)]
                check_buckets = [t.floor('6min') for t in check_times]

                total_flashes = sum(flash_counts.get(bucket, 0) for bucket in check_buckets)

                if total_flashes >= min_events:
                    rich_indices.append(batch_start + local_idx)

        dist_print(f"Found {len(rich_indices)} lightning-rich samples (>={min_events} events)")
        return rich_indices

    def create_lightning_grid(self, timestamp, field='FLASH', flash_type=0, time_window_minutes=6, smoothing=False,
                              spatial_radius_km=16, binary=True, use_torch_expansion=True):
        """
        为指定时间戳创建闪电网格 (Vectorized version)

        参数:
        timestamp: 目标时间戳
        field: 使用的字段 ('FLASH'或'CGFLASH')
        flash_type: 当field='CGFLASH'时，指定闪电类型 (0=地闪, 1=云闪)
        time_window_minutes: 时间窗口（分钟）
        smoothing: 是否应用高斯平滑
        spatial_radius_km: 空间扩展半径（公里），0表示不扩展
        binary: 是否输出二值化结果

        返回:
        torch.Tensor: 闪电网格 (H, W)
        """
        # Check cache
        cache_key = f"{timestamp}_{field}_{flash_type}_{time_window_minutes}_{smoothing}_{spatial_radius_km}_{binary}_{use_torch_expansion}"
        if cache_key in self.grid_cache:
            return self.grid_cache[cache_key]

        # Manage cache size to prevent OOM
        if len(self.grid_cache) > 1000:
            # Simple eviction: remove oldest 200 entries (assuming dict is ordered in Python 3.7+)
            keys = list(self.grid_cache.keys())
            for k in keys[:200]:
                self.grid_cache.pop(k, None)

        # Initialize grid
        grid = np.zeros((self.grid_info['height'], self.grid_info['width']), dtype=np.float32)

        # Define time window
        time_start = timestamp - timedelta(minutes=time_window_minutes)

        # Using binary search (searchsorted) instead of boolean indexing for >100x speedup
        # Convert timestamp to numpy datetime64 for compatibility
        ts_np = pd.Timestamp(timestamp).to_datetime64()
        start_np = pd.Timestamp(time_start).to_datetime64()

        # Find indices using binary search on pre-sorted array
        # This reduces complexity from O(N) to O(log N)
        start_idx = np.searchsorted(self.time_buckets, start_np)
        end_idx = np.searchsorted(self.time_buckets, ts_np, side='right')

        # Slice using iloc (zero-copy view if possible)
        window_data = self.flash_data.iloc[start_idx:end_idx]

        if len(window_data) == 0:
            tensor = torch.from_numpy(grid)
            self.grid_cache[cache_key] = tensor
            return tensor

        # === Vectorized processing ===
        # 1. Collect all lightning points and weights
        all_lons = []
        all_lats = []
        all_weights = []

        # Grid parameters for faster computation
        lon_min, lon_max = self.grid_info['lon_min'], self.grid_info['lon_max']
        lat_min, lat_max = self.grid_info['lat_min'], self.grid_info['lat_max']
        width, height = self.grid_info['width'], self.grid_info['height']

        # Pre-calculate conversion factors
        lon_factor = (width - 1) / (lon_max - lon_min)
        lat_factor = (height - 1) / (lat_max - lat_min)

        # Process each time bucket
        for row in window_data.itertuples():
            # Calculate time decay factor once per row
            time_diff = (timestamp - row.time_bucket).total_seconds() / 60.0
            time_weight = max(0.1, 1.0 - (time_diff / time_window_minutes))

            lons = row.LONGITUDE
            lats = row.LATITUDE

            # Filter by flash type if needed
            if field == 'CGFLASH':
                cgflash_values = row.CGFLASH
                # Vectorized filtering
                cgflash_array = np.array(cgflash_values)
                mask_flash = cgflash_array == flash_type
                if not np.any(mask_flash):
                    continue
                lons = np.array(lons)[mask_flash].tolist()
                lats = np.array(lats)[mask_flash].tolist()

            if not lons:
                continue

            all_lons.extend(lons)
            all_lats.extend(lats)
            all_weights.extend([time_weight] * len(lons))

        if not all_lons:
            tensor = torch.from_numpy(grid)
            self.grid_cache[cache_key] = tensor
            return tensor

        # Convert to numpy arrays for vectorization
        lons_arr = np.array(all_lons)
        lats_arr = np.array(all_lats)
        weights_arr = np.array(all_weights)

        # 2. Vectorized coordinate mapping
        # X: Longitude (West -> East) maps to 0 -> Width (Left -> Right)
        x_indices = np.clip(((lons_arr - lon_min) * lon_factor).astype(np.int32), 0, width - 1)
        
        # Y: Latitude (South -> North). 
        # CAUTION: Image coordinates usually start from Top-Left (North-West).
        # So Lat_Max should map to Y=0, and Lat_Min to Y=Height.
        # Original: ((lats_arr - lat_min) * lat_factor) -> Maps South to 0 (Bottom-up coordinate)
        # Correct for Image: ((lat_max - lats_arr) * lat_factor) -> Maps North to 0 (Top-down coordinate)
        
        # Let's assume standard image coordinates (Top-down) unless radar data is known to be flipped
        y_indices = np.clip(((lat_max - lats_arr) * lat_factor).astype(np.int32), 0, height - 1)

        # 3. Apply spatial expansion if requested
        if spatial_radius_km > 0:
            # Accumulate points first
            point_grid = np.zeros_like(grid)
            np.add.at(point_grid, (y_indices, x_indices), weights_arr)

            # [OPTIMIZATION] Use Distance Transform for large radius dilation
            # This is O(N) independent of radius, much faster than convolution/max_pool for large kernels

            # Calculate pixel sizes in km
            avg_lat = (lat_min + lat_max) / 2
            km_per_deg_lat = 111.0
            km_per_deg_lon = 111.0 * np.cos(np.radians(avg_lat))

            deg_per_pixel_lat = (lat_max - lat_min) / height
            deg_per_pixel_lon = (lon_max - lon_min) / width

            pixel_size_km_lat = deg_per_pixel_lat * km_per_deg_lat
            pixel_size_km_lon = deg_per_pixel_lon * km_per_deg_lon

            if binary and spatial_radius_km >= 7:  # Use distance transform for large radii
                from scipy.ndimage import distance_transform_edt

                # Create binary mask of lightning points
                point_mask = point_grid > 0

                # Distance transform expects True/1 for background, False/0 for features
                # We want distance FROM lightning points
                dist_map = distance_transform_edt(~point_mask, sampling=[pixel_size_km_lat, pixel_size_km_lon])

                # Create expanded mask where distance <= radius
                grid = (dist_map <= spatial_radius_km).astype(np.float32)

            elif use_torch_expansion:
                # Fast PyTorch-based expansion (CPU only) - keep for small radii
                import torch.nn.functional as F

                # Convert to tensor on CPU
                device = torch.device('cpu')
                grid_tensor = torch.from_numpy(point_grid).unsqueeze(0).unsqueeze(0).to(device)

                # Calculate kernel size
                kernel_size = max(3, 2 * int(spatial_radius_km / pixel_size_km_lon) + 1)

                # Only use max_pool for smaller kernels (< 50 pixels)
                if kernel_size < 50:
                    if binary:
                        # Binary dilation using max pooling
                        expanded = F.max_pool2d(grid_tensor, kernel_size=kernel_size,
                                                stride=1, padding=kernel_size // 2)
                        grid = (expanded.squeeze().cpu().numpy() > 0).astype(np.float32)
                    else:
                        # Weighted expansion using average pooling
                        expanded = F.avg_pool2d(grid_tensor, kernel_size=kernel_size,
                                                stride=1, padding=kernel_size // 2) * (kernel_size ** 2)
                        grid = expanded.squeeze().cpu().numpy()
                else:
                    # Fall back to distance transform for large kernels
                    from scipy.ndimage import distance_transform_edt
                    point_mask = point_grid > 0
                    dist_map = distance_transform_edt(~point_mask, sampling=[pixel_size_km_lat, pixel_size_km_lon])

                    if binary:
                        grid = (dist_map <= spatial_radius_km).astype(np.float32)
                    else:
                        # For non-binary, use distance-based weighting
                        grid = np.maximum(0, 1 - dist_map / spatial_radius_km) * point_grid.max()
        else:
            # No spatial expansion - direct accumulation
            np.add.at(grid, (y_indices, x_indices), weights_arr)

        # Apply Gaussian smoothing if requested
        if smoothing:
            from scipy.ndimage import gaussian_filter
            sigma = 0.5 if spatial_radius_km > 0 else 1.0
            grid = gaussian_filter(grid, sigma=sigma)

        # Final processing
        if binary:
            grid = (grid > 0).astype(np.float32)
        elif grid.max() > 0:
            grid = grid / grid.max()

        # Convert to tensor and cache
        tensor = torch.from_numpy(grid)
        self.grid_cache[cache_key] = tensor
        return tensor

    def visualize_lightning_distribution(self, output_dir=None, field='CGFLASH', flash_type=None):
        """
        Visualize the distribution of lightning events across time buckets

        Parameters:
        output_dir: Directory to save visualization, if None will just display
        field: Field to analyze ('FLASH' or 'CGFLASH')
        flash_type: If field='CGFLASH', specify flash type (0=ground, 1=cloud) or None for all
        """
        import matplotlib.pyplot as plt
        import os

        if self.flash_data is None:
            dist_print("No lightning data loaded. Call load_and_preprocess() first.")
            return

        # Prepare data
        counts = []
        timestamps = []

        for _, row in self.flash_data.iterrows():
            if field == 'CGFLASH' and flash_type is not None:
                # Count only specific flash type
                cgflash_values = row['CGFLASH']
                filtered_count = sum(1 for val in cgflash_values if val == flash_type)
                counts.append(filtered_count)
            else:
                # Count all flashes
                counts.append(len(row[field]))

            timestamps.append(row['time_bucket'])

        # Create filtered data without zeros for visualization
        non_zero_indices = [i for i, count in enumerate(counts) if count > 0]
        non_zero_counts = [counts[i] for i in non_zero_indices]
        non_zero_timestamps = [timestamps[i] for i in non_zero_indices]

        # Create plot
        plt.figure(figsize=(12, 6))

        # Histogram of counts (excluding zeros)
        plt.subplot(1, 2, 1)
        plt.hist(non_zero_counts, bins=30)
        plt.title(f'Distribution of Lightning Counts per 6-min Bucket\n(Excluding Zero Values)')
        plt.xlabel('Number of Lightning Events')
        plt.ylabel('Frequency')

        # Time series of counts
        plt.subplot(1, 2, 2)
        plt.plot(non_zero_timestamps, non_zero_counts)
        plt.title(f'Lightning Events Over Time\n(Excluding Zero Values)')
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save or display
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            flash_type_str = f"_type{flash_type}" if flash_type is not None else ""
            filename = os.path.join(output_dir, f"lightning_distribution_{field}{flash_type_str}_nonzero.png")
            plt.savefig(filename)
            dist_print(f"Saved lightning distribution visualization to {filename}")
        else:
            plt.show()

        # Print statistics
        total_events = sum(counts)
        max_count = max(counts) if counts else 0
        non_zero_buckets = sum(1 for c in counts if c > 0)

        dist_print(f"Lightning Distribution Statistics ({field}):")
        dist_print(f"  Total events: {total_events}")
        dist_print(f"  Time buckets with data: {non_zero_buckets}/{len(counts)}")
        dist_print(f"  Average events per bucket: {total_events / non_zero_buckets if non_zero_buckets else 0:.2f}")
        dist_print(f"  Maximum events in a bucket: {max_count}")

        return counts, timestamps