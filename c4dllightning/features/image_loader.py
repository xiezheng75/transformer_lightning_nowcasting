import os
import glob
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from datetime import datetime, timedelta
from collections import defaultdict
import re
import concurrent.futures
from functools import partial
import torch.nn.functional as F
from sklearn.cluster import KMeans
from skimage.transform import resize


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

class RadarImageGenerator:
    """雷达图像数据生成器"""

    def __init__(self, data_root, products, img_size, timesteps, filename_pattern=None,
                 parallel_workers=None, max_files_per_product=None, use_background_mask=False,
                 prefetch_factor=2, cache_dir=None):



        """
        data_root: 数据根目录
        products: 雷达产品列表 (['dbz', 'dbzh', 'vil'])
        img_size: 图像尺寸 (height, width)
        timesteps: 时间步数 (past, future)
        filename_pattern: 文件名模式（用于提取时间戳）
        parallel_workers: 并行工作进程数
        prefetch_factor: 数据预取因子
        cache_dir: 缓存目录路径，None表示禁用磁盘缓存
        """

        self.data_root = data_root
        self.products = products
        self.img_size = img_size
        self.past_timesteps, self.future_timesteps = timesteps
        self.parallel_workers = parallel_workers
        self.max_files_per_product = max_files_per_product
        self.use_background_mask = use_background_mask
        self.prefetch_factor = prefetch_factor

        self.background_mask = None
        # A reference CAPPI frame can be supplied here to derive the background
        # mask, e.g. <data_root>/newCAPPI/YYYY/YYYYMMDD/AWS_GD_CAPPI_*_2500m.png
        self.use_background_mask = use_background_mask

        # 产品路径映射
        self.product_paths = {
            'dbz': os.path.join(data_root, "CR"),
            'dbzh': os.path.join(data_root, "Thunder/PONDSImage/dbzh"),
            'vil': os.path.join(data_root, "Thunder/PONDSImage/vil")
        }

        # 文件名模式（用于提取时间戳）
        self.filename_patterns = filename_pattern if filename_pattern else [
            r'(\d{12})',  # 默认尝试匹配任何12位数字作为时间戳
        ]
        if isinstance(self.filename_patterns, str):
            self.filename_patterns = [self.filename_patterns]

        # 建立时间索引 - 确保只包含所有产品都有的时间点
        self.time_index = self._build_time_index()

        # 过滤掉任何不包含所有产品的时间点
        filtered_index = {ts: data for ts, data in self.time_index.items()
                          if all(product in data for product in self.products)}

        if not filtered_index:
            dist_print("严重警告: 过滤后没有任何时间点同时包含所有产品!")
            dist_print("请检查数据路径和文件命名模式，确保所有产品在某些时间点上有重叠。")
            # 尝试找出最早的共同时间点
            product_start_times = {}
            for product in self.products:
                product_times = [ts for ts, data in self.time_index.items() if product in data]
                if product_times:
                    product_start_times[product] = min(product_times)

            if product_start_times:
                dist_print("各产品最早时间点:")
                for prod, time in product_start_times.items():
                    dist_print(f"  {prod}: {time}")

        self.time_index = filtered_index
        dist_print(f"过滤后，时间索引中有 {len(self.time_index)} 个包含所有产品的时间点")

        # 归一化参数
        self.normalization_params = {
            'dbz': (0, 75),  # 雷达反射率范围 5-75 dbz
            'dbzh': (0, 18.5),  # 高度反射率范围 0.5-18.5 km
            'vil': (0, 70)  # 垂直液态水含量范围 1-70 kg/m²
        }

        # 设置缓存目录
        # 修改逻辑：默认禁用磁盘缓存，除非显式提供路径
        # 这可以防止默认写入网络挂载点导致的性能问题
        self.cache_dir = cache_dir

        if self.cache_dir:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                dist_print(f"Processed data cache directory: {self.cache_dir}")
                # Warn if the cache sits on a network mount: write latency there
                # can dominate training time.
                if os.path.ismount(self.cache_dir):
                    dist_print(
                        "NOTE: Cache directory appears to be on a network mount. This may slow down training due to write latency.")
            except Exception as e:
                dist_print(f"Warning: Could not create cache directory {self.cache_dir}: {e}")
                self.cache_dir = None
        else:
            dist_print("Disk caching disabled. Using in-memory processing only (fastest for high-latency storage).")

        # Initialize Color Lookup Tables (LUTs) for fast mapping
        self._init_color_luts()

        # 清理可能遗留的临时文件（仅主进程执行）
        if self.cache_dir and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            temp_files = glob.glob(os.path.join(self.cache_dir, "*.tmp.*"))
            if temp_files:
                dist_print(f"Cleaning up {len(temp_files)} temporary files from previous runs...")
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass

    def _init_color_luts(self):
        """Initialize lookup tables for fast color-to-value mapping"""
        self.luts = {}
        dist_print("Initializing color lookup tables...")
        
        for product in ['dbzh', 'vil']:
            if product not in self.products:
                continue
                
            # Define color maps (same as in _extract_colorbar)
            if product == 'dbzh':
                color_to_value_map = {
                    (2, 255, 255): 0.5, (2, 190, 190): 1.5, (1, 160, 95): 2.5,
                    (0, 130, 0): 3.5, (0, 255, 0): 4.5, (81, 255, 19): 5.5,
                    (162, 255, 38): 6.5, (255, 244, 130): 7.5, (255, 249, 65): 8.5,
                    (255, 255, 0): 9.5, (255, 154, 2): 10.5, (255, 107, 1): 11.5,
                    (255, 60, 0): 12.5, (255, 0, 0): 13.5, (227, 0, 0): 14.5,
                    (200, 0, 0): 15.5, (255, 0, 255): 16.5, (226, 16, 194): 17.5,
                    (197, 33, 133): 18.5
                }
            elif product == 'vil':
                color_to_value_map = {
                    (147, 147, 147): 1, (107, 107, 107): 5, (249, 162, 162): 10,
                    (236, 130, 130): 15, (195, 101, 101): 20, (0, 251, 134): 25,
                    (0, 180, 0): 30, (255, 255, 101): 35, (203, 203, 85): 40,
                    (255, 85, 85): 45, (214, 0, 0): 50, (166, 0, 0): 55,
                    (0, 0, 255): 60, (255, 255, 255): 65, (228, 0, 255): 70
                }

            # Create LUT: 16M entries float32 array (~64MB)
            # Use int32 for RGB indexing: (R<<16) | (G<<8) | B
            lut = np.zeros(16777216, dtype=np.float32)
            
            for color, value in color_to_value_map.items():
                r, g, b = color
                idx = (int(r) << 16) | (int(g) << 8) | int(b)
                if idx < 16777216:
                    lut[idx] = value
                
            self.luts[product] = lut
            dist_print(f"Initialized fast LUT for {product}")

    def _build_time_index(self):
        """建立时间戳到文件路径的映射"""
        dist_print(f"[{datetime.now()}] Starting to build time index...")
        time_index = defaultdict(dict)
        corrupted_files = []

        # 确定并行工作进程数
        n_workers = getattr(self, 'parallel_workers', None) or os.cpu_count()
        dist_print(f"In image_loader.py,Building time index with {n_workers} parallel workers...")

        # 收集需要处理的文件列表
        file_tasks = []

        # 首先收集所有产品的时间信息，以确定共同的时间起点
        all_product_timestamps = {}
        earliest_common_time = None

        for product in self.products:
            product_dir = self.product_paths[product]
            dist_print(f"Collecting timestamp info for product: {product} from {product_dir}")

            product_timestamps = []
            for root, _, files in os.walk(product_dir):
                for file in files:
                    # Handle different file types based on the product
                    if product == 'dbz' and file.endswith('.ref') and 'ref_all_' in file:
                        # Parse .ref files for dbz product
                        try:
                            # Extract timestamp from filename format: ref_all_YYYYMMDDHHMM_14.ref
                            timestamp_match = re.search(r'ref_all_(\d{12})_', file)
                            if timestamp_match:
                                timestamp_str = timestamp_match.group(1)
                                dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M")
                                full_path = os.path.join(root, file)

                                # Validate file size for .ref files
                                file_size = os.path.getsize(full_path)
                                expected_size = 700 * 900  # Expected bytes for uint8 array

                                if file_size == expected_size:
                                    product_timestamps.append((dt, product, full_path, file))
                                else:
                                    dist_print(f"Skipping corrupted DBZ file (wrong size): {file} (size: {file_size})")

                        except Exception as e:
                            dist_print(f"无法解析dbz .ref文件时间戳 (文件: {file}): {e}")
                    elif file.endswith('.png'):
                        # Process PNG files for dbzh and vil products
                        # no need to process GD files for dbz
                        if product == 'dbz' and 'GD' not in file:
                            continue

                        full_path = os.path.join(root, file)
                        # 提取时间戳信息
                        timestamp_str = None
                        for pattern in self.filename_patterns:
                            match = re.search(pattern, file)
                            if match:
                                timestamp_str = match.group(1)
                                break

                        if timestamp_str:
                            try:
                                dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M")
                                product_timestamps.append((dt, product, full_path, file))
                            except Exception as e:
                                dist_print(f"无法解析时间戳 {timestamp_str} (文件: {file}): {e}")

            if product_timestamps:
                product_timestamps.sort(key=lambda x: x[0])
                all_product_timestamps[product] = product_timestamps
                dist_print(f"产品 {product} 时间范围: {product_timestamps[0][0]} 到 {product_timestamps[-1][0]}")

                # 额外检查并打印前几个文件，以便更好地了解数据结构
                for i in range(min(3, len(product_timestamps))):
                    dist_print(f"  {product} 样本 #{i}: {product_timestamps[i][0]} - {product_timestamps[i][3]}")

        # 确定所有产品共同的最早时间点
        # 获取每个产品的起始时间
        # 获取每个产品的时间点集合
        product_time_sets = {}
        for prod, times in all_product_timestamps.items():
            if times:
                # 收集该产品的所有时间点
                product_time_sets[prod] = set(item[0] for item in times)

        # 找到共同时间点
        common_times = None
        for prod, time_set in product_time_sets.items():
            if common_times is None:
                common_times = time_set
            else:
                common_times &= time_set  # 取交集

        if common_times and len(common_times) > 0:
            earliest_common_time = min(common_times)  # 使用最早的共同时间点
            dist_print(f"确定共同起始时间点: {earliest_common_time}")
            dist_print(f"共有 {len(common_times)} 个时间点同时拥有所有产品")
        else:
            # 退而求其次，尝试使用最晚的起始时间
            start_times = {prod: times[0][0] for prod, times in all_product_timestamps.items() if times}
            if len(start_times) == len(self.products):
                earliest_common_time = max(start_times.values())
                dist_print(f"警告: 没有找到同时包含所有产品的时间点！")
                dist_print(f"使用最晚的起始时间作为近似共同起点: {earliest_common_time}")

        # 基于共同起始时间过滤并收集文件
        for product in self.products:
            product_dir = self.product_paths[product]
            product_files = []

            if product not in all_product_timestamps or not all_product_timestamps[product]:
                dist_print(f"产品 {product} 没有可用的时间戳信息")
                continue

            # 过滤出共同时间点之后的文件
            filtered_timestamps = []
            if earliest_common_time:
                filtered_timestamps = [
                    item for item in all_product_timestamps[product]
                    if item[0] >= earliest_common_time
                ]
            else:
                filtered_timestamps = all_product_timestamps[product]

            dist_print(f"产品 {product} 在共同起始时间 {earliest_common_time} 之后有 {len(filtered_timestamps)} 个文件")

            # 收集该产品的文件，最多收集max_files_per_product个
            file_count = 0
            max_files = self.max_files_per_product or float('inf')

            # 添加到product_files，保持时间顺序
            for i, (dt, prod, path, fname) in enumerate(filtered_timestamps):
                if i < max_files:
                    product_files.append((prod, path, fname))
                    file_count += 1
                    # 每100个文件打印一次，用于验证
                    if i % 10000 == 0 or i == len(filtered_timestamps) - 1:
                        dist_print(f"添加 {prod} 文件 #{i}: {fname} (时间戳: {dt})")
                else:
                    break

            dist_print(f"已为产品 {product} 收集 {file_count}/{len(filtered_timestamps)} 个时间排序文件")
            dist_print(f"Collected {len(product_files)} files for product: {product}")
            file_tasks.extend(product_files)

        dist_print(f"Total files to process: {len(file_tasks)}")


        def process_file(task):
            # dist_print(f"Processing file...")
            product, full_path, file = task
            result = {"product": product, "path": full_path, "timestamp": None, "error": None}
            # dist_print(f"Processing {file} for product {product}...")

            # 验证文件是否可读
            try:
                if product == 'dbz' and file.endswith('.ref'):
                    # For .ref files, check if file exists and can be opened
                    with open(full_path, 'rb') as f:
                        # Just check if it can be opened
                        pass
                    # dist_print(f"Opened {file} successfully.")
                else:
                    # For image files, check if they can be opened
                    with Image.open(full_path) as img:
                        # dist_print(f"Opened {file} successfully.")
                        # Just check if it can be opened
                        pass
            except Exception as e:
                result["error"] = f"Cannot open file: {str(e)}"
                return result

            # 尝试匹配文件名中的时间戳
            timestamp_str = None
            for pattern in self.filename_patterns:
                match = re.search(pattern, file)
                if match:
                    timestamp_str = match.group(1)
                    break

            if timestamp_str:
                try:
                    dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M")
                    result["timestamp"] = dt
                except Exception as e:
                    result["error"] = f"Error processing timestamp: {str(e)}"
            else:
                result["error"] = f"Could not extract timestamp"

            return result

        # 使用进程池并行处理文件
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            dist_print(f"Starting parallel processing of {len(file_tasks)} files...")
            results = list(executor.map(process_file, file_tasks))

            # 分析每个产品的时间覆盖情况
            product_timestamps = defaultdict(list)
            for result in results:
                if not result["error"] and result["timestamp"]:
                    product = result["product"]
                    product_timestamps[product].append(result["timestamp"])
                    # 存储到时间索引
                    time_index[result["timestamp"]][product] = result["path"]

            # 打印每个产品的时间覆盖信息
            dist_print("\n时间覆盖分析:")
            for product, timestamps in product_timestamps.items():
                timestamps.sort()
                dist_print(f"产品 {product}: {len(timestamps)} 个时间点")
                if timestamps:
                    dist_print(f"  起始时间: {timestamps[0]}")
                    dist_print(f"  结束时间: {timestamps[-1]}")
                    dist_print(f"  时间跨度: {timestamps[-1] - timestamps[0]}")
                    dist_print(f"  平均时间间隔: {(timestamps[-1] - timestamps[0]) / max(1, len(timestamps) - 1)}")

            # Collect the unreadable files for reporting.
            #
            # This loop used to re-assign paths into time_index using the loop
            # variables `product` and `full_path` left over from the enclosing
            # scopes rather than result["product"]. Because `product` was frozen
            # at whatever the preceding summary loop happened to end on, every
            # result was written into that one slot, so e.g. dbzh paths could land
            # in the vil slot. The assignment is redundant -- the loop above
            # already stores result["path"] under result["product"] -- so it is
            # removed rather than repaired.
            for result in results:
                if result["error"]:
                    if "Cannot open file" in result["error"]:
                        corrupted_files.append((result["path"], result["error"]))
                    else:
                        dist_print(f"Error with {os.path.basename(result['path'])}: {result['error']}")

        # 输出有问题的文件
        if corrupted_files:
            dist_print(f"Found {len(corrupted_files)} corrupted or unreadable images. First 10:")
            for i, (path, error) in enumerate(corrupted_files[:10]):
                dist_print(f"  - {path}: {error}")

        dist_print(f"[{datetime.now()}] Finished building time index with {len(time_index)} unique timestamps.")
        # 按时间排序
        sorted_time_index = dict(sorted(time_index.items()))

        # 找出时间戳中所有三种产品都具备的时间点
        complete_timestamps = [ts for ts, data in sorted_time_index.items()
                               if all(product in data for product in self.products)]

        dist_print(f"时间索引中总共有 {len(sorted_time_index)} 个时间点")
        dist_print(f"其中包含所有产品的时间点有 {len(complete_timestamps)} 个")

        if complete_timestamps:
            earliest_complete = min(complete_timestamps)
            latest_complete = max(complete_timestamps)
            dist_print(f"所有产品共有的时间范围: {earliest_complete} 到 {latest_complete}")

            # 打印样本完整时间点
            sample_count = min(5, len(complete_timestamps))
            dist_print(f"样本完整时间点:")
            for i in range(sample_count):
                idx = i * (len(complete_timestamps) // max(1, sample_count))
                if idx < len(complete_timestamps):
                    ts = complete_timestamps[idx]
                    dist_print(f"  {ts}: {', '.join(sorted_time_index[ts].keys())}")
        else:
            dist_print("警告: 没有任何时间点同时包含所有产品!")

        return sorted_time_index

    def _read_dbz_numerical_file(self, timestamp):
        """
        Read numerical DBZ data file based on timestamp
        timestamp: datetime object
        returns: numpy array of DBZ values
        """
        # Check if timestamp is before 2022 (when dbz data starts)
        if timestamp.year < 2022:
            dist_print(f"Warning: Attempting to load DBZ data from {timestamp}, but DBZ data only starts from 2022")
            return None

        # Construct file path: <data_root>/CR/YYYY/YYYYMMDD/ref_all_YYYYMMDDHHMM_14.ref
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        hour = timestamp.hour
        minute = timestamp.minute

        date_str = f"{year}{month:02d}{day:02d}"
        datetime_str = f"{date_str}{hour:02d}{minute:02d}"

        import os
        file_path = os.path.join(self.data_root, f"CR/{year}/{date_str}/ref_all_{datetime_str}_14.ref")

        try:
            # Check if file exists and has the correct size
            import os
            if not os.path.exists(file_path):
                dist_print(f"DBZ data file not found: {file_path}")
                return None

            file_size = os.path.getsize(file_path)
            expected_size = 700 * 900  # Expected number of bytes for uint8 array

            if file_size != expected_size:
                dist_print(f"DBZ file has incorrect size: {file_path} (size: {file_size}, expected: {expected_size})")
                return None

            # Read the binary file (700 rows x 900 columns)
            with open(file_path, 'rb') as file:
                data = np.frombuffer(file.read(), dtype=np.uint8).reshape((700, 900))

            # Original data bounds
            orig_lon_range = (108.505, 117.495)
            orig_lat_range = (19.0519, 26.0419)

            # Target bounds to match DBZH and VIL
            target_lon_range = (109.505, 117.495)
            target_lat_range = (19.0519, 26.0419)

            # Calculate indices for cropping
            # For longitude: we need to crop from the left side
            lon_resolution = (orig_lon_range[1] - orig_lon_range[0]) / 900  # degrees per pixel
            lon_start_idx = int((target_lon_range[0] - orig_lon_range[0]) / lon_resolution)

            # Crop the data (we only need to crop longitude since latitude range is the same)
            cropped_data = data[:, lon_start_idx:]

            # Resize to match our standard image size if needed
            if cropped_data.shape != self.img_size:
                from skimage.transform import resize
                cropped_data = resize(cropped_data, self.img_size, preserve_range=True).astype(np.uint8)

            # dist_print(f"Successfully loaded DBZ numerical data from {file_path}")
            # dist_print(f"  - Data shape: {cropped_data.shape}")
            # dist_print(f"  - Value range: {cropped_data.min()} to {cropped_data.max()}")

            return cropped_data

        except Exception as e:
            dist_print(f"Error reading DBZ data file {file_path}: {str(e)}")
            return None

    def _normalize_product(self, data, product):
        """
        对不同的产品进行归一化处理
        data: numpy数组，产品数据
        product: 产品类型
        返回: 归一化后的numpy数组
        """
        if product == 'dbz':
            # For numerical DBZ data, we normalize from 0-75 range to 0-1
            # Clip values below 5 to 0 (typically noise or no precipitation)
            data = np.clip(data, 0, 75)
            data = np.where(data < 5, 0, data)
            return data / 75.0
        elif product == 'dbzh':
            # 对于回波顶高，通常是0-20km，归一化到0-1
            data = np.clip(data, 0, 20)
            return data / 20.0
        elif product == 'vil':
            # 对于液态水，通常是0-70kg/m²，归一化到0-1
            data = np.clip(data, 0, 70)
            return data / 70.0
        else:
            # 默认归一化到0-1
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val > min_val:
                return (data - min_val) / (max_val - min_val)
            else:
                return np.zeros_like(data)


    def _get_cache_path(self, timestamp):
        """Generate cache filename for a specific timestamp"""
        timestamp_str = timestamp.strftime("%Y%m%d%H%M")
        # Cache includes all products in one file [H, W, C]
        return os.path.join(self.cache_dir, f"radar_{timestamp_str}.npy")

    def load_time_series(self, start_time, time_steps=6, interval=timedelta(minutes=6), use_gpu=True):
        """
        加载时间序列雷达数据
        start_time: 起始时间 (datetime)
        time_steps: 时间步数
        interval: 时间间隔
        """
        # 初始化输出张量 [T, H, W, C]
        series = np.zeros((time_steps, self.img_size[0], self.img_size[1], len(self.products)), dtype=np.float32)

        # 检查时间索引是否为空
        if not self.time_index:
            dist_print("Warning: Empty time index. No valid images found.")
            return series

        # 定义各产品的colorbar区域掩码 (将这些区域从最终数据中排除)
        colorbar_masks = {}
        for product in self.products:
            mask = np.ones((self.img_size[0], self.img_size[1]), dtype=bool)

            # 根据产品类型定义colorbar区域
            if product == 'dbz':
                # 右侧colorbar区域
                x_start = int(self.img_size[1] * 0.85)
                y_start = int(self.img_size[0] * 0.65)
                mask[y_start:, x_start:] = False
            elif product == 'dbzh':
                # 底部colorbar区域
                y_start = int(self.img_size[0] * 0.88)
                mask[y_start:, :] = False
                # 右侧区域也需要排除
                x_start = int(self.img_size[1] * 0.85)
                y_start = int(self.img_size[0] * 0.65)
                mask[y_start:, x_start:] = False
            elif product == 'vil':
                # 右侧colorbar区域
                x_start = int(self.img_size[1] * 0.85)
                y_start = int(self.img_size[0] * 0.52)
                mask[y_start:, x_start:] = False

            colorbar_masks[product] = mask

        # 检查时间索引键是否存在
        if not self.time_index.keys():
            dist_print("Warning: No time keys available")
            return series

        # 找到有效的时间点
        valid_times = [t for t in self.time_index.keys()
                       if all(p in self.time_index[t] for p in self.products)]

        if not valid_times:
            dist_print("Warning: No valid times with all products available")
            return series

        # 定义加载单个时间点的函数
        def load_timestep(t):
            target_time = start_time + t * interval
            closest_time = min(valid_times, key=lambda x: abs(x - target_time))

            # -------------------------------------------------------------------
            # 1. Check Cache First (Optimization #1)
            # -------------------------------------------------------------------
            if self.cache_dir:
                cache_path = self._get_cache_path(closest_time)
                if os.path.exists(cache_path):
                    try:
                        # Use memory mapping for near-instant load
                        # mmap_mode='r' lets OS handle paging, very efficient
                        # Copying to memory to avoid keeping file handle open excessively
                        cached_data = np.load(cache_path, mmap_mode='r')
                        return t, np.array(cached_data)  # Convert to array to detach from file
                    except Exception as e:
                        dist_print(f"Error reading cache {cache_path}: {e}")

            # -------------------------------------------------------------------
            # 2. Slow Loading (if not cached)
            # -------------------------------------------------------------------

            # 记录时间差异，用于调试
            time_diff = abs(closest_time - target_time)
            if time_diff > timedelta(hours=1):
                dist_print(
                    f"Warning: Large time difference ({time_diff}) at step {t}. Target: {target_time}, Using: {closest_time}")

            # 创建当前时间步的数据
            timestep_data = np.zeros((self.img_size[0], self.img_size[1], len(self.products)), dtype=np.float32)

            # 加载所有产品图像
            for c, product in enumerate(self.products):
                if product == 'dbz':
                    # 直接从数值文件加载DBZ数据
                    dbz_data = self._read_dbz_numerical_file(closest_time)

                    # If failed, try nearby timestamps (within 30 minutes)
                    if dbz_data is None:
                        for offset in [timedelta(minutes=6), timedelta(minutes=-6),
                                     timedelta(minutes=12), timedelta(minutes=-12),
                                     timedelta(minutes=18), timedelta(minutes=-18),
                                     timedelta(minutes=24), timedelta(minutes=-24),
                                     timedelta(minutes=30), timedelta(minutes=-30)]:
                            alternative_time = closest_time + offset
                            if alternative_time in valid_times:
                                dbz_data = self._read_dbz_numerical_file(alternative_time)
                                if dbz_data is not None:
                                    dist_print(f"Using alternative DBZ data from {alternative_time} instead of {closest_time}")
                                    break

                    if dbz_data is not None:
                        # 应用产品特定归一化
                        timestep_data[:, :, c] = self._normalize_product(dbz_data, product)
                    else:
                        dist_print(f"Warning: Could not load DBZ numerical data for {closest_time}")
                        timestep_data[:, :, c] = 0.0
                elif product in self.time_index[closest_time]:
                    # 处理其他产品 (DBZH, VIL)
                    img_path = self.time_index[closest_time][product]
                    try:
                        # 使用with语句确保资源正确释放，快速验证图片是否可读
                        with Image.open(img_path) as img:
                            # 尝试转换为RGB，如果失败说明图片有问题
                            try:
                                img_test = img.convert('RGB')
                                test_array = np.array(img_test)
                                if test_array.size == 0:
                                    dist_print(f"Warning: Empty image array, skipping: {img_path}")
                                    timestep_data[:, :, c] = 0.0
                                    continue
                            except Exception as e:
                                dist_print(f"Warning: Cannot convert image to RGB, skipping: {img_path} - {e}")
                                timestep_data[:, :, c] = 0.0
                                continue

                            # 图片验证通过，继续处理
                            # OPTIMIZATION: Use Fast Lookup Table (LUT) if available
                            if hasattr(self, 'luts') and product in self.luts:
                                try:
                                    # Instant color mapping using pre-computed LUT
                                    img_array = np.array(img.convert('RGB'))
                                    
                                    # Convert RGB to 24-bit integer index: (R<<16)|(G<<8)|B
                                    r = img_array[..., 0].astype(np.int32)
                                    g = img_array[..., 1].astype(np.int32)
                                    b = img_array[..., 2].astype(np.int32)
                                    idx = (r << 16) | (g << 8) | b
                                    
                                    # Direct lookup O(1)
                                    value_array = self.luts[product][idx]
                                    
                                    # Apply colorbar mask
                                    if product in colorbar_masks:
                                        value_array[~colorbar_masks[product]] = 0
                                        
                                    timestep_data[:, :, c] = self._normalize_product(value_array, product)
                                    continue # Skip slow path
                                except Exception as e:
                                    dist_print(f"Fast LUT failed for {product}: {e}, falling back")

                            # 尝试提取colorbar并创建颜色到数值的映射
                            color_to_value_func, valid_colorbar_colors = self._extract_colorbar(img, product)

                            if color_to_value_func:
                                # 如果成功提取了colorbar，使用颜色映射
                                img_rgb = img.convert('RGB')
                                img_array = np.array(img_rgb)

                                # 创建一个只包含colorbar中有效颜色的过滤后RGB图像
                                filtered_array = img_array.copy()

                                # 创建颜色到值的映射
                                color_value_map = {color: color_to_value_func(color) for color in valid_colorbar_colors}

                                # # 对每个像素进行处理 - 仅保留colorbar中的颜色
                                # for i in range(filtered_array.shape[0]):
                                #     for j in range(filtered_array.shape[1]):
                                #         pixel_color = filtered_array[i, j]
                                #
                                #         # 如果像素颜色不在有效colorbar颜色集合中，设置为黑色(0,0,0)
                                #         if tuple(pixel_color) not in valid_colorbar_colors:
                                #             filtered_array[i, j] = [0, 0, 0]

                                # -------------------------------------------------------------------
                                # 3. Optimized Color Filtering (Optimization #2)
                                # -------------------------------------------------------------------
                                # ORIGINAL SLOW CODE (nested loops):
                                # for i in range(filtered_array.shape[0]):
                                #     for j in range(filtered_array.shape[1]):
                                #         ...

                                # NEW VECTORIZED CODE:
                                # Create a mask of valid pixels using NumPy broadcasting
                                # This is 100x faster than Python loops
                                valid_mask = np.zeros(filtered_array.shape[:2], dtype=bool)

                                # Iterate over valid colors (only ~20 iterations) instead of pixels (560,000 iterations)
                                for color in valid_colorbar_colors:
                                    # Check where image pixels match this color exactly
                                    # img_array is (H, W, 3), color is (3,)
                                    # np.all checks match across RGB channels
                                    color_match = np.all(img_array == color, axis=2)
                                    valid_mask |= color_match

                                # Set invalid pixels to black (0,0,0)
                                filtered_array[~valid_mask] = [0, 0, 0]

                                # 将colorbar区域也设为黑色
                                if product in colorbar_masks:
                                    # 将colorbar区域设为黑色
                                    filtered_array[~colorbar_masks[product]] = [0, 0, 0]

                                # 创建值数组
                                value_array = np.zeros((filtered_array.shape[0], filtered_array.shape[1]),
                                                       dtype=np.float32)

                                # 应用映射到过滤后图像（只包含有效colorbar颜色）
                                for color, value in color_value_map.items():
                                    mask = np.all(filtered_array == color, axis=2)
                                    value_array[mask] = value

                                # 额外步骤: 应用colorbar掩码，将colorbar区域设为0
                                if product in colorbar_masks:
                                    value_array[~colorbar_masks[product]] = 0

                                # 应用产品特定归一化
                                timestep_data[:, :, c] = self._normalize_product(value_array, product)
                            else:
                                # 如果没有成功提取colorbar，回退到原始方法
                                img = img.convert('L')  # 转为灰度
                                array = np.array(img, dtype=np.float32)
                                # 应用colorbar掩码
                                if product in colorbar_masks:
                                    array[~colorbar_masks[product]] = 0
                                timestep_data[:, :, c] = self._normalize_product(array, product)
                    except Exception as e:
                        dist_print(f"Error loading {img_path}: {str(e)}")
                        timestep_data[:, :, c] = 0.0
                else:
                    dist_print(f"Warning: Product {product} not available for time {closest_time}")
                    timestep_data[:, :, c] = 0.0

            # -------------------------------------------------------------------
            # 4. Save to Cache (Optimization #1 continued)
            # -------------------------------------------------------------------
            if self.cache_dir:
                cache_path = self._get_cache_path(closest_time)
                # Double-check to avoid redundant writes in multi-process scenario
                if not os.path.exists(cache_path):
                    try:
                        # Use temp file + rename for atomic write safety in multi-process
                        temp_path = cache_path + f".tmp.{os.getpid()}.npy"
                        np.save(temp_path, timestep_data)
                        os.replace(temp_path, cache_path)
                    except Exception as e:
                        # Clean up temp file if write failed (e.g., race condition)
                        if 'temp_path' in locals() and os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except:
                                pass
                        # Silently continue - cache is just optimization
                        pass

            return t, timestep_data

        # 优化：移除ThreadPoolExecutor，直接顺序加载
        # 在DataLoader多进程模式下，进程内的多线程只会增加开销和I/O争用
        # 尤其是读取.npy缓存时，顺序读取效率更高
        for t in range(time_steps):
            try:
                _, timestep_data = load_timestep(t)
                series[t] = timestep_data
            except Exception as e:
                dist_print(f"Error loading timestep {t}: {e}")
                # Keep zero initialized data for this timestep
                continue

        return series  # [T, H, W, C]

    def _extract_colorbar(self, img, product):
        """
        获取产品的颜色到数值映射
        img: PIL图像对象 (保留参数以保持接口兼容性)
        product: 产品类型
        返回: (color_to_value_func, valid_colorbar_colors)
        """
        # DBZ产品使用数值文件，不需要颜色映射
        if product == 'dbz':
            return None, None

        # DBZH 产品的固定颜色映射
        if product == 'dbzh':
            color_to_value_map = {
                (2, 255, 255): 0.5,
                (2, 190, 190): 1.5,
                (1, 160, 95): 2.5,
                (0, 130, 0): 3.5,
                (0, 255, 0): 4.5,
                (81, 255, 19): 5.5,
                (162, 255, 38): 6.5,
                (255, 244, 130): 7.5,
                (255, 249, 65): 8.5,
                (255, 255, 0): 9.5,
                (255, 154, 2): 10.5,
                (255, 107, 1): 11.5,
                (255, 60, 0): 12.5,
                (255, 0, 0): 13.5,
                (227, 0, 0): 14.5,
                (200, 0, 0): 15.5,
                (255, 0, 255): 16.5,
                (226, 16, 194): 17.5,
                (197, 33, 133): 18.5
            }
            distance_threshold = 30

        # VIL 产品的固定颜色映射
        elif product == 'vil':
            color_to_value_map = {
                (147, 147, 147): 1,
                (107, 107, 107): 5,
                (249, 162, 162): 10,
                (236, 130, 130): 15,
                (195, 101, 101): 20,
                (0, 251, 134): 25,
                (0, 180, 0): 30,
                (255, 255, 101): 35,
                (203, 203, 85): 40,
                (255, 85, 85): 45,
                (214, 0, 0): 50,
                (166, 0, 0): 55,
                (0, 0, 255): 60,
                (255, 255, 255): 65,
                (228, 0, 255): 70
            }
            distance_threshold = 40  # VIL可能需要稍大的容差

        else:
            # 未知产品类型，返回None使用灰度转换作为后备方案
            dist_print(f"Warning: Unknown product type {product}, will use grayscale conversion")
            return None, None

        # 创建有效colorbar颜色集合
        valid_colorbar_colors = set(color_to_value_map.keys())

        # 创建颜色映射函数
        def color_to_value(rgb_color):
            if isinstance(rgb_color, np.ndarray):
                rgb_tuple = tuple(map(int, rgb_color))
            else:
                rgb_tuple = tuple(map(int, rgb_color))

            # 直接查找颜色映射
            if rgb_tuple in color_to_value_map:
                return color_to_value_map[rgb_tuple]
            else:
                # 找到最接近的颜色（容错处理）
                min_distance = float('inf')
                closest_color = None
                for color in color_to_value_map:
                    # 计算RGB颜色距离
                    distance = np.sqrt(sum((np.array(rgb_tuple) - np.array(color)) ** 2))
                    if distance < min_distance:
                        min_distance = distance
                        closest_color = color

                if closest_color and min_distance < distance_threshold:
                    return color_to_value_map[closest_color]
                else:
                    # 如果找不到合适的颜色，返回0（背景值）
                    return 0

        return color_to_value, valid_colorbar_colors

    # def _create_background_mask(self):
    #     """
    #     创建一个背景掩码，用于从雷达图像中移除背景元素但保留气象数据
    #     返回: 背景掩码（二维布尔数组，True表示背景像素）
    #     """
    #     # 如果已经创建了背景掩码，直接返回
    #     if 'background_mask' in self.__dict__ and self.background_mask is not None:
    #         return self.background_mask
    #
    #     print("Creating background mask from reference image...")
    #
    #     try:
    #         # 加载背景参考图像
    #         with Image.open(self.background_image_path) as bg_img:
    #             # 转换为RGB并调整为模型所需的尺寸
    #             bg_rgb = bg_img.convert('RGB')
    #             bg_array = np.array(bg_rgb)
    #
    #             # 转换为灰度以简化比较
    #             bg_gray = np.mean(bg_array, axis=2).astype(np.uint8)
    #
    #             # 创建掩码（全为True，表示所有像素初始都视为背景）
    #             mask = np.ones((self.img_size[0], self.img_size[1]), dtype=bool)
    #
    #             # 排除右侧的colorbar区域（基于产品类型可能需要调整）
    #             x_start = int(self.img_size[1] * 0.85)
    #             y_start = int(self.img_size[0] * 0.65)
    #             x_end = self.img_size[1]
    #             y_end = self.img_size[0]
    #
    #             # 将colorbar区域在掩码中标记为False（不是背景）
    #             mask[y_start:y_end, x_start:x_end] = False
    #
    #             # 保存背景图像的灰度值和创建的掩码
    #             self.background_gray = bg_gray
    #             self.background_mask = mask
    #             print(f"Background mask created, {np.sum(mask)} background pixels identified")
    #             return mask
    #
    #     except Exception as e:
    #         print(f"Error creating background mask: {str(e)}")
    #         # 返回空掩码（全False，表示没有背景）
    #         return np.zeros((self.img_size[0], self.img_size[1]), dtype=bool)
    #
    # def _apply_background_mask(self, img_array):
    #     """
    #     应用背景掩码到图像数组，移除背景元素（地图、文字等）但保留气象数据和colorbar
    #     利用背景图像与产品图像的精确对齐进行更精确的背景去除
    #     img_array: 图像数组
    #     返回: 应用掩码后的图像数组
    #     """
    #     if not self.use_background_mask:
    #         return img_array
    #
    #     # 获取背景掩码
    #     bg_mask = self._create_background_mask()
    #
    #     # 创建掩码后的数组副本
    #     masked_array = img_array.copy()
    #
    #     # 计算当前图像的灰度值（如果是RGB图像）
    #     if len(img_array.shape) == 3 and img_array.shape[2] == 3:
    #         img_gray = np.mean(img_array, axis=2).astype(np.uint8)
    #     else:
    #         img_gray = img_array.astype(np.uint8)
    #
    #     # 在背景区域中识别并移除背景元素
    #     if hasattr(self, 'background_gray'):
    #         # 计算与背景的差异
    #         diff = np.abs(img_gray - self.background_gray)
    #
    #         # 根据产品类型调整阈值
    #         # 由于所有产品都与背景图像具有相同尺寸和坐标，可以使用统一的阈值
    #         threshold = 25  # 默认阈值
    #
    #         # 创建一个动态掩码，识别与背景相似的像素
    #         similar_to_bg = diff < threshold
    #
    #         # 仅在原始背景掩码区域内，将与背景相似的像素设为0
    #         pixels_to_mask = bg_mask & similar_to_bg
    #
    #         if len(img_array.shape) == 3:
    #             # 对RGB图像，将所有通道设为0
    #             masked_array[pixels_to_mask, :] = 0
    #         else:
    #             # 对灰度图像，直接设为0
    #             masked_array[pixels_to_mask] = 0
    #
    #         # 打印统计信息以便调试
    #         masked_pixels_count = np.sum(pixels_to_mask)
    #         total_pixels = np.prod(img_array.shape[:2])
    #         print(f"背景掩码: 移除了{masked_pixels_count}个像素 ({masked_pixels_count / total_pixels * 100:.1f}%)")
    #
    #     else:
    #         # 如果没有背景灰度图，则使用原始掩码
    #         if len(img_array.shape) == 3:
    #             masked_array[bg_mask, :] = 0
    #         else:
    #             masked_array[bg_mask] = 0
    #         print("警告: 使用简单掩码，而不是基于背景图像比较")
    #
    #     return masked_array

    def validate_data_loading(self, output_dir=None):
        """
        验证雷达图像数据加载是否正确
        output_dir: 输出目录，如果提供则保存示例图像
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        dist_print("Validating radar data loading...")

        # 创建输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 从时间索引中选择一些时间点进行验证
        time_points = list(self.time_index.keys())
        if not time_points:
            dist_print("No valid time points found in the index.")
            return

        # 统计每个产品的时间点数量
        product_counts = {product: 0 for product in self.products}
        for timestamp in time_points:
            for product in self.products:
                if product in self.time_index[timestamp]:
                    product_counts[product] += 1

        dist_print("\n各产品可用时间点统计:")
        for product, count in product_counts.items():
            dist_print(f"- {product}: {count}/{len(time_points)} 个时间点 ({count / len(time_points) * 100:.1f}%)")

        # 过滤只保留所有产品都可用的时间点
        complete_time_points = []
        for timestamp in time_points:
            if all(product in self.time_index[timestamp] for product in self.products):
                complete_time_points.append(timestamp)

        dist_print(
            f"\n找到 {len(complete_time_points)}/{len(time_points)} 个时间点同时包含所有产品 ({len(complete_time_points) / len(time_points) * 100:.1f}%)")

        if complete_time_points:
            dist_print(f"第一个完整时间点: {complete_time_points[0]}")
            dist_print(f"最后一个完整时间点: {complete_time_points[-1]}")
            dist_print(f"时间跨度: {complete_time_points[-1] - complete_time_points[0]}")
        else:
            dist_print("警告: 没有找到同时包含所有产品的时间点!")
            return

        # 使用完整的时间点进行后续处理
        time_points = complete_time_points

        # 选择至少3个时间点（开始、中间、结束）进行检查
        check_indices = [0, len(time_points) // 2, -1]
        check_times = [time_points[i] for i in check_indices if i < len(time_points)]

        for t_idx, timestamp in enumerate(check_times):
            fig, axes = plt.subplots(len(self.products), 3, figsize=(15, 5 * len(self.products)))
            fig.suptitle(f"Radar Data Validation - {timestamp}")

            for p_idx, product in enumerate(self.products):
                if product in self.time_index[timestamp]:
                    img_path = self.time_index[timestamp][product]
                    dist_print(f"Checking {product} at {timestamp}: {img_path}")

                    try:
                        # 1. 加载原始图像
                        with Image.open(img_path) as img:
                            raw_img = img.convert('L')
                            raw_array = np.array(raw_img)

                            # 2. 不需要调整大小，直接使用原始图像
                            original_img = raw_img
                            original_array = np.array(original_img)

                            # 3. 归一化后的图像
                            norm_array = self._normalize_product(np.array(original_img, dtype=np.float32), product)

                            # 打印统计信息
                            dist_print(
                                f"  - Raw stats: min={raw_array.min()}, max={raw_array.max()}, mean={raw_array.mean():.2f}")

                            dist_print(
                                f"  - Normalized stats: min={norm_array.min():.4f}, max={norm_array.max():.4f}, mean={norm_array.mean():.4f}")

                            # 可视化
                            if len(self.products) == 1:
                                ax_raw, ax_original, ax_norm = axes
                            else:
                                ax_raw, ax_original, ax_norm = axes[p_idx]

                            # 原始图像
                            im_raw = ax_raw.imshow(raw_array, cmap='viridis')
                            ax_raw.set_title(f"{product} - Raw")
                            plt.colorbar(im_raw, ax=ax_raw)

                            # 调整大小的图像
                            im_original = ax_original.imshow(original_array, cmap='viridis')
                            ax_original.set_title(f"{product} - Original")
                            plt.colorbar(im_original, ax=ax_original)

                            # 归一化后的图像
                            im_norm = ax_norm.imshow(norm_array, cmap='viridis')
                            ax_norm.set_title(f"{product} - Normalized")
                            plt.colorbar(im_norm, ax=ax_norm)

                    except Exception as e:
                        dist_print(f"Error validating {img_path}: {str(e)}")
                        if len(self.products) == 1:
                            for ax in axes:
                                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                        else:
                            for ax in axes[p_idx]:
                                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                else:
                    dist_print(f"Product {product} not available for timestamp {timestamp}")
                    if len(self.products) == 1:
                        for ax in axes:
                            ax.text(0.5, 0.5, f"Product {product} not available", ha='center', va='center')
                    else:
                        for ax in axes[p_idx]:
                            ax.text(0.5, 0.5, f"Product {product} not available", ha='center', va='center')

            plt.tight_layout()

            if output_dir:
                output_file = os.path.join(output_dir, f"radar_validation_{t_idx}.png")
                plt.savefig(output_file)
                dist_print(f"Saved validation image to {output_file}")
            else:
                plt.show()

            plt.close(fig)

        # 额外检查: 加载一个时间序列并分析
        if time_points:
            start_time = time_points[0]
            dist_print(f"\nLoading time series starting at {start_time}...")
            series = self.load_time_series(start_time)

            # 分析时间序列数据
            dist_print(f"Time series shape: {series.shape}")
            for c, product in enumerate(self.products):
                channel_data = series[:, :, :, c]
                dist_print(f"Product {product} stats:")
                dist_print(
                    f"  - Min: {channel_data.min():.4f}, Max: {channel_data.max():.4f}, Mean: {channel_data.mean():.4f}")
                dist_print(
                    f"  - Non-zero values: {np.count_nonzero(channel_data)}/{channel_data.size} ({np.count_nonzero(channel_data) / channel_data.size * 100:.2f}%)")

                # 检查是否有相同值
                unique_values = np.unique(channel_data)
                dist_print(f"  - Unique values: {len(unique_values)} (first 5: {unique_values[:5]})")

                if output_dir:
                    # 保存时间序列的第一个时间步
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(channel_data[0], cmap='viridis')
                    ax.set_title(f"{product} - First Timestep")
                    plt.colorbar(im, ax=ax)
                    plt.savefig(os.path.join(output_dir, f"timeseries_{product}_first.png"))
                    plt.close(fig)

                    # 保存时间序列的最后一个时间步
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(channel_data[-1], cmap='viridis')
                    ax.set_title(f"{product} - Last Timestep")
                    plt.colorbar(im, ax=ax)
                    plt.savefig(os.path.join(output_dir, f"timeseries_{product}_last.png"))
                    plt.close(fig)

    # def validate_colorbar_extraction(self, output_dir=None):
    #     """
    #     验证colorbar提取和颜色到数值的映射是否正确
    #     output_dir: 输出目录，如果提供则保存示例图像
    #     """
    #     import matplotlib.pyplot as plt
    #     from matplotlib.colors import Normalize
    #
    #     print("Validating colorbar extraction...")
    #
    #     # 创建输出目录
    #     if output_dir:
    #         os.makedirs(output_dir, exist_ok=True)
    #
    #     # 从时间索引中选择一些时间点进行验证
    #     time_points = list(self.time_index.keys())
    #     if not time_points:
    #         print("No valid time points found in the index.")
    #         return
    #
    #     # 选择至少一个时间点进行检查
    #     if time_points:
    #         timestamp = time_points[0]
    #
    #         fig, axes = plt.subplots(len(self.products), 3, figsize=(18, 6 * len(self.products)))
    #         fig.suptitle(f"Colorbar Extraction Validation - {timestamp}")
    #
    #         for p_idx, product in enumerate(self.products):
    #             if product in self.time_index[timestamp]:
    #                 img_path = self.time_index[timestamp][product]
    #                 print(f"Checking colorbar for {product} at {timestamp}: {img_path}")
    #
    #                 try:
    #                     # 1. 加载原始图像
    #                     with Image.open(img_path) as img:
    #                         # 原始RGB图像
    #                         rgb_img = img.convert('RGB')
    #                         rgb_array = np.array(rgb_img)
    #
    #                         # 2. 提取colorbar并创建映射
    #                         color_to_value_func = self._extract_colorbar(img, product)
    #
    #                         if color_to_value_func:
    #                             # 3. 应用映射到调整大小后的图像
    #                             resized_rgb = rgb_img.resize((self.img_size[1], self.img_size[0]))
    #                             resized_rgb_array = np.array(resized_rgb)
    #
    #                             # 创建值数组
    #                             value_array = np.zeros((resized_rgb_array.shape[0], resized_rgb_array.shape[1]),
    #                                                    dtype=np.float32)
    #
    #                             # 对每个像素应用映射（为了演示，只处理少量像素）
    #                             sample_step = 1  # 每1个像素采样一次以加快处理
    #                             for i in range(0, resized_rgb_array.shape[0], sample_step):
    #                                 for j in range(0, resized_rgb_array.shape[1], sample_step):
    #                                     value_array[i, j] = color_to_value_func(resized_rgb_array[i, j])
    #
    #                             # 4. 应用产品特定归一化
    #                             norm_array = self._normalize_product(value_array, product)
    #
    #                             # 打印统计信息
    #                             print(f"  - RGB stats: shape={rgb_array.shape}")
    #                             print(
    #                                 f"  - Value array stats: min={value_array.min():.2f}, max={value_array.max():.2f}, mean={value_array.mean():.2f}")
    #                             print(
    #                                 f"  - Normalized stats: min={norm_array.min():.4f}, max={norm_array.max():.4f}, mean={norm_array.mean():.4f}")
    #
    #                             # 可视化
    #                             if len(self.products) == 1:
    #                                 ax_rgb, ax_value, ax_norm = axes
    #                             else:
    #                                 ax_rgb, ax_value, ax_norm = axes[p_idx]
    #
    #                             # 原始RGB图像
    #                             ax_rgb.imshow(rgb_array)
    #                             ax_rgb.set_title(f"{product} - Original RGB")
    #
    #                             # 颜色到数值映射结果
    #                             im_value = ax_value.imshow(value_array, cmap='viridis')
    #                             ax_value.set_title(f"{product} - Color to Value Mapping")
    #                             plt.colorbar(im_value, ax=ax_value)
    #
    #                             # 归一化后的图像
    #                             im_norm = ax_norm.imshow(norm_array, cmap='viridis')
    #                             ax_norm.set_title(f"{product} - Normalized")
    #                             plt.colorbar(im_norm, ax=ax_norm)
    #                         else:
    #                             print(f"Could not extract colorbar for {product}")
    #                             if len(self.products) == 1:
    #                                 for ax in axes:
    #                                     ax.text(0.5, 0.5, "Colorbar extraction failed", ha='center', va='center')
    #                             else:
    #                                 for ax in axes[p_idx]:
    #                                     ax.text(0.5, 0.5, "Colorbar extraction failed", ha='center', va='center')
    #
    #                 except Exception as e:
    #                     print(f"Error validating colorbar extraction for {img_path}: {str(e)}")
    #                     if len(self.products) == 1:
    #                         for ax in axes:
    #                             ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
    #                     else:
    #                         for ax in axes[p_idx]:
    #                             ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
    #             else:
    #                 print(f"Product {product} not available for timestamp {timestamp}")
    #                 if len(self.products) == 1:
    #                     for ax in axes:
    #                         ax.text(0.5, 0.5, f"Product {product} not available", ha='center', va='center')
    #                 else:
    #                     for ax in axes[p_idx]:
    #                         ax.text(0.5, 0.5, f"Product {product} not available", ha='center', va='center')
    #
    #         plt.tight_layout()
    #
    #         if output_dir:
    #             output_file = os.path.join(output_dir, f"colorbar_validation.png")
    #             plt.savefig(output_file)
    #             print(f"Saved colorbar validation image to {output_file}")
    #         else:
    #             plt.show()
    #
    #         plt.close(fig)


    def validate_colorbar_extraction(self, output_dir=None):
        """
        验证colorbar提取和颜色到数值的映射是否正确
        output_dir: 输出目录，如果提供则保存示例图像
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.patches import Rectangle

        dist_print("Validating colorbar extraction...")

        # 创建输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 从时间索引中选择一个时间点进行验证
        time_points = list(self.time_index.keys())
        if not time_points:
            dist_print("No valid time points found in the index.")
            return

        timestamp = time_points[0]

        # 每个产品创建一个图
        for product in self.products:
            if product not in self.time_index[timestamp]:
                dist_print(f"Product {product} not available for timestamp {timestamp}")
                continue

            img_path = self.time_index[timestamp][product]
            dist_print(f"Validating colorbar extraction for {product}: {img_path}")

            try:
                # 获取产品的预期颜色数量和值范围
                expected_colors = 0
                if product == 'dbz':
                    # expected_colors = 15
                    # expected_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
                    # direction = "vertical"
                    dist_print("Skipping colorbar validation for DBZ as it now uses numerical data")
                    continue
                elif product == 'dbzh':
                    expected_colors = 19
                    expected_values = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5,
                                       15.5, 16.5, 17.5, 18.5]
                    direction = "horizontal"
                elif product == 'vil':
                    expected_colors = 15
                    expected_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
                    direction = "vertical"

                with Image.open(img_path) as img:
                    # 保存原始图像
                    original_rgb = img.convert('RGB')
                    original_array = np.array(original_rgb)

                    # 提取colorbar和颜色映射
                    color_to_value_func, valid_colorbar_colors = self._extract_colorbar(img, product)

                    if not color_to_value_func or not valid_colorbar_colors:
                        dist_print(f"Failed to extract colorbar for {product}")
                        continue

                    # 创建包含所有提取颜色的图表
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    fig.suptitle(f"Colorbar Extraction Validation - {product} - {timestamp}")

                    # 原始图像
                    axes[0, 0].imshow(original_array)
                    axes[0, 0].set_title("Original Image")

                    # 标记colorbar区域
                    if product == 'dbz':
                        # 右侧colorbar区域
                        x_start = int(original_array.shape[1] * 0.85)
                        y_start = int(original_array.shape[0] * 0.65)
                        width = original_array.shape[1] - x_start
                        height = original_array.shape[0] - y_start
                        rect = Rectangle((x_start, y_start), width, height,
                                         linewidth=2, edgecolor='r', facecolor='none')
                        axes[0, 0].add_patch(rect)
                        axes[0, 0].text(x_start + width / 2, y_start - 10, "Expected: Top-to-Bottom",
                                        color='red', ha='center', fontsize=10)
                    elif product == 'dbzh':
                        # 底部colorbar区域
                        y_start = int(original_array.shape[0] * 0.88)
                        width = original_array.shape[1]
                        height = original_array.shape[0] - y_start
                        rect = Rectangle((0, y_start), width, height,
                                         linewidth=2, edgecolor='r', facecolor='none')
                        axes[0, 0].add_patch(rect)
                        axes[0, 0].text(width / 2, y_start - 10, "Expected: Left-to-Right",
                                        color='red', ha='center', fontsize=10)
                    elif product == 'vil':
                        # 右侧colorbar区域
                        x_start = int(original_array.shape[1] * 0.85)
                        y_start = int(original_array.shape[0] * 0.52)
                        width = original_array.shape[1] - x_start
                        height = original_array.shape[0] - y_start
                        rect = Rectangle((x_start, y_start), width, height,
                                         linewidth=2, edgecolor='r', facecolor='none')
                        axes[0, 0].add_patch(rect)
                        axes[0, 0].text(x_start + width / 2, y_start - 10, "Expected: Bottom-to-Top",
                                        color='red', ha='center', fontsize=10)

                    # 显示提取的colorbar颜色及其对应值
                    color_list = list(valid_colorbar_colors)
                    value_list = [color_to_value_func(color) for color in color_list]

                    # 打印提取的颜色和值的数量
                    dist_print(f"  - Extracted {len(color_list)}/{expected_colors} colors from colorbar")
                    dist_print(f"  - Value range: {min(value_list):.2f} to {max(value_list):.2f}")

                    # 添加预期和实际颜色数量比较
                    if len(color_list) != expected_colors:
                        dist_print(f"  - WARNING: Expected {expected_colors} colors, but extracted {len(color_list)}")

                    # 检查值是否按预期排序 (应该是递增的)
                    if not all(value_list[i] <= value_list[i + 1] for i in range(len(value_list) - 1)):
                        dist_print("  - WARNING: Extracted values are not in ascending order!")
                        dist_print(f"  - Extracted values: {value_list}")

                    # 颜色样本显示
                    color_samples = np.zeros((50, len(color_list), 3), dtype=np.uint8)
                    for i, color in enumerate(color_list):
                        color_samples[:, i] = color

                    axes[0, 1].imshow(color_samples)
                    axes[0, 1].set_title(f"Extracted Colors ({len(color_list)}/{expected_colors} colors)")
                    axes[0, 1].set_xticks(range(len(color_list)))
                    axes[0, 1].set_xticklabels([f"{value:.1f}" for value in value_list], rotation=90)

                    # 创建彩色映射图像
                    # # 使用原始图像尺寸，不进行resize
                    original_rgb = original_rgb
                    original_array = np.array(original_rgb)

                    # 创建一个掩码标识哪些像素颜色在有效colorbar颜色集合中
                    color_match_mask = np.zeros(original_array.shape[:2], dtype=bool)

                    # 检查每个像素是否匹配任何colorbar颜色
                    for color in valid_colorbar_colors:
                        mask = np.all(original_array == color, axis=2)
                        color_match_mask |= mask

                    # 创建一个展示哪些像素匹配colorbar颜色的图像
                    match_image = np.zeros_like(original_array)
                    match_image[color_match_mask] = original_array[color_match_mask]

                    axes[1, 0].imshow(original_array)
                    axes[1, 0].set_title("Original Image (700x800)")

                    axes[1, 1].imshow(match_image)
                    axes[1, 1].set_title("Pixels Matching Colorbar Colors")

                    # 计算统计信息
                    total_pixels = np.prod(original_array.shape[:2])
                    match_pixels = np.sum(color_match_mask)
                    match_percentage = match_pixels / total_pixels * 100

                    dist_print(f"  - Exact color matches: {match_pixels}/{total_pixels} pixels ({match_percentage:.2f}%)")

                    # 添加统计信息文本框
                    stats_text = (f"Total colors extracted: {len(color_list)}/{expected_colors}\n"
                                  f"Value range: {min(value_list):.2f} to {max(value_list):.2f}\n"
                                  f"Expected direction: {direction}\n"
                                  f"Exact color matches: {match_percentage:.2f}% of pixels")

                    fig.text(0.5, 0.02, stats_text, ha='center', bbox=dict(facecolor='white', alpha=0.8))

                    plt.tight_layout()

                    if output_dir:
                        output_file = os.path.join(output_dir, f"colorbar_validation_{product}.png")
                        plt.savefig(output_file)
                        dist_print(f"Saved colorbar validation to {output_file}")
                    else:
                        plt.show()

                    plt.close(fig)

                    # 第二个图：测试不同的颜色距离阈值
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    fig.suptitle(f"Color Distance Threshold Testing - {product}")

                    # 将所有有效colorbar颜色转换为数组，以便快速计算
                    valid_colors_array = np.array(list(valid_colorbar_colors))

                    # 测试不同的阈值
                    thresholds = [0,1,2,3,4,5,6,10,20]

                    for i, threshold in enumerate(thresholds):
                        row, col = divmod(i, 3)

                        # 创建一个过滤后的图像副本
                        filtered_array = original_array.copy()

                        # 对每个像素应用颜色距离过滤
                        match_count = 0
                        for y in range(filtered_array.shape[0]):
                            for x in range(filtered_array.shape[1]):
                                pixel_color = filtered_array[y, x]

                                # 跳过黑色像素
                                if np.array_equal(pixel_color, [0, 0, 0]):
                                    continue

                                # 如果已经是精确匹配，跳过
                                if tuple(pixel_color) in valid_colorbar_colors:
                                    match_count += 1
                                    continue

                                # 计算与所有有效颜色的距离
                                distances = np.sqrt(np.sum((valid_colors_array - pixel_color) ** 2, axis=1))
                                min_distance = np.min(distances)

                                # 如果距离大于阈值，设为黑色
                                if min_distance > threshold:
                                    filtered_array[y, x] = [0, 0, 0]
                                else:
                                    match_count += 1

                        axes[row, col].imshow(filtered_array)
                        match_percentage = match_count / total_pixels * 100
                        axes[row, col].set_title(f"Threshold = {threshold} ({match_percentage:.2f}% matched)")

                    # 计算实际在图像中使用的颜色数量
                    unique_colors = np.unique(original_array.reshape(-1, 3), axis=0)
                    unique_count = len(unique_colors)

                    # 添加信息文本
                    threshold_text = (f"Colorbar colors: {len(valid_colorbar_colors)}\n"
                                      f"Unique colors in image: {unique_count}\n"
                                      f"Current threshold in code: 40")

                    fig.text(0.5, 0.02, threshold_text, ha='center', bbox=dict(facecolor='white', alpha=0.8))

                    plt.tight_layout()

                    if output_dir:
                        output_file = os.path.join(output_dir, f"threshold_testing_{product}.png")
                        plt.savefig(output_file)
                        dist_print(f"Saved threshold testing to {output_file}")
                    else:
                        plt.show()

                    plt.close(fig)

            except Exception as e:
                dist_print(f"Error validating colorbar extraction for {product}: {str(e)}")
                import traceback
                traceback.print_exc()

    def validate_background_masking(self, output_dir=None):
        """
        验证数据加载和颜色映射功能是否正确处理了雷达数据
        output_dir: 输出目录，如果提供则保存示例图像
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        dist_print("Validating data processing functionality...")

        # 创建输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 从时间索引中选择一个时间点进行验证
        time_points = list(self.time_index.keys())
        if not time_points:
            dist_print("No valid time points found in the index.")
            return

        timestamp = time_points[0]  # 选择第n个时间点

        # 定义各产品的colorbar区域掩码 (与load_time_series保持一致)
        colorbar_masks = {}
        for product in self.products:
            mask = np.ones((self.img_size[0], self.img_size[1]), dtype=bool)

            # 根据产品类型定义colorbar区域
            if product == 'dbz':
                # DBZ uses numerical data (.ref format), no colorbar masking needed
                pass
            elif product == 'dbzh':
                # 底部colorbar区域
                y_start = int(self.img_size[0] * 0.88)
                mask[y_start:, :] = False
                # 右侧区域也需要排除
                x_start = int(self.img_size[1] * 0.85)
                y_start = int(self.img_size[0] * 0.65)
                mask[y_start:, x_start:] = False
            elif product == 'vil':
                # 右侧colorbar区域
                x_start = int(self.img_size[1] * 0.85)
                y_start = int(self.img_size[0] * 0.52)
                mask[y_start:, x_start:] = False

            colorbar_masks[product] = mask

        # 定义图表布局
        fig, axes = plt.subplots(len(self.products), 4, figsize=(24, 6 * len(self.products)))
        fig.suptitle(f"Data Processing Validation - {timestamp}")

        for p_idx, product in enumerate(self.products):
            if product == 'dbz':
                # 处理DBZ数值数据
                dbz_data = self._read_dbz_numerical_file(timestamp)
                if dbz_data is not None:
                    # 获取子图数组
                    if len(self.products) == 1:
                        ax_orig_data, ax_normalized, ax_histogram, ax_info = axes
                    else:
                        ax_orig_data, ax_normalized, ax_histogram, ax_info = axes[p_idx]

                    # 1. 显示原始数值数据
                    im = ax_orig_data.imshow(dbz_data, cmap='jet')
                    ax_orig_data.set_title(f"{product} - Original Numerical Data")
                    plt.colorbar(im, ax=ax_orig_data)

                    # 2. 显示归一化后的数据
                    normalized_data = self._normalize_product(dbz_data, product)
                    im = ax_normalized.imshow(normalized_data, cmap='jet', vmin=0, vmax=1)
                    ax_normalized.set_title(f"{product} - Normalized Data")
                    plt.colorbar(im, ax=ax_normalized)

                    # 3. 显示数值分布直方图
                    non_zero_values = dbz_data[dbz_data > 0]
                    if len(non_zero_values) > 0:
                        ax_histogram.hist(non_zero_values.flatten(), bins=50)
                        ax_histogram.set_title(f"{product} - Value Distribution")
                        ax_histogram.set_xlabel("DBZ Value")
                        ax_histogram.set_ylabel("Frequency")
                    else:
                        ax_histogram.text(0.5, 0.5, "No data values > 0",
                                          ha='center', va='center', transform=ax_histogram.transAxes)

                    # 4. 显示统计信息
                    ax_info.axis('off')
                    stats_text = (
                        f"DBZ Statistics:\n"
                        f"Min: {dbz_data.min()}\n"
                        f"Max: {dbz_data.max()}\n"
                        f"Mean: {dbz_data.mean():.2f}\n"
                        f"Non-zero pixels: {np.count_nonzero(dbz_data)}/{dbz_data.size} "
                        f"({np.count_nonzero(dbz_data) / dbz_data.size * 100:.2f}%)\n"
                        f"Shape: {dbz_data.shape}\n"
                        f"Source: Numerical .ref file"
                    )
                    ax_info.text(0.1, 0.5, stats_text, va='center', fontsize=12)
                else:
                    # 如果无法加载DBZ数据，显示错误信息
                    if len(self.products) == 1:
                        for ax in axes:
                            ax.text(0.5, 0.5, "Could not load DBZ numerical data",
                                    ha='center', va='center')
                    else:
                        for ax in axes[p_idx]:
                            ax.text(0.5, 0.5, "Could not load DBZ numerical data",
                                    ha='center', va='center')
            elif product in self.time_index[timestamp]:
                # 处理基于图像的产品 (DBZH, VIL)
                img_path = self.time_index[timestamp][product]
                try:
                    # 加载原始图像
                    with Image.open(img_path) as img:
                        # 保存原始尺寸图像
                        original_rgb = img.convert('RGB')
                        original_array = np.array(original_rgb)

                        # 提取colorbar并创建映射
                        color_to_value_func, valid_colorbar_colors = self._extract_colorbar(img, product)

                        # 获取子图数组
                        if len(self.products) == 1:
                            ax_orig_img, ax_color_extract, ax_value_map, ax_info = axes
                        else:
                            ax_orig_img, ax_color_extract, ax_value_map, ax_info = axes[p_idx]

                        # 1. 显示原始图像
                        ax_orig_img.imshow(original_array)
                        ax_orig_img.set_title(f"{product} - Original Image")

                        # 显示扫描线位置
                        line_viz = original_array.copy()
                        if product == 'dbzh':
                            # 在水平线位置670画一条红线
                            y_pos = 670
                            if y_pos < line_viz.shape[0]:
                                line_viz[y_pos, :] = [255, 0, 0]
                            ax_orig_img.axhline(y=y_pos, color='r', linestyle='-', linewidth=1)
                        elif product == 'vil':
                            # 在垂直线位置735画一条红线
                            x_pos = 735
                            if x_pos < line_viz.shape[1]:
                                line_viz[:, x_pos] = [255, 0, 0]
                            ax_orig_img.axvline(x=x_pos, color='r', linestyle='-', linewidth=1)

                        # 标记colorbar区域
                        if product in colorbar_masks:
                            if product == 'dbzh':
                                # 底部colorbar区域
                                y_start = int(self.img_size[0] * 0.88)
                                width = self.img_size[1]
                                height = self.img_size[0] - y_start
                                rect = Rectangle((0, y_start), width, height,
                                                 linewidth=2, edgecolor='r', facecolor='none')
                                ax_orig_img.add_patch(rect)

                                # 右侧区域
                                x_start = int(self.img_size[1] * 0.85)
                                y_start = int(self.img_size[0] * 0.65)
                                width = self.img_size[1] - x_start
                                height = self.img_size[0] - y_start
                                rect = Rectangle((x_start, y_start), width, height,
                                                 linewidth=2, edgecolor='r', facecolor='none')
                                ax_orig_img.add_patch(rect)
                            elif product == 'vil':
                                # 右侧colorbar区域
                                x_start = int(self.img_size[1] * 0.85)
                                y_start = int(self.img_size[0] * 0.52)
                                width = self.img_size[1] - x_start
                                height = self.img_size[0] - y_start
                                rect = Rectangle((x_start, y_start), width, height,
                                                 linewidth=2, edgecolor='r', facecolor='none')
                                ax_orig_img.add_patch(rect)

                        if color_to_value_func:
                            # 2. 显示颜色提取结果
                            # 创建一个只包含colorbar中有效颜色的过滤后RGB图像
                            filtered_array = original_array.copy()

                            # 创建颜色到值的映射
                            color_value_map = {color: color_to_value_func(color) for color in valid_colorbar_colors}

                            # 对每个像素进行处理 - 仅保留colorbar中的颜色
                            for i in range(filtered_array.shape[0]):
                                for j in range(filtered_array.shape[1]):
                                    pixel_color = filtered_array[i, j]
                                    # 如果像素颜色不在有效colorbar颜色集合中，设置为黑色(0,0,0)
                                    if tuple(pixel_color) not in valid_colorbar_colors:
                                        filtered_array[i, j] = [0, 0, 0]

                            # 将colorbar区域也设为黑色
                            if product in colorbar_masks:
                                filtered_array[~colorbar_masks[product]] = [0, 0, 0]

                            ax_color_extract.imshow(filtered_array)
                            ax_color_extract.set_title(f"{product} - Extracted Valid Colors")

                            # 3. 显示颜色到值的映射结果
                            # 创建值数组
                            value_array = np.zeros((filtered_array.shape[0], filtered_array.shape[1]),
                                                   dtype=np.float32)

                            # 应用映射到过滤后图像（只包含有效colorbar颜色）
                            for color, value in color_value_map.items():
                                mask = np.all(filtered_array == color, axis=2)
                                value_array[mask] = value

                            # 应用colorbar掩码，将colorbar区域设为0
                            if product in colorbar_masks:
                                value_array[~colorbar_masks[product]] = 0

                            im = ax_value_map.imshow(value_array, cmap='jet')
                            ax_value_map.set_title(f"{product} - Mapped Values")
                            plt.colorbar(im, ax=ax_value_map)

                            # 4. 显示提取的colorbar颜色和统计信息
                            ax_info.axis('off')

                            # 提取唯一颜色和对应的值
                            unique_colors = list(color_value_map.keys())
                            unique_values = [color_value_map[c] for c in unique_colors]

                            # 按值排序
                            sorted_indices = np.argsort(unique_values)
                            sorted_colors = [unique_colors[i] for i in sorted_indices]
                            sorted_values = [unique_values[i] for i in sorted_indices]

                            # 计算非零值的统计信息
                            non_zero_values = value_array[value_array > 0]
                            stats_text = (
                                f"{product} Statistics:\n"
                                f"Unique colors: {len(unique_colors)}\n"
                                f"Value range: {min(unique_values):.2f} to {max(unique_values):.2f}\n"
                                f"Non-zero pixels: {len(non_zero_values)}/{value_array.size} "
                                f"({len(non_zero_values) / value_array.size * 100:.2f}%)\n"
                            )

                            ax_info.text(0.1, 0.9, stats_text, va='top', fontsize=12)

                            # 在右侧显示一部分颜色块示例
                            num_display_colors = min(25, len(sorted_colors))
                            display_step = max(1, len(sorted_colors) // num_display_colors)

                            for i, idx in enumerate(range(0, len(sorted_colors), display_step)[:num_display_colors]):
                                color = sorted_colors[idx]
                                value = sorted_values[idx]

                                # 创建颜色块
                                y_pos = 0.7 - (i * 0.05)
                                rect = plt.Rectangle((0.1, y_pos), 0.05, 0.04, color=[c / 255 for c in color])
                                ax_info.add_patch(rect)

                                # 添加对应值
                                ax_info.text(0.2, y_pos + 0.02, f"{value:.2f}", va='center', fontsize=10)

                            ax_info.text(0.1, 0.75, "Color samples:", va='center', fontsize=12)
                        else:
                            # 如果没有成功提取colorbar
                            for ax in [ax_color_extract, ax_value_map, ax_info]:
                                ax.text(0.5, 0.5, "Could not extract colorbar",
                                        ha='center', va='center')
                except Exception as e:
                    # 处理错误
                    dist_print(f"Error processing {product}: {str(e)}")
                    if len(self.products) == 1:
                        for ax in axes:
                            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                    else:
                        for ax in axes[p_idx]:
                            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            else:
                # 产品不可用
                dist_print(f"Product {product} not available for timestamp {timestamp}")
                if len(self.products) == 1:
                    for ax in axes:
                        ax.text(0.5, 0.5, f"Product {product} not available",
                                ha='center', va='center')
                else:
                    for ax in axes[p_idx]:
                        ax.text(0.5, 0.5, f"Product {product} not available",
                                ha='center', va='center')

        plt.tight_layout()

        if output_dir:
            output_file = os.path.join(output_dir, "data_processing_validation.png")
            plt.savefig(output_file)
            dist_print(f"Saved data processing validation to {output_file}")
        else:
            plt.show()

        plt.close(fig)