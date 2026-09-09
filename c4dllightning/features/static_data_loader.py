# static_data_loader.py
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import logging


# Setup logging
import os
local_rank = int(os.environ.get('LOCAL_RANK', -1))

# Only configure logging for rank 0 or non-distributed runs
logger = logging.getLogger('static_data')
if local_rank <= 0:
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)
else:
    # Mute logging for other ranks
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL + 1)


class StaticDataLoader:
    """Base class for loading and processing static geographical data"""

    def __init__(self, data_root, target_shape=(700, 800),
                 lon_range=(109.505, 117.495), lat_range=(19.0519, 26.0419),
                 cache_file=None):
        self.data_root = data_root
        self.target_shape = target_shape
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.cache_file = cache_file
        self.data = None

    def load(self):
        """Load data - to be implemented by subclasses"""
        raise NotImplementedError

    def preprocess(self):
        """Preprocess data - to be implemented by subclasses"""
        raise NotImplementedError

    def get_data(self):
        """Return processed data"""
        if self.data is None:
            # Try to load from cache first
            if self.cache_file and os.path.exists(self.cache_file):
                try:
                    logger.info(f"Loading data from cache: {self.cache_file}")
                    self.data = np.load(self.cache_file)
                    logger.info(f"Data loaded from cache, shape: {self.data.shape}")
                    return self.data
                except Exception as e:
                    logger.warning(f"Error loading from cache: {e}. Will load from raw files.")

            # If cache loading failed, load from raw files
            self.load()
            self.preprocess()

            # Save to cache if specified
            # Only allow rank 0 or non-distributed processes to save
            local_rank = int(os.environ.get('LOCAL_RANK', -1))
            if self.cache_file and (local_rank == -1 or local_rank == 0):
                try:
                    logger.info(f"Saving data to cache: {self.cache_file}")
                    os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

                    # Atomic write
                    temp_path = self.cache_file + f".tmp.{os.getpid()}.npy"
                    np.save(temp_path, self.data)

                    # Atomic rename
                    if os.path.exists(self.cache_file):
                        try:
                            os.remove(self.cache_file)
                        except OSError:
                            pass

                    try:
                        os.rename(temp_path, self.cache_file)
                    except OSError:
                        # Fallback cleanup
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                except Exception as e:
                    logger.error(f"Error saving to cache: {e}")
                    # Cleanup
                    if 'temp_path' in locals() and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass

        return self.data

    def visualize(self, output_dir=None, filename=None):
        """Visualize the data"""
        if self.data is None:
            self.get_data()

        plt.figure(figsize=(10, 8))
        plt.imshow(self.data, cmap='viridis')
        plt.colorbar(label=self.__class__.__name__)
        plt.title(f"Static Data: {self.__class__.__name__}")

        if output_dir and filename:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path)
            logger.info(f"Visualization saved to {output_path}")

        plt.close()


class LandCoverLoader(StaticDataLoader):
    """Loader for Land Cover Type data"""

    def __init__(self, data_root, target_shape=(700, 800),
                 lon_range=(109.505, 117.495), lat_range=(19.0519, 26.0419),
                 cache_file=None, landcover_map=None):
        super().__init__(data_root, target_shape, lon_range, lat_range, cache_file)
        # Optional mapping to reclassify land cover classes
        self.landcover_map = landcover_map or {}

    def load(self):
        """Load Land Cover Type data from files (supports .tar.gz and .hdf)"""
        logger.info(f"Loading Land Cover Type data from {self.data_root}")

        try:
            import tarfile
            import rasterio
            from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
            from rasterio.transform import from_bounds
            from rasterio.crs import CRS

            # Find all land cover files (check for HDF first, then tar.gz, then TIF)
            lc_files = []
            for root, _, files in os.walk(self.data_root):
                for f in files:
                    # Check for HDF, tar.gz, or directly accessible TIFFs
                    if f.lower().endswith(('.hdf', '.tar.gz', '.tif', '.tiff')):
                        lc_files.append(os.path.join(root, f))

            if not lc_files:
                raise FileNotFoundError(f"No Land Cover files (.hdf, .tar.gz, or .tif) found in {self.data_root}")

            logger.info(f"Found {len(lc_files)} land cover files")

            # Initialize an empty array for the composite land cover map
            self.data = np.zeros(self.target_shape, dtype=np.uint8)

            # Process each file
            for lc_file in lc_files:
                logger.info(f"Processing Land Cover file: {os.path.basename(lc_file)}")
                
                src_context = None
                temp_files_to_clean = []

                try:
                    # Handle MODIS HDF files
                    if lc_file.lower().endswith('.hdf'):
                        try:
                            # Open the HDF container
                            src_container = rasterio.open(lc_file)
                            subdatasets = src_container.subdatasets
                            
                            # Find LC_Type1 (IGBP) subdataset
                            target_sds = None
                            for sds in subdatasets:
                                # Rasterio returns subdatasets as strings (paths)
                                if 'LC_Type1' in sds:
                                    target_sds = sds
                                    break
                            
                            if not target_sds and subdatasets:
                                # Fallback: try first subdataset if explicit name not found
                                logger.warning(f"Could not find 'LC_Type1' in {lc_file}, using first subdataset.")
                                target_sds = subdatasets[0]
                            
                            if target_sds:
                                logger.info(f"Opening subdataset: {target_sds}")
                                src_context = rasterio.open(target_sds)
                            else:
                                logger.warning(f"No subdatasets found in {lc_file}")
                                continue
                        except Exception as e:
                            if "not recognized as being in a supported file format" in str(e):
                                logger.error(f"GDAL HDF4 driver missing! Cannot open {os.path.basename(lc_file)}.")
                                logger.error("Solution: Install gdal with hdf4 support OR convert files to GeoTIFF.")
                            raise e

                    # Handle loose GeoTIFF files (e.g. converted from HDF)
                    elif lc_file.lower().endswith(('.tif', '.tiff')):
                        # Skip if it looks like a raw satellite band (unless it has land cover keywords)
                        fname = os.path.basename(lc_file).upper()
                        is_band = any(x in fname for x in ['_B1', '_B2', '_B3', '_B4', '_BQA'])
                        is_lc = any(ind in fname for ind in ['_LC', 'LANDCOVER', 'CLASS', 'CDL', 'IGBP', 'MOD12Q1', 'TYPE'])
                        
                        if is_band and not is_lc:
                            continue
                            
                        logger.info(f"Opening GeoTIFF directly: {lc_file}")
                        src_context = rasterio.open(lc_file)

                    # Handle legacy tar.gz files
                    elif lc_file.lower().endswith('.tar.gz'):
                        # Extract the tar.gz file to a temporary directory
                        with tarfile.open(lc_file, 'r:gz') as tar:
                            # Find valid TIF files
                            tif_files = [f for f in tar.getnames() 
                                       if (f.upper().endswith('.TIF') or f.upper().endswith('.TIFF'))
                                       and not any(x in f.upper() for x in ['_B1', '_B2', '_B3', '_B4', '_BQA'])]
                            
                            # Prioritize _LC files
                            lc_tifs = [f for f in tif_files if '_LC' in f.upper()]
                            if lc_tifs:
                                tif_files = lc_tifs

                            if not tif_files:
                                logger.warning(f"No valid Land Cover TIF found in {lc_file}, skipping.")
                                continue

                            # Pick the best candidate
                            tif_file = tif_files[0]
                            # Extract beside the archive, unless that sits on a
                            # read-only or network mount, in which case use scratch.
                            tmp_dir = os.path.dirname(lc_file)
                            if not os.access(tmp_dir, os.W_OK) or os.path.ismount(tmp_dir):
                                tmp_dir = os.path.join(tempfile.gettempdir(), "land_cover_extract")
                            os.makedirs(tmp_dir, exist_ok=True)
                            
                            tar.extract(tif_file, tmp_dir)
                            tif_path = os.path.join(tmp_dir, tif_file)
                            temp_files_to_clean.append(tif_path)
                            
                            src_context = rasterio.open(tif_path)

                    # Common Processing Logic
                    if src_context:
                        with src_context as src:
                            # Get metadata for debugging
                            logger.info(f"Source CRS: {src.crs}")
                            logger.info(f"Source bounds: {src.bounds}")
                            logger.info(f"Source shape: {src.shape}")

                            # Calculate transform (handle coordinate systems)
                            src_crs = src.crs
                            target_crs = CRS.from_epsg(4326)
                            
                            try:
                                left, bottom, right, top = transform_bounds(
                                    src_crs, target_crs, 
                                    src.bounds.left, src.bounds.bottom, 
                                    src.bounds.right, src.bounds.top
                                )
                            except Exception as e:
                                # If transform fails, assume already compatible or fallback
                                logger.warning(f"Transform bounds warning: {e}")
                                left, bottom, right, top = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top

                            # Check intersection in Lat/Lon
                            # Relaxed check: if file is anywhere nearby
                            if (right < self.lon_range[0] or
                                    left > self.lon_range[1] or
                                    top < self.lat_range[0] or
                                    bottom > self.lat_range[1]):
                                logger.warning(f"File {os.path.basename(lc_file)} bounds ({left:.2f}, {bottom:.2f}, {right:.2f}, {top:.2f}) do not intersect with target region, skipping.")
                                continue

                            # Define the target CRS and transform
                            dst_crs = 'EPSG:4326'  # WGS84

                            dst_width, dst_height = self.target_shape[1], self.target_shape[0]
                            dst_transform = from_bounds(
                                self.lon_range[0], self.lat_range[0],
                                self.lon_range[1], self.lat_range[1],
                                dst_width, dst_height
                            )

                            # Initialize destination array
                            dst_array = np.zeros((dst_height, dst_width), dtype=np.uint8)

                            # Reproject
                            reproject(
                                source=rasterio.band(src, 1),
                                destination=dst_array,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=dst_transform,
                                dst_crs=dst_crs,
                                resampling=Resampling.nearest # Use nearest for classes
                            )

                            # Update composite map (Smart Merge)
                            # Logic: 
                            # 1. If dst_array has valid land class (>=1 and <255), overwrite whatever is there.
                            # 2. If dst_array is water (0) or background (255), only write if current map is empty/background.
                            # We treat 0 as Water (valid class) but also potentially background after conversion.
                            # But since we initialize self.data with 0s, we should prioritize Non-Zero values.
                            
                            # Valid land classes in new tile (assuming 0 is water/background, 255 is fill)
                            valid_land_mask = (dst_array > 0) & (dst_array < 255)
                            
                            if np.any(valid_land_mask):
                                self.data[valid_land_mask] = dst_array[valid_land_mask]
                                
                            # If we have explicit Water (0) in the new tile, we should strictly speaking trust it,
                            # BUT to avoid "background 0" overwriting "valid land" from a previous tile due to reprojection overlap artifacts,
                            # we only write 0s where we don't already have land.
                            # (This assumes land takes precedence over water in edge cases)
                            water_mask = (dst_array == 0)
                            # Only overwrite if current pixel is not already marked as land
                            safe_to_write_water = water_mask & (self.data == 0)
                            # self.data[safe_to_write_water] = 0 # No-op since it's already 0, but logical for clarity
                                
                except Exception as e:
                    logger.error(f"Error processing {lc_file}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Cleanup temp files
                    for f in temp_files_to_clean:
                        try:
                            if os.path.exists(f):
                                os.remove(f)
                        except:
                            pass

            # If no data was loaded, raise an error
            if np.all(self.data == 0):
                logger.warning("No valid land cover data was loaded. Using fallback data.")
                self.data = self._generate_fallback_data()

        except ImportError as e:
            logger.error(f"Required package not installed: {e}")
            logger.info("Please install with: pip install rasterio")
            self.data = self._generate_fallback_data()
        except Exception as e:
            logger.error(f"Error loading Land Cover Type data: {e}", exc_info=True)
            self.data = self._generate_fallback_data()

    def _generate_fallback_data(self):
        """Generate fallback data if loading fails"""
        logger.warning("Generating synthetic land cover data as fallback")
        # Create a synthetic pattern with 10 land cover classes (1-10)
        fallback = np.zeros(self.target_shape, dtype=np.uint8)

        # Create some patterns
        y, x = np.indices(self.target_shape)

        # Mountains in the north
        fallback[(y < self.target_shape[0] * 0.3) & (x > self.target_shape[1] * 0.3) &
                 (x < self.target_shape[1] * 0.7)] = 1

        # Forest in the middle
        fallback[(y >= self.target_shape[0] * 0.3) & (y < self.target_shape[0] * 0.6) &
                 (x > self.target_shape[1] * 0.2) & (x < self.target_shape[1] * 0.8)] = 2

        # Urban areas (scattered)
        urban_centers = [
            (int(self.target_shape[0] * 0.7), int(self.target_shape[1] * 0.3)),
            (int(self.target_shape[0] * 0.8), int(self.target_shape[1] * 0.7))
        ]

        for cy, cx in urban_centers:
            dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            fallback[(dist < self.target_shape[0] * 0.1)] = 3

        # Water bodies
        water_y = int(self.target_shape[0] * 0.4)
        water_width = int(self.target_shape[1] * 0.05)
        fallback[(y > water_y - water_width) & (y < water_y + water_width)] = 4

        return fallback

    def preprocess(self):
        """Preprocess land cover data"""
        if self.data is None:
            self.load()

        # Apply class mapping if provided
        if self.landcover_map:
            mapped_data = np.zeros_like(self.data)
            for original, new in self.landcover_map.items():
                mapped_data[self.data == original] = new
            self.data = mapped_data
            
        # Clean up MODIS fill values (255 -> 0 for water/background)
        # This fixes visualization issues where 255 skews the colormap
        if np.any(self.data == 255):
            logger.info("Replacing fill value 255 with 0 (Water/Background)")
            self.data[self.data == 255] = 0

        # Ensure data is in the correct format
        if self.data.shape != self.target_shape:
            logger.info(f"Resizing land cover data from {self.data.shape} to {self.target_shape}")
            self.data = resize(self.data, self.target_shape,
                               order=0, preserve_range=True).astype(np.uint8)

        # Display statistics
        logger.info(f"Land Cover Type data shape: {self.data.shape}")
        logger.info(f"Land Cover Type unique values: {np.unique(self.data)}")
        logger.info(f"Land Cover Type data range: [{np.min(self.data)}, {np.max(self.data)}]")

    def visualize(self, output_dir=None, filename="land_cover.png"):
        """Visualize land cover data with a discrete colormap"""
        if self.data is None:
            self.get_data()

        unique_values = np.unique(self.data)
        n_classes = len(unique_values)

        plt.figure(figsize=(12, 10))

        # Use a discrete colormap appropriate for categorical data
        cmap = plt.cm.get_cmap('tab20', n_classes)
        im = plt.imshow(self.data, cmap=cmap, interpolation='nearest')

        # Create colorbar with class labels
        cbar = plt.colorbar(im, ticks=unique_values)
        cbar.set_label('Land Cover Class')

        # Add class statistics to the title
        title = f"Land Cover Type Data ({n_classes} classes)"
        title += "\nClass distribution:"
        for val in unique_values:
            count = np.sum(self.data == val)
            pct = count / self.data.size * 100
            if pct > 1:  # Only show classes with significant presence
                title += f" Class {val}: {pct:.1f}%,"

        plt.title(title)

        if output_dir and filename:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path)
            logger.info(f"Land cover visualization saved to {output_path}")

        plt.close()


class DEMLoader(StaticDataLoader):
    """Loader for Digital Elevation Model data"""

    def __init__(self, data_root, target_shape=(700, 800),
                 lon_range=(109.505, 117.495), lat_range=(19.0519, 26.0419),
                 cache_file=None, normalize=True):
        super().__init__(data_root, target_shape, lon_range, lat_range, cache_file)
        self.normalize = normalize

    def load(self):
        """Load DEM data from .img files"""
        logger.info(f"Loading DEM data from {self.data_root}")

        try:
            import rasterio
            from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
            from rasterio.transform import from_bounds
            from rasterio.crs import CRS

            # Find all .img files
            dem_files = []
            for root, _, files in os.walk(self.data_root):
                dem_files.extend([os.path.join(root, f) for f in files if f.endswith('.img')])

            if not dem_files:
                raise FileNotFoundError(f"No DEM .img files found in {self.data_root}")

            logger.info(f"Found {len(dem_files)} DEM files")

            # Initialize an empty array for the composite DEM
            self.data = np.zeros(self.target_shape, dtype=np.float32)
            self.data.fill(np.nan)  # Fill with NaN to identify unfilled areas

            # Track coverage to combine multiple files
            coverage = np.zeros(self.target_shape, dtype=bool)

            # Process each file
            for dem_file in dem_files:
                logger.info(f"Processing DEM file: {os.path.basename(dem_file)}")

                try:
                    with rasterio.open(dem_file) as src:
                        # Get metadata for debugging
                        logger.info(f"Source CRS: {src.crs}")
                        logger.info(f"Source bounds: {src.bounds}")
                        logger.info(f"Source shape: {src.shape}")

                        # Transform source bounds to EPSG:4326 (Lat/Lon) for intersection check
                        src_crs = src.crs
                        target_crs = CRS.from_epsg(4326)
                        
                        try:
                            left, bottom, right, top = transform_bounds(
                                src_crs, target_crs, 
                                src.bounds.left, src.bounds.bottom, 
                                src.bounds.right, src.bounds.top
                            )
                        except Exception as e:
                            logger.warning(f"Could not transform bounds: {e}. Assuming intersection.")
                            # Fallback: assume intersection if transformation fails
                            left, bottom, right, top = self.lon_range[0], self.lat_range[0], self.lon_range[1], self.lat_range[1]

                        # Check intersection in Lat/Lon
                        if (right < self.lon_range[0] or
                                left > self.lon_range[1] or
                                top < self.lat_range[0] or
                                bottom > self.lat_range[1]):
                            logger.warning(f"File {dem_file} does not intersect with target region (Lat/Lon), skipping.")
                            continue

                        # Define the target CRS and transform
                        dst_crs = 'EPSG:4326'  # WGS84

                        # [OPTIMIZATION] Define transform based on target shape directly
                        dst_width, dst_height = self.target_shape[1], self.target_shape[0]  # (W, H)
                        dst_transform = from_bounds(
                            self.lon_range[0], self.lat_range[0],
                            self.lon_range[1], self.lat_range[1],
                            dst_width, dst_height
                        )

                        # Initialize the destination array with target shape
                        dst_array = np.zeros((dst_height, dst_width), dtype=np.float32)
                        dst_array.fill(np.nan)

                        # Reproject directly to target shape
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=dst_array,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear
                        )

                        # Resize to target shape using bilinear interpolation for smooth elevation
                        dst_array = resize(dst_array, self.target_shape,
                                           order=1, preserve_range=True).astype(np.float32)

                        # Update the composite DEM where we have new data
                        valid_mask = ~np.isnan(dst_array)
                        
                        # Use maximum for overlapping areas to avoid averaging artifacts
                        overlap_mask = valid_mask & ~np.isnan(self.data)
                        if np.any(overlap_mask):
                            self.data[overlap_mask] = np.maximum(self.data[overlap_mask], dst_array[overlap_mask])
                            
                        # Update non-overlapping new data
                        new_only_mask = valid_mask & np.isnan(self.data)
                        self.data[new_only_mask] = dst_array[new_only_mask]

                        # Update coverage mask
                        coverage = coverage | valid_mask

                except Exception as e:
                    logger.error(f"Error processing DEM file {dem_file}: {e}")
                    continue

            # If no data was loaded or coverage is too low, use fallback
            coverage_pct = np.sum(coverage) / coverage.size * 100
            logger.info(f"DEM coverage: {coverage_pct:.2f}%")

            if coverage_pct < 10:
                logger.warning("DEM coverage too low. Using fallback data.")
                self.data = self._generate_fallback_dem()
            else:
                # Fill remaining NaN values with nearest valid values
                if np.any(~coverage):
                    logger.info("Filling NaN values in DEM")
                    from scipy import ndimage

                    # Create a mask of valid values
                    mask = ~np.isnan(self.data)

                    # Fill NaN with the nearest valid values
                    indices = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
                    self.data = self.data[tuple(indices)]

        except ImportError as e:
            logger.error(f"Required package not installed: {e}")
            logger.info("Please install with: pip install rasterio scipy")
            self.data = self._generate_fallback_dem()
        except Exception as e:
            logger.error(f"Error loading DEM data: {e}", exc_info=True)
            self.data = self._generate_fallback_dem()

    def _generate_fallback_dem(self):
        """Generate fallback DEM if loading fails"""
        logger.warning("Generating synthetic DEM data as fallback")

        # Create a synthetic DEM with some mountains and valleys
        y, x = np.indices(self.target_shape)

        # Normalize coordinates to [0, 1]
        y = y / self.target_shape[0]
        x = x / self.target_shape[1]

        # Create mountains in the north
        mountains = 2000 * np.exp(-30 * ((x - 0.5) ** 2 + (y - 0.2) ** 2))

        # Create a ridge
        ridge = 1500 * np.exp(-100 * (x - 0.7) ** 2) * np.sin(10 * y * np.pi) ** 2

        # Create a valley
        valley = -500 * np.exp(-100 * (x - 0.3) ** 2) * (0.5 + 0.5 * np.sin(8 * y * np.pi))

        # Combine features
        dem = mountains + ridge + valley

        # Add some noise for texture
        noise = np.random.normal(0, 50, self.target_shape)
        dem += noise

        # Ensure reasonable elevation range (0 to 3000 meters)
        dem = np.clip(dem, 0, 3000)

        return dem

    def preprocess(self):
        """Preprocess DEM data"""
        if self.data is None:
            self.load()

        # Handle any remaining NaN values
        if np.any(np.isnan(self.data)):
            logger.warning(f"Found {np.sum(np.isnan(self.data))} NaN values in DEM after loading")
            self.data = np.nan_to_num(self.data, nan=0.0)

        # Ensure data is in the correct format
        if self.data.shape != self.target_shape:
            logger.info(f"Resizing DEM data from {self.data.shape} to {self.target_shape}")
            self.data = resize(self.data, self.target_shape,
                               order=1, preserve_range=True).astype(np.float32)

        # Normalize elevation to [0, 1] range for better model convergence
        if self.normalize and np.max(self.data) > np.min(self.data):
            self.data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
            logger.info(f"Normalized DEM to range [0, 1]")

        # Display statistics
        logger.info(f"DEM data shape: {self.data.shape}")
        logger.info(f"DEM data range: [{np.min(self.data):.2f}, {np.max(self.data):.2f}]")
        logger.info(f"DEM mean elevation: {np.mean(self.data):.2f}")

    def visualize(self, output_dir=None, filename="dem.png"):
        """Visualize DEM data with terrain colormap"""
        if self.data is None:
            self.get_data()

        plt.figure(figsize=(12, 10))

        # Use a terrain colormap for elevation
        im = plt.imshow(self.data, cmap='terrain')
        cbar = plt.colorbar(im)

        if self.normalize:
            cbar.set_label('Normalized Elevation')
        else:
            cbar.set_label('Elevation (m)')

        plt.title(f"Digital Elevation Model\nElevation range: {np.min(self.data):.2f} - {np.max(self.data):.2f}")

        if output_dir and filename:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path)
            logger.info(f"DEM visualization saved to {output_path}")

        plt.close()


def load_static_data(config, results_dir=None):
    """Utility function to load multiple static data sources"""
    static_data = {}

    # Load Land Cover Type if enabled
    if config.get('land_cover', {}).get('enabled', False):
        lc_config = config['land_cover']
        lc_loader = LandCoverLoader(
            data_root=lc_config['data_root'],
            target_shape=config.get('target_shape', (700, 800)),
            lon_range=config.get('lon_range', (109.505, 117.495)),
            lat_range=config.get('lat_range', (19.0519, 26.0419)),
            cache_file=lc_config.get('cache_file')
        )
        static_data['land_cover'] = lc_loader.get_data()

        # Visualize if results_dir is provided
        if results_dir:
            lc_loader.visualize(results_dir, "land_cover.png")

    # Load DEM if enabled
    if config.get('dem', {}).get('enabled', False):
        dem_config = config['dem']
        dem_loader = DEMLoader(
            data_root=dem_config['data_root'],
            target_shape=config.get('target_shape', (700, 800)),
            lon_range=config.get('lon_range', (109.505, 117.495)),
            lat_range=config.get('lat_range', (19.0519, 26.0419)),
            cache_file=dem_config.get('cache_file'),
            normalize=dem_config.get('normalize', True)
        )
        static_data['dem'] = dem_loader.get_data()

        # Visualize if results_dir is provided
        if results_dir:
            dem_loader.visualize(results_dir, "dem.png")

    return static_data


if __name__ == "__main__":
    """Test the static data loaders"""
    import sys
    import os

    # Project root from the command line, otherwise inferred from this file's location
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print(f"Testing static data loaders with project root: {project_root}")

    # Configure static data sources
    static_config = {
        'target_shape': (700, 800),
        'lon_range': (109.505, 117.495),
        'lat_range': (19.0519, 26.0419),
        'land_cover': {
            'enabled': True,
            'data_root': os.path.join(project_root, "data/Owned_data/Land_Cover_Type"),
            'cache_file': os.path.join(project_root, "data/land_cover_cache.npy")
        },
        'dem': {
            'enabled': True,
            'data_root': os.path.join(project_root, "data/Owned_data/DEM"),
            'cache_file': os.path.join(project_root, "data/dem_cache.npy"),
            'normalize': True
        }
    }

    # Create results directory for visualizations
    results_dir = os.path.join(project_root, "radar_results/static_data")
    os.makedirs(results_dir, exist_ok=True)

    # Load and visualize all static data
    static_data = load_static_data(static_config, results_dir)

    # Print summary
    print("\n=== Static Data Summary ===")
    for name, data in static_data.items():
        print(f"{name} shape: {data.shape}")
        print(f"{name} type: {data.dtype}")
        print(f"{name} range: [{np.min(data)}, {np.max(data)}]")
        if name == 'land_cover':
            unique_values = np.unique(data)
            print(f"{name} unique classes: {unique_values}")
