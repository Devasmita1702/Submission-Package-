import numpy as np
import pyproj
import rasterio
from rasterio.crs import CRS


class GeoTransformer:
    def __init__(self, src_dataset: rasterio.DatasetReader):
        self.transform = src_dataset.transform
        self.src_crs = src_dataset.crs
        self.wgs84_crs = CRS.from_epsg(4326)
        
        if self.src_crs != self.wgs84_crs:
            self.reprojector = pyproj.Transformer.from_crs(
                self.src_crs, self.wgs84_crs, always_xy=True
            )
        else:
            self.reprojector = None

    def pixel_to_wgs84(self, px: float, py: float) -> tuple[float, float]:
        """Converts continuous raster pixel (x=col, y=row) to WGS84 (lon, lat)."""
        # Map pixel (col, row) to raster CRS coordinates using Affine transform
        map_x, map_y = self.transform * (px, py)
        
        # Reproject to WGS84 EPSG:4326 if necessary
        if self.reprojector:
            lon, lat = self.reprojector.transform(map_x, map_y)
        else:
            lon, lat = map_x, map_y
            
        return float(lon), float(lat)
