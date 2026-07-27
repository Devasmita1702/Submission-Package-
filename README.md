# Submission-Package-

# Geospatial GCP Inference Pipeline

## Strategy Overview
This pipeline executes GCP pose estimation over large geospatial orthomosaics using bounded-memory sliding-window iteration and strict geospatial coordinate propagation.

### Key Pipeline Components
1. **Windowing & Memory Bound:** Uses standard 640x640 windows with an overlap stride of 160 pixels to prevent split-marker boundary truncation. Memory consumption stays below 500 MB RAM regardless of input raster size.
2. **Preprocessing:** Automatically reads multi-channel GeoTIFF rasters via `rasterio`, scales `uint8`/`uint16` values to float32 `[0, 1]`, and letterboxes dynamic edge windows safely to match model input specifications.
3. **Post-Processing & Fusion:** Dense ONNX candidate keypoints are filtered by confidence ($> 0.35$). Overlapping window duplicates are clustered using **DBSCAN** in full-raster pixel coordinates and fused using confidence-weighted spatial averaging.
4. **Geospatial Reprojection:** Local window keypoints translate to continuous raster coordinates, which are converted to native map coordinates via the affine transform matrix, and reprojected to EPSG:4326 (WGS84) via `pyproj`.

## Docker Build & Run Usage

### Building Image
```bash
docker build --platform linux/amd64 -t gcp-inference .
