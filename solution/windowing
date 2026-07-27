import cv2
import numpy as np
import rasterio
from rasterio.windows import Window


def generate_sliding_windows(height: int, width: int, tile_size: int = 640, overlap: int = 128):
    stride = tile_size - overlap
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            yield Window(col_off=x, row_off=y, width=w, height=h)


def preprocess_window(
    src_dataset: rasterio.DatasetReader, 
    window: Window, 
    target_size: int = 640
) -> tuple[np.ndarray, float, float, float]:
    """
    Reads tile window from raster, converts to RGB float32 [0, 1], and pads to 640x640.
    """
    # Read specified window bands
    count = src_dataset.count
    if count >= 3:
        arr = src_dataset.read([1, 2, 3], window=window)  # RGB
    else:
        single = src_dataset.read(1, window=window)
        arr = np.repeat(single[np.newaxis, :, :], 3, axis=0)

    # Transpose CHW -> HWC
    img = np.transpose(arr, (1, 2, 0)).astype(np.float32)

    # Scale range to [0, 1] based on dtype
    if src_dataset.dtypes[0] == 'uint8':
        img /= 255.0
    elif src_dataset.dtypes[0] == 'uint16':
        img /= 65535.0
    else:
        img = np.clip(img, 0.0, 1.0)

    h, w, _ = img.shape
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top_pad, left_pad = pad_h // 2, pad_w // 2

    # Canvas creation with constant padding
    canvas = np.zeros((target_size, target_size, 3), dtype=np.float32)
    canvas[top_pad:top_pad + new_h, left_pad:left_pad + new_w] = img_resized

    # Format into NCHW tensor layout
    tensor = np.transpose(canvas, (2, 0, 1))[np.newaxis, ...]
    return tensor, scale, top_pad, left_pad
