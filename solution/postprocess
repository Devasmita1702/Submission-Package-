import numpy as np
from sklearn.cluster import DBSCAN


def parse_onnx_output(
    output_tensor: np.ndarray, 
    conf_thresh: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses [1, 9, 8400] model output tensor.
    Returns:
        kpts_xy: shape (N, 2) in model image coords [640, 640]
        scores: shape (N,)
    """
    pred = output_tensor[0].T  # Shape: [8400, 9]
    class_probs = pred[:, 4:7]
    scores = class_probs.max(axis=1)
    
    mask = scores >= conf_thresh
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
        
    kpts_xy = pred[mask, 7:9]
    scores = scores[mask]
    
    return kpts_xy, scores


def fuse_duplicate_detections(
    detections: list[dict], 
    dist_threshold_px: float = 15.0
) -> list[dict]:
    """
    DBSCAN clustering to consolidate duplicate predictions across overlapping tiles.
    """
    if not detections:
        return []

    coords = np.array([[d['pixel_x'], d['pixel_y']] for d in detections])
    confidences = np.array([d['confidence'] for d in detections])

    db = DBSCAN(eps=dist_threshold_px, min_samples=1, metric='euclidean').fit(coords)
    labels = db.labels_

    fused_detections = []
    unique_labels = set(labels)

    for label in unique_labels:
        cluster_mask = (labels == label)
        c_coords = coords[cluster_mask]
        c_confs = confidences[cluster_mask]

        # Weighted average center localization
        weights = c_confs / np.sum(c_confs)
        fused_px = float(np.sum(c_coords[:, 0] * weights))
        fused_py = float(np.sum(c_coords[:, 1] * weights))
        fused_conf = float(np.max(c_confs))  # Maximum probability hypothesis

        # Retain original geometric reference
        sample_item = detections[np.where(cluster_mask)[0][0]]
        
        fused_detections.append({
            'pixel_x': round(fused_px, 2),
            'pixel_y': round(fused_py, 2),
            'confidence': round(fused_conf, 4),
            '_sample': sample_item
        })

    return fused_detections
