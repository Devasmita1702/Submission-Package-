import argparse
import json
import os
import sys
import numpy as np
import onnxruntime as ort
import rasterio

from solution.geo import GeoTransformer
from solution.postprocess import parse_onnx_output, fuse_duplicate_detections
from solution.windowing import generate_sliding_windows, preprocess_window


def process_scene(
    scene_info: dict, 
    manifest_dir: str, 
    session: ort.InferenceSession,
    conf_thresh: float = 0.35
) -> dict:
    raster_rel_path = scene_info['raster_path']
    raster_full_path = os.path.join(manifest_dir, raster_rel_path)

    raw_candidates = []

    with rasterio.open(raster_full_path) as src:
        geo_trans = GeoTransformer(src)
        h, w = src.height, src.width
        input_name = session.get_inputs()[0].name

        for window in generate_sliding_windows(h, w, tile_size=640, overlap=160):
            tensor, scale, pad_top, pad_left = preprocess_window(src, window)
            outputs = session.run(None, {input_name: tensor})
            kpts, scores = parse_onnx_output(outputs[0], conf_thresh=conf_thresh)

            for (kx, ky), score in zip(kpts, scores):
                # Reverse letterbox padding and scaling
                win_x = (kx - pad_left) / scale
                win_y = (ky - pad_top) / scale

                # Ensure detection falls within actual unpadded tile bounds
                if 0 <= win_x <= window.width and 0 <= win_y <= window.height:
                    full_px = window.col_off + win_x
                    full_py = window.row_off + win_y

                    # Verify detection strictly lies within full-raster boundaries
                    if 0 <= full_px <= w and 0 <= full_py <= h:
                        raw_candidates.append({
                            'pixel_x': full_px,
                            'pixel_y': full_py,
                            'confidence': score
                        })

        # Deduplicate predictions across overlapping window boundaries
        fused = fuse_duplicate_detections(raw_candidates, dist_threshold_px=15.0)

        formatted_detections = []
        for det in fused:
            lon, lat = geo_trans.pixel_to_wgs84(det['pixel_x'], det['pixel_y'])
            formatted_detections.append({
                "pixel_x": det['pixel_x'],
                "pixel_y": det['pixel_y'],
                "longitude": round(lon, 7),
                "latitude": round(lat, 7),
                "confidence": det['confidence']
            })

    return {
        "scene_id": scene_info['scene_id'],
        "detections": formatted_detections
    }


def main():
    parser = argparse.ArgumentParser(description="GCP Pose Inference Pipeline CLI")
    parser.add_argument("--manifest", required=True, help="Path to input manifest.json")
    parser.add_argument("--model", required=True, help="Path to gcp_pose.onnx")
    parser.add_argument("--output", required=True, help="Path to output predictions.json")
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    manifest_dir = os.path.dirname(manifest_path)

    with open(manifest_path, 'r') as f:
        manifest_data = json.load(f)

    # Initialize ONNX Runtime Session using CPU execution provider
    opts = ort.SessionOptions()
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(args.model, opts, providers=['CPUExecutionProvider'])

    out_scenes = []
    for scene in manifest_data.get('scenes', []):
        scene_result = process_scene(scene, manifest_dir, session)
        out_scenes.append(scene_result)

    output_payload = {
        "schema_version": "1.0",
        "scenes": out_scenes
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_payload, f, indent=2)

    print(f"Inference successfully finished. Output saved to: {args.output}")


if __name__ == "__main__":
    main()
