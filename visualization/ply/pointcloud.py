#!/usr/bin/env python3
"""
Converte point clouds .npy → .ply (formato CloudCompare).

Each .ply contains:
  - the scene points (grey, from the depth image, in the camera frame)
  - the drone points (red, from the 3D box in the label JSON)

Uso:
    python3 convert_pointclouds_to_ply.py <dataset_dir> [--limit N]

Output:
    <dataset_dir>/pointnet_ply/frame_XXXXXX.ply
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path


def write_ply(path, points, colors=None):
    """
    Escreve PLY binary little-endian.
    points: Nx3 float32
    colors: Nx3 uint8 (R, G, B) ou None
    """
    n = len(points)
    has_color = colors is not None
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header.append("end_header\n")
    header_bytes = "\n".join(header).encode("ascii")
    with open(path, "wb") as f:
        f.write(header_bytes)
        if has_color:
            # Interleaved: x y z r g b per point
            rec = np.zeros(n, dtype=[
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("r", "u1"), ("g", "u1"), ("b", "u1"),
            ])
            rec["x"] = points[:, 0]
            rec["y"] = points[:, 1]
            rec["z"] = points[:, 2]
            rec["r"] = colors[:, 0]
            rec["g"] = colors[:, 1]
            rec["b"] = colors[:, 2]
            f.write(rec.tobytes())
        else:
            f.write(points.astype("<f4").tobytes())


def airsim_to_cv_frame(p):
    """
    Converte AirSim NED camera frame (X=forward, Y=right, Z=down)
    into the CV camera frame (X=right, Y=down, Z=forward), which is what
    the one depth_to_pointcloud uses.
    """
    return [p[1], p[2], p[0]]


def points_from_box3d(box_min, box_max, n_per_edge=10):
    """Generate points along the 12 edges of a 3D box, to view it as a wireframe.
    Converts AirSim -> CV frame first."""
    # Convert the corners to the CV frame
    box_min = airsim_to_cv_frame(box_min)
    box_max = airsim_to_cv_frame(box_max)
    # Reorder min/max: after the axis swap they may have exchanged sign
    bmin = [min(box_min[i], box_max[i]) for i in range(3)]
    bmax = [max(box_min[i], box_max[i]) for i in range(3)]
    pts = []
    for axis in range(3):
        for i in (0, 1):
            for j in (0, 1):
                for k in range(n_per_edge):
                    t = k / max(1, n_per_edge - 1)
                    p = [0.0, 0.0, 0.0]
                    p[axis] = bmin[axis] + t * (bmax[axis] - bmin[axis])
                    other = [a for a in range(3) if a != axis]
                    p[other[0]] = bmin[other[0]] if i == 0 else bmax[other[0]]
                    p[other[1]] = bmin[other[1]] if j == 0 else bmax[other[1]]
                    pts.append(p)
    return np.array(pts, dtype=np.float32)


def convert_frame(npy_path, json_path, out_path):
    scene = np.load(npy_path).astype(np.float32)  # Nx3 (camera frame)
    n_scene = len(scene)
    # Grey for the scene points
    colors_scene = np.full((n_scene, 3), 150, dtype=np.uint8)

    # Read the 3D labels if present -- add the 3D boxes as red points
    extra_pts = []
    extra_colors = []
    if json_path and json_path.exists():
        try:
            data = json.load(open(json_path))
            for d in data:
                if "box3D_min" in d and "box3D_max" in d:
                    # box3D comes in the camera frame (the same one as the scene)
                    bb_pts = points_from_box3d(d["box3D_min"], d["box3D_max"])
                    extra_pts.append(bb_pts)
                    # Red for the drone box
                    extra_colors.append(
                        np.full((len(bb_pts), 3), [255, 0, 0], dtype=np.uint8))
                # Mark the drone centre (relative_pose) with a green point
                # Convert AirSim NED -> CV frame so it aligns with the point cloud
                if "center" in d:
                    c = np.array([airsim_to_cv_frame(d["center"])], dtype=np.float32)
                    extra_pts.append(c)
                    extra_colors.append(np.array([[0, 255, 0]], dtype=np.uint8))
        except Exception as e:
            print(f"  warning ler {json_path}: {e}")

    if extra_pts:
        extra_pts = np.concatenate(extra_pts, axis=0)
        extra_colors = np.concatenate(extra_colors, axis=0)
        all_pts = np.concatenate([scene, extra_pts], axis=0)
        all_colors = np.concatenate([colors_scene, extra_colors], axis=0)
    else:
        all_pts = scene
        all_colors = colors_scene

    write_ply(out_path, all_pts, all_colors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", help="ex: dataset_urban_city")
    ap.add_argument("--limit", type=int, default=None,
                    help="convert only the first N frames (default: all)")
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    npy_dir = root / "pointnet" / "point_clouds"
    json_dir = root / "pointnet" / "labels_3d"
    out_dir = root / "pointnet_ply"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(npy_dir.glob("*.npy"))
    if args.limit:
        files = files[:args.limit]

    print(f"Convertendo {len(files)} point clouds...")
    for i, npy_path in enumerate(files):
        frame_name = npy_path.stem
        json_path = json_dir / f"{frame_name}.json"
        out_path = out_dir / f"{frame_name}.ply"
        convert_frame(npy_path, json_path, out_path)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}")

    print(f"\nFeito. Output: {out_dir}")
    print(f"To open in CloudCompare: right-click -> Open -> select the .ply")
    print(f"Colours: grey = scene, red = the drone 3D box, green = the drone centre")


if __name__ == "__main__":
    main()
