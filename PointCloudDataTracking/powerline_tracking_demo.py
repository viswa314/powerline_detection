"""
Powerline Tracking Demo - Fast version for Toronto-3D Dataset

This script demonstrates powerline detection and cable tracking 
using geometric features from the Toronto-3D point cloud dataset.

Modes:
1. Ground Truth Mode: Uses labeled utility line points (fast)
2. Detection Mode: Detects powerlines using geometric features

Author: GitHub Copilot
"""

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from pathlib import Path
import time


def load_toronto3d_ply(filepath: str) -> dict:
    """
    Load a Toronto-3D PLY file with all attributes.
    
    Returns dict with: points, colors, labels, intensity
    """
    print(f"Loading: {filepath}")
    
    with open(filepath, 'rb') as f:
        # Skip header
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if line == 'end_header':
                break
        
        # Define dtype for Toronto-3D format
        dtype = np.dtype([
            ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('intensity', 'f4'), ('gpstime', 'f4'), 
            ('scanangle', 'f4'), ('label', 'f4')
        ])
        
        data = np.fromfile(f, dtype=dtype)
    
    # Apply UTM offset
    UTM_OFFSET = np.array([627285, 4842000, 0])
    points = np.column_stack([data['x'], data['y'], data['z']]) - UTM_OFFSET
    colors = np.column_stack([data['red'], data['green'], data['blue']]) / 255.0
    labels = data['label'].astype(np.int32)
    
    print(f"  Loaded {len(points):,} points")
    
    return {
        'points': points,
        'colors': colors,
        'labels': labels,
        'intensity': data['intensity']
    }


def extract_powerlines_gt(data: dict) -> np.ndarray:
    """Extract powerline points using ground truth labels."""
    mask = data['labels'] == 5  # Utility_line label
    points = data['points'][mask]
    print(f"  Utility line points: {len(points):,}")
    return points


def cluster_cables(points: np.ndarray, eps: float = 1.0, 
                   min_samples: int = 10) -> tuple:
    """
    Cluster powerline points into individual cables using DBSCAN.
    
    Returns (cluster_labels, n_clusters)
    """
    print(f"Clustering {len(points):,} points...")
    start = time.time()
    
    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = clustering.fit_predict(points)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    print(f"  Found {n_clusters} cable clusters, {n_noise:,} noise points")
    print(f"  Time: {time.time() - start:.1f}s")
    
    return labels, n_clusters


def track_cable_segments(points: np.ndarray, cluster_labels: np.ndarray,
                         min_length: float = 5.0) -> list:
    """
    Track and fit line segments to cable clusters.
    
    For each cluster:
    - Apply PCA to find main direction
    - Extract start/end points
    - Compute cable properties
    """
    print("Fitting cable segments...")
    
    segments = []
    unique_labels = set(cluster_labels) - {-1}
    
    for label in unique_labels:
        mask = cluster_labels == label
        cluster_pts = points[mask]
        
        if len(cluster_pts) < 20:
            continue
        
        # PCA for main direction
        pca = PCA(n_components=3)
        pca.fit(cluster_pts)
        direction = pca.components_[0]
        
        # Ensure consistent direction
        if direction[1] < 0:  # Point in +Y direction
            direction = -direction
        
        # Project onto principal axis to find endpoints
        centroid = cluster_pts.mean(axis=0)
        projections = np.dot(cluster_pts - centroid, direction)
        
        start_pt = cluster_pts[np.argmin(projections)]
        end_pt = cluster_pts[np.argmax(projections)]
        length = np.linalg.norm(end_pt - start_pt)
        
        if length < min_length:
            continue
        
        # Height range
        z_min, z_max = cluster_pts[:, 2].min(), cluster_pts[:, 2].max()
        
        segments.append({
            'id': len(segments),
            'start': start_pt,
            'end': end_pt,
            'direction': direction,
            'length': length,
            'n_points': len(cluster_pts),
            'height_range': (z_min, z_max),
            'linearity': pca.explained_variance_ratio_[0],
            'points': cluster_pts
        })
    
    # Sort by length
    segments.sort(key=lambda x: x['length'], reverse=True)
    
    print(f"  Tracked {len(segments)} cable segments")
    
    return segments


def analyze_cables(segments: list) -> None:
    """Print cable analysis."""
    if not segments:
        print("No cable segments found!")
        return
    
    print("\n" + "="*60)
    print("CABLE ANALYSIS")
    print("="*60)
    
    total_length = sum(s['length'] for s in segments)
    total_points = sum(s['n_points'] for s in segments)
    
    print(f"Total cables: {len(segments)}")
    print(f"Total length: {total_length:.1f}m")
    print(f"Total points: {total_points:,}")
    
    print("\nTop 10 Cable Segments:")
    print("-"*60)
    print(f"{'ID':>3} {'Length(m)':>10} {'Points':>8} {'Height(m)':>12} {'Linearity':>10}")
    print("-"*60)
    
    for seg in segments[:10]:
        h_range = f"{seg['height_range'][0]:.1f}-{seg['height_range'][1]:.1f}"
        print(f"{seg['id']:>3} {seg['length']:>10.1f} {seg['n_points']:>8,} "
              f"{h_range:>12} {seg['linearity']:>10.2%}")
    
    print("="*60)


def visualize_powerlines(all_points: np.ndarray, all_colors: np.ndarray,
                         powerline_points: np.ndarray, segments: list,
                         save_path: str = None) -> None:
    """
    Visualize detected powerlines with Open3D.
    
    - Gray: Original point cloud
    - Red: Powerline points
    - Green lines: Tracked cable centerlines
    """
    print("\nPreparing visualization...")
    
    geometries = []
    
    # Original cloud (faded gray)
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(all_points)
    pcd_all.colors = o3d.utility.Vector3dVector(all_colors * 0.3 + 0.2)
    geometries.append(pcd_all)
    
    # Powerline points (red)
    pcd_power = o3d.geometry.PointCloud()
    pcd_power.points = o3d.utility.Vector3dVector(powerline_points)
    pcd_power.paint_uniform_color([1, 0, 0])  # Red
    geometries.append(pcd_power)
    
    # Cable centerlines (green)
    for seg in segments:
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([seg['start'], seg['end']])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.paint_uniform_color([0, 1, 0])  # Green
        geometries.append(line)
        
        # Also add spheres at endpoints
        for pt in [seg['start'], seg['end']]:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
            sphere.translate(pt)
            sphere.paint_uniform_color([0, 1, 0])
            geometries.append(sphere)
    
    # Save combined point cloud if path provided
    if save_path:
        # Combine powerlines with colors
        pcd_save = o3d.geometry.PointCloud()
        pcd_save.points = o3d.utility.Vector3dVector(powerline_points)
        pcd_save.paint_uniform_color([1, 0, 0])
        o3d.io.write_point_cloud(save_path, pcd_save)
        print(f"Saved powerlines to: {save_path}")
    
    # Visualize
    print("Opening visualization window...")
    print("Controls: Left-drag to rotate, Right-drag to pan, Scroll to zoom")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Powerline Detection - Toronto-3D",
        width=1400,
        height=900
    )


def detect_powerlines_geometric(data: dict, 
                                min_height: float = 5.0,
                                max_height: float = 25.0,
                                voxel_size: float = 0.15) -> np.ndarray:
    """
    Detect powerlines using geometric features (no ground truth).
    
    Method:
    1. Filter by height (above ground)
    2. Compute linearity using PCA
    3. Filter high-linearity points
    """
    print("\nDetecting powerlines using geometric features...")
    
    points = data['points']
    labels = data['labels']
    
    # Downsample for faster processing
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pts = np.asarray(pcd_down.points)
    print(f"  Downsampled to {len(pts):,} points")
    
    # Height filter (relative to ground)
    ground_level = np.percentile(pts[:, 2], 5)
    rel_height = pts[:, 2] - ground_level
    height_mask = (rel_height >= min_height) & (rel_height <= max_height)
    print(f"  After height filter: {height_mask.sum():,} points")
    
    # Compute linearity for elevated points
    elevated_pts = pts[height_mask]
    if len(elevated_pts) == 0:
        return np.array([])
    
    # Build KD-tree for neighbor search
    tree = KDTree(elevated_pts)
    
    linearity = np.zeros(len(elevated_pts))
    
    print("  Computing linearity features...")
    for i in range(len(elevated_pts)):
        # Get 20 nearest neighbors
        dist, idx = tree.query(elevated_pts[i], k=20)
        neighbors = elevated_pts[idx]
        
        # PCA on neighbors
        if len(neighbors) >= 3:
            cov = np.cov(neighbors.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            if eigenvalues.sum() > 1e-10:
                linearity[i] = (eigenvalues[0] - eigenvalues[1]) / eigenvalues.sum()
        
        if (i + 1) % 50000 == 0:
            print(f"    Processed {i+1:,}/{len(elevated_pts):,}")
    
    # Filter by linearity
    linear_mask = linearity > 0.7
    powerline_pts = elevated_pts[linear_mask]
    
    print(f"  Detected {len(powerline_pts):,} powerline candidate points")
    
    return powerline_pts


def main():
    """Main entry point."""
    
    # Find PLY files
    data_dir = Path(__file__).parent
    ply_files = list(data_dir.glob("L*.ply"))
    
    if not ply_files:
        print("No PLY files found!")
        print(f"Please place Toronto-3D PLY files (L001.ply, etc.) in: {data_dir}")
        return
    
    print(f"Found {len(ply_files)} PLY files: {[f.name for f in ply_files]}")
    
    # Process L001
    filepath = str(ply_files[0])
    
    # Load data
    data = load_toronto3d_ply(filepath)
    
    # Mode selection
    print("\n" + "="*60)
    print("POWERLINE TRACKING SYSTEM")
    print("="*60)
    
    # Use ground truth labels (fast and accurate)
    use_ground_truth = True
    
    if use_ground_truth:
        print("\nMode: Using ground truth labels")
        powerline_pts = extract_powerlines_gt(data)
    else:
        print("\nMode: Geometric feature detection")
        powerline_pts = detect_powerlines_geometric(data)
    
    if len(powerline_pts) == 0:
        print("No powerline points found!")
        return
    
    # Cluster into cables
    cluster_labels, n_clusters = cluster_cables(
        powerline_pts, 
        eps=1.5,          # 1.5m clustering distance
        min_samples=15    # Minimum 15 points per cluster
    )
    
    # Track cable segments
    segments = track_cable_segments(
        powerline_pts, 
        cluster_labels,
        min_length=5.0    # Minimum 5m cable length
    )
    
    # Analyze results
    analyze_cables(segments)
    
    # Save results
    output_path = str(data_dir / "detected_powerlines.ply")
    
    # Visualize
    try:
        visualize_powerlines(
            data['points'], 
            data['colors'],
            powerline_pts, 
            segments,
            save_path=output_path
        )
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Saving results without visualization...")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(powerline_pts)
        pcd.paint_uniform_color([1, 0, 0])
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved to: {output_path}")
    
    print("\nDone!")
    
    return data, powerline_pts, segments


if __name__ == "__main__":
    data, powerline_pts, segments = main()
