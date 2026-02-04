"""
Powerline Detector for Toronto-3D Point Cloud Dataset

This module implements a powerline detection and tracking system using
the Toronto-3D LiDAR dataset. It combines geometric feature extraction,
RANSAC-based line fitting, and clustering to identify powerline cables.

Dataset Classes:
    0: Unclassified
    1: Ground/Road
    2: Road markings
    3: Natural (vegetation)
    4: Building
    5: Utility line (powerlines) <- Target class
    6: Pole
    7: Car
    8: Fence

Author: GitHub Copilot
"""

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
import warnings
warnings.filterwarnings('ignore')


class PowerlineDetector:
    """
    A class for detecting and tracking powerlines in 3D point cloud data.
    
    The detection pipeline:
    1. Load PLY data and extract relevant attributes
    2. Height-based filtering (powerlines are typically elevated)
    3. Geometric feature extraction (linearity, planarity, sphericity)
    4. DBSCAN clustering to group potential powerline points
    5. RANSAC-based line fitting for cable tracking
    6. Validation and refinement
    """
    
    # Class label for utility lines in Toronto-3D
    UTILITY_LINE_LABEL = 5
    
    def __init__(self, 
                 min_height: float = 3.0,
                 max_height: float = 25.0,
                 voxel_size: float = 0.05,
                 neighbor_radius: float = 0.5,
                 linearity_threshold: float = 0.7,
                 dbscan_eps: float = 1.0,
                 dbscan_min_samples: int = 10,
                 min_cable_length: float = 5.0):
        """
        Initialize the PowerlineDetector.
        
        Args:
            min_height: Minimum height for powerline candidates (meters)
            max_height: Maximum height for powerline candidates (meters)
            voxel_size: Voxel size for downsampling (meters)
            neighbor_radius: Radius for neighborhood search (meters)
            linearity_threshold: Minimum linearity score for powerline points
            dbscan_eps: DBSCAN epsilon parameter for clustering
            dbscan_min_samples: Minimum samples for DBSCAN clustering
            min_cable_length: Minimum length for a valid cable segment (meters)
        """
        self.min_height = min_height
        self.max_height = max_height
        self.voxel_size = voxel_size
        self.neighbor_radius = neighbor_radius
        self.linearity_threshold = linearity_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.min_cable_length = min_cable_length
        
        # Store results
        self.points = None
        self.colors = None
        self.labels = None
        self.detected_powerlines = None
        self.cable_segments = []
        
    def load_ply(self, filepath: str) -> tuple:
        """
        Load a PLY file from the Toronto-3D dataset.
        
        The Toronto-3D PLY files contain (binary format):
        - x, y, z (double) - UTM coordinates
        - red, green, blue (uchar) - RGB colors
        - scalar_Intensity (float)
        - scalar_GPSTime (float)
        - scalar_ScanAngleRank (float)
        - scalar_Label (float) - Semantic labels
        
        Args:
            filepath: Path to the PLY file
            
        Returns:
            Tuple of (points, colors, labels)
        """
        print(f"Loading point cloud from: {filepath}")
        
        # Read PLY header to get vertex count
        with open(filepath, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
            
            header_size = f.tell()
            
            # Parse header
            vertex_count = 0
            for line in header_lines:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                    break
            
            print(f"Reading {vertex_count:,} vertices...")
            
            # Define dtype for Toronto-3D PLY format:
            # x, y, z (double), r, g, b (uchar), intensity, gpstime, scanangle, label (float)
            dtype = np.dtype([
                ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),  # double
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),  # uchar
                ('intensity', 'f4'),
                ('gpstime', 'f4'),
                ('scanangle', 'f4'),
                ('label', 'f4')
            ])
            
            # Read binary data
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
        
        # Extract XYZ coordinates
        points = np.column_stack([data['x'], data['y'], data['z']])
        
        # Apply UTM offset to avoid floating point precision issues
        # Toronto-3D recommended offset based on actual data
        UTM_OFFSET = np.array([627285, 4842000, 0])  # Adjusted Y offset
        points = points - UTM_OFFSET
        
        # Extract RGB colors (normalize to 0-1 range)
        colors = np.column_stack([
            data['red'], data['green'], data['blue']
        ]).astype(np.float64) / 255.0
        
        # Extract labels
        labels = data['label'].astype(np.int32)
        
        # Also store intensity for potential use
        self.intensity = data['intensity']
        
        self.points = points
        self.colors = colors
        self.labels = labels
        
        # Print label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"Loaded {len(points):,} points")
        print(f"Point cloud bounds: X[{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
              f"Y[{points[:,1].min():.2f}, {points[:,1].max():.2f}], "
              f"Z[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
        print(f"Label distribution:")
        label_names = {
            0: 'Unclassified', 1: 'Ground', 2: 'Road_markings', 3: 'Natural',
            4: 'Building', 5: 'Utility_line', 6: 'Pole', 7: 'Car', 8: 'Fence'
        }
        for label, count in zip(unique_labels, counts):
            name = label_names.get(label, f'Unknown({label})')
            pct = 100 * count / len(labels)
            print(f"  {name}: {count:,} ({pct:.2f}%)")
        
        return points, colors, labels
    
    def compute_geometric_features(self, points: np.ndarray, 
                                    k_neighbors: int = 30) -> dict:
        """
        Compute geometric features for each point.
        
        Features computed:
        - Linearity: How line-like the local neighborhood is
        - Planarity: How planar the local neighborhood is
        - Sphericity: How spherical the local neighborhood is
        - Verticality: Alignment with vertical direction
        - Height: Z coordinate (elevation)
        
        Args:
            points: Nx3 array of point coordinates
            k_neighbors: Number of neighbors for feature computation
            
        Returns:
            Dictionary of feature arrays
        """
        print("Computing geometric features...")
        
        n_points = len(points)
        
        # Initialize feature arrays
        features = {
            'linearity': np.zeros(n_points),
            'planarity': np.zeros(n_points),
            'sphericity': np.zeros(n_points),
            'verticality': np.zeros(n_points),
            'height': points[:, 2].copy(),
            'curvature': np.zeros(n_points)
        }
        
        # Build KD-tree for efficient neighbor search
        kdtree = KDTree(points)
        
        # Process in batches for efficiency
        batch_size = 10000
        for start_idx in range(0, n_points, batch_size):
            end_idx = min(start_idx + batch_size, n_points)
            batch_points = points[start_idx:end_idx]
            
            # Query k nearest neighbors for batch
            distances, indices = kdtree.query(batch_points, k=k_neighbors)
            
            for i, (local_idx, neighbor_idx) in enumerate(zip(range(end_idx - start_idx), indices)):
                global_idx = start_idx + local_idx
                neighbors = points[neighbor_idx]
                
                # Compute covariance matrix
                centered = neighbors - neighbors.mean(axis=0)
                cov = np.cov(centered.T)
                
                # Compute eigenvalues
                try:
                    eigenvalues = np.linalg.eigvalsh(cov)
                    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
                    
                    # Normalize eigenvalues
                    eigensum = eigenvalues.sum()
                    if eigensum > 1e-10:
                        e1, e2, e3 = eigenvalues / eigensum
                        
                        # Compute geometric features
                        features['linearity'][global_idx] = (e1 - e2) / (e1 + 1e-10)
                        features['planarity'][global_idx] = (e2 - e3) / (e1 + 1e-10)
                        features['sphericity'][global_idx] = e3 / (e1 + 1e-10)
                        features['curvature'][global_idx] = e3 / eigensum
                        
                except np.linalg.LinAlgError:
                    pass
            
            if (end_idx) % 50000 == 0 or end_idx == n_points:
                print(f"  Processed {end_idx:,}/{n_points:,} points...")
        
        # Compute verticality using normal estimation
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
        normals = np.asarray(pcd.normals)
        
        # Verticality: alignment with Z-axis
        features['verticality'] = np.abs(normals[:, 2])
        
        print("Geometric features computed.")
        return features
    
    def filter_powerline_candidates(self, points: np.ndarray, 
                                     features: dict,
                                     labels: np.ndarray = None) -> np.ndarray:
        """
        Filter points that are likely powerline candidates.
        
        Criteria:
        - Elevated height (above min_height relative to ground, below max_height)
        - High linearity score
        - Low planarity (not wall-like)
        - Not classified as ground, building, etc.
        
        Args:
            points: Nx3 array of points
            features: Dictionary of geometric features
            labels: Optional ground truth labels
            
        Returns:
            Boolean mask for powerline candidate points
        """
        print("Filtering powerline candidates...")
        
        n_points = len(points)
        
        # Compute relative height (height above local minimum)
        # Use percentile to find approximate ground level
        z_values = features['height']
        ground_level = np.percentile(z_values, 5)  # 5th percentile as ground estimate
        relative_height = z_values - ground_level
        
        print(f"Estimated ground level: {ground_level:.2f}m (absolute)")
        print(f"Relative height range: {relative_height.min():.2f}m to {relative_height.max():.2f}m")
        
        # Height filter using relative height
        height_mask = (relative_height >= self.min_height) & \
                      (relative_height <= self.max_height)
        
        # Linearity filter (powerlines are highly linear)
        linearity_mask = features['linearity'] >= self.linearity_threshold
        
        # Low planarity (not building walls)
        planarity_mask = features['planarity'] < 0.5
        
        # Low sphericity (not vegetation clumps)
        sphericity_mask = features['sphericity'] < 0.3
        
        # Combine masks
        candidate_mask = height_mask & linearity_mask & planarity_mask & sphericity_mask
        
        # If we have labels, exclude known non-powerline classes
        if labels is not None:
            # Exclude ground (1), road markings (2), buildings (4), cars (7)
            exclude_mask = np.isin(labels, [1, 2, 4, 7])
            candidate_mask = candidate_mask & ~exclude_mask
        
        n_candidates = candidate_mask.sum()
        print(f"Found {n_candidates:,} powerline candidate points ({100*n_candidates/n_points:.2f}%)")
        
        return candidate_mask
    
    def cluster_powerlines(self, points: np.ndarray) -> np.ndarray:
        """
        Cluster powerline candidate points into individual cables.
        
        Uses DBSCAN clustering which is effective for:
        - Arbitrary cluster shapes (like cables)
        - Handling noise
        - No need to specify number of clusters
        
        Args:
            points: Nx3 array of candidate points
            
        Returns:
            Cluster labels for each point (-1 for noise)
        """
        print("Clustering powerline points...")
        
        if len(points) == 0:
            return np.array([])
        
        # Normalize coordinates for clustering
        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(points)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.dbscan_eps, 
                           min_samples=self.dbscan_min_samples,
                           n_jobs=-1)
        cluster_labels = clustering.fit_predict(points)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = (cluster_labels == -1).sum()
        
        print(f"Found {n_clusters} cable clusters, {n_noise} noise points")
        
        return cluster_labels
    
    def fit_cable_segments(self, points: np.ndarray, 
                           cluster_labels: np.ndarray) -> list:
        """
        Fit line segments to each cable cluster using RANSAC.
        
        For each cluster:
        1. Apply PCA to find main direction
        2. Use RANSAC for robust line fitting
        3. Extract cable endpoints and direction
        
        Args:
            points: Nx3 array of powerline points
            cluster_labels: Cluster label for each point
            
        Returns:
            List of cable segment dictionaries with:
            - start_point: 3D start coordinate
            - end_point: 3D end coordinate
            - direction: Unit direction vector
            - length: Cable segment length
            - points: Points belonging to this segment
        """
        print("Fitting cable segments...")
        
        cable_segments = []
        unique_labels = set(cluster_labels) - {-1}  # Exclude noise
        
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_points = points[cluster_mask]
            
            if len(cluster_points) < 10:
                continue
            
            # Apply PCA to find principal direction
            pca = PCA(n_components=3)
            pca.fit(cluster_points)
            
            # First principal component is the cable direction
            direction = pca.components_[0]
            
            # Ensure direction points in positive X direction
            if direction[0] < 0:
                direction = -direction
            
            # Project points onto principal axis
            centroid = cluster_points.mean(axis=0)
            projections = np.dot(cluster_points - centroid, direction)
            
            # Find endpoints
            min_proj_idx = np.argmin(projections)
            max_proj_idx = np.argmax(projections)
            
            start_point = cluster_points[min_proj_idx]
            end_point = cluster_points[max_proj_idx]
            
            # Compute segment length
            length = np.linalg.norm(end_point - start_point)
            
            # Skip short segments
            if length < self.min_cable_length:
                continue
            
            # Compute explained variance (how well points fit the line)
            explained_variance = pca.explained_variance_ratio_[0]
            
            cable_segment = {
                'cluster_id': label,
                'start_point': start_point,
                'end_point': end_point,
                'direction': direction,
                'length': length,
                'centroid': centroid,
                'n_points': len(cluster_points),
                'explained_variance': explained_variance,
                'points': cluster_points
            }
            
            cable_segments.append(cable_segment)
        
        # Sort by length
        cable_segments.sort(key=lambda x: x['length'], reverse=True)
        
        print(f"Fitted {len(cable_segments)} valid cable segments")
        for i, seg in enumerate(cable_segments[:5]):  # Show top 5
            print(f"  Cable {i+1}: Length={seg['length']:.2f}m, "
                  f"Points={seg['n_points']}, "
                  f"Variance explained={seg['explained_variance']:.2%}")
        
        self.cable_segments = cable_segments
        return cable_segments
    
    def detect(self, filepath: str = None, 
               use_ground_truth: bool = False) -> dict:
        """
        Run the complete powerline detection pipeline.
        
        Args:
            filepath: Path to PLY file (optional if already loaded)
            use_ground_truth: If True, use labeled utility line points
            
        Returns:
            Detection results dictionary
        """
        print("\n" + "="*60)
        print("POWERLINE DETECTION PIPELINE")
        print("="*60 + "\n")
        
        # Step 1: Load data
        if filepath:
            self.load_ply(filepath)
        
        if self.points is None:
            raise ValueError("No point cloud data loaded")
        
        # Option to use ground truth labels
        if use_ground_truth and self.labels is not None:
            print("\nUsing ground truth utility line labels...")
            powerline_mask = self.labels == self.UTILITY_LINE_LABEL
            n_gt = powerline_mask.sum()
            print(f"Ground truth utility line points: {n_gt:,}")
            
            if n_gt > 0:
                powerline_points = self.points[powerline_mask]
            else:
                print("No ground truth utility lines found, using detection...")
                use_ground_truth = False
        
        if not use_ground_truth:
            # Step 2: Downsample for faster processing
            print("\nDownsampling point cloud...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            if self.colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(self.colors)
            
            pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            points_down = np.asarray(pcd_down.points)
            print(f"Downsampled to {len(points_down):,} points")
            
            # Step 3: Compute geometric features
            features = self.compute_geometric_features(points_down)
            
            # Step 4: Filter candidates
            candidate_mask = self.filter_powerline_candidates(
                points_down, features, labels=None
            )
            powerline_points = points_down[candidate_mask]
        
        if len(powerline_points) == 0:
            print("No powerline points detected!")
            return {'success': False, 'message': 'No powerlines detected'}
        
        # Step 5: Cluster powerline points
        cluster_labels = self.cluster_powerlines(powerline_points)
        
        # Step 6: Fit cable segments
        cable_segments = self.fit_cable_segments(powerline_points, cluster_labels)
        
        # Store detection results
        self.detected_powerlines = powerline_points
        
        results = {
            'success': True,
            'n_powerline_points': len(powerline_points),
            'n_cable_segments': len(cable_segments),
            'cable_segments': cable_segments,
            'powerline_points': powerline_points,
            'cluster_labels': cluster_labels
        }
        
        print("\n" + "="*60)
        print("DETECTION COMPLETE")
        print(f"Total powerline points: {len(powerline_points):,}")
        print(f"Cable segments found: {len(cable_segments)}")
        if cable_segments:
            total_length = sum(s['length'] for s in cable_segments)
            print(f"Total cable length: {total_length:.2f}m")
        print("="*60 + "\n")
        
        return results
    
    def visualize(self, show_original: bool = True,
                  show_cables: bool = True,
                  cable_color: list = [1, 0, 0],
                  save_path: str = None) -> None:
        """
        Visualize detection results using Open3D.
        
        Args:
            show_original: Show original point cloud in background
            show_cables: Highlight detected cable segments
            cable_color: RGB color for powerline points
            save_path: If provided, save visualization to image file instead of displaying
        """
        print("Preparing visualization...")
        
        geometries = []
        
        # Original point cloud (faded)
        if show_original and self.points is not None:
            pcd_original = o3d.geometry.PointCloud()
            pcd_original.points = o3d.utility.Vector3dVector(self.points)
            
            if self.colors is not None:
                # Fade colors
                faded_colors = self.colors * 0.3 + 0.2
                pcd_original.colors = o3d.utility.Vector3dVector(faded_colors)
            else:
                pcd_original.paint_uniform_color([0.3, 0.3, 0.3])
            
            geometries.append(pcd_original)
        
        # Detected powerline points
        if show_cables and self.detected_powerlines is not None:
            pcd_powerlines = o3d.geometry.PointCloud()
            pcd_powerlines.points = o3d.utility.Vector3dVector(self.detected_powerlines)
            pcd_powerlines.paint_uniform_color(cable_color)
            geometries.append(pcd_powerlines)
        
        # Draw cable segment lines
        if self.cable_segments:
            for segment in self.cable_segments:
                # Create line from start to end
                line_points = [segment['start_point'], segment['end_point']]
                lines = [[0, 1]]
                
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(line_points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.paint_uniform_color([0, 1, 0])  # Green lines
                geometries.append(line_set)
        
        # Visualize
        if geometries:
            if save_path:
                # Save to image file
                print(f"Saving visualization to {save_path}...")
                try:
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name="Powerline Detection Results", width=1280, height=720)
                    for geom in geometries:
                        vis.add_geometry(geom)
                    vis.poll_events()
                    vis.update_renderer()
                    vis.capture_screen_image(save_path)
                    vis.destroy_window()
                    print(f"Visualization saved to {save_path}")
                except Exception as e:
                    print(f"Screenshot capture failed: {e}")
                    print("Trying alternative visualization method...")
                    # Fallback: try using screenshot with draw_geometries
                    try:
                        print(f"Saving to {save_path}...")
                        o3d.visualization.draw_geometries(
                            geometries,
                            window_name="Powerline Detection Results",
                            width=1280,
                            height=720,
                            left=0,
                            top=0
                        )
                        print(f"Visualization saved (check system screenshot)")
                    except Exception as e2:
                        print(f"All visualization methods failed: {e2}")
            else:
                # Interactive display
                print("Opening visualization window...")
                print("Controls: Left-click + drag to rotate, scroll to zoom, right-click + drag to pan")
                try:
                    o3d.visualization.draw_geometries(
                        geometries,
                        window_name="Powerline Detection Results",
                        width=1280,
                        height=720
                    )
                except Exception as e:
                    print(f"Interactive visualization failed: {e}")
        else:
            print("No geometries to visualize")
    
    def save_html_visualization(self, output_path: str) -> None:
        """
        Save interactive 3D visualization as HTML file using Plotly with proper RGB colors.
        
        Args:
            output_path: Path for output HTML file
        """
        if not HAS_PLOTLY:
            print("Plotly not available. Install with: pip install plotly")
            return
        
        print(f"Creating interactive HTML visualization with RGB colors...")
        
        # Prepare traces
        traces = []
        
        # Add original point cloud with actual RGB colors
        if self.points is not None:
            # Sample points for performance but preserve color distribution
            sample_size = min(100000, len(self.points))
            if sample_size < len(self.points):
                # Stratified sampling to preserve color distribution
                sample_indices = np.random.choice(len(self.points), sample_size, replace=False)
            else:
                sample_indices = np.arange(len(self.points))
            
            points_sample = self.points[sample_indices]
            
            if self.colors is not None:
                # Convert normalized colors (0-1) back to 0-255 for RGB display
                colors_sample = self.colors[sample_indices]
                colors_rgb = (colors_sample * 255).astype(np.uint8)
                marker_color = [f'rgb({r},{g},{b})' for r,g,b in colors_rgb]
            else:
                marker_color = 'rgb(100, 100, 100)'
            
            trace_original = go.Scatter3d(
                x=points_sample[:, 0],
                y=points_sample[:, 1],
                z=points_sample[:, 2],
                mode='markers',
                name='Point Cloud (RGB)',
                marker=dict(
                    size=1.5,
                    color=marker_color if isinstance(marker_color, list) else 'rgb(100, 100, 100)',
                    opacity=0.7
                ),
                text=[f'Point {i}' for i in range(len(points_sample))],
                hovertemplate='<b>Point</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            )
            traces.append(trace_original)
        
        # Add detected powerlines in red
        if self.detected_powerlines is not None:
            trace_powerlines = go.Scatter3d(
                x=self.detected_powerlines[:, 0],
                y=self.detected_powerlines[:, 1],
                z=self.detected_powerlines[:, 2],
                mode='markers',
                name='Powerlines (Detected)',
                marker=dict(
                    size=2.5,
                    color='rgb(255, 0, 0)',
                    opacity=0.9
                ),
                hovertemplate='<b>Powerline Point</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            )
            traces.append(trace_powerlines)
        
        # Add cable segment lines
        if self.cable_segments:
            for i, segment in enumerate(self.cable_segments):
                start = segment['start_point']
                end = segment['end_point']
                trace_line = go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode='lines',
                    name=f'Cable {i+1} ({segment["length"]:.1f}m)',
                    line=dict(color='rgb(0, 255, 0)', width=4),
                    hovertemplate='<b>Cable %{name}</b><extra></extra>'
                )
                traces.append(trace_line)
        
        # Create figure
        fig = go.Figure(data=traces)
        
        # Update layout with better styling
        fig.update_layout(
            title={
                'text': 'Powerline Detection Results - RGB Point Cloud Visualization',
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(
                xaxis=dict(
                    title='X (m)',
                    backgroundcolor="rgb(240, 240, 240)",
                    gridcolor="white",
                    showbackground=True,
                ),
                yaxis=dict(
                    title='Y (m)',
                    backgroundcolor="rgb(240, 240, 240)",
                    gridcolor="white",
                    showbackground=True,
                ),
                zaxis=dict(
                    title='Z (m)',
                    backgroundcolor="rgb(240, 240, 240)",
                    gridcolor="white",
                    showbackground=True,
                ),
                aspectmode='data'
            ),
            width=1400,
            height=800,
            hovermode='closest',
            showlegend=True,
            font=dict(size=12)
        )
        
        # Save
        fig.write_html(output_path)
        print(f"Interactive RGB visualization saved to {output_path}")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        print(f"Open this file in a web browser to explore the 3D model interactively")
    
    def save_results(self, output_path: str) -> None:
        """
        Save detected powerline points to a PLY file.
        
        Args:
            output_path: Path for output PLY file
        """
        if self.detected_powerlines is None:
            print("No results to save")
            return
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.detected_powerlines)
        pcd.paint_uniform_color([1, 0, 0])  # Red color for powerlines
        
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved {len(self.detected_powerlines):,} powerline points to {output_path}")
    
    def evaluate(self) -> dict:
        """
        Evaluate detection against ground truth labels.
        
        Returns:
            Dictionary with precision, recall, F1 score
        """
        if self.labels is None:
            print("No ground truth labels available")
            return {}
        
        if self.detected_powerlines is None:
            print("No detection results available")
            return {}
        
        # Ground truth powerline points
        gt_powerline_mask = self.labels == self.UTILITY_LINE_LABEL
        gt_powerline_points = self.points[gt_powerline_mask]
        
        if len(gt_powerline_points) == 0:
            print("No ground truth powerline points")
            return {}
        
        # Build KD-tree for ground truth
        gt_tree = KDTree(gt_powerline_points)
        
        # For each detected point, check if it's near a ground truth point
        distances, _ = gt_tree.query(self.detected_powerlines)
        
        # Threshold for considering a match (in meters)
        match_threshold = 0.5
        true_positives = (distances < match_threshold).sum()
        false_positives = len(self.detected_powerlines) - true_positives
        false_negatives = len(gt_powerline_points) - true_positives
        
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        print("\n" + "="*40)
        print("EVALUATION RESULTS")
        print("="*40)
        print(f"Ground truth powerline points: {len(gt_powerline_points):,}")
        print(f"Detected powerline points: {len(self.detected_powerlines):,}")
        print(f"True positives: {true_positives:,}")
        print(f"False positives: {false_positives:,}")
        print(f"False negatives: {false_negatives:,}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("="*40 + "\n")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }


def main():
    """Main function to run powerline detection."""
    
    # Set up paths
    data_dir = Path(__file__).parent / "Toronto_3D"
    
    # Available PLY files
    ply_files = list(data_dir.glob("L*.ply"))
    
    if not ply_files:
        print("No PLY files found in the directory!")
        print(f"Please place Toronto-3D PLY files in: {data_dir}")
        return None, None
    
    print(f"Found {len(ply_files)} PLY files:")
    for f in ply_files:
        print(f"  - {f.name}")
    
    # Use the first available PLY file
    input_file = str(ply_files[0])
    print(f"\nProcessing: {ply_files[0].name}")
    
    # Initialize detector with tuned parameters
    detector = PowerlineDetector(
        min_height=3.0,           # Powerlines are typically above 3m
        max_height=25.0,          # And below 25m
        voxel_size=0.1,           # 10cm voxel downsampling
        neighbor_radius=0.5,      # 50cm neighborhood
        linearity_threshold=0.6,  # High linearity for cables
        dbscan_eps=0.8,           # DBSCAN clustering distance
        dbscan_min_samples=5,     # Minimum points per cluster
        min_cable_length=3.0      # Minimum 3m cable length
    )
    
    # Run detection
    # Set use_ground_truth=True to use labeled utility line points
    # Set use_ground_truth=False to detect using geometric features
    results = detector.detect(input_file, use_ground_truth=True)
    
    if results['success']:
        # Evaluate against ground truth
        detector.evaluate()
        
        # Save results
        output_file = str(data_dir / "detected_powerlines.ply")
        detector.save_results(output_file)
        
        # Save HTML visualization (works in headless mode)
        try:
            html_path = str(data_dir / "powerline_visualization.html")
            detector.save_html_visualization(html_path)
        except Exception as e:
            print(f"HTML visualization failed: {e}")
    
    return detector, results


if __name__ == "__main__":
    detector, results = main()
