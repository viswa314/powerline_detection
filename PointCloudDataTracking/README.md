# Toronto-3D Powerline Detection and Tracking

This folder contains scripts and utilities to detect and track powerline cables
in the Toronto-3D LiDAR point cloud dataset. The implementation uses Open3D
for point cloud I/O and visualization, NumPy/SciPy for numeric ops, and
scikit-learn for clustering and PCA.

**Key Files**
- **Source:** [Toronto_3D/powerline_detector.py](Toronto_3D/powerline_detector.py) : Main detector class `PowerlineDetector` implementing the full pipeline (load, feature extraction, candidate filtering, clustering, line fitting, evaluation, visualization, save results).
- **Demo:** [Toronto_3D/powerline_tracking_demo.py](Toronto_3D/powerline_tracking_demo.py) : Example script showing a faster demo-mode workflow (ground-truth mode and geometric detection mode), clustering, tracking and analysis utilities.
- **Visualizer:** [Toronto_3D/visualize_saved_results.py](Toronto_3D/visualize_saved_results.py) : Load `detected_powerlines.ply` (saved output) and visualize it alongside the original cloud.
- **Classes:** [Toronto_3D/Mavericks_classes_9.txt](Toronto_3D/Mavericks_classes_9.txt) : Label mapping used by the Toronto-3D dataset.
- **Dependencies:** [Toronto_3D/requirements.txt](Toronto_3D/requirements.txt) : Python packages required to run the code.

**Overview — How It Works**
- **Loading:** The scripts read Toronto-3D PLY files using a custom binary parser that expects the dataset's vertex layout (x,y,z doubles; r,g,b bytes; intensity, gpstime, scanangle, label floats). A UTM offset (approx. `[627285, 4842000, 0]`) is subtracted to reduce numeric magnitudes.
- **Preprocessing:** Point clouds are optionally voxel-downsampled (`voxel_size` parameter) to speed up processing.
- **Geometric Features:** For each point (or downsampled subset) local geometric features are computed using neighborhood covariance and PCA: linearity, planarity, sphericity, curvature and verticality (via estimated normals).
- **Candidate Filtering:** Points are filtered by relative height (above an estimated ground percentile), high linearity, low planarity/sphericity, and optional label-based exclusion to form powerline candidates.
- **Clustering:** DBSCAN groups candidate points into cable clusters. Parameters `eps` and `min_samples` are configurable.
- **Line Fitting / Tracking:** For each cluster, PCA estimates the main axis and endpoints are derived by projecting points onto that axis. Short segments (below `min_cable_length`) are discarded. Results include segment start/end, direction, length, explained variance and constituent points.
- **Evaluation:** If ground-truth labels are loaded, detected points are matched (KD-tree nearest neighbor) to label=5 (`Utility_line`) points to compute precision, recall and F1 score.
- **Visualization & Saving:** Results can be visualized with Open3D (`PowerlineDetector.visualize()` or the demo visualizer). Detected powerlines can be saved to `detected_powerlines.ply` with `save_results()` and later reloaded with `visualize_saved_results.py`.

**How to Run**
Prerequisites: create and activate a Python environment, then install dependencies from `requirements.txt`.

Windows (PowerShell) example:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r Toronto_3D\requirements.txt
```

Quick run (uses the first `L*.ply` file in the folder):
```powershell
python Toronto_3D\powerline_tracking_demo.py
# or run the full pipeline
python Toronto_3D\powerline_detector.py
```

Notes:
- The scripts expect Toronto-3D PLY files in the `Toronto_3D` folder (files named like `L001.ply`).
- Visualization requires a GUI display. When running headless (remote server), skip visualization or use off-screen rendering.

**Tunable Parameters**
- `PowerlineDetector` constructor: `min_height`, `max_height`, `voxel_size`, `neighbor_radius`, `linearity_threshold`, `dbscan_eps`, `dbscan_min_samples`, `min_cable_length`.
- Demo script clustering: `eps` and `min_samples` for `DBSCAN`, `min_length` for segment filtering.

**Tips for improving results**
- Adjust `voxel_size` to balance speed vs. detail (smaller yields better detail but slower).
- Increase `k_neighbors` when computing geometric features for more robust PCA estimates on dense scans.
- Tune `linearity_threshold` and `dbscan_eps` when cable spacing or point density changes.
- If ground truth labels are available, use `use_ground_truth=True` (in the demo or detector) as a fast, accurate baseline.

**Outputs**
- `detected_powerlines.ply` — point cloud containing detected powerline points (red in visualizations).
- Console logs with statistics: number of candidate points, clusters found, fitted cable segments, evaluation metrics (precision/recall/F1 when ground truth used).

**Where to look in the code**
- Detection and pipeline orchestration: [Toronto_3D/powerline_detector.py](Toronto_3D/powerline_detector.py)
- Fast demo and analysis helper: [Toronto_3D/powerline_tracking_demo.py](Toronto_3D/powerline_tracking_demo.py)
- Saved-results visualization: [Toronto_3D/visualize_saved_results.py](Toronto_3D/visualize_saved_results.py)

**License & Author**
The code in this folder was authored as part of a research/demo project. No explicit license file is included — add one if you intend to redistribute.

**Next steps you might want**
- Add a small example PLY (or a subset) for quick testing.
- Add a `README_QUICK_START.md` with smaller commands for headless servers.
- Add unit tests for data loading and geometric feature computations.

---
Generated README for quick onboarding and reproducibility.
