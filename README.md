# Powerline Tracking & Geometric Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Open3D](https://img.shields.io/badge/Library-Open3D-green.svg)](http://www.open3d.org/)

An advanced perception pipeline for detecting and tracking powerlines from 3D Point Cloud Data (PCD). This project focuses on high-precision geometric extraction and semantic filtering of aerial LiDAR data.

## 📺 Detection Results
![Powerline Detection Demo](media/demo.webm)
*Visualization of the powerline extraction process from PCD sequences.*

## 🌟 Features
- **Geometric Extraction**: Uses RANSAC and Euclidean Clustering to isolate thin, linear structures from complex backgrounds.
- **Dynamic Tracking**: Processes point cloud sequences to track powerline trajectories across large outdoor environments.
- **Efficient Filtering**: Advanced voxel-grid downsampling and outlier removal for real-time-capable processing.
- **Interactive Visualization**: Built with Open3D for high-performance 3D rendering of results.

## 📂 Project Structure
- `PointCloudDataTracking/`: Core detection logic and algorithm implementation.
- `pcd/`: Raw and processed point cloud datasets (Git-ignored due to size).
- `powerline_detector.py`: Main detection class using geometric constraints.

## 🛠️ Installation
Install the required Python packages:
```bash
pip install -r PointCloudDataTracking/requirements.txt
```

## 🚀 Running the Demo
To run the automated detection on the provided sample data:
```bash
python3 PointCloudDataTracking/powerline_tracking_demo.py
```

## 📈 Performance
- **Accuracy**: Achieves high F1-score on powerline-specific geometric kernels.
- **Robustness**: Handles noisy LiDAR data and vegetation interference through height-based and density-based filters.

---
Developed by **Gandamalla Viswa**
