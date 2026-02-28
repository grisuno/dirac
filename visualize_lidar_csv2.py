#!/usr/bin/env python3
"""
Quick visualization script for Weight Space LiDAR CSV output.
Usage: python3 visualize_lidar_csv.py your_point_cloud.csv
"""

import argparse
import sys
import csv
import numpy as np
from pathlib import Path


def visualize_csv(csv_path: str, output_path: str = None, colormap: str = "viridis"):
    """
    Visualize a point cloud CSV file.
    
    CSV must have columns: x, y, z (and optionally: intensity, range)
    """
    print(f"Loading: {csv_path}")
    
    # Read CSV
    x, y, z, intensity = [], [], [], []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Check required columns
        fieldnames = reader.fieldnames or []
        required = ['x', 'y', 'z']
        
        missing = [c for c in required if c not in fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        
        has_intensity = 'intensity' in fieldnames
        
        for row in reader:
            x.append(float(row['x']))
            y.append(float(row['y']))
            z.append(float(row['z']))
            if has_intensity:
                intensity.append(float(row['intensity']))
    
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    intensity = np.array(intensity) if intensity else np.ones(len(x))
    
    print(f"Loaded {len(x):,} points")
    
    # Import matplotlib
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("ERROR: matplotlib required. Install with: pip install matplotlib")
        sys.exit(1)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot
    scatter = ax.scatter(
        x, y, z,
        c=intensity,
        cmap=colormap,
        s=1,
        alpha=0.7,
    )
    
    ax.set_xlabel('X (Weight Dimension 1)', fontsize=10)
    ax.set_ylabel('Y (Weight Dimension 2)', fontsize=10)
    ax.set_zlabel('Z (Weight Dimension 3)', fontsize=10)
    ax.set_title(f'Weight Space LiDAR - {Path(csv_path).stem}', fontsize=12)
    
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Intensity', fontsize=10)
    
    # Stats text
    stats_text = (
        f"Points: {len(x):,}\n"
        f"X range: [{x.min():.3f}, {x.max():.3f}]\n"
        f"Y range: [{y.min():.3f}, {y.max():.3f}]\n"
        f"Z range: [{z.min():.3f}, {z.max():.3f}]"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=8, family='monospace',
             verticalalignment='bottom', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        # Auto-generate output path
        out_path = csv_path.rsplit('.', 1)[0] + '_visualization.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
    
    plt.close()
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Weight Space LiDAR CSV point cloud'
    )
    parser.add_argument('csv_file', type=str, help='Path to CSV file')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output PNG path (default: same as input with _visualization.png)')
    parser.add_argument('-c', '--colormap', type=str, default='viridis',
                        help='Matplotlib colormap (default: viridis)')
    
    args = parser.parse_args()
    
    visualize_csv(args.csv_file, args.output, args.colormap)


if __name__ == '__main__':
    main()
