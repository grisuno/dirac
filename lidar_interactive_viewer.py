#!/usr/bin/env python3
"""
Interactive 3D LiDAR Point Cloud Viewer

Creates an interactive HTML visualization for navigating weight space point clouds.
Supports rotation, zoom, pan, and point selection.

Usage:
    python3 lidar_interactive_viewer.py your_point_cloud.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_csv_point_cloud(csv_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load point cloud from CSV file.
    
    Returns:
        points: Nx3 array of coordinates
        attributes: dict of additional attributes (intensity, range, etc.)
    """
    print(f"[Loading] {csv_path}")
    
    columns = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        
        # Initialize lists
        for col in fieldnames:
            columns[col] = []
        
        for row in reader:
            for col in fieldnames:
                columns[col].append(float(row[col]))
    
    # Convert to numpy
    for col in columns:
        columns[col] = np.array(columns[col])
    
    # Extract XYZ
    if not all(c in columns for c in ['x', 'y', 'z']):
        raise ValueError("CSV must have x, y, z columns")
    
    points = np.column_stack([columns['x'], columns['y'], columns['z']])
    
    # Remove XYZ from attributes
    attributes = {k: v for k, v in columns.items() if k not in ['x', 'y', 'z']}
    
    print(f"[Loaded] {len(points):,} points")
    
    return points, attributes


def generate_interactive_html(
    points: np.ndarray,
    attributes: Dict[str, np.ndarray],
    output_path: str,
    title: str = "Weight Space LiDAR",
    point_size: float = 2.0,
    colormap: str = "Viridis",
    intensity_col: str = "intensity",
) -> str:
    """
    Generate interactive HTML using Three.js for 3D navigation.
    """
    
    # Normalize intensity for coloring
    if intensity_col in attributes:
        intensity = attributes[intensity_col]
    else:
        intensity = np.linalg.norm(points, axis=1)
    
    # Normalize to 0-1
    i_min, i_max = intensity.min(), intensity.max()
    if i_max > i_min:
        intensity_norm = (intensity - i_min) / (i_max - i_min)
    else:
        intensity_norm = np.ones_like(intensity) * 0.5
    
    # Convert points to JSON
    points_json = json.dumps(points.tolist())
    intensity_json = json.dumps(intensity_norm.tolist())
    
    # Calculate bounds for camera positioning
    center = points.mean(axis=0)
    extent = points.max(axis=0) - points.min(axis=0)
    max_extent = extent.max()
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #eee;
            overflow: hidden;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
        }}
        #info-panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 8px;
            font-size: 12px;
            max-width: 280px;
            z-index: 100;
        }}
        #info-panel h2 {{
            color: #00d4ff;
            margin-bottom: 10px;
            font-size: 16px;
        }}
        #info-panel p {{
            margin: 5px 0;
            color: #aaa;
        }}
        #info-panel .stat {{
            color: #0f0;
            font-family: monospace;
        }}
        #controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 8px;
            z-index: 100;
        }}
        #controls label {{
            display: block;
            margin: 8px 0;
            font-size: 12px;
        }}
        #controls input[type="range"] {{
            width: 150px;
            vertical-align: middle;
        }}
        #controls select, #controls button {{
            width: 100%;
            padding: 5px;
            margin: 5px 0;
            background: #333;
            color: #fff;
            border: 1px solid #555;
            border-radius: 4px;
        }}
        #controls button {{
            cursor: pointer;
            background: #0066cc;
            border: none;
            padding: 8px;
        }}
        #controls button:hover {{
            background: #0088ff;
        }}
        #crosshair {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
            z-index: 50;
        }}
        #crosshair::before, #crosshair::after {{
            content: '';
            position: absolute;
            background: rgba(255,255,255,0.3);
        }}
        #crosshair::before {{
            width: 20px;
            height: 1px;
            left: -10px;
            top: 0;
        }}
        #crosshair::after {{
            width: 1px;
            height: 20px;
            left: 0;
            top: -10px;
        }}
        #tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.9);
            color: #0f0;
            padding: 8px 12px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 11px;
            pointer-events: none;
            display: none;
            z-index: 200;
        }}
        #colorbar {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            padding: 10px;
            border-radius: 8px;
            z-index: 100;
        }}
        #colorbar-gradient {{
            width: 20px;
            height: 150px;
            background: linear-gradient(to top, #440154, #31688e, #35b779, #fde725);
            border-radius: 4px;
        }}
        #colorbar-labels {{
            position: absolute;
            left: 30px;
            top: 0;
            height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            font-size: 10px;
            color: #aaa;
        }}
        .keyboard-hint {{
            background: #333;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 10px;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    
    <div id="info-panel">
        <h2>Weight Space LiDAR</h2>
        <p>Points: <span class="stat" id="point-count">{len(points):,}</span></p>
        <p>Center: <span class="stat" id="center-info">({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})</span></p>
        <p>Extent: <span class="stat" id="extent-info">{max_extent:.3f}</span></p>
        <hr style="margin: 10px 0; border-color: #444;">
        <p><span class="keyboard-hint">LMB</span> Rotate</p>
        <p><span class="keyboard-hint">RMB</span> Pan</p>
        <p><span class="keyboard-hint">Scroll</span> Zoom</p>
        <p><span class="keyboard-hint">R</span> Reset View</p>
        <p><span class="keyboard-hint">A</span> Auto Rotate</p>
    </div>
    
    <div id="controls">
        <label>Point Size: <input type="range" id="point-size" min="0.5" max="10" step="0.5" value="{point_size}"></label>
        <label>Opacity: <input type="range" id="opacity" min="0.1" max="1" step="0.1" value="0.8"></label>
        <label>Colormap:
            <select id="colormap">
                <option value="viridis">Viridis</option>
                <option value="plasma">Plasma</option>
                <option value="inferno">Inferno</option>
                <option value="magma">Magma</option>
                <option value="cividis">Cividis</option>
                <option value="rainbow">Rainbow</option>
            </select>
        </label>
        <label>Point Shape:
            <select id="point-shape">
                <option value="circle">Circle</option>
                <option value="square">Square</option>
            </select>
        </label>
        <button id="reset-btn">Reset View</button>
        <button id="top-view">Top View (XY)</button>
        <button id="front-view">Front View (XZ)</button>
        <button id="side-view">Side View (YZ)</button>
    </div>
    
    <div id="crosshair"></div>
    <div id="tooltip"></div>
    
    <div id="colorbar">
        <div style="position: relative;">
            <div id="colorbar-gradient"></div>
            <div id="colorbar-labels">
                <span>{i_max:.4f}</span>
                <span>{(i_max+i_min)/2:.4f}</span>
                <span>{i_min:.4f}</span>
            </div>
        </div>
        <div style="text-align: center; font-size: 10px; margin-top: 5px;">Intensity</div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        // Data
        const pointsData = {points_json};
        const intensityData = {intensity_json};
        
        // Three.js setup
        let scene, camera, renderer, controls, pointCloud;
        let autoRotate = false;
        
        const colorMaps = {{
            viridis: [[0.267, 0.004, 0.329], [0.282, 0.140, 0.458], [0.253, 0.265, 0.529], [0.206, 0.371, 0.553], [0.163, 0.471, 0.558], [0.127, 0.566, 0.550], [0.134, 0.658, 0.517], [0.266, 0.746, 0.440], [0.477, 0.821, 0.318], [0.741, 0.873, 0.150], [0.993, 0.906, 0.144]],
            plasma: [[0.050, 0.030, 0.528], [0.291, 0.148, 0.516], [0.485, 0.204, 0.405], [0.656, 0.266, 0.322], [0.804, 0.350, 0.269], [0.897, 0.452, 0.224], [0.948, 0.576, 0.163], [0.972, 0.722, 0.140], [0.972, 0.873, 0.132], [0.940, 0.975, 0.131]],
            inferno: [[0.001, 0.000, 0.014], [0.122, 0.047, 0.282], [0.281, 0.087, 0.433], [0.448, 0.106, 0.446], [0.604, 0.131, 0.398], [0.745, 0.198, 0.307], [0.863, 0.319, 0.213], [0.942, 0.488, 0.159], [0.979, 0.684, 0.084], [0.988, 0.998, 0.645]],
            magma: [[0.001, 0.000, 0.014], [0.101, 0.035, 0.276], [0.208, 0.068, 0.445], [0.336, 0.092, 0.508], [0.476, 0.114, 0.504], [0.613, 0.153, 0.447], [0.742, 0.227, 0.361], [0.854, 0.340, 0.262], [0.941, 0.497, 0.184], [0.988, 0.998, 0.645]],
            cividis: [[0.000, 0.126, 0.301], [0.075, 0.210, 0.402], [0.161, 0.292, 0.459], [0.254, 0.370, 0.492], [0.353, 0.446, 0.497], [0.462, 0.519, 0.476], [0.579, 0.588, 0.439], [0.704, 0.654, 0.387], [0.836, 0.715, 0.330], [0.976, 0.983, 0.320]],
            rainbow: [[0.278, 0.000, 0.490], [0.000, 0.000, 1.000], [0.000, 1.000, 1.000], [0.000, 1.000, 0.000], [1.000, 1.000, 0.000], [1.000, 0.000, 0.000], [0.500, 0.000, 0.500]]
        }};
        
        function getColorFromMap(value, mapName) {{
            const map = colorMaps[mapName] || colorMaps.viridis;
            const idx = Math.min(Math.floor(value * (map.length - 1)), map.length - 1);
            return map[Math.max(0, idx)];
        }}
        
        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);
            
            // Camera
            const extent = {max_extent};
            camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, extent * 100);
            camera.position.set({center[0]}, {center[1]}, {center[2]} + extent * 2);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set({center[0]}, {center[1]}, {center[2]});
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = true;
            controls.minDistance = extent * 0.1;
            controls.maxDistance = extent * 50;
            
            // Point cloud
            createPointCloud();
            
            // Grid helper
            const gridHelper = new THREE.GridHelper(extent * 3, 30, 0x444444, 0x333333);
            gridHelper.position.set({center[0]}, {center[1]} - extent/2, {center[2]});
            scene.add(gridHelper);
            
            // Axes helper
            const axesHelper = new THREE.AxesHelper(extent);
            axesHelper.position.set({center[0]} - extent, {center[1]} - extent/2, {center[2]} - extent);
            scene.add(axesHelper);
            
            // Event listeners
            setupEventListeners();
            
            // Start animation
            animate();
        }}
        
        function createPointCloud() {{
            if (pointCloud) {{
                scene.remove(pointCloud);
                pointCloud.geometry.dispose();
                pointCloud.material.dispose();
            }}
            
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(pointsData.length * 3);
            const colors = new Float32Array(pointsData.length * 3);
            
            const currentColormap = document.getElementById('colormap').value;
            
            for (let i = 0; i < pointsData.length; i++) {{
                positions[i * 3] = pointsData[i][0];
                positions[i * 3 + 1] = pointsData[i][1];
                positions[i * 3 + 2] = pointsData[i][2];
                
                const color = getColorFromMap(intensityData[i], currentColormap);
                colors[i * 3] = color[0];
                colors[i * 3 + 1] = color[1];
                colors[i * 3 + 2] = color[2];
            }}
            
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            
            const pointSize = parseFloat(document.getElementById('point-size').value);
            const opacity = parseFloat(document.getElementById('opacity').value);
            const pointShape = document.getElementById('point-shape').value;
            
            const material = new THREE.PointsMaterial({{
                size: pointSize,
                vertexColors: true,
                transparent: true,
                opacity: opacity,
                sizeAttenuation: true,
                map: createPointTexture(pointShape),
                alphaTest: 0.01
            }});
            
            pointCloud = new THREE.Points(geometry, material);
            scene.add(pointCloud);
        }}
        
        function createPointTexture(shape) {{
            const canvas = document.createElement('canvas');
            canvas.width = 32;
            canvas.height = 32;
            const ctx = canvas.getContext('2d');
            
            const gradient = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
            gradient.addColorStop(0, 'rgba(255,255,255,1)');
            gradient.addColorStop(0.5, 'rgba(255,255,255,0.8)');
            gradient.addColorStop(1, 'rgba(255,255,255,0)');
            
            ctx.fillStyle = gradient;
            if (shape === 'circle') {{
                ctx.beginPath();
                ctx.arc(16, 16, 16, 0, Math.PI * 2);
                ctx.fill();
            }} else {{
                ctx.fillRect(4, 4, 24, 24);
            }}
            
            const texture = new THREE.CanvasTexture(canvas);
            return texture;
        }}
        
        function setupEventListeners() {{
            window.addEventListener('resize', onWindowResize);
            document.addEventListener('keydown', onKeyDown);
            
            document.getElementById('point-size').addEventListener('input', createPointCloud);
            document.getElementById('opacity').addEventListener('input', createPointCloud);
            document.getElementById('colormap').addEventListener('change', () => {{
                createPointCloud();
                updateColorbar();
            }});
            document.getElementById('point-shape').addEventListener('change', createPointCloud);
            
            document.getElementById('reset-btn').addEventListener('click', resetView);
            document.getElementById('top-view').addEventListener('click', () => setView('top'));
            document.getElementById('front-view').addEventListener('click', () => setView('front'));
            document.getElementById('side-view').addEventListener('click', () => setView('side'));
            
            // Tooltip on hover
            renderer.domElement.addEventListener('mousemove', onMouseMove);
        }}
        
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}
        
        function onKeyDown(event) {{
            switch(event.key.toLowerCase()) {{
                case 'r':
                    resetView();
                    break;
                case 'a':
                    autoRotate = !autoRotate;
                    controls.autoRotate = autoRotate;
                    controls.autoRotateSpeed = 2.0;
                    break;
            }}
        }}
        
        function resetView() {{
            const extent = {max_extent};
            camera.position.set({center[0]}, {center[1]}, {center[2]} + extent * 2);
            controls.target.set({center[0]}, {center[1]}, {center[2]});
            controls.update();
        }}
        
        function setView(view) {{
            const extent = {max_extent};
            const center = [{center[0]}, {center[1]}, {center[2]}];
            
            switch(view) {{
                case 'top':
                    camera.position.set(center[0], center[1] + extent * 2, center[2]);
                    break;
                case 'front':
                    camera.position.set(center[0], center[1], center[2] + extent * 2);
                    break;
                case 'side':
                    camera.position.set(center[0] + extent * 2, center[1], center[2]);
                    break;
            }}
            controls.target.set(...center);
            controls.update();
        }}
        
        function updateColorbar() {{
            const currentColormap = document.getElementById('colormap').value;
            const gradientMap = {{
                viridis: 'linear-gradient(to top, #440154, #31688e, #35b779, #fde725)',
                plasma: 'linear-gradient(to top, #0d0887, #cc4778, #f89540, #f0f921)',
                inferno: 'linear-gradient(to top, #000004, #57106e, #bc3754, #f98e09, #fcffa4)',
                magma: 'linear-gradient(to top, #000004, #51127c, #b63679, #fb8861, #fcfdbf)',
                cividis: 'linear-gradient(to top, #00204d, #414287, #7c7b78, #bcab60, #fde725)',
                rainbow: 'linear-gradient(to top, #47006f, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000, #800080)'
            }};
            document.getElementById('colorbar-gradient').style.background = gradientMap[currentColormap] || gradientMap.viridis;
        }}
        
        function onMouseMove(event) {{
            const mouse = new THREE.Vector2(
                (event.clientX / window.innerWidth) * 2 - 1,
                -(event.clientY / window.innerHeight) * 2 + 1
            );
            
            const raycaster = new THREE.Raycaster();
            raycaster.setFromCamera(mouse, camera);
            
            if (pointCloud) {{
                const intersects = raycaster.intersectObject(pointCloud);
                const tooltip = document.getElementById('tooltip');
                
                if (intersects.length > 0) {{
                    const idx = intersects[0].index;
                    const point = pointsData[idx];
                    const intensity = intensityData[idx];
                    
                    tooltip.innerHTML = `Point #${{idx}}<br>` +
                        `X: ${{point[0].toFixed(6)}}<br>` +
                        `Y: ${{point[1].toFixed(6)}}<br>` +
                        `Z: ${{point[2].toFixed(6)}}<br>` +
                        `Intensity: ${{intensity.toFixed(6)}}`;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 15) + 'px';
                    tooltip.style.top = (event.clientY + 15) + 'px';
                }} else {{
                    tooltip.style.display = 'none';
                }}
            }}
        }}
        
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        
        // Initialize
        init();
    </script>
</body>
</html>'''
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"[Created] {output_path}")
    return output_path


def generate_plotly_html(
    points: np.ndarray,
    attributes: Dict[str, np.ndarray],
    output_path: str,
    title: str = "Weight Space LiDAR",
) -> str:
    """
    Generate interactive HTML using Plotly.js (alternative viewer).
    Sometimes more compatible with large point clouds.
    """
    
    # Normalize intensity
    if 'intensity' in attributes:
        intensity = attributes['intensity']
    else:
        intensity = np.linalg.norm(points, axis=1)
    
    # Sample if too large
    max_points = 50000
    if len(points) > max_points:
        print(f"[Sampling] Reducing from {len(points):,} to {max_points:,} points for web performance")
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        intensity = intensity[indices]
    
    points_json = json.dumps(points.tolist())
    intensity_json = json.dumps(intensity.tolist())
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #plot {{ width: 100vw; height: 100vh; }}
    </style>
</head>
<body>
    <div id="plot"></div>
    <script>
        const points = {points_json};
        const intensity = {intensity_json};
        
        const x = points.map(p => p[0]);
        const y = points.map(p => p[1]);
        const z = points.map(p => p[2]);
        
        const trace = {{
            x: x,
            y: y,
            z: z,
            mode: 'markers',
            marker: {{
                size: 2,
                color: intensity,
                colorscale: 'Viridis',
                opacity: 0.8,
                colorbar: {{
                    title: 'Intensity'
                }}
            }},
            type: 'scatter3d',
            hovertemplate: 'X: %{{x:.4f}}<br>Y: %{{y:.4f}}<br>Z: %{{z:.4f}}<extra></extra>'
        }};
        
        const layout = {{
            title: '{title}',
            autosize: true,
            scene: {{
                aspectmode: 'data',
                xaxis: {{ title: 'X' }},
                yaxis: {{ title: 'Y' }},
                zaxis: {{ title: 'Z' }},
                camera: {{
                    eye: {{ x: 1.5, y: 1.5, z: 1.5 }}
                }}
            }},
            margin: {{ l: 0, r: 0, b: 0, t: 30 }}
        }};
        
        Plotly.newPlot('plot', [trace], layout);
    </script>
</body>
</html>'''
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"[Created] {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Create interactive 3D visualization for LiDAR point cloud CSV'
    )
    parser.add_argument('csv_file', type=str, help='Path to point cloud CSV file')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output HTML path (default: same as input with .html)')
    parser.add_argument('-t', '--title', type=str, default='Weight Space LiDAR',
                        help='Title for visualization')
    parser.add_argument('-s', '--point-size', type=float, default=2.0,
                        help='Point size (default: 2.0)')
    parser.add_argument('--plotly', action='store_true',
                        help='Use Plotly.js instead of Three.js (better for large clouds)')
    
    args = parser.parse_args()
    
    # Load data
    points, attributes = load_csv_point_cloud(args.csv_file)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.csv_file.rsplit('.', 1)[0] + '_interactive.html'
    
    # Generate visualization
    if args.plotly:
        generate_plotly_html(points, attributes, output_path, args.title)
    else:
        generate_interactive_html(
            points, attributes, output_path, args.title,
            point_size=args.point_size
        )
    
    print(f"\n[Done] Open in browser: file://{Path(output_path).absolute()}")


if __name__ == '__main__':
    main()
