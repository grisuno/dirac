#!/usr/bin/env python3
"""
Standard 3D Weight Space Visualization

Typical approach without "LiDAR physics":
- Direct PCA/t-SNE reduction
- Layer-based coloring
- Per-layer statistics
- Simple distance metrics

Usage:
    python3 weight_3d_standard.py checkpoint.pth
    python3 weight_3d_standard.py checkpoint.pth --method tsne
    python3 weight_3d_standard.py checkpoint.pth --per-layer
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class StandardWeightVisualizer:
    """
    Standard 3D weight visualization without LiDAR physics.
    
    Each point represents:
    - Per-layer mode: one layer (weights flattened)
    - Per-neuron mode: one neuron/filter
    - Sliding window mode: consecutive weight chunks
    """
    
    def __init__(
        self,
        max_samples: int = 50000,
        random_seed: int = 42,
    ):
        self.max_samples = max_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load PyTorch checkpoint."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        return torch.load(path, map_location="cpu", weights_only=False)
    
    def extract_weights_per_layer(
        self,
        checkpoint: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], List[str], List[Dict]]:
        """
        Extract weights organized by layer.
        
        Returns:
            weights: List of flattened weight arrays per layer
            names: Layer names
            stats: Statistics per layer
        """
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        weights = []
        names = []
        stats = []
        
        for name, tensor in state_dict.items():
            # Only weight tensors
            if not any(kw in name.lower() for kw in ["weight", "kernel", "bias"]):
                continue
            
            if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
                arr = tensor.detach().cpu().numpy()
            else:
                arr = np.array(tensor)
            
            # Flatten
            flat = arr.flatten()
            
            weights.append(flat)
            names.append(name)
            
            # Compute statistics
            stats.append({
                "shape": arr.shape,
                "num_params": len(flat),
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat)),
                "min": float(np.min(flat)),
                "max": float(np.max(flat)),
                "abs_mean": float(np.mean(np.abs(flat))),
                "sparsity": float(np.sum(np.abs(flat) < 1e-6) / len(flat)),
            })
        
        print(f"Extracted {len(weights)} layers")
        return weights, names, stats
    
    def extract_weights_per_neuron(
        self,
        checkpoint: Dict[str, Any],
        max_neurons: int = 10000,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Extract weights organized per neuron/filter.
        
        Each row = one neuron's incoming weights.
        """
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        neurons = []
        layer_ids = []
        
        for name, tensor in state_dict.items():
            if "weight" not in name.lower():
                continue
            
            if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
                arr = tensor.detach().cpu().numpy()
            else:
                arr = np.array(tensor)
            
            # For 2D weights (Linear): each row is a neuron
            # For 4D weights (Conv2d): each filter is a neuron
            if arr.ndim == 2:
                # Linear: (out_features, in_features)
                for i in range(arr.shape[0]):
                    neurons.append(arr[i])
                    layer_ids.append(name)
            
            elif arr.ndim == 4:
                # Conv2d: (out_channels, in_channels, H, W)
                for i in range(arr.shape[0]):
                    neurons.append(arr[i].flatten())
                    layer_ids.append(name)
            
            elif arr.ndim == 1:
                # Bias or 1D: treat as single neuron
                neurons.append(arr)
                layer_ids.append(name)
        
        if not neurons:
            raise ValueError("No neurons found")
        
        # Pad to same length
        max_len = max(len(n) for n in neurons)
        padded = np.zeros((len(neurons), max_len), dtype=np.float32)
        for i, n in enumerate(neurons):
            padded[i, :len(n)] = n
        
        # Sample if too many
        if len(padded) > max_neurons:
            idx = np.random.choice(len(padded), max_neurons, replace=False)
            padded = padded[idx]
            layer_ids = [layer_ids[i] for i in idx]
        
        # Create layer index array
        unique_layers = list(set(layer_ids))
        layer_to_idx = {l: i for i, l in enumerate(unique_layers)}
        layer_indices = np.array([layer_to_idx[l] for l in layer_ids])
        
        print(f"Extracted {len(padded)} neurons from {len(unique_layers)} layers")
        return padded, unique_layers, layer_indices
    
    def extract_weights_sliding_window(
        self,
        checkpoint: Dict[str, Any],
        window_size: int = 256,
        num_windows: int = 5000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract weights using sliding window approach.
        
        Each point = consecutive chunk of weights.
        """
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        # Concatenate all weights
        all_weights = []
        for name, tensor in state_dict.items():
            if "weight" in name.lower() or "kernel" in name.lower():
                if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
                    all_weights.append(tensor.detach().cpu().numpy().flatten())
                else:
                    all_weights.append(np.array(tensor).flatten())
        
        concatenated = np.concatenate(all_weights)
        print(f"Total parameters: {len(concatenated):,}")
        
        # Sample windows
        actual_windows = min(num_windows, len(concatenated) // window_size)
        windows = np.zeros((actual_windows, window_size), dtype=np.float32)
        positions = np.zeros(actual_windows, dtype=np.float32)
        
        step = (len(concatenated) - window_size) / max(1, actual_windows - 1)
        
        for i in range(actual_windows):
            start = int(i * step)
            windows[i] = concatenated[start:start + window_size]
            positions[i] = start / len(concatenated)  # Relative position
        
        print(f"Created {actual_windows} windows of size {window_size}")
        return windows, positions
    
    def reduce_dimensions(
        self,
        data: np.ndarray,
        method: str = "pca",
        n_components: int = 3,
    ) -> np.ndarray:
        """Apply dimensionality reduction."""
        if not SKLEARN_AVAILABLE:
            return self._simple_projection(data)
        
        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        if method == "tsne":
            perplexity = min(30, data.shape[0] - 1)
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=self.random_seed,
                max_iter=1000,
            )
        else:  # PCA
            reducer = PCA(n_components=n_components, random_state=self.random_seed)
        
        reduced = reducer.fit_transform(data_scaled)
        
        if method == "pca":
            explained = np.sum(reducer.explained_variance_ratio_)
            print(f"PCA explained variance: {explained:.4f}")
        
        return reduced.astype(np.float32)
    
    def _simple_projection(self, data: np.ndarray) -> np.ndarray:
        """Fallback projection without sklearn."""
        n = data.shape[0]
        projected = np.zeros((n, 3), dtype=np.float32)
        projected[:, 0] = np.mean(data, axis=1)
        projected[:, 1] = np.std(data, axis=1)
        projected[:, 2] = np.linalg.norm(data, axis=1)
        return projected


def generate_standard_html(
    coordinates: np.ndarray,
    colors: np.ndarray,
    labels: List[str],
    output_path: str,
    title: str = "Weight Space 3D",
    hover_data: Optional[np.ndarray] = None,
) -> str:
    """
    Generate interactive 3D visualization using Plotly.
    Standard approach - simple and effective.
    """
    
    # Sample if too large
    max_points = 30000
    if len(coordinates) > max_points:
        print(f"Sampling from {len(coordinates):,} to {max_points:,} points")
        idx = np.random.choice(len(coordinates), max_points, replace=False)
        coordinates = coordinates[idx]
        colors = colors[idx]
        labels = [labels[i] if isinstance(colors[i], (int, float)) or colors[i] < len(labels) else labels[0] for i in range(len(colors))]
        if hover_data is not None:
            hover_data = hover_data[idx]
    
    # Convert to list for JSON
    x = coordinates[:, 0].tolist()
    y = coordinates[:, 1].tolist()
    z = coordinates[:, 2].tolist()
    colors_list = colors.tolist()
    
    # Create unique labels for legend
    unique_labels = list(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    color_indices = [label_to_idx[l] for l in labels]
    
    # Generate hover text
    if hover_data is not None:
        hover_text = [
            f"Point {i}<br>Value: {hover_data[i]:.4f}"
            for i in range(len(x))
        ]
    else:
        hover_text = [f"Point {i}" for i in range(len(x))]
    
    labels_json = json.dumps(unique_labels)
    color_indices_json = json.dumps(color_indices)
    x_json = json.dumps(x)
    y_json = json.dumps(y)
    z_json = json.dumps(z)
    hover_json = json.dumps(hover_text)
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #plot {{ width: 100vw; height: 100vh; }}
        #info {{ position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.8); 
                 color: white; padding: 15px; border-radius: 8px; font-size: 12px; z-index: 1000; }}
    </style>
</head>
<body>
    <div id="info">
        <strong>{title}</strong><br>
        Points: <span id="count">{len(x):,}</span><br>
        Method: Standard PCA/t-SNE<br>
        <hr>
        Controls:<br>
        - Drag to rotate<br>
        - Scroll to zoom<br>
        - Double-click to reset
    </div>
    <div id="plot"></div>
    <script>
        const labels = {labels_json};
        const colorIndices = {color_indices_json};
        const x = {x_json};
        const y = {y_json};
        const z = {z_json};
        const hoverText = {hover_json};
        
        // Color palette
        const colors = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
            '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
            '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'
        ];
        
        // Create traces per label
        const traces = [];
        for (let i = 0; i < labels.length; i++) {{
            const idx = colorIndices.map((c, j) => c === i ? j : -1).filter(j => j >= 0);
            
            traces.push({{
                name: labels[i],
                x: idx.map(j => x[j]),
                y: idx.map(j => y[j]),
                z: idx.map(j => z[j]),
                text: idx.map(j => hoverText[j]),
                mode: 'markers',
                type: 'scatter3d',
                marker: {{
                    size: 3,
                    color: colors[i % colors.length],
                    opacity: 0.7
                }},
                hoverinfo: 'text+z'
            }});
        }}
        
        const layout = {{
            title: '',
            autosize: true,
            showlegend: true,
            scene: {{
                aspectmode: 'data',
                xaxis: {{ title: 'PC1', gridcolor: '#333' }},
                yaxis: {{ title: 'PC2', gridcolor: '#333' }},
                zaxis: {{ title: 'PC3', gridcolor: '#333' }},
                bgcolor: '#1a1a2e'
            }},
            paper_bgcolor: '#1a1a2e',
            plot_bgcolor: '#1a1a2e',
            legend: {{
                font: {{ color: '#fff' }},
                bgcolor: 'rgba(0,0,0,0.5)'
            }},
            margin: {{ l: 0, r: 0, b: 0, t: 0 }}
        }};
        
        const config = {{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['toImage', 'sendDataToCloud']
        }};
        
        Plotly.newPlot('plot', traces, layout, config);
    </script>
</body>
</html>'''
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Created: {output_path}")
    return output_path


def generate_continuous_html(
    coordinates: np.ndarray,
    color_values: np.ndarray,
    output_path: str,
    title: str = "Weight Space 3D",
    colorbar_title: str = "Value",
) -> str:
    """
    Generate visualization with continuous color scale.
    """
    
    max_points = 30000
    if len(coordinates) > max_points:
        idx = np.random.choice(len(coordinates), max_points, replace=False)
        coordinates = coordinates[idx]
        color_values = color_values[idx]
    
    x_json = json.dumps(coordinates[:, 0].tolist())
    y_json = json.dumps(coordinates[:, 1].tolist())
    z_json = json.dumps(coordinates[:, 2].tolist())
    c_json = json.dumps(color_values.tolist())
    
    html = f'''<!DOCTYPE html>
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
        const trace = {{
            x: {x_json},
            y: {y_json},
            z: {z_json},
            mode: 'markers',
            type: 'scatter3d',
            marker: {{
                size: 2,
                color: {c_json},
                colorscale: 'Viridis',
                colorbar: {{ title: '{colorbar_title}' }},
                opacity: 0.8
            }},
            hovertemplate: 'X: %{{x:.4f}}<br>Y: %{{y:.4f}}<br>Z: %{{z:.4f}}<br>{colorbar_title}: %{{marker.color:.4f}}<extra></extra>'
        }};
        
        const layout = {{
            autosize: true,
            scene: {{
                aspectmode: 'data',
                xaxis: {{ title: 'PC1', gridcolor: '#444' }},
                yaxis: {{ title: 'PC2', gridcolor: '#444' }},
                zaxis: {{ title: 'PC3', gridcolor: '#444' }},
                bgcolor: '#0d1117'
            }},
            paper_bgcolor: '#0d1117',
            margin: {{ l: 0, r: 0, b: 0, t: 0 }}
        }};
        
        Plotly.newPlot('plot', [trace], layout, {{ responsive: true }});
    </script>
</body>
</html>'''
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Created: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Standard 3D Weight Space Visualization'
    )
    parser.add_argument('checkpoint', type=str, help='Path to .pth checkpoint')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output HTML path')
    parser.add_argument('-m', '--method', type=str, default='pca',
                        choices=['pca', 'tsne'],
                        help='Dimensionality reduction method')
    parser.add_argument('--per-layer', action='store_true',
                        help='One point per layer (default: sliding window)')
    parser.add_argument('--per-neuron', action='store_true',
                        help='One point per neuron/filter')
    parser.add_argument('--window-size', type=int, default=256,
                        help='Sliding window size')
    parser.add_argument('--num-windows', type=int, default=5000,
                        help='Number of sliding windows')
    
    args = parser.parse_args()
    
    # Initialize
    viz = StandardWeightVisualizer()
    
    # Load checkpoint
    print(f"Loading: {args.checkpoint}")
    checkpoint = viz.load_checkpoint(args.checkpoint)
    
    # Extract weights based on mode
    if args.per_layer:
        print("\n[Mode: Per-Layer]")
        weights, names, stats = viz.extract_weights_per_layer(checkpoint)
        
        # Pad to same length
        max_len = max(len(w) for w in weights)
        padded = np.zeros((len(weights), max_len), dtype=np.float32)
        for i, w in enumerate(weights):
            padded[i, :len(w)] = w
        
        # Reduce
        coords = viz.reduce_dimensions(padded, args.method)
        
        # Use layer names as labels
        labels = names
        
        generate_standard_html(
            coords, 
            np.arange(len(names)),
            labels,
            args.output or args.checkpoint.replace('.pth', '_per_layer.html'),
            f"Weight Space - Per Layer ({args.method.upper()})"
        )
    
    elif args.per_neuron:
        print("\n[Mode: Per-Neuron]")
        neurons, layer_names, layer_indices = viz.extract_weights_per_neuron(checkpoint)
        
        coords = viz.reduce_dimensions(neurons, args.method)
        
        labels = [layer_names[i] for i in layer_indices]
        
        generate_standard_html(
            coords,
            layer_indices,
            layer_names,
            args.output or args.checkpoint.replace('.pth', '_per_neuron.html'),
            f"Weight Space - Per Neuron ({args.method.upper()})"
        )
    
    else:
        print("\n[Mode: Sliding Window]")
        windows, positions = viz.extract_weights_sliding_window(
            checkpoint,
            args.window_size,
            args.num_windows
        )
        
        coords = viz.reduce_dimensions(windows, args.method)
        
        generate_continuous_html(
            coords,
            positions,
            args.output or args.checkpoint.replace('.pth', '_standard.html'),
            f"Weight Space - Sliding Window ({args.method.upper()})",
            "Position in Network"
        )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
