#!/usr/bin/env python3
"""
WeightSpaceLiDAR: Synthetic LiDAR for Neural Network Weight Space Navigation

A production-grade implementation applying real LiDAR mathematics to neural network
weight space analysis. Enables 3D/4D navigation through .pth checkpoint files with
temporal evolution visualization.

Mathematical Foundation:
    The LiDAR equation P(r) = P_t * η * A_r * ρ / (π * R²) * exp(-2γR) is adapted
    to weight space where:
    - R: distance in weight space (Frobenius norm between weight tensors)
    - β: Hessian curvature coefficient (local loss landscape sensitivity)
    - τ: optical depth (accumulated gradient integral along trajectory)
    - T: transmission (gradient flow preservation through layers)

Author: Synthetic LiDAR Research
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
from scipy import linalg
from scipy.linalg import eigh, expm
from scipy.spatial.distance import cdist, pdist, squareform

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass(frozen=True)
class WeightSpaceLiDARConfig:
    """
    Master configuration for Weight Space LiDAR system.
    All parameters are immutable and type-safe.
    """

    SPEED_OF_LIGHT_WEIGHT_SPACE: float = 1.0
    REFRACTIVE_INDEX_WEIGHT_SPACE: float = 1.0

    LASER_ENERGY_JOULES: float = 1.0
    RECEIVER_AREA_M2: float = 1.0
    TRANSMISSION_EFFICIENCY: float = 0.85
    TARGET_REFLECTIVITY: float = 0.5

    EXTINCTION_COEFFICIENT_BASE: float = 0.01
    BACKSCATTER_COEFFICIENT_BASE: float = 0.1
    MULTIPLE_SCATTERING_FACTOR: float = 0.05
    BACKGROUND_NOISE_LEVEL: float = 1e-6

    TIME_OF_FLIGHT_BINS: int = 1024
    MAX_RANGE_WEIGHT_SPACE: float = 100.0
    RANGE_RESOLUTION: float = 0.01
    AZIMUTH_RESOLUTION_RAD: float = 0.01745
    ELEVATION_RESOLUTION_RAD: float = 0.01745

    HORIZONTAL_FOV_RAD: float = 6.28319
    VERTICAL_FOV_RAD: float = 3.14159
    AZIMUTH_STEPS: int = 360
    ELEVATION_STEPS: int = 180

    WAVELENGTH_NM: float = 905.0
    PULSE_DURATION_NS: float = 10.0
    PULSE_REPETITION_FREQ_HZ: float = 10000.0

    WEIGHT_SAMPLING_STRATEGY: str = "stratified"
    WEIGHT_SAMPLE_RATIO: float = 0.1
    MAX_WEIGHT_DIMENSION: int = 10000
    LAYER_SAMPLING_WEIGHTS: Tuple[float, ...] = field(default_factory=lambda: (1.0,))

    PCA_COMPONENTS: int = 3
    TSNE_PERPLEXITY: float = 30.0
    TSNE_LEARNING_RATE: float = 200.0
    TSNE_N_ITER: int = 1000
    UMAP_N_NEIGHBORS: int = 15
    UMAP_MIN_DIST: float = 0.1

    TEMPORAL_INTERPOLATION_METHOD: str = "linear"
    TEMPORAL_SMOOTHING_WINDOW: int = 5
    TEMPORAL_DECAY_FACTOR: float = 0.95
    CHECKPOINT_SORTING_METHOD: str = "epoch"

    HESSIAN_SAMPLE_SIZE: int = 1000
    HESSIAN_EPSILON: float = 1e-5
    GRADIENT_SAMPLE_BATCHES: int = 5
    CURVATURE_ESTIMATION_METHOD: str = "finite_difference"

    POINT_CLOUD_DENSITY: int = 10000
    VOXEL_SIZE: float = 0.1
    OUTLIER_REMOVAL_THRESHOLD: float = 3.0
    POINT_NORMAL_ESTIMATION_K: int = 10

    INTENSITY_NORMALIZATION: str = "minmax"
    RANGE_NORMALIZATION: str = "unit"
    COORDINATE_SYSTEM: str = "cartesian"

    OUTPUT_FORMAT: str = "las"
    OUTPUT_COMPRESSION: bool = True
    OUTPUT_PRECISION: int = 6
    METADATA_EMBEDDING: bool = True

    DEVICE: str = "cuda" if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available() else "cpu"
    DTYPE: str = "float32"
    RANDOM_SEED: int = 42
    LOG_LEVEL: str = "INFO"
    NUM_WORKERS: int = 4
    BATCH_SIZE: int = 32

    FIGURE_DPI: int = 150
    FIGURE_SIZE_X: int = 12
    FIGURE_SIZE_Y: int = 10
    COLORMAP: str = "viridis"
    RENDERER: str = "matplotlib"

    MIN_VALID_RANGE: float = 1e-10
    MAX_VALID_RANGE: float = 1e10
    NUMERICAL_PRECISION: float = 1e-12
    CONVERGENCE_TOLERANCE: float = 1e-8
    MAX_ITERATIONS: int = 1000

    CHECKPOINT_FILE_PATTERN: str = "*.pth"
    SAVE_INTERMEDIATE_RESULTS: bool = True
    OUTPUT_DIRECTORY: str = "lidar_output"
    TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"


class ILogger(Protocol):
    """Protocol for logger implementations."""

    def debug(self, msg: str) -> None: ...
    def info(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...


class LoggerFactory:
    """Factory for creating configured logger instances."""

    @staticmethod
    def create(name: str, level: str = "INFO") -> ILogger:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


@runtime_checkable
class IWeightExtractor(Protocol):
    """Protocol for weight extraction strategies."""

    def extract(self, checkpoint: Dict[str, Any]) -> np.ndarray:
        """Extract weight vector from checkpoint."""
        ...

    def get_layer_names(self, checkpoint: Dict[str, Any]) -> List[str]:
        """Get list of layer names from checkpoint."""
        ...


@runtime_checkable
class IDimensionalityReducer(Protocol):
    """Protocol for dimensionality reduction strategies."""

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data to lower dimensions."""
        ...

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform new data using fitted model."""
        ...


@runtime_checkable
class IRangeCalculator(Protocol):
    """Protocol for range calculation strategies."""

    def calculate(self, origin: np.ndarray, target: np.ndarray) -> float:
        """Calculate range between two points in weight space."""
        ...


@runtime_checkable
class ITransmissionCalculator(Protocol):
    """Protocol for transmission calculation strategies."""

    def calculate(self, path_integral: float, extinction: float) -> float:
        """Calculate transmission along a path."""
        ...


@runtime_checkable
class IPointCloudGenerator(Protocol):
    """Protocol for point cloud generation strategies."""

    def generate(
        self,
        weights: np.ndarray,
        intensities: np.ndarray,
        ranges: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Generate point cloud from weight data."""
        ...


class DefaultWeightExtractor:
    """Default implementation for weight extraction from PyTorch checkpoints."""

    def __init__(self, config: WeightSpaceLiDARConfig):
        self._config = config
        self._logger = LoggerFactory.create("DefaultWeightExtractor", config.LOG_LEVEL)

    def extract(self, checkpoint: Dict[str, Any]) -> np.ndarray:
        state_dict = self._resolve_state_dict(checkpoint)
        weight_tensors = []

        for name, tensor in state_dict.items():
            if self._is_weight_tensor(name, tensor):
                flattened = self._flatten_and_sample(tensor)
                weight_tensors.append(flattened)

        if not weight_tensors:
            raise ValueError("No weight tensors found in checkpoint")

        return np.concatenate(weight_tensors)

    def get_layer_names(self, checkpoint: Dict[str, Any]) -> List[str]:
        state_dict = self._resolve_state_dict(checkpoint)
        return [
            name for name, tensor in state_dict.items()
            if self._is_weight_tensor(name, tensor)
        ]

    def _resolve_state_dict(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint

    def _is_weight_tensor(self, name: str, tensor: Any) -> bool:
        if not TORCH_AVAILABLE or not isinstance(tensor, Tensor):
            return False
        weight_keywords = ("weight", "kernel", "bias")
        return any(kw in name.lower() for kw in weight_keywords)

    def _flatten_and_sample(self, tensor: Any) -> np.ndarray:
        flattened = tensor.detach().cpu().numpy().flatten()

        if len(flattened) > self._config.MAX_WEIGHT_DIMENSION:
            sample_size = int(len(flattened) * self._config.WEIGHT_SAMPLE_RATIO)
            sample_size = max(sample_size, self._config.MAX_WEIGHT_DIMENSION)
            indices = np.random.choice(len(flattened), sample_size, replace=False)
            return flattened[indices]

        return flattened


class PCAReducer:
    """PCA-based dimensionality reduction for weight space."""

    def __init__(self, config: WeightSpaceLiDARConfig):
        self._config = config
        self._model: Optional[PCA] = None
        self._logger = LoggerFactory.create("PCAReducer", config.LOG_LEVEL)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for PCA reduction")

        n_components = min(self._config.PCA_COMPONENTS, data.shape[1], data.shape[0])
        self._model = PCA(n_components=n_components)

        reduced = self._model.fit_transform(data)
        explained_variance = np.sum(self._model.explained_variance_ratio_)

        self._logger.info(
            f"PCA reduction: {data.shape[1]} -> {n_components} dimensions, "
            f"explained variance: {explained_variance:.4f}"
        )

        return reduced

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit_transform first.")
        return self._model.transform(data)

    def get_explained_variance(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.explained_variance_ratio_


class TSNEReducer:
    """t-SNE based dimensionality reduction for weight space visualization."""

    def __init__(self, config: WeightSpaceLiDARConfig):
        self._config = config
        self._model: Optional[TSNE] = None
        self._logger = LoggerFactory.create("TSNEReducer", config.LOG_LEVEL)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for t-SNE reduction")

        n_components = min(self._config.PCA_COMPONENTS, 3)

        self._model = TSNE(
            n_components=n_components,
            perplexity=min(self._config.TSNE_PERPLEXITY, data.shape[0] - 1),
            learning_rate=self._config.TSNE_LEARNING_RATE,
            n_iter=self._config.TSNE_N_ITER,
            random_state=self._config.RANDOM_SEED,
        )

        reduced = self._model.fit_transform(data)

        self._logger.info(
            f"t-SNE reduction: {data.shape[1]} -> {n_components} dimensions"
        )

        return reduced

    def transform(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("t-SNE does not support transform on new data")


class FrobeniusRangeCalculator:
    """Calculate range using Frobenius norm in weight space."""

    def __init__(self, config: WeightSpaceLiDARConfig):
        self._config = config
        self._scale_factor = config.SPEED_OF_LIGHT_WEIGHT_SPACE / (
            2.0 * config.REFRACTIVE_INDEX_WEIGHT_SPACE
        )

    def calculate(self, origin: np.ndarray, target: np.ndarray) -> float:
        diff = target - origin
        euclidean_dist = np.linalg.norm(diff)

        range_value = euclidean_dist * self._scale_factor

        return float(np.clip(
            range_value,
            self._config.MIN_VALID_RANGE,
            self._config.MAX_VALID_RANGE
        ))

    def calculate_batch(
        self,
        origin: np.ndarray,
        targets: np.ndarray
    ) -> np.ndarray:
        diff = targets - origin
        distances = np.linalg.norm(diff, axis=1)
        ranges = distances * self._scale_factor

        return np.clip(
            ranges,
            self._config.MIN_VALID_RANGE,
            self._config.MAX_VALID_RANGE
        )


class BeerLambertTransmissionCalculator:
    """Calculate transmission using Beer-Lambert law adapted for weight space."""

    def __init__(self, config: WeightSpaceLiDARConfig):
        self._config = config
        self._extinction_base = config.EXTINCTION_COEFFICIENT_BASE

    def calculate(self, path_integral: float, extinction: float) -> float:
        effective_extinction = self._extinction_base * (1.0 + extinction)
        transmission = np.exp(-2.0 * effective_extinction * path_integral)

        return float(np.clip(transmission, 0.0, 1.0))

    def calculate_optical_depth(
        self,
        gradients: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        if len(gradients) < 2:
            return 0.0

        gradient_norms = np.linalg.norm(gradients, axis=1)
        weight_norms = np.linalg.norm(weights, axis=1) + self._config.NUMERICAL_PRECISION

        normalized_gradients = gradient_norms / weight_norms

        optical_depth = np.trapz(normalized_gradients)

        return float(optical_depth)


class HessianCurvatureEstimator:
    """Estimate local curvature (backscatter coefficient) using Hessian approximation."""

    def __init__(self, config: WeightSpaceLiDARConfig):
        self._config = config
        self._epsilon = config.HESSIAN_EPSILON
        self._logger = LoggerFactory.create("HessianCurvatureEstimator", config.LOG_LEVEL)

    def estimate(
        self,
        weights: np.ndarray,
        loss_fn: Optional[Callable] = None,
    ) -> float:
        n_dims = min(weights.shape[0], self._config.HESSIAN_SAMPLE_SIZE)

        if n_dims < 2:
            return self._config.BACKSCATTER_COEFFICIENT_BASE

        subsample_indices = np.random.choice(
            weights.shape[0],
            size=n_dims,
            replace=False
        )
        w_sub = weights[subsample_indices]

        hessian_diag = self._estimate_hessian_diagonal(w_sub, loss_fn)

        curvature = np.mean(np.abs(hessian_diag))

        return float(curvature * self._config.BACKSCATTER_COEFFICIENT_BASE)

    def _estimate_hessian_diagonal(
        self,
        weights: np.ndarray,
        loss_fn: Optional[Callable],
    ) -> np.ndarray:
        if loss_fn is not None and TORCH_AVAILABLE:
            return self._numerical_hessian_diag(weights, loss_fn)

        return self._empirical_curvature_estimate(weights)

    def _numerical_hessian_diag(
        self,
        weights: np.ndarray,
        loss_fn: Callable,
    ) -> np.ndarray:
        if not TORCH_AVAILABLE:
            return self._empirical_curvature_estimate(weights)

        w_tensor = torch.from_numpy(weights).requires_grad_(True)
        loss = loss_fn(w_tensor)

        grad = torch.autograd.grad(loss, w_tensor, create_graph=True)[0]

        hessian_diag = torch.zeros_like(w_tensor)
        for i in range(len(w_tensor)):
            grad_i = grad[i]
            hessian_diag[i] = torch.autograd.grad(
                grad_i, w_tensor, retain_graph=True
            )[0][i]

        return hessian_diag.detach().numpy()

    def _empirical_curvature_estimate(self, weights: np.ndarray) -> np.ndarray:
        sorted_weights = np.sort(weights)
        second_diff = np.diff(sorted_weights, n=2)

        if len(second_diff) == 0:
            return np.ones(len(weights)) * self._config.BACKSCATTER_COEFFICIENT_BASE

        mean_curvature = np.mean(np.abs(second_diff))

        return np.abs(weights - np.mean(weights)) * mean_curvature


class LiDARPhysicsEngine:
    """
    Core LiDAR physics engine adapted for weight space analysis.

    Implements the full LiDAR equation:
    P(r) = (E_L * c / 2) * A * [β_a * P_a + β_m * P_m] * exp(-2∫σ(r')dr') / r² + M(r) + b

    Adapted for weight space where:
    - E_L: laser energy -> probing intensity
    - β: backscatter coefficient -> Hessian curvature
    - σ: extinction coefficient -> gradient magnitude
    - r: range -> weight space distance
    """

    def __init__(self, config: WeightSpaceLiDARConfig):
        self._config = config
        self._range_calc = FrobeniusRangeCalculator(config)
        self._transmission_calc = BeerLambertTransmissionCalculator(config)
        self._curvature_est = HessianCurvatureEstimator(config)
        self._logger = LoggerFactory.create("LiDARPhysicsEngine", config.LOG_LEVEL)

    def compute_return_signal(
        self,
        origin_weights: np.ndarray,
        target_weights: np.ndarray,
        hessian_estimate: Optional[float] = None,
        gradient_integral: float = 0.0,
    ) -> Dict[str, float]:
        range_value = self._range_calc.calculate(origin_weights, target_weights)

        if hessian_estimate is None:
            hessian_estimate = self._curvature_est.estimate(target_weights)

        backscatter = self._compute_backscatter(hessian_estimate, range_value)

        transmission = self._transmission_calc.calculate(
            range_value,
            gradient_integral
        )

        geometric_factor = self._compute_geometric_factor(range_value)

        power_received = self._compute_received_power(
            backscatter,
            transmission,
            geometric_factor,
            range_value
        )

        intensity = self._compute_intensity(power_received, range_value)

        return {
            "range": range_value,
            "intensity": intensity,
            "backscatter": backscatter,
            "transmission": transmission,
            "power_received": power_received,
            "geometric_factor": geometric_factor,
        }

    def compute_point_cloud(
        self,
        origin_weights: np.ndarray,
        weight_matrix: np.ndarray,
        reduction_result: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        n_points = weight_matrix.shape[0]

        ranges = self._range_calc.calculate_batch(origin_weights, weight_matrix)

        azimuth = np.arctan2(reduction_result[:, 1], reduction_result[:, 0])
        elevation = np.arctan2(
            reduction_result[:, 2],
            np.sqrt(reduction_result[:, 0]**2 + reduction_result[:, 1]**2)
        )

        intensities = self._compute_intensity_field(
            weight_matrix,
            ranges,
            origin_weights
        )

        x = ranges * np.cos(elevation) * np.cos(azimuth)
        y = ranges * np.cos(elevation) * np.sin(azimuth)
        z = ranges * np.sin(elevation)

        return {
            "x": x,
            "y": y,
            "z": z,
            "range": ranges,
            "intensity": intensities,
            "azimuth": azimuth,
            "elevation": elevation,
        }

    def _compute_backscatter(
        self,
        hessian_estimate: float,
        range_value: float,
    ) -> float:
        base_backscatter = self._config.BACKSCATTER_COEFFICIENT_BASE

        curvature_factor = 1.0 + np.abs(hessian_estimate)

        range_factor = 1.0 / (1.0 + range_value * self._config.EXTINCTION_COEFFICIENT_BASE)

        return base_backscatter * curvature_factor * range_factor

    def _compute_geometric_factor(self, range_value: float) -> float:
        r_safe = max(range_value, self._config.NUMERICAL_PRECISION)

        solid_angle = self._config.RECEIVER_AREA_M2 / (r_safe ** 2)

        return solid_angle * self._config.TRANSMISSION_EFFICIENCY

    def _compute_received_power(
        self,
        backscatter: float,
        transmission: float,
        geometric_factor: float,
        range_value: float,
    ) -> float:
        transmitted_power = self._config.LASER_ENERGY_JOULES * self._config.SPEED_OF_LIGHT_WEIGHT_SPACE / 2.0

        reflected_power = transmitted_power * self._config.TARGET_REFLECTIVITY * backscatter

        received_power = reflected_power * geometric_factor * transmission

        noise = self._config.BACKGROUND_NOISE_LEVEL * np.random.randn()

        multiple_scatter = self._config.MULTIPLE_SCATTERING_FACTOR * backscatter

        return float(max(received_power + noise + multiple_scatter, 0.0))

    def _compute_intensity(
        self,
        power_received: float,
        range_value: float,
    ) -> float:
        r_safe = max(range_value, self._config.NUMERICAL_PRECISION)

        intensity = power_received / (r_safe ** 2)

        if self._config.INTENSITY_NORMALIZATION == "minmax":
            intensity = np.clip(intensity, 0.0, 1.0)
        elif self._config.INTENSITY_NORMALIZATION == "log":
            intensity = np.log1p(intensity)

        return float(intensity)

    def _compute_intensity_field(
        self,
        weight_matrix: np.ndarray,
        ranges: np.ndarray,
        origin: np.ndarray,
    ) -> np.ndarray:
        weight_norms = np.linalg.norm(weight_matrix, axis=1)
        origin_norm = np.linalg.norm(origin) + self._config.NUMERICAL_PRECISION

        relative_magnitude = weight_norms / origin_norm

        range_factor = 1.0 / (1.0 + ranges / self._config.MAX_RANGE_WEIGHT_SPACE)

        intensities = relative_magnitude * range_factor

        intensities = (intensities - intensities.min()) / (
            intensities.max() - intensities.min() + self._config.NUMERICAL_PRECISION
        )

        return intensities


class TemporalCheckpointScanner:
    """
    Scanner for temporal evolution of checkpoints.
    Provides 4D visualization (3D space + time) of weight space dynamics.
    """

    def __init__(self, config: WeightSpaceLiDARConfig):
        self._config = config
        self._weight_extractor = DefaultWeightExtractor(config)
        self._physics_engine = LiDARPhysicsEngine(config)
        self._logger = LoggerFactory.create("TemporalCheckpointScanner", config.LOG_LEVEL)

    def scan_directory(
        self,
        checkpoint_dir: str,
        sort_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        checkpoint_files = self._find_checkpoint_files(checkpoint_dir)

        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

        sort_method = sort_by or self._config.CHECKPOINT_SORTING_METHOD
        checkpoint_files = self._sort_checkpoints(checkpoint_files, sort_method)

        temporal_data = self._extract_temporal_weights(checkpoint_files)

        temporal_signals = self._compute_temporal_signals(temporal_data)

        trajectories = self._compute_trajectories(temporal_data)

        return {
            "checkpoints": checkpoint_files,
            "weights": temporal_data["weights"],
            "timestamps": temporal_data["timestamps"],
            "epochs": temporal_data["epochs"],
            "signals": temporal_signals,
            "trajectories": trajectories,
            "metadata": {
                "num_checkpoints": len(checkpoint_files),
                "scan_timestamp": datetime.now().isoformat(),
                "config": self._config,
            }
        }

    def _find_checkpoint_files(self, directory: str) -> List[str]:
        pattern = self._config.CHECKPOINT_FILE_PATTERN
        checkpoint_files = list(Path(directory).glob(pattern))

        return [str(f) for f in checkpoint_files]

    def _sort_checkpoints(
        self,
        files: List[str],
        method: str,
    ) -> List[str]:
        if method == "epoch":
            return sorted(files, key=self._extract_epoch)
        elif method == "modified":
            return sorted(files, key=lambda f: os.path.getmtime(f))
        elif method == "name":
            return sorted(files)
        else:
            return files

    def _extract_epoch(self, filepath: str) -> int:
        import re
        match = re.search(r"epoch[_]?(\d+)", filepath)
        if match:
            return int(match.group(1))
        match = re.search(r"_(\d+)\.pth$", filepath)
        if match:
            return int(match.group(1))
        return 0

    def _extract_temporal_weights(
        self,
        checkpoint_files: List[str],
    ) -> Dict[str, Any]:
        weights_list = []
        timestamps = []
        epochs = []

        for filepath in checkpoint_files:
            try:
                checkpoint = self._load_checkpoint(filepath)
                weights = self._weight_extractor.extract(checkpoint)
                weights_list.append(weights)

                timestamp = os.path.getmtime(filepath)
                timestamps.append(timestamp)

                epoch = self._extract_epoch(filepath)
                epochs.append(epoch)

            except Exception as e:
                self._logger.warning(f"Failed to load {filepath}: {e}")

        if not weights_list:
            raise ValueError("No valid checkpoints could be loaded")

        return {
            "weights": np.array(weights_list),
            "timestamps": np.array(timestamps),
            "epochs": np.array(epochs),
        }

    def _load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for checkpoint loading")

        return torch.load(filepath, map_location="cpu", weights_only=False)

    def _compute_temporal_signals(
        self,
        temporal_data: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        weights = temporal_data["weights"]
        n_checkpoints = len(weights)

        ranges = np.zeros(n_checkpoints)
        velocities = np.zeros(n_checkpoints)
        accelerations = np.zeros(n_checkpoints)

        origin = weights[0]

        for i, w in enumerate(weights):
            signal = self._physics_engine.compute_return_signal(origin, w)
            ranges[i] = signal["range"]

        for i in range(1, n_checkpoints):
            velocities[i] = ranges[i] - ranges[i - 1]

        for i in range(2, n_checkpoints):
            accelerations[i] = velocities[i] - velocities[i - 1]

        return {
            "ranges": ranges,
            "velocities": velocities,
            "accelerations": accelerations,
        }

    def _compute_trajectories(
        self,
        temporal_data: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        weights = temporal_data["weights"]

        if not SKLEARN_AVAILABLE:
            self._logger.warning("scikit-learn not available, using simple projection")
            return self._simple_trajectory(weights)

        reducer = PCAReducer(self._config)
        reduced_weights = reducer.fit_transform(weights)

        if reduced_weights.shape[1] < 3:
            padding = np.zeros((reduced_weights.shape[0], 3 - reduced_weights.shape[1]))
            reduced_weights = np.hstack([reduced_weights, padding])

        origin = reduced_weights[0]
        point_cloud = self._physics_engine.compute_point_cloud(
            weights[0],
            weights,
            reduced_weights
        )

        return {
            "positions": reduced_weights,
            "point_cloud": point_cloud,
            "explained_variance": reducer.get_explained_variance(),
        }

    def _simple_trajectory(self, weights: np.ndarray) -> Dict[str, np.ndarray]:
        n = weights.shape[0]
        positions = np.zeros((n, 3))

        for i in range(n):
            positions[i, 0] = np.mean(weights[i])
            positions[i, 1] = np.std(weights[i])
            positions[i, 2] = np.linalg.norm(weights[i])

        return {
            "positions": positions,
            "point_cloud": {
                "x": positions[:, 0],
                "y": positions[:, 1],
                "z": positions[:, 2],
                "range": np.linalg.norm(positions, axis=1),
                "intensity": np.ones(n),
            },
            "explained_variance": np.array([1.0, 0.5, 0.25]),
        }


class PointCloudGenerator:
    """Generate point cloud representations of weight space."""

    def __init__(self, config: WeightSpaceLiDARConfig):
        self._config = config
        self._physics_engine = LiDARPhysicsEngine(config)
        self._logger = LoggerFactory.create("PointCloudGenerator", config.LOG_LEVEL)

    def generate_from_checkpoint(
        self,
        checkpoint_path: str,
        reduction_method: str = "pca",
    ) -> Dict[str, Any]:
        weight_extractor = DefaultWeightExtractor(self._config)

        checkpoint = self._load_checkpoint(checkpoint_path)
        weights = weight_extractor.extract(checkpoint)

        layer_weights = self._extract_layer_weights(checkpoint)

        return self.generate_from_weights(weights, layer_weights, reduction_method)

    def generate_from_weights(
        self,
        flat_weights: np.ndarray,
        layer_weights: Optional[Dict[str, np.ndarray]] = None,
        reduction_method: str = "pca",
    ) -> Dict[str, Any]:
        weight_vectors = self._create_weight_vectors(flat_weights, layer_weights)

        reduced = self._apply_reduction(weight_vectors, reduction_method)

        origin = np.zeros(reduced.shape[1])

        point_cloud = self._physics_engine.compute_point_cloud(
            origin,
            weight_vectors,
            reduced
        )

        point_cloud = self._post_process(point_cloud)

        return {
            "points": point_cloud,
            "reduced_coordinates": reduced,
            "weight_vectors": weight_vectors,
            "metadata": {
                "num_points": len(point_cloud["x"]),
                "reduction_method": reduction_method,
                "generation_timestamp": datetime.now().isoformat(),
            }
        }

    def _load_checkpoint(self, path: str) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")

        return torch.load(path, map_location="cpu", weights_only=False)

    def _extract_layer_weights(
        self,
        checkpoint: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        layer_weights = {}

        for name, tensor in state_dict.items():
            if "weight" in name.lower() or "kernel" in name.lower():
                if TORCH_AVAILABLE and isinstance(tensor, Tensor):
                    layer_weights[name] = tensor.detach().cpu().numpy().flatten()

        return layer_weights

    def _create_weight_vectors(
        self,
        flat_weights: np.ndarray,
        layer_weights: Optional[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        if layer_weights is None or len(layer_weights) == 0:
            return self._create_synthetic_points(flat_weights)

        vectors = []
        for name, weights in layer_weights.items():
            n_points = max(
                1,
                int(len(weights) * self._config.WEIGHT_SAMPLE_RATIO)
            )

            for _ in range(n_points):
                indices = np.random.choice(
                    len(weights),
                    min(len(weights), self._config.PCA_COMPONENTS),
                    replace=False
                )
                vectors.append(weights[indices])

        return np.array(vectors) if vectors else self._create_synthetic_points(flat_weights)

    def _create_synthetic_points(self, weights: np.ndarray) -> np.ndarray:
        n_points = min(
            self._config.POINT_CLOUD_DENSITY,
            len(weights) // self._config.PCA_COMPONENTS
        )
        n_points = max(n_points, 100)

        points = []
        dim = self._config.PCA_COMPONENTS

        for _ in range(n_points):
            indices = np.random.choice(len(weights), dim, replace=False)
            points.append(weights[indices])

        return np.array(points)

    def _apply_reduction(
        self,
        weight_vectors: np.ndarray,
        method: str,
    ) -> np.ndarray:
        if not SKLEARN_AVAILABLE:
            return self._simple_projection(weight_vectors)

        if method == "pca":
            reducer = PCAReducer(self._config)
        elif method == "tsne":
            reducer = TSNEReducer(self._config)
        else:
            reducer = PCAReducer(self._config)

        return reducer.fit_transform(weight_vectors)

    def _simple_projection(self, vectors: np.ndarray) -> np.ndarray:
        n = vectors.shape[0]
        projected = np.zeros((n, 3))

        projected[:, 0] = np.mean(vectors, axis=1)
        projected[:, 1] = np.std(vectors, axis=1)
        projected[:, 2] = np.linalg.norm(vectors, axis=1)

        return projected

    def _post_process(
        self,
        point_cloud: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        point_cloud = self._remove_outliers(point_cloud)

        point_cloud = self._normalize_coordinates(point_cloud)

        return point_cloud

    def _remove_outliers(
        self,
        point_cloud: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        threshold = self._config.OUTLIER_REMOVAL_THRESHOLD

        ranges = point_cloud["range"]
        mean_range = np.mean(ranges)
        std_range = np.std(ranges)

        mask = np.abs(ranges - mean_range) < threshold * std_range

        return {
            key: values[mask] if isinstance(values, np.ndarray) else values
            for key, values in point_cloud.items()
        }

    def _normalize_coordinates(
        self,
        point_cloud: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        if self._config.RANGE_NORMALIZATION == "unit":
            max_range = np.max(point_cloud["range"]) + self._config.NUMERICAL_PRECISION

            point_cloud["x"] = point_cloud["x"] / max_range
            point_cloud["y"] = point_cloud["y"] / max_range
            point_cloud["z"] = point_cloud["z"] / max_range

        return point_cloud


class WeightSpaceNavigator:
    """
    Main navigation interface for weight space LiDAR.
    Provides high-level API for exploring neural network checkpoints.
    """

    def __init__(self, config: Optional[WeightSpaceLiDARConfig] = None):
        self._config = config or WeightSpaceLiDARConfig()
        self._temporal_scanner = TemporalCheckpointScanner(self._config)
        self._point_cloud_gen = PointCloudGenerator(self._config)
        self._physics_engine = LiDARPhysicsEngine(self._config)
        self._logger = LoggerFactory.create("WeightSpaceNavigator", self._config.LOG_LEVEL)

        self._cached_scan: Optional[Dict[str, Any]] = None

    def scan_checkpoints(
        self,
        checkpoint_dir: str,
        sort_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._logger.info(f"Scanning checkpoint directory: {checkpoint_dir}")

        scan_result = self._temporal_scanner.scan_directory(checkpoint_dir, sort_by)
        self._cached_scan = scan_result

        return scan_result

    def generate_point_cloud(
        self,
        checkpoint_path: str,
        reduction_method: str = "pca",
    ) -> Dict[str, Any]:
        self._logger.info(f"Generating point cloud from: {checkpoint_path}")

        return self._point_cloud_gen.generate_from_checkpoint(
            checkpoint_path,
            reduction_method
        )

    def compute_range_map(
        self,
        checkpoint_path: str,
        reference_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        weight_extractor = DefaultWeightExtractor(self._config)

        checkpoint = self._load_checkpoint(checkpoint_path)
        target_weights = weight_extractor.extract(checkpoint)

        if reference_path:
            ref_checkpoint = self._load_checkpoint(reference_path)
            origin_weights = weight_extractor.extract(ref_checkpoint)
        else:
            origin_weights = np.zeros_like(target_weights)

        signal = self._physics_engine.compute_return_signal(
            origin_weights,
            target_weights
        )

        layer_ranges = self._compute_layer_ranges(checkpoint, origin_weights)

        return {
            "global_range": signal["range"],
            "global_intensity": signal["intensity"],
            "transmission": signal["transmission"],
            "backscatter": signal["backscatter"],
            "layer_ranges": layer_ranges,
        }

    def temporal_evolution(
        self,
        checkpoint_dir: str,
    ) -> Dict[str, Any]:
        if self._cached_scan is None:
            self._cached_scan = self.scan_checkpoints(checkpoint_dir)

        trajectories = self._cached_scan["trajectories"]
        signals = self._cached_scan["signals"]

        epochs = self._cached_scan["epochs"]

        evolution_metrics = self._compute_evolution_metrics(trajectories, signals, epochs)

        return {
            "positions": trajectories["positions"],
            "ranges": signals["ranges"],
            "velocities": signals["velocities"],
            "epochs": epochs,
            "metrics": evolution_metrics,
        }

    def export_point_cloud(
        self,
        point_cloud: Dict[str, np.ndarray],
        output_path: str,
        format: Optional[str] = None,
    ) -> str:
        output_format = format or self._config.OUTPUT_FORMAT

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if output_format == "las":
            return self._export_las(point_cloud, output_path)
        elif output_format == "ply":
            return self._export_ply(point_cloud, output_path)
        elif output_format == "csv":
            return self._export_csv(point_cloud, output_path)
        elif output_format == "json":
            return self._export_json(point_cloud, output_path)
        else:
            return self._export_csv(point_cloud, output_path)

    def visualize_3d(
        self,
        point_cloud: Dict[str, np.ndarray],
        title: str = "Weight Space LiDAR",
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(self._config.FIGURE_SIZE_X, self._config.FIGURE_SIZE_Y))
            ax = fig.add_subplot(111, projection="3d")

            scatter = ax.scatter(
                point_cloud["x"],
                point_cloud["y"],
                point_cloud["z"],
                c=point_cloud["intensity"],
                cmap=self._config.COLORMAP,
                s=1,
                alpha=0.6,
            )

            ax.set_xlabel("X (Weight Dimension 1)")
            ax.set_ylabel("Y (Weight Dimension 2)")
            ax.set_zlabel("Z (Weight Dimension 3)")
            ax.set_title(title)

            fig.colorbar(scatter, ax=ax, label="Intensity")

            if save_path:
                plt.savefig(save_path, dpi=self._config.FIGURE_DPI, bbox_inches="tight")
                self._logger.info(f"Figure saved to {save_path}")

            return fig

        except ImportError:
            self._logger.warning("matplotlib not available for 3D visualization")
            return None

    def visualize_temporal(
        self,
        evolution_data: Dict[str, Any],
        title: str = "Temporal Evolution",
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            positions = evolution_data["positions"]
            epochs = evolution_data["epochs"]
            ranges = evolution_data["ranges"]

            ax1 = axes[0, 0]
            ax1.plot(epochs, positions[:, 0], "r-", label="PC1")
            ax1.plot(epochs, positions[:, 1], "g-", label="PC2")
            ax1.plot(epochs, positions[:, 2], "b-", label="PC3")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Position")
            ax1.set_title("Trajectory in Reduced Space")
            ax1.legend()

            ax2 = axes[0, 1]
            ax2.plot(epochs, ranges, "k-")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Range (Weight Space)")
            ax2.set_title("Range Evolution")

            ax3 = axes[1, 0]
            ax3.plot(epochs, evolution_data["velocities"], "m-")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Velocity")
            ax3.set_title("Training Velocity")

            ax4 = axes[1, 1]
            ax4.plot(positions[:, 0], positions[:, 1], "c-", alpha=0.7)
            ax4.scatter(positions[0, 0], positions[0, 1], c="g", s=100, label="Start")
            ax4.scatter(positions[-1, 0], positions[-1, 1], c="r", s=100, label="End")
            ax4.set_xlabel("PC1")
            ax4.set_ylabel("PC2")
            ax4.set_title("2D Trajectory")
            ax4.legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=self._config.FIGURE_DPI, bbox_inches="tight")
                self._logger.info(f"Figure saved to {save_path}")

            return fig

        except ImportError:
            self._logger.warning("matplotlib not available for temporal visualization")
            return None

    def _load_checkpoint(self, path: str) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for checkpoint loading")

        return torch.load(path, map_location="cpu", weights_only=False)

    def _compute_layer_ranges(
        self,
        checkpoint: Dict[str, Any],
        origin: np.ndarray,
    ) -> Dict[str, float]:
        weight_extractor = DefaultWeightExtractor(self._config)
        layer_names = weight_extractor.get_layer_names(checkpoint)
        layer_ranges = {}

        state_dict = checkpoint.get("model_state_dict", checkpoint)

        for name in layer_names:
            if name in state_dict:
                tensor = state_dict[name]
                if TORCH_AVAILABLE and isinstance(tensor, Tensor):
                    weights = tensor.detach().cpu().numpy().flatten()

                    if len(origin) >= len(weights):
                        signal = self._physics_engine.compute_return_signal(
                            origin[:len(weights)],
                            weights
                        )
                        layer_ranges[name] = signal["range"]

        return layer_ranges

    def _compute_evolution_metrics(
        self,
        trajectories: Dict[str, Any],
        signals: Dict[str, Any],
        epochs: np.ndarray,
    ) -> Dict[str, float]:
        positions = trajectories["positions"]
        ranges = signals["ranges"]
        velocities = signals["velocities"]

        total_distance = np.sum(np.abs(velocities))

        if len(ranges) > 1:
            mean_velocity = np.mean(np.abs(velocities[1:]))
            max_velocity = np.max(np.abs(velocities))
        else:
            mean_velocity = 0.0
            max_velocity = 0.0

        if len(positions) > 1:
            displacement = np.linalg.norm(positions[-1] - positions[0])
        else:
            displacement = 0.0

        path_efficiency = displacement / (total_distance + self._config.NUMERICAL_PRECISION)

        return {
            "total_distance": float(total_distance),
            "displacement": float(displacement),
            "path_efficiency": float(path_efficiency),
            "mean_velocity": float(mean_velocity),
            "max_velocity": float(max_velocity),
            "final_range": float(ranges[-1]) if len(ranges) > 0 else 0.0,
        }

    def _export_las(
        self,
        point_cloud: Dict[str, np.ndarray],
        output_path: str,
    ) -> str:
        try:
            import laspy

            header = laspy.LasHeader(point_format=3, version="1.2")
            las = laspy.LasData(header)

            las.x = point_cloud["x"].astype(np.float32)
            las.y = point_cloud["y"].astype(np.float32)
            las.z = point_cloud["z"].astype(np.float32)
            las.intensity = (point_cloud["intensity"] * 65535).astype(np.uint16)

            las.write(output_path)

        except ImportError:
            self._logger.warning("laspy not available, falling back to CSV")
            return self._export_csv(point_cloud, output_path.replace(".las", ".csv"))

        return output_path

    def _export_ply(
        self,
        point_cloud: Dict[str, np.ndarray],
        output_path: str,
    ) -> str:
        n_points = len(point_cloud["x"])

        with open(output_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float intensity\n")
            f.write("end_header\n")

            for i in range(n_points):
                f.write(
                    f"{point_cloud['x'][i]:.{self._config.OUTPUT_PRECISION}f} "
                    f"{point_cloud['y'][i]:.{self._config.OUTPUT_PRECISION}f} "
                    f"{point_cloud['z'][i]:.{self._config.OUTPUT_PRECISION}f} "
                    f"{point_cloud['intensity'][i]:.{self._config.OUTPUT_PRECISION}f}\n"
                )

        return output_path

    def _export_csv(
        self,
        point_cloud: Dict[str, np.ndarray],
        output_path: str,
    ) -> str:
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z", "intensity", "range", "azimuth", "elevation"])

            n_points = len(point_cloud["x"])
            for i in range(n_points):
                writer.writerow([
                    f"{point_cloud['x'][i]:.{self._config.OUTPUT_PRECISION}f}",
                    f"{point_cloud['y'][i]:.{self._config.OUTPUT_PRECISION}f}",
                    f"{point_cloud['z'][i]:.{self._config.OUTPUT_PRECISION}f}",
                    f"{point_cloud['intensity'][i]:.{self._config.OUTPUT_PRECISION}f}",
                    f"{point_cloud['range'][i]:.{self._config.OUTPUT_PRECISION}f}",
                    f"{point_cloud['azimuth'][i]:.{self._config.OUTPUT_PRECISION}f}",
                    f"{point_cloud['elevation'][i]:.{self._config.OUTPUT_PRECISION}f}",
                ])

        return output_path

    def _export_json(
        self,
        point_cloud: Dict[str, np.ndarray],
        output_path: str,
    ) -> str:
        data = {
            "points": [
                {
                    "x": float(point_cloud["x"][i]),
                    "y": float(point_cloud["y"][i]),
                    "z": float(point_cloud["z"][i]),
                    "intensity": float(point_cloud["intensity"][i]),
                    "range": float(point_cloud["range"][i]),
                }
                for i in range(len(point_cloud["x"]))
            ],
            "metadata": {
                "num_points": len(point_cloud["x"]),
                "export_timestamp": datetime.now().isoformat(),
            }
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path


class WeightSpaceLiDARCLI:
    """Command-line interface for Weight Space LiDAR."""

    def __init__(self):
        self._parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Weight Space LiDAR: Synthetic LiDAR for Neural Network Analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        scan_parser = subparsers.add_parser(
            "scan",
            help="Scan checkpoint directory for temporal analysis"
        )
        scan_parser.add_argument(
            "checkpoint_dir",
            type=str,
            help="Directory containing .pth checkpoint files"
        )
        scan_parser.add_argument(
            "--sort-by",
            type=str,
            default="epoch",
            choices=["epoch", "modified", "name"],
            help="Method to sort checkpoints"
        )
        scan_parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output file path for results"
        )

        cloud_parser = subparsers.add_parser(
            "cloud",
            help="Generate point cloud from checkpoint"
        )
        cloud_parser.add_argument(
            "checkpoint",
            type=str,
            help="Path to .pth checkpoint file"
        )
        cloud_parser.add_argument(
            "--reduction",
            type=str,
            default="pca",
            choices=["pca", "tsne"],
            help="Dimensionality reduction method"
        )
        cloud_parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output file path"
        )
        cloud_parser.add_argument(
            "--format",
            type=str,
            default="csv",
            choices=["csv", "ply", "json", "las"],
            help="Output format"
        )
        cloud_parser.add_argument(
            "--visualize",
            action="store_true",
            help="Generate 3D visualization"
        )

        range_parser = subparsers.add_parser(
            "range",
            help="Compute range map between checkpoints"
        )
        range_parser.add_argument(
            "target",
            type=str,
            help="Target checkpoint path"
        )
        range_parser.add_argument(
            "--reference",
            type=str,
            default=None,
            help="Reference checkpoint path (default: zero vector)"
        )

        evolution_parser = subparsers.add_parser(
            "evolution",
            help="Analyze temporal evolution of checkpoints"
        )
        evolution_parser.add_argument(
            "checkpoint_dir",
            type=str,
            help="Directory containing .pth checkpoint files"
        )
        evolution_parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output file path"
        )
        evolution_parser.add_argument(
            "--visualize",
            action="store_true",
            help="Generate visualization"
        )

        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        parsed = self._parser.parse_args(args)

        if parsed.command is None:
            self._parser.print_help()
            return 1

        config = WeightSpaceLiDARConfig()
        navigator = WeightSpaceNavigator(config)

        try:
            if parsed.command == "scan":
                return self._handle_scan(navigator, parsed)
            elif parsed.command == "cloud":
                return self._handle_cloud(navigator, parsed)
            elif parsed.command == "range":
                return self._handle_range(navigator, parsed)
            elif parsed.command == "evolution":
                return self._handle_evolution(navigator, parsed)
            else:
                self._parser.print_help()
                return 1

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    def _handle_scan(
        self,
        navigator: WeightSpaceNavigator,
        args: argparse.Namespace,
    ) -> int:
        print(f"Scanning directory: {args.checkpoint_dir}")

        result = navigator.scan_checkpoints(args.checkpoint_dir, args.sort_by)

        print(f"Found {result['metadata']['num_checkpoints']} checkpoints")
        print(f"Epochs: {result['epochs'].tolist()}")
        print(f"Range evolution: {result['signals']['ranges'].tolist()}")

        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime(WeightSpaceLiDARConfig.TIMESTAMP_FORMAT)
            output_path = f"scan_results_{timestamp}.json"

        with open(output_path, "w") as f:
            serializable = {
                "epochs": result["epochs"].tolist(),
                "ranges": result["signals"]["ranges"].tolist(),
                "velocities": result["signals"]["velocities"].tolist(),
                "positions": result["trajectories"]["positions"].tolist(),
                "metadata": result["metadata"],
            }
            json.dump(serializable, f, indent=2)

        print(f"Results saved to: {output_path}")
        return 0

    def _handle_cloud(
        self,
        navigator: WeightSpaceNavigator,
        args: argparse.Namespace,
    ) -> int:
        print(f"Generating point cloud from: {args.checkpoint}")

        result = navigator.generate_point_cloud(
            args.checkpoint,
            args.reduction
        )

        print(f"Generated {result['metadata']['num_points']} points")

        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime(WeightSpaceLiDARConfig.TIMESTAMP_FORMAT)
            output_path = f"point_cloud_{timestamp}.{args.format}"

        navigator.export_point_cloud(result["points"], output_path, args.format)
        print(f"Point cloud saved to: {output_path}")

        if args.visualize:
            viz_path = output_path.rsplit(".", 1)[0] + ".png"
            navigator.visualize_3d(result["points"], save_path=viz_path)
            print(f"Visualization saved to: {viz_path}")

        return 0

    def _handle_range(
        self,
        navigator: WeightSpaceNavigator,
        args: argparse.Namespace,
    ) -> int:
        print(f"Computing range map for: {args.target}")

        result = navigator.compute_range_map(args.target, args.reference)

        print(f"Global range: {result['global_range']:.6f}")
        print(f"Intensity: {result['global_intensity']:.6f}")
        print(f"Transmission: {result['transmission']:.6f}")
        print(f"Backscatter: {result['backscatter']:.6f}")

        if result['layer_ranges']:
            print("\nLayer ranges:")
            for name, r in result['layer_ranges'].items():
                print(f"  {name}: {r:.6f}")

        return 0

    def _handle_evolution(
        self,
        navigator: WeightSpaceNavigator,
        args: argparse.Namespace,
    ) -> int:
        print(f"Analyzing temporal evolution: {args.checkpoint_dir}")

        evolution = navigator.temporal_evolution(args.checkpoint_dir)

        print("\nEvolution Metrics:")
        for key, value in evolution['metrics'].items():
            print(f"  {key}: {value:.6f}")

        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime(WeightSpaceLiDARConfig.TIMESTAMP_FORMAT)
            output_path = f"evolution_{timestamp}.json"

        with open(output_path, "w") as f:
            serializable = {
                "epochs": evolution["epochs"].tolist(),
                "positions": evolution["positions"].tolist(),
                "ranges": evolution["ranges"].tolist(),
                "velocities": evolution["velocities"].tolist(),
                "metrics": evolution["metrics"],
            }
            json.dump(serializable, f, indent=2)

        print(f"\nResults saved to: {output_path}")

        if args.visualize:
            viz_path = output_path.rsplit(".", 1)[0] + ".png"
            navigator.visualize_temporal(evolution, save_path=viz_path)
            print(f"Visualization saved to: {viz_path}")

        return 0


def main() -> int:
    """Entry point for Weight Space LiDAR CLI."""
    cli = WeightSpaceLiDARCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
