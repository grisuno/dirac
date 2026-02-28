#!/usr/bin/env python3
"""
dirac_crystallography_suite.py

Comprehensive crystallographic and physical analysis suite for Dirac equation
neural network checkpoints. Integrates Berry phase, MBL analysis, Ricci flow,
control theory, Schrodinger analysis, and thermodynamic metrics.

Author: Gris Iscomeback
Email: grisiscomeback@gmail.com
Date: 2024
License: AGPL v3
"""

import argparse
import copy
import glob
import json
import logging
import math
import os
import re
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any, Union, Protocol, runtime_checkable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import stats, signal, linalg
from scipy.stats import entropy as scipy_entropy
from scipy.linalg import eigh, expm, eigvals
from scipy.optimize import fsolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')


@dataclass(frozen=True)
class CrystallographySuiteConfig:
    """Master configuration for the complete crystallography suite."""
    
    # Dirac Network Architecture
    GRID_SIZE: int = 16
    HIDDEN_DIM: int = 32
    NUM_SPECTRAL_LAYERS: int = 2
    EXPANSION_DIM: int = 64
    SPINOR_COMPONENTS: int = 4
    
    # Physical Constants
    DIRAC_MASS: float = 1.0
    DIRAC_C: float = 1.0
    DIRAC_GAMMA_REPRESENTATION: str = 'dirac'
    POTENTIAL_DEPTH: float = 5.0
    POTENTIAL_WIDTH: float = 0.3
    NUM_EIGENSTATES: int = 8
    ENERGY_SCALE: float = 1.0
    SPINOR_NORM_TARGET: float = 1.0
    HBAR: float = 1e-6
    HBAR_PHYSICAL: float = 1.054571817e-34
    
    # Training Parameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.005
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 5000
    TIME_STEPS: int = 2
    DT: float = 0.01
    TRAIN_RATIO: float = 0.7
    NUM_SAMPLES: int = 200
    GRADIENT_CLIP_NORM: float = 1.0
    
    # MBL Analysis Thresholds
    LEVEL_SPACING_WIGNER_DYSON: float = 0.5307
    LEVEL_SPACING_POISSON: float = 0.3863
    LEVEL_SPACING_TOLERANCE: float = 0.05
    BRODY_THERMAL: float = 1.0
    BRODY_MBL: float = 0.0
    BRODY_TOLERANCE: float = 0.1
    PR_LOCALIZATION_THRESHOLD: float = 0.8
    PR_DELIMITED_THRESHOLD: float = 0.1
    PR_RENYI_INDEX: int = 2
    
    # Crystallography Thresholds
    ALPHA_CRYSTAL_THRESHOLD: float = 7.0
    ALPHA_PERFECT_CRYSTAL_THRESHOLD: float = 10.0
    DELTA_CRYSTAL_THRESHOLD: float = 0.1
    DELTA_OPTICAL_THRESHOLD: float = 0.01
    DELTA_GLASS_THRESHOLD: float = 0.4
    KAPPA_CRYSTAL_THRESHOLD: float = 1.5
    TEMPERATURE_CRYSTAL_THRESHOLD: float = 1e-9
    
    # Topological Analysis
    TORUS_GRID_SIZE: int = 16
    TOPO_ALIGNMENT_THRESHOLD: float = 0.7
    TOPO_HYSTERESIS_WIDTH: float = 0.1
    TOPO_COUPLING_STRENGTH: float = 0.5
    TOPO_LAMBDA_BASE: float = 1e20
    TOPO_LAMBDA_CRITICAL: float = 1e34
    TOPO_ALIGNMENT_HISTORY_LEN: int = 100
    TOPO_PHASE_SMOOTHING: float = 0.95
    TOPO_LOCALIZATION_LIQUID: float = 0.2
    TOPO_LOCALIZATION_CRYSTAL: float = 1.0
    TOPO_CRYSTALLIZATION_PRESSURE_DECAY: float = 0.999
    TOPO_ENABLED: bool = True
    
    # Ricci Flow Parameters
    RICCI_FLOW_TEMPORAL_STEPS: int = 50
    RICCI_FLOW_TIME_STEP_SIZE: float = 0.001
    RICCI_FLOW_REGULARIZATION_ALPHA: float = 0.1
    NECK_EIGENVALUE_RATIO_THRESHOLD: float = 100.0
    CURVATURE_COLLAPSE_ABSOLUTE_THRESHOLD: float = 1e-6
    ENTROPY_SINGULARITY_THRESHOLD: float = 0.05
    RICCI_SCALAR_DIVERGENCE_THRESHOLD: float = 1e6
    RICCI_CURVATURE_SAMPLES: int = 100
    RICCI_MAX_DIMENSION: int = 5000
    
    # Control Theory Parameters
    POLE_ZERO_TOLERANCE: float = 1e-6
    STABILITY_MARGIN: float = 0.01
    FREQUENCY_SAMPLES: int = 1000
    FREQUENCY_MIN: float = 1e-3
    FREQUENCY_MAX: float = 1e3
    TIME_SAMPLES: int = 500
    TIME_MAX: float = 10.0
    
    # Berry Phase Parameters
    BERRY_PHASE_TOLERANCE: float = 0.1
    
    # Thermodynamic Parameters
    GIBBS_T0: float = 1e-3
    GIBBS_C: float = 0.5
    ENTROPY_BINS: int = 50
    PCA_COMPONENTS: int = 3
    ENTROPY_EPS: float = 1e-10
    MIN_VARIANCE_THRESHOLD: float = 1e-8
    EIGENVALUE_TOL: float = 1e-10
    KAPPA_MAX_DIM: int = 10000
    KAPPA_GRADIENT_BATCHES: int = 5
    
    # Analysis Limits
    SPECTRAL_PEAK_LIMIT: int = 20
    SPECTRAL_POWER_LIMIT: int = 100
    PARAM_FLATTEN_LIMIT: int = 2000
    GRADIENT_BUFFER_LIMIT: int = 500
    WEIGHT_METRIC_DIM_LIMIT: int = 256
    HEAT_KERNEL_SPECTRAL_CUTOFF: int = 100
    COMPRESSED_DIMENSION: int = 512
    
    # Checkpointing
    CHECKPOINT_INTERVAL_MINUTES: int = 5
    MAX_CHECKPOINTS: int = 10
    CHECKPOINT_LATEST_PATH: str = 'latest.pth'
    
    # Device and Logging
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED: int = 42
    LOG_LEVEL: str = 'INFO'
    FIGURE_DPI: int = 300
    SAVE_FORMAT: str = 'png'


class LoggerFactory:
    """Factory for creating configured logger instances."""
    
    @staticmethod
    def create_logger(name: str, level: str = None, config: CrystallographySuiteConfig = None) -> logging.Logger:
        logger = logging.getLogger(name)
        effective_level = level or (config.LOG_LEVEL if config else 'INFO')
        logger.setLevel(getattr(logging, effective_level.upper()))
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


class IMetricCalculator(Protocol):
    """Protocol for metric calculation strategies."""
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """Compute metrics for the given model."""
        ...


class IPhaseDetector(Protocol):
    """Protocol for phase detection strategies."""
    
    def detect(self, spectral_field: torch.Tensor) -> Dict[str, Any]:
        """Detect phase from spectral field."""
        ...


class GammaMatrices:
    """Dirac gamma matrices in various representations."""
    
    def __init__(self, representation: str, device: str, config: CrystallographySuiteConfig):
        self.representation = representation
        self.device = device
        self.config = config
        self._init_matrices()
    
    def _init_matrices(self):
        if self.representation == 'dirac':
            self.gamma0 = torch.tensor([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1]
            ], dtype=torch.complex64, device=self.device)
            
            self.gamma1 = torch.tensor([
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [-1, 0, 0, 0]
            ], dtype=torch.complex64, device=self.device)
            
            self.gamma2 = torch.tensor([
                [0, 0, 0, -1j],
                [0, 0, 1j, 0],
                [0, 1j, 0, 0],
                [-1j, 0, 0, 0]
            ], dtype=torch.complex64, device=self.device)
            
            self.gamma3 = torch.tensor([
                [0, 0, 1, 0],
                [0, 0, 0, -1],
                [-1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=torch.complex64, device=self.device)
            
            self.gamma5 = torch.tensor([
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=torch.complex64, device=self.device)
        else:
            raise ValueError(f"Unsupported gamma representation: {self.representation}")
        
        self.gammas = [self.gamma0, self.gamma1, self.gamma2, self.gamma3]


class DiracHamiltonianOperator:
    """Dirac Hamiltonian operator for 4-component spinors."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
        self.grid_size = config.GRID_SIZE
        self.mass = config.DIRAC_MASS
        self.c = config.DIRAC_C
        self.gamma = GammaMatrices(config.DIRAC_GAMMA_REPRESENTATION, config.DEVICE, config)
        self._precompute_operators()
    
    def _precompute_operators(self):
        kx = torch.fft.fftfreq(self.grid_size, d=1.0) * 2 * np.pi
        ky = torch.fft.fftfreq(self.grid_size, d=1.0) * 2 * np.pi
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        
        self.kx_grid = KX.to(self.config.DEVICE)
        self.ky_grid = KY.to(self.config.DEVICE)
        
        self.alpha_x = self.gamma.gamma0 @ self.gamma.gamma1
        self.alpha_y = self.gamma.gamma0 @ self.gamma.gamma2
        self.beta = self.gamma.gamma0
        
        self.alpha_x = self.alpha_x.to(self.config.DEVICE)
        self.alpha_y = self.alpha_y.to(self.config.DEVICE)
        self.beta = self.beta.to(self.config.DEVICE)

    def apply_dirac_hamiltonian(self, spinor: torch.Tensor) -> torch.Tensor:
        """Apply Dirac Hamiltonian to 4-component spinor."""
        device = spinor.device
        
        if spinor.dim() == 3:
            spinor = spinor.unsqueeze(0)
        
        batch_size = spinor.shape[0]
        result = torch.zeros_like(spinor, dtype=torch.complex64)
        
        for c in range(4):
            psi_c = spinor[:, c, :, :]
            psi_c_fft = torch.fft.fft2(psi_c)
            
            px_psi_fft = self.kx_grid * psi_c_fft
            py_psi_fft = self.ky_grid * psi_c_fft
            
            px_psi = torch.fft.ifft2(px_psi_fft)
            py_psi = torch.fft.ifft2(py_psi_fft)
            
            for d in range(4):
                alpha_x_cd = self.alpha_x[c, d].item()
                alpha_y_cd = self.alpha_y[c, d].item()
                
                result[:, c, :, :] += self.c * (
                    alpha_x_cd * px_psi +
                    alpha_y_cd * py_psi
                )
        
        mass_term = self.mass * self.c**2
        for c in range(4):
            for d in range(4):
                beta_cd = self.beta[c, d].item()
                result[:, c, :, :] += beta_cd * mass_term * spinor[:, d, :, :]
        
        return result


class SpectralLayer(nn.Module):
    """Spectral convolution layer operating in Fourier space."""
    
    def __init__(self, channels: int, grid_size: int, config: CrystallographySuiteConfig):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.config = config
        self.kernel_real = nn.Parameter(
            torch.randn(channels, channels, grid_size // 2 + 1, grid_size) * 0.1
        )
        self.kernel_imag = nn.Parameter(
            torch.randn(channels, channels, grid_size // 2 + 1, grid_size) * 0.1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.rfft2(x)
        batch, channels, freq_h, freq_w = x_fft.shape
        kernel_real = self.kernel_real.mean(dim=0)
        kernel_imag = self.kernel_imag.mean(dim=0)
        kernel_real_exp = kernel_real.unsqueeze(0).unsqueeze(0).squeeze(0)
        kernel_imag_exp = kernel_imag.unsqueeze(0).unsqueeze(0).squeeze(0)
        kernel_real_interp = F.interpolate(
            kernel_real_exp, size=(freq_h, freq_w), mode='bilinear', align_corners=False
        )
        kernel_imag_interp = F.interpolate(
            kernel_imag_exp, size=(freq_h, freq_w), mode='bilinear', align_corners=False
        )
        real_part = x_fft.real * kernel_real_interp - x_fft.imag * kernel_imag_interp
        imag_part = x_fft.real * kernel_imag_interp + x_fft.imag * kernel_real_interp
        output_fft = torch.complex(real_part, imag_part)
        output = torch.fft.irfft2(output_fft, s=(self.grid_size, self.grid_size))
        return output


class DiracSpectralNetwork(nn.Module):
    """Neural network for learning Dirac equation dynamics."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        super().__init__()
        self.config = config
        self.grid_size = config.GRID_SIZE
        self.spinor_components = config.SPINOR_COMPONENTS
        self.input_channels = config.SPINOR_COMPONENTS * 2
        self.output_channels = config.SPINOR_COMPONENTS * 2
        
        self.input_proj = nn.Conv2d(self.input_channels, config.HIDDEN_DIM, kernel_size=1)
        self.expansion_proj = nn.Conv2d(config.HIDDEN_DIM, config.EXPANSION_DIM, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            SpectralLayer(config.EXPANSION_DIM, config.GRID_SIZE, config)
            for _ in range(config.NUM_SPECTRAL_LAYERS)
        ])
        self.contraction_proj = nn.Conv2d(config.EXPANSION_DIM, config.HIDDEN_DIM, kernel_size=1)
        self.output_proj = nn.Conv2d(config.HIDDEN_DIM, self.output_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = F.gelu(self.input_proj(x))
        x = F.gelu(self.expansion_proj(x))
        for spectral_layer in self.spectral_layers:
            x = F.gelu(spectral_layer(x))
        x = F.gelu(self.contraction_proj(x))
        return self.output_proj(x)


class WeightIntegrityCalculator:
    """Calculator for weight integrity metrics."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        has_nan = False
        has_inf = False
        total_params = 0
        nan_count = 0
        inf_count = 0
        
        for param in model.parameters():
            data = param.data
            numel = data.numel()
            total_params += numel
            n_nan = torch.isnan(data).sum().item()
            n_inf = torch.isinf(data).sum().item()
            if n_nan > 0:
                has_nan = True
                nan_count += n_nan
            if n_inf > 0:
                has_inf = True
                inf_count += n_inf
        
        corruption_ratio = (nan_count + inf_count) / total_params if total_params > 0 else 0.0
        
        return {
            'is_valid': not (has_nan or has_inf),
            'has_nan': has_nan,
            'has_inf': has_inf,
            'total_params': total_params,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'corruption_ratio': corruption_ratio
        }


class DiscretizationCalculator:
    """Calculator for discretization margin and alpha purity metrics."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        margins = []
        all_params = []
        layer_deltas = {}
        
        for name, param in model.named_parameters():
            if param.numel() > 0:
                p_data = param.data.detach()
                all_params.append(p_data.flatten())
                margin = (p_data - p_data.round()).abs().max().item()
                margins.append(margin)
                layer_deltas[name] = margin
        
        delta = max(margins) if margins else 0.0
        alpha = -np.log(delta + self.config.ENTROPY_EPS) if delta > 0 else 20.0
        
        flat_params = torch.cat(all_params)[:self.config.PARAM_FLATTEN_LIMIT]
        spectral_entropy = self._compute_spectral_entropy(flat_params)
        
        return {
            'delta': delta,
            'alpha': alpha,
            'spectral_entropy': spectral_entropy,
            'is_discrete': delta < self.config.DELTA_CRYSTAL_THRESHOLD,
            'layer_deltas': layer_deltas
        }
    
    def _compute_spectral_entropy(self, weights: torch.Tensor) -> float:
        if weights.numel() == 0:
            return 0.0
        w = weights.detach().cpu()
        fft_spectrum = torch.fft.fft(w)
        power_spectrum = torch.abs(fft_spectrum)**2
        
        ps_normalized = power_spectrum / (torch.sum(power_spectrum) + 1e-10)
        ps_normalized = ps_normalized[ps_normalized > 1e-10]
        if len(ps_normalized) == 0:
            return 0.0
        entropy = -torch.sum(ps_normalized * torch.log(ps_normalized + 1e-10))
        return float(entropy.item())


class SpectralGeometryCalculator:
    """Calculator for spectral geometry metrics including MBL level spacing."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        all_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
        all_weights = all_weights[:self.config.PARAM_FLATTEN_LIMIT].cpu().numpy()
        
        n = len(all_weights)
        outer_product = np.outer(all_weights, all_weights) / n
        outer_product += np.eye(n) * self.config.EIGENVALUE_TOL
        
        try:
            eigenvalues = eigh(outer_product, eigvals_only=True)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            effective_dim = np.sum(eigenvalues > self.config.EIGENVALUE_TOL)
            spectral_gap = eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0.0
            participation_ratio = (np.sum(eigenvalues)**2) / (np.sum(eigenvalues**2) + 1e-10)
            level_spacing = np.diff(eigenvalues)
            level_spacing_ratio = self._compute_level_spacing_ratio(level_spacing)
            
            return {
                'spectral_gap': float(spectral_gap),
                'effective_dimension': int(effective_dim),
                'participation_ratio': float(participation_ratio),
                'level_spacing_ratio': float(level_spacing_ratio),
                'largest_eigenvalue': float(eigenvalues[0]),
                'smallest_eigenvalue': float(eigenvalues[-1])
            }
        except Exception as e:
            return {
                'spectral_gap': 0.0,
                'effective_dimension': 0,
                'participation_ratio': 0.0,
                'level_spacing_ratio': 0.0,
                'largest_eigenvalue': 0.0,
                'smallest_eigenvalue': 0.0,
                'error': str(e)
            }
    
    def _compute_level_spacing_ratio(self, spacings: np.ndarray) -> float:
        if len(spacings) < 2:
            return 0.0
        ratios = []
        for i in range(len(spacings) - 1):
            s1 = abs(spacings[i])
            s2 = abs(spacings[i+1])
            if s1 > 1e-15 and s2 > 1e-15:
                ratios.append(min(s1, s2) / max(s1, s2))
        return np.mean(ratios) if ratios else 0.0


class RicciCurvatureCalculator:
    """Calculator for Ricci curvature estimation in weight space."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        all_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
        n = min(len(all_weights), self.config.PARAM_FLATTEN_LIMIT)
        w = all_weights[:n].cpu().numpy()
        
        metric_tensor = np.outer(w, w) / n
        metric_tensor += np.eye(n) * self.config.EIGENVALUE_TOL
        
        ricci_scalar = self._compute_ricci_scalar(metric_tensor)
        sectional_curvatures = self._estimate_sectional_curvatures(metric_tensor)
        
        return {
            'ricci_scalar': float(ricci_scalar),
            'mean_sectional_curvature': float(np.mean(sectional_curvatures)),
            'curvature_variance': float(np.var(sectional_curvatures))
        }
    
    def _compute_ricci_scalar(self, metric: np.ndarray) -> float:
        eigenvalues = eigh(metric, eigvals_only=True)
        eigenvalues = eigenvalues[eigenvalues > self.config.EIGENVALUE_TOL]
        n = len(eigenvalues)
        if n < 2:
            return 0.0
        ricci_scalar = n * np.sum(1.0 / eigenvalues)
        return ricci_scalar
    
    def _estimate_sectional_curvatures(self, metric: np.ndarray, samples: int = None) -> np.ndarray:
        samples = samples or self.config.RICCI_CURVATURE_SAMPLES
        curvatures = []
        n = metric.shape[0]
        for _ in range(samples):
            i, j = np.random.choice(n, 2, replace=False)
            block = metric[np.ix_([i, j], [i, j])]
            det = np.linalg.det(block)
            if det > self.config.EIGENVALUE_TOL:
                curvatures.append(1.0 / det)
        return np.array(curvatures) if curvatures else np.array([0.0])


class BerryPhaseCalculator:
    """Calculates Berry phase from training checkpoint trajectory."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
        self.logger = LoggerFactory.create_logger("BerryPhaseCalculator", config=config)
    
    def load_checkpoints(self, checkpoint_dir: str) -> List[Dict[str, Any]]:
        pattern = os.path.join(checkpoint_dir, "*.pth")
        files = sorted(glob.glob(pattern), key=self._extract_epoch)
        
        checkpoints = []
        for f in files:
            try:
                ckpt = torch.load(f, map_location='cpu', weights_only=False)
                checkpoints.append({
                    'path': f,
                    'epoch': self._extract_epoch(f),
                    'state_dict': ckpt.get('model_state_dict', ckpt),
                    'metrics': ckpt.get('metrics', {})
                })
            except Exception as e:
                self.logger.warning(f"Could not load {f}: {e}")
        
        return checkpoints
    
    def _extract_epoch(self, filepath: str) -> int:
        match = re.search(r'epoch[_]?(\d+)', filepath)
        return int(match.group(1)) if match else 0
    
    def flatten_kernel_params(self, state_dict: Dict) -> Optional[torch.Tensor]:
        kernels = []
        layer_indices = set()
        for key in state_dict.keys():
            if 'spectral_layers' in key:
                parts = key.split('.')
                if len(parts) >= 2:
                    try:
                        idx = int(parts[1])
                        layer_indices.add(idx)
                    except:
                        pass
        
        for idx in sorted(layer_indices):
            real_key = f'spectral_layers.{idx}.kernel_real'
            imag_key = f'spectral_layers.{idx}.kernel_imag'
            
            if real_key in state_dict and imag_key in state_dict:
                kr = state_dict[real_key]
                ki = state_dict[imag_key]
                kernels.append(torch.complex(kr, ki).flatten())
        
        return torch.cat(kernels) if kernels else None
    
    def compute_berry_connection_discrete(self, theta_prev: torch.Tensor, theta_curr: torch.Tensor) -> float:
        if theta_prev is None or theta_curr is None:
            return 0.0
        
        theta_prev_norm = theta_prev / (torch.norm(theta_prev) + 1e-10)
        theta_curr_norm = theta_curr / (torch.norm(theta_curr) + 1e-10)
        
        overlap = torch.sum(torch.conj(theta_prev_norm) * theta_curr_norm)
        
        if torch.abs(overlap) < 1e-10:
            return 0.0
        
        phase = torch.angle(overlap).item()
        return phase
    
    def calculate_berry_phase(self, checkpoint_dir: str) -> Dict[str, Any]:
        checkpoints = self.load_checkpoints(checkpoint_dir)
        
        if len(checkpoints) < 2:
            return {'error': f'Need at least 2 checkpoints, found {len(checkpoints)}'}
        
        kernels = []
        epochs = []
        for ckpt in checkpoints:
            kernel = self.flatten_kernel_params(ckpt['state_dict'])
            kernels.append(kernel)
            epochs.append(ckpt['epoch'])
        
        berry_phases = []
        for i in range(1, len(kernels)):
            if kernels[i-1] is not None and kernels[i] is not None:
                phase = self.compute_berry_connection_discrete(kernels[i-1], kernels[i])
                berry_phases.append(phase)
            else:
                berry_phases.append(0.0)
        
        cumulative_phase = np.cumsum(berry_phases)
        total_phase = cumulative_phase[-1] if len(cumulative_phase) > 0 else 0.0
        
        phase_mod_2pi = total_phase % (2 * np.pi)
        if phase_mod_2pi > np.pi:
            phase_mod_2pi -= 2 * np.pi
        
        winding_number = int(round(total_phase / (2 * np.pi)))
        
        return {
            'total_berry_phase': total_phase,
            'berry_phase_mod_2pi': phase_mod_2pi,
            'winding_number': winding_number,
            'num_checkpoints': len(checkpoints),
            'epochs': epochs
        }


class ControlSystemAnalyzer:
    """Control theory analysis for neural network dynamics."""

    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config

    def extract_state_space(self, model: nn.Module) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_weights = []
        all_shapes = []
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                w = param.data.cpu().numpy()
                all_weights.append(w)
                all_shapes.append(w.shape)
        
        if len(all_weights) == 0:
            n = 10
            return np.eye(n), np.random.randn(n, 1), np.random.randn(1, n), np.zeros((1, 1))
        
        # Build state matrix by taking the first square block of each layer
        # This handles layers with different dimensions
        max_dim = max(min(w.shape[0], w.shape[1]) for w in all_weights)
        
        A_blocks = []
        for w in all_weights:
            # Reshape 4D tensors to 2D
            if w.ndim > 2:
                w = w.reshape(w.shape[0], -1)
            
            # Extract square block
            n = min(w.shape[0], w.shape[1])
            if n > 0:
                A_blocks.append(w[:n, :n])
        
        if not A_blocks:
            n = 10
            return np.eye(n), np.random.randn(n, 1), np.random.randn(1, n), np.zeros((1, 1))
        
        # Find the common dimension by padding/trimming to the median size
        sizes = [min(b.shape[0], b.shape[1]) for b in A_blocks]
        target_dim = int(np.median(sizes))
        
        if target_dim < 2:
            target_dim = 2
        
        # Create a composite state matrix
        A_composite = np.zeros((target_dim, target_dim))
        
        for b in A_blocks:
            n = min(b.shape[0], b.shape[1], target_dim)
            if n > 0:
                A_composite[:n, :n] += b[:n, :n]
        
        A_composite = A_composite / len(A_blocks)
        
        # Add identity for stability
        A = A_composite + np.eye(target_dim) * 0.01
        
        n_states = A.shape[0]
        B = np.random.randn(n_states, 1) * 0.01
        C = np.random.randn(1, n_states) * 0.01
        D = np.zeros((1, 1))
        
        return A, B, C, D

    def analyze_stability(self, A: np.ndarray) -> Dict[str, Any]:
        try:
            eigenvalues = np.linalg.eigvals(A)
            real_parts = np.real(eigenvalues)
            
            is_stable = np.all(real_parts < -self.config.STABILITY_MARGIN)
            stability_margin = -np.max(real_parts) if len(real_parts) > 0 else float('inf')
            
            return {
                'is_stable': is_stable,
                'stability_margin': float(stability_margin),
                'eigenvalues': [complex(e) for e in eigenvalues[:10]],
                'dominant_pole': complex(eigenvalues[np.argmax(real_parts)]) if len(eigenvalues) > 0 else None
            }
        except Exception as e:
            return {'is_stable': False, 'error': str(e)}

    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        A, B, C, D = self.extract_state_space(model)
        stability = self.analyze_stability(A)
        
        return {
            'system_dimension': A.shape[0],
            'stability_analysis': stability
        }

class ThermodynamicCalculator:
    """Calculator for thermodynamic potentials."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        delta = kwargs.get('delta', 1.0)
        alpha = kwargs.get('alpha', 0.0)
        kappa = kwargs.get('kappa', 1.0)
        t_eff = kwargs.get('effective_temperature', 1.0)
        
        internal_energy_proxy = delta
        entropy_proxy = -alpha
        gibbs_free_energy = internal_energy_proxy - (t_eff * entropy_proxy) if t_eff > 0 else internal_energy_proxy
        
        T_0 = self.config.GIBBS_T0
        c = self.config.GIBBS_C
        T_critical_predicted = T_0 * np.exp(-c * alpha)
        
        phase_stability = "stable" if t_eff < T_critical_predicted else "unstable"
        phase_type = self._classify_phase(delta, kappa, t_eff, alpha)
        
        return {
            'gibbs_free_energy': float(gibbs_free_energy),
            'entropy_proxy': float(entropy_proxy),
            'critical_temperature_estimate': float(T_critical_predicted),
            'phase_stability': phase_stability,
            'phase_type': phase_type
        }
    
    def _classify_phase(self, delta: float, kappa: float, temp: float, alpha: float) -> str:
        if delta < self.config.DELTA_CRYSTAL_THRESHOLD and kappa < self.config.KAPPA_CRYSTAL_THRESHOLD and temp < self.config.TEMPERATURE_CRYSTAL_THRESHOLD:
            return "Perfect Crystal"
        if delta < self.config.DELTA_CRYSTAL_THRESHOLD and kappa >= self.config.KAPPA_CRYSTAL_THRESHOLD:
            return "Polycrystalline"
        if delta >= self.config.DELTA_GLASS_THRESHOLD and temp < self.config.TEMPERATURE_CRYSTAL_THRESHOLD:
            return "Cold Glass"
        if kappa > 1e6:
            return "Amorphous Glass"
        if alpha > self.config.ALPHA_CRYSTAL_THRESHOLD:
            return "Topological Insulator"
        return "Functional Glass"


class FullFourierAnalyzer:
    """Complete Fourier analysis for spectral fields."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
        self.grid_size = config.TORUS_GRID_SIZE
        kx = torch.fft.fftfreq(self.grid_size) * 2 * np.pi
        ky = torch.fft.fftfreq(self.grid_size) * 2 * np.pi
        self.KX, self.KY = torch.meshgrid(kx, ky, indexing='ij')
        self.K_MAG = torch.sqrt(self.KX**2 + self.KY**2)
    
    def compute_full_spectrum(self, spectral_field: torch.Tensor) -> Dict[str, torch.Tensor]:
        if spectral_field.dim() == 3:
            spectral_field = spectral_field.unsqueeze(0)
        
        B, C, H, W = spectral_field.shape
        device = spectral_field.device
        
        fft_2d = torch.fft.fft2(spectral_field, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft_2d, dim=(-2, -1))
        
        magnitude = torch.abs(fft_shifted)
        phase = torch.angle(fft_shifted)
        power_spectrum = magnitude**2
        
        power_spectrum_sum = power_spectrum.sum(dim=(-2, -1), keepdim=True) + 1e-10
        power_normalized = power_spectrum / power_spectrum_sum
        
        total_power = power_spectrum.sum(dim=(-2, -1))
        spectral_concentration = (power_spectrum.max(dim=-1)[0].max(dim=-1)[0]) / (total_power + 1e-10)
        
        return {
            'fft_2d': fft_shifted,
            'magnitude': magnitude,
            'phase': phase,
            'power_spectrum': power_spectrum,
            'power_normalized': power_normalized,
            'spectral_concentration': spectral_concentration,
            'total_power': total_power
        }
    
    def compute_resonance_metrics(self, spectral_field: torch.Tensor) -> Dict[str, Any]:
        spectrum = self.compute_full_spectrum(spectral_field)
        
        spectral_conc = spectrum['spectral_concentration'].mean().item()
        power_2d = spectrum['power_spectrum'].mean(dim=1)
        
        resonance_score = spectral_conc
        
        return {
            'spectral_concentration': spectral_conc,
            'resonance_score': float(resonance_score),
            'total_power': float(spectrum['total_power'].mean().item())
        }


class FourierMassCenterAnalyzer:
    """Analyzer for center of mass in Fourier space."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
        self.grid_size = config.TORUS_GRID_SIZE
        self.full_fourier = FullFourierAnalyzer(config)
        kx = torch.fft.fftfreq(config.TORUS_GRID_SIZE) * 2 * np.pi
        ky = torch.fft.fftfreq(config.TORUS_GRID_SIZE) * 2 * np.pi
        self.KX, self.KY = torch.meshgrid(kx, ky, indexing='ij')
    
    def compute_mass_center(self, spectral_field: torch.Tensor) -> Dict[str, torch.Tensor]:
        if spectral_field.dim() == 3:
            spectral_field = spectral_field.unsqueeze(0)
        
        B, C, H, W = spectral_field.shape
        device = spectral_field.device
        
        kx = self.KX.to(device)
        ky = self.KY.to(device)
        
        if H != self.grid_size or W != self.grid_size:
            kx = F.interpolate(
                kx.unsqueeze(0).unsqueeze(0).float(),
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
            ky = F.interpolate(
                ky.unsqueeze(0).unsqueeze(0).float(),
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
        
        density = torch.abs(spectral_field)**2
        density = density.mean(dim=1)
        
        total_mass = density.sum(dim=(-2, -1), keepdim=True) + 1e-10
        
        R_x = (kx * density).sum(dim=(-2, -1)) / total_mass.squeeze(-1).squeeze(-1)
        R_y = (ky * density).sum(dim=(-2, -1)) / total_mass.squeeze(-1).squeeze(-1)
        
        resonance = self.full_fourier.compute_resonance_metrics(spectral_field)
        
        return {
            'R_cm': torch.stack([R_x, R_y], dim=-1),
            'R_cm_x': float(R_x.mean().item()),
            'R_cm_y': float(R_y.mean().item()),
            'total_mass': total_mass.squeeze(),
            'resonance_score': resonance['resonance_score'],
            'spectral_concentration': resonance['spectral_concentration']
        }


class TopologicalPhaseDetector:
    """Detects topological phases from spectral field analysis."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
        self.mass_analyzer = FourierMassCenterAnalyzer(config)
        self.phase_state = 0.0
        self.alignment_history = np.zeros(config.TOPO_ALIGNMENT_HISTORY_LEN)
        self.history_ptr = 0
    
    def detect(self, spectral_field: torch.Tensor) -> Dict[str, Any]:
        mass_analysis = self.mass_analyzer.compute_mass_center(spectral_field)
        
        R_cm = mass_analysis['R_cm']
        alignment = R_cm[..., 0] < -0.5
        resonance_score = mass_analysis['resonance_score']
        
        alignment_val = alignment.float().mean().item() if alignment.dim() > 0 else alignment.float().item()
        self.alignment_history[self.history_ptr] = alignment_val
        self.history_ptr = (self.history_ptr + 1) % len(self.alignment_history)
        
        is_aligned = float(alignment_val > self.config.TOPO_ALIGNMENT_THRESHOLD)
        
        alpha = self.config.TOPO_PHASE_SMOOTHING
        self.phase_state = alpha * self.phase_state + (1 - alpha) * is_aligned
        
        return {
            'R_cm': R_cm.detach(),
            'R_cm_x': mass_analysis['R_cm_x'],
            'R_cm_y': mass_analysis['R_cm_y'],
            'alignment_score': float(alignment_val),
            'phase_state': float(self.phase_state),
            'is_crystalline': float(self.phase_state > 0.7),
            'resonance_score': float(resonance_score),
            'spectral_concentration': mass_analysis['spectral_concentration']
        }


class SpectralFieldExtractor:
    """Extracts spectral fields from neural network layers."""
    
    @staticmethod
    def extract(model: nn.Module, grid_size: int = 16) -> Optional[torch.Tensor]:
        if hasattr(model, 'spectral_layers'):
            layers = model.spectral_layers
        else:
            return None
        
        spectral_weights = []
        for layer in layers:
            if hasattr(layer, 'kernel_real') and hasattr(layer, 'kernel_imag'):
                kr_avg = layer.kernel_real.data.mean(dim=0)
                ki_avg = layer.kernel_imag.data.mean(dim=0)
                spectral_weights.append(torch.complex(kr_avg, ki_avg))
        
        if not spectral_weights:
            return None
        return torch.stack(spectral_weights).mean(dim=0)


class TopologicalMetricsCalculator:
    """Calculates topological metrics from model spectral fields."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
        self.phase_detector = TopologicalPhaseDetector(config)
        self.field_extractor = SpectralFieldExtractor()
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        if not self.config.TOPO_ENABLED:
            return self._empty_metrics()
        
        spectral_field = self.field_extractor.extract(model, self.config.GRID_SIZE)
        if spectral_field is None:
            return self._empty_metrics()
        
        phase_info = self.phase_detector.detect(spectral_field)
        
        return {
            'topo_R_cm_x': phase_info['R_cm_x'],
            'topo_R_cm_y': phase_info['R_cm_y'],
            'topo_phase_state': phase_info['phase_state'],
            'topo_is_crystalline': phase_info['is_crystalline'],
            'topo_alignment_score': phase_info['alignment_score'],
            'topo_resonance_score': phase_info['resonance_score'],
            'topo_spectral_concentration': phase_info['spectral_concentration']
        }
    
    @staticmethod
    def _empty_metrics() -> Dict[str, Any]:
        return {
            'topo_R_cm_x': 0.0,
            'topo_R_cm_y': 0.0,
            'topo_phase_state': 0.0,
            'topo_is_crystalline': 0.0,
            'topo_alignment_score': 0.0,
            'topo_resonance_score': 0.0,
            'topo_spectral_concentration': 0.0
        }


class GradientDynamicsCalculator:
    """Calculator for gradient-based metrics."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        val_x = kwargs.get('val_x')
        val_y = kwargs.get('val_y')
        
        if val_x is None or val_y is None:
            return {
                'kappa': float('inf'),
                'effective_temperature': 0.0,
                'gradient_variance': 0.0
            }
        
        model.train()
        grads = []
        
        for i in range(self.config.KAPPA_GRADIENT_BATCHES):
            try:
                model.zero_grad()
                outputs = model(val_x)
                loss = F.mse_loss(outputs, val_y)
                loss.backward()
                
                grad_list = []
                for p in model.parameters():
                    if p.grad is not None and p.grad.numel() > 0:
                        grad_list.append(p.grad.flatten())
                
                if grad_list:
                    grad_vector = torch.cat(grad_list)
                    if torch.isfinite(grad_vector).all():
                        grads.append(grad_vector.detach().clone())
            except Exception:
                continue
        
        model.eval()
        
        if len(grads) < 2:
            return {
                'kappa': float('inf'),
                'effective_temperature': 0.0,
                'gradient_variance': 0.0
            }
        
        grads_tensor = torch.stack(grads)
        n_samples, n_dims = grads_tensor.shape
        
        if n_dims > self.config.KAPPA_MAX_DIM:
            indices = torch.randperm(n_dims, device=grads_tensor.device)[:self.config.KAPPA_MAX_DIM]
            grads_tensor = grads_tensor[:, indices]
        
        try:
            if n_samples < n_dims:
                gram = torch.mm(grads_tensor, grads_tensor.t()) / max(n_samples - 1, 1)
                eigenvals = torch.linalg.eigvalsh(gram)
            else:
                cov = torch.cov(grads_tensor.t())
                eigenvals = torch.linalg.eigvalsh(cov).real
            
            eigenvals = eigenvals[eigenvals > self.config.EIGENVALUE_TOL]
            
            if len(eigenvals) == 0:
                return {
                    'kappa': float('inf'),
                    'effective_temperature': 0.0,
                    'gradient_variance': 0.0
                }
            
            kappa = (eigenvals.max() / eigenvals.min()).item()
            
            second_moment = torch.mean(torch.norm(grads_tensor, dim=1)**2)
            first_moment_sq = torch.norm(torch.mean(grads_tensor, dim=0))**2
            variance = second_moment - first_moment_sq
            
            temperature = float(variance / (2.0 * grads_tensor.shape[1]))
            
            return {
                'kappa': kappa,
                'effective_temperature': temperature,
                'gradient_variance': float(variance)
            }
        except Exception:
            return {
                'kappa': float('inf'),
                'effective_temperature': 0.0,
                'gradient_variance': 0.0
            }


class SchrodingerAnalyzer:
    """Quantum mechanical analysis of network parameters."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
        self.target_dim = config.COMPRESSED_DIMENSION
        self.projection_matrix = None
    
    def extract_compressed_wavefunction(self, model: nn.Module) -> torch.Tensor:
        all_params = []
        
        for name, param in model.named_parameters():
            if param.numel() > 0:
                flat = param.data.flatten()
                all_params.append(flat)
        
        full_vector = torch.cat(all_params)
        total_params = full_vector.numel()
        
        if total_params <= self.target_dim:
            if len(full_vector) < self.target_dim:
                padding = torch.zeros(self.target_dim - len(full_vector), 
                                    dtype=full_vector.dtype, 
                                    device=full_vector.device)
                full_vector = torch.cat([full_vector, padding])
            return full_vector[:self.target_dim]
        
        return self._compress_johnson_lindenstrauss(full_vector)
    
    def _compress_johnson_lindenstrauss(self, vector: torch.Tensor) -> torch.Tensor:
        if self.projection_matrix is None or self.projection_matrix.shape[1] != len(vector):
            self.projection_matrix = torch.randn(
                self.target_dim, 
                len(vector), 
                device=vector.device,
                dtype=torch.float32
            ) / np.sqrt(self.target_dim)
        
        compressed = torch.matmul(self.projection_matrix, vector.float())
        return compressed
    
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        wave_function = self.extract_compressed_wavefunction(model)
        n_dim = len(wave_function)
        
        probability_density = torch.abs(wave_function) ** 2
        probability_density = probability_density / (torch.sum(probability_density) + 1e-10)
        
        entropy = -torch.sum(probability_density * torch.log(probability_density + 1e-10)).item()
        
        participation_ratio = 1.0 / torch.sum(probability_density ** 2).item()
        
        coherence = torch.sum(torch.abs(torch.outer(wave_function, wave_function.conj()))).item() - torch.sum(torch.abs(wave_function)**2).item()
        
        return {
            'wavefunction_entropy': entropy,
            'participation_ratio': participation_ratio,
            'quantum_coherence': coherence,
            'compressed_dimension': n_dim
        }


class ComprehensiveVisualizer:
    """Generates comprehensive visualizations for all metrics."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
    
    def visualize_checkpoint_analysis(self, results: Dict[str, Any], output_path: str):
        fig = plt.figure(figsize=(20, 16), dpi=self.config.FIGURE_DPI)
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        epoch = results.get('metadata', {}).get('epoch', 'unknown')
        fig.suptitle(f'Comprehensive Crystallographic Analysis - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        self._plot_weight_distribution(results, fig.add_subplot(gs[0, 0]))
        self._plot_spectral_analysis(results, fig.add_subplot(gs[0, 1]))
        self._plot_phase_diagram(results, fig.add_subplot(gs[0, 2]))
        self._plot_curvature_distribution(results, fig.add_subplot(gs[0, 3]))
        self._plot_level_spacing(results, fig.add_subplot(gs[1, 0]))
        self._plot_eigenvalue_spectrum(results, fig.add_subplot(gs[1, 1]))
        self._plot_thermodynamic_potentials(results, fig.add_subplot(gs[1, 2]))
        self._plot_topological_metrics(results, fig.add_subplot(gs[1, 3]))
        self._plot_berry_phase(results, fig.add_subplot(gs[2, 0]))
        self._plot_control_stability(results, fig.add_subplot(gs[2, 1]))
        self._plot_quantum_metrics(results, fig.add_subplot(gs[2, 2]))
        self._plot_summary_table(results, fig.add_subplot(gs[2, 3]))
        self._plot_layer_deltas(results, fig.add_subplot(gs[3, 0]))
        self._plot_resonance_metrics(results, fig.add_subplot(gs[3, 1]))
        self._plot_spectral_concentration(results, fig.add_subplot(gs[3, 2]))
        self._plot_health_score(results, fig.add_subplot(gs[3, 3]))
        
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, format=self.config.SAVE_FORMAT, bbox_inches='tight')
        plt.close()
    
    def _plot_weight_distribution(self, results: Dict, ax):
        if 'weight_integrity' in results:
            wi = results['weight_integrity']
            total = wi.get('total_params', 1)
            valid = total - wi.get('nan_count', 0) - wi.get('inf_count', 0)
            labels = ['Valid', 'NaN', 'Inf']
            sizes = [valid, wi.get('nan_count', 0), wi.get('inf_count', 0)]
            colors = ['#2E86AB', '#D62828', '#F18F01']
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Weight Integrity', fontweight='bold')
    
    def _plot_spectral_analysis(self, results: Dict, ax):
        if 'spectral_geometry' in results:
            sg = results['spectral_geometry']
            metrics = ['Spectral Gap', 'Participation Ratio', 'Eff. Dimension']
            values = [
                sg.get('spectral_gap', 0),
                sg.get('participation_ratio', 0),
                sg.get('effective_dimension', 0) / 100
            ]
            colors = ['#06A77D', '#A23B72', '#2E86AB']
            ax.bar(metrics, values, color=colors, alpha=0.7)
            ax.set_title('Spectral Geometry', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_phase_diagram(self, results: Dict, ax):
        if 'thermodynamics' in results and 'discretization' in results:
            thermo = results['thermodynamics']
            disc = results['discretization']
            alpha = disc.get('alpha', 0)
            temp = thermo.get('effective_temperature', 1) if 'effective_temperature' in thermo else 1
            
            ax.scatter([alpha], [temp], s=200, c='#D62828', marker='*', zorder=5)
            ax.axhline(y=self.config.TEMPERATURE_CRYSTAL_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=self.config.ALPHA_CRYSTAL_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Alpha (Purity)')
            ax.set_ylabel('Temperature')
            ax.set_title('Phase Diagram', fontweight='bold')
            ax.set_xlim(0, max(15, alpha * 1.2))
            ax.set_ylim(1e-12, max(1e-3, temp * 10))
            ax.set_yscale('log')
    
    def _plot_curvature_distribution(self, results: Dict, ax):
        if 'ricci_curvature' in results:
            rc = results['ricci_curvature']
            metrics = ['Ricci Scalar', 'Mean Sectional', 'Variance']
            values = [
                rc.get('ricci_scalar', 0),
                rc.get('mean_sectional_curvature', 0),
                rc.get('curvature_variance', 0)
            ]
            colors = ['#F18F01', '#06A77D', '#A23B72']
            ax.bar(metrics, values, color=colors, alpha=0.7)
            ax.set_title('Ricci Curvature', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_level_spacing(self, results: Dict, ax):
        if 'spectral_geometry' in results:
            sg = results['spectral_geometry']
            lsr = sg.get('level_spacing_ratio', 0)
            
            ax.bar(['Level Spacing Ratio'], [lsr], color='#2E86AB', alpha=0.7)
            ax.axhline(y=self.config.LEVEL_SPACING_WIGNER_DYSON, color='red', linestyle='--', 
                      label=f'Wigner-Dyson: {self.config.LEVEL_SPACING_WIGNER_DYSON}')
            ax.axhline(y=self.config.LEVEL_SPACING_POISSON, color='green', linestyle='--',
                      label=f'Poisson: {self.config.LEVEL_SPACING_POISSON}')
            ax.set_ylabel('Ratio')
            ax.set_title('MBL Level Spacing', fontweight='bold')
            ax.legend(fontsize=8)
    
    def _plot_eigenvalue_spectrum(self, results: Dict, ax):
        if 'spectral_geometry' in results:
            sg = results['spectral_geometry']
            largest = sg.get('largest_eigenvalue', 0)
            smallest = sg.get('smallest_eigenvalue', 0)
            
            if largest > 0 and smallest > 0:
                ax.bar(['Largest', 'Smallest'], [np.log10(largest + 1e-10), np.log10(smallest + 1e-10)], 
                      color=['#2E86AB', '#A23B72'], alpha=0.7)
                ax.set_ylabel('log10(Eigenvalue)')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
            ax.set_title('Eigenvalue Spectrum', fontweight='bold')
    
    def _plot_thermodynamic_potentials(self, results: Dict, ax):
        if 'thermodynamics' in results:
            thermo = results['thermodynamics']
            metrics = ['Gibbs Free Energy', 'Entropy', 'Critical T']
            values = [
                thermo.get('gibbs_free_energy', 0),
                thermo.get('entropy_proxy', 0),
                thermo.get('critical_temperature_estimate', 0)
            ]
            colors = ['#06A77D', '#D62828', '#F18F01']
            ax.bar(metrics, values, color=colors, alpha=0.7)
            ax.set_title('Thermodynamics', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_topological_metrics(self, results: Dict, ax):
        if 'topological' in results:
            topo = results['topological']
            metrics = ['Phase State', 'Alignment', 'Resonance']
            values = [
                topo.get('topo_phase_state', 0),
                topo.get('topo_alignment_score', 0),
                topo.get('topo_resonance_score', 0)
            ]
            colors = ['#2E86AB', '#A23B72', '#06A77D']
            ax.bar(metrics, values, color=colors, alpha=0.7)
            ax.set_title('Topological Metrics', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_berry_phase(self, results: Dict, ax):
        if 'berry_phase' in results:
            bp = results['berry_phase']
            total = bp.get('total_berry_phase', 0)
            mod = bp.get('berry_phase_mod_2pi', 0)
            winding = bp.get('winding_number', 0)
            
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'gray', linestyle='--', alpha=0.5)
            
            angle = mod
            ax.arrow(0, 0, 0.9*np.cos(angle), 0.9*np.sin(angle), 
                    head_width=0.1, head_length=0.05, fc='blue', ec='blue')
            ax.plot(np.cos(angle), np.sin(angle), 'ro', markersize=10)
            
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_aspect('equal')
            ax.set_title(f'Berry Phase: {mod:.3f} rad\nWinding: {winding}', fontweight='bold')
    
    def _plot_control_stability(self, results: Dict, ax):
        if 'control_theory' in results:
            ct = results['control_theory']
            stability = ct.get('stability_analysis', {})
            
            is_stable = stability.get('is_stable', False)
            margin = stability.get('stability_margin', 0)
            
            color = '#06A77D' if is_stable else '#D62828'
            ax.bar(['Stability Margin', 'Is Stable'], [margin, float(is_stable)], color=[color, color], alpha=0.7)
            ax.set_title(f'Control Theory\n{"Stable" if is_stable else "Unstable"}', fontweight='bold')
    
    def _plot_quantum_metrics(self, results: Dict, ax):
        if 'schrodinger' in results:
            sch = results['schrodinger']
            metrics = ['Entropy', 'Participation', 'Coherence']
            values = [
                sch.get('wavefunction_entropy', 0),
                sch.get('participation_ratio', 0) / 10,
                sch.get('quantum_coherence', 0)
            ]
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            ax.bar(metrics, values, color=colors, alpha=0.7)
            ax.set_title('Quantum Metrics', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_summary_table(self, results: Dict, ax):
        ax.axis('off')
        
        summary_text = "Summary\n" + "="*30 + "\n"
        
        if 'discretization' in results:
            disc = results['discretization']
            summary_text += f"Alpha: {disc.get('alpha', 0):.4f}\n"
            summary_text += f"Delta: {disc.get('delta', 0):.6f}\n"
            summary_text += f"Phase: {results.get('thermodynamics', {}).get('phase_type', 'Unknown')}\n"
        
        if 'topological' in results:
            topo = results['topological']
            summary_text += f"Crystalline: {'Yes' if topo.get('topo_is_crystalline', 0) > 0.5 else 'No'}\n"
        
        if 'health_score' in results:
            summary_text += f"Health Score: {results['health_score']:.2f}\n"
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.set_title('Summary', fontweight='bold')
    
    def _plot_layer_deltas(self, results: Dict, ax):
        if 'discretization' in results and 'layer_deltas' in results['discretization']:
            layer_deltas = results['discretization']['layer_deltas']
            names = list(layer_deltas.keys())[:10]
            values = [layer_deltas[n] for n in names]
            
            ax.barh(range(len(names)), values, color='#2E86AB', alpha=0.7)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([n[:20] for n in names], fontsize=8)
            ax.set_xlabel('Delta')
            ax.set_title('Layer Discretization', fontweight='bold')
    
    def _plot_resonance_metrics(self, results: Dict, ax):
        if 'topological' in results:
            topo = results['topological']
            metrics = ['Spectral Conc.', 'Resonance Score']
            values = [
                topo.get('topo_spectral_concentration', 0),
                topo.get('topo_resonance_score', 0)
            ]
            colors = ['#F18F01', '#06A77D']
            ax.bar(metrics, values, color=colors, alpha=0.7)
            ax.set_title('Resonance', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_spectral_concentration(self, results: Dict, ax):
        if 'spectral_geometry' in results:
            sg = results['spectral_geometry']
            gap = sg.get('spectral_gap', 0)
            pr = sg.get('participation_ratio', 0)
            
            ax.scatter([gap], [pr], s=200, c='#D62828', marker='*', zorder=5)
            ax.set_xlabel('Spectral Gap')
            ax.set_ylabel('Participation Ratio')
            ax.set_title('Spectral Metrics', fontweight='bold')
    
    def _plot_health_score(self, results: Dict, ax):
        if 'health_score' in results:
            score = results['health_score']
            
            colors = ['#D62828', '#F18F01', '#06A77D']
            color = colors[0] if score < 0.33 else colors[1] if score < 0.67 else colors[2]
            
            ax.bar(['Health Score'], [score], color=color, alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_title(f'Health: {score:.2f}', fontweight='bold')


class CheckpointAnalyzer:
    """Main analyzer that orchestrates all metric calculations."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
        self.logger = LoggerFactory.create_logger("CheckpointAnalyzer", config=config)
        
        self.weight_integrity_calc = WeightIntegrityCalculator(config)
        self.discretization_calc = DiscretizationCalculator(config)
        self.spectral_geometry_calc = SpectralGeometryCalculator(config)
        self.ricci_curvature_calc = RicciCurvatureCalculator(config)
        self.berry_phase_calc = BerryPhaseCalculator(config)
        self.control_analyzer = ControlSystemAnalyzer(config)
        self.thermodynamic_calc = ThermodynamicCalculator(config)
        self.topological_calc = TopologicalMetricsCalculator(config)
        self.gradient_dynamics_calc = GradientDynamicsCalculator(config)
        self.schrodinger_analyzer = SchrodingerAnalyzer(config)
        self.visualizer = ComprehensiveVisualizer(config)
    
    def analyze_checkpoint(self, checkpoint_path: str, val_data: Optional[Tuple] = None) -> Dict[str, Any]:
        self.logger.info(f"Analyzing checkpoint: {checkpoint_path}")
        
        start_time = time.time()
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return {'error': str(e), 'checkpoint_path': checkpoint_path}
        
        model = DiracSpectralNetwork(self.config)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown') if isinstance(checkpoint, dict) else 'unknown'
        metrics_history = checkpoint.get('metrics', {}) if isinstance(checkpoint, dict) else {}
        
        results = {
            'metadata': {
                'checkpoint_path': checkpoint_path,
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'analysis_duration_seconds': 0
            }
        }
        
        val_x, val_y = val_data if val_data else (None, None)
        
        self.logger.info("Computing weight integrity...")
        results['weight_integrity'] = self.weight_integrity_calc.compute(model)
        
        self.logger.info("Computing discretization metrics...")
        results['discretization'] = self.discretization_calc.compute(model)
        
        self.logger.info("Computing spectral geometry...")
        results['spectral_geometry'] = self.spectral_geometry_calc.compute(model)
        
        self.logger.info("Computing Ricci curvature...")
        results['ricci_curvature'] = self.ricci_curvature_calc.compute(model)
        
        self.logger.info("Computing control theory analysis...")
        results['control_theory'] = self.control_analyzer.compute(model)
        
        self.logger.info("Computing topological metrics...")
        results['topological'] = self.topological_calc.compute(model)
        
        self.logger.info("Computing Schrodinger analysis...")
        results['schrodinger'] = self.schrodinger_analyzer.compute(model)
        
        self.logger.info("Computing gradient dynamics...")
        grad_results = self.gradient_dynamics_calc.compute(model, val_x=val_x, val_y=val_y)
        results['gradient_dynamics'] = grad_results
        
        self.logger.info("Computing thermodynamic metrics...")
        results['thermodynamics'] = self.thermodynamic_calc.compute(
            model,
            delta=results['discretization'].get('delta', 1.0),
            alpha=results['discretization'].get('alpha', 0.0),
            kappa=grad_results.get('kappa', 1.0),
            effective_temperature=grad_results.get('effective_temperature', 1.0)
        )
        
        results['health_score'] = self._compute_health_score(results)
        
        results['metadata']['analysis_duration_seconds'] = time.time() - start_time
        
        self.logger.info(f"Analysis completed in {results['metadata']['analysis_duration_seconds']:.2f}s")
        
        return results
    
    def _compute_health_score(self, results: Dict[str, Any]) -> float:
        score = 0.0
        count = 0
        
        if 'weight_integrity' in results:
            wi = results['weight_integrity']
            score += float(wi.get('is_valid', False))
            count += 1
        
        if 'discretization' in results:
            disc = results['discretization']
            alpha = disc.get('alpha', 0)
            score += min(alpha / self.config.ALPHA_CRYSTAL_THRESHOLD, 1.0)
            count += 1
        
        if 'spectral_geometry' in results:
            sg = results['spectral_geometry']
            lsr = sg.get('level_spacing_ratio', 0)
            mbl_score = 1.0 - abs(lsr - self.config.LEVEL_SPACING_POISSON) / self.config.LEVEL_SPACING_TOLERANCE
            score += max(0, min(1, mbl_score))
            count += 1
        
        if 'topological' in results:
            topo = results['topological']
            score += topo.get('topo_phase_state', 0)
            count += 1
        
        return score / count if count > 0 else 0.0


class BatchProcessor:
    """Processes multiple checkpoints in batch mode."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
        self.logger = LoggerFactory.create_logger("BatchProcessor", config=config)
        self.analyzer = CheckpointAnalyzer(config)
        self.visualizer = ComprehensiveVisualizer(config)
    
    def process_directory(self, checkpoint_dir: str, output_dir: str, val_data: Optional[Tuple] = None):
        self.logger.info(f"Processing directory: {checkpoint_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        pattern = os.path.join(checkpoint_dir, "*.pth")
        checkpoint_files = sorted(glob.glob(pattern))
        
        if not checkpoint_files:
            self.logger.warning(f"No checkpoint files found in {checkpoint_dir}")
            return
        
        self.logger.info(f"Found {len(checkpoint_files)} checkpoints")
        
        all_results = []
        
        for i, checkpoint_path in enumerate(checkpoint_files):
            self.logger.info(f"Processing checkpoint {i+1}/{len(checkpoint_files)}: {checkpoint_path}")
            
            try:
                results = self.analyzer.analyze_checkpoint(checkpoint_path, val_data)
                all_results.append(results)
                
                checkpoint_name = Path(checkpoint_path).stem
                
                viz_path = os.path.join(output_dir, f"{checkpoint_name}_analysis.{self.config.SAVE_FORMAT}")
                self.visualizer.visualize_checkpoint_analysis(results, viz_path)
                
                json_path = os.path.join(output_dir, f"{checkpoint_name}_metrics.json")
                with open(json_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
            except Exception as e:
                self.logger.error(f"Error processing {checkpoint_path}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        summary = self._generate_summary(all_results)
        summary_path = os.path.join(output_dir, "analysis_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self._generate_evolution_plots(all_results, output_dir)
        
        self.logger.info(f"Batch processing complete. Results saved to {output_dir}")
    
    def _generate_summary(self, all_results: List[Dict]) -> Dict[str, Any]:
        if not all_results:
            return {}

        alphas = [r.get('discretization', {}).get('alpha', 0) for r in all_results]
        deltas = [r.get('discretization', {}).get('delta', 0) for r in all_results]
        health_scores = [r.get('health_score', 0) for r in all_results]
        ricci_scalars = [r.get('ricci_curvature', {}).get('ricci_scalar', 0) for r in all_results]
        lsrs = [r.get('spectral_geometry', {}).get('level_spacing_ratio', 0) for r in all_results]
        phases = [r.get('thermodynamics', {}).get('phase_type', 'Unknown') for r in all_results]

        # Fixed: use dictionary comprehension instead of Counter
        phase_distribution = {k: phases.count(k) for k in set(phases)}

        return {
            'total_checkpoints': len(all_results),
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'alpha': {
                    'mean': float(np.mean(alphas)) if alphas else 0,
                    'std': float(np.std(alphas)) if alphas else 0,
                    'min': float(np.min(alphas)) if alphas else 0,
                    'max': float(np.max(alphas)) if alphas else 0
                },
                'delta': {
                    'mean': float(np.mean(deltas)) if deltas else 0,
                    'std': float(np.std(deltas)) if deltas else 0,
                    'min': float(np.min(deltas)) if deltas else 0,
                    'max': float(np.max(deltas)) if deltas else 0
                },
                'health_score': {
                    'mean': float(np.mean(health_scores)) if health_scores else 0,
                    'std': float(np.std(health_scores)) if health_scores else 0,
                    'min': float(np.min(health_scores)) if health_scores else 0,
                    'max': float(np.max(health_scores)) if health_scores else 0
                },
                'ricci_scalar': {
                    'mean': float(np.mean(ricci_scalars)) if ricci_scalars else 0,
                    'std': float(np.std(ricci_scalars)) if ricci_scalars else 0,
                    'min': float(np.min(ricci_scalars)) if ricci_scalars else 0,
                    'max': float(np.max(ricci_scalars)) if ricci_scalars else 0
                },
                'level_spacing_ratio': {
                    'mean': float(np.mean(lsrs)) if lsrs else 0,
                    'std': float(np.std(lsrs)) if lsrs else 0
                },
                'phase_distribution': phase_distribution
            },
            'epochs': [r.get('metadata', {}).get('epoch', i) for i, r in enumerate(all_results)]
        }

    def _generate_evolution_plots(self, all_results: List[Dict], output_dir: str):
        if not all_results or len(all_results) < 2:
            return
        
        epochs = [r.get('metadata', {}).get('epoch', i) for i, r in enumerate(all_results)]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=self.config.FIGURE_DPI)
        fig.suptitle('Metric Evolution Across Checkpoints', fontsize=14, fontweight='bold')
        
        metrics_to_plot = [
            ('discretization', 'alpha', 'Alpha (Purity)', axes[0, 0]),
            ('discretization', 'delta', 'Delta (Discretization)', axes[0, 1]),
            ('health_score', None, 'Health Score', axes[0, 2]),
            ('ricci_curvature', 'ricci_scalar', 'Ricci Scalar', axes[1, 0]),
            ('spectral_geometry', 'level_spacing_ratio', 'Level Spacing Ratio', axes[1, 1]),
            ('spectral_geometry', 'spectral_gap', 'Spectral Gap', axes[1, 2]),
            ('topological', 'topo_phase_state', 'Phase State', axes[2, 0]),
            ('schrodinger', 'wavefunction_entropy', 'Wavefunction Entropy', axes[2, 1]),
            ('gradient_dynamics', 'effective_temperature', 'Effective Temperature', axes[2, 2])
        ]
        
        for category, key, label, ax in metrics_to_plot:
            if category == 'health_score':
                values = [r.get('health_score', 0) for r in all_results]
            else:
                values = [r.get(category, {}).get(key, 0) for r in all_results]
            
            ax.plot(epochs, values, 'o-', color='#2E86AB', linewidth=2, markersize=6)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(label)
            ax.set_title(label, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        evolution_path = os.path.join(output_dir, f"metric_evolution.{self.config.SAVE_FORMAT}")
        plt.savefig(evolution_path, dpi=self.config.FIGURE_DPI, format=self.config.SAVE_FORMAT)
        plt.close()
        
        self.logger.info(f"Evolution plots saved to {evolution_path}")


class DiracCrystallographySuite:
    """Main entry point for the crystallography analysis suite."""
    
    def __init__(self, config: CrystallographySuiteConfig):
        self.config = config
        self.logger = LoggerFactory.create_logger("DiracCrystallographySuite", config=config)
        self.batch_processor = BatchProcessor(config)
        self.berry_phase_calc = BerryPhaseCalculator(config)
    
    def run_analysis(self, checkpoint_dir: str, output_dir: str):
        self.logger.info("=" * 70)
        self.logger.info("DIRAC CRYSTALLOGRAPHY ANALYSIS SUITE")
        self.logger.info("=" * 70)
        
        start_time = time.time()
        
        self.batch_processor.process_directory(checkpoint_dir, output_dir)
        
        berry_results = self.berry_phase_calc.calculate_berry_phase(checkpoint_dir)
        
        berry_path = os.path.join(output_dir, "berry_phase_analysis.json")
        with open(berry_path, 'w') as f:
            json.dump(berry_results, f, indent=2, default=str)
        
        self._generate_berry_phase_visualization(berry_results, output_dir)
        
        total_time = time.time() - start_time
        self.logger.info("=" * 70)
        self.logger.info(f"ANALYSIS COMPLETE")
        self.logger.info(f"Total duration: {total_time:.2f} seconds")
        self.logger.info(f"Results saved to: {output_dir}")
        self.logger.info("=" * 70)
    
    def _generate_berry_phase_visualization(self, berry_results: Dict, output_dir: str):
        if 'error' in berry_results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.config.FIGURE_DPI)
        
        total_phase = berry_results.get('total_berry_phase', 0)
        mod_2pi = berry_results.get('berry_phase_mod_2pi', 0)
        winding = berry_results.get('winding_number', 0)
        
        theta = np.linspace(0, 2*np.pi, 100)
        axes[0].plot(np.cos(theta), np.sin(theta), 'gray', linestyle='--', alpha=0.5)
        
        angle = mod_2pi
        axes[0].arrow(0, 0, 0.9*np.cos(angle), 0.9*np.sin(angle), 
                     head_width=0.1, head_length=0.05, fc='blue', ec='blue')
        axes[0].plot(np.cos(angle), np.sin(angle), 'ro', markersize=15)
        
        axes[0].set_xlim(-1.5, 1.5)
        axes[0].set_ylim(-1.5, 1.5)
        axes[0].set_aspect('equal')
        axes[0].set_title(f'Berry Phase: {mod_2pi:.4f} rad ({np.degrees(mod_2pi):.1f} deg)', fontweight='bold')
        axes[0].set_xlabel('Re(e^{iγ})')
        axes[0].set_ylabel('Im(e^{iγ})')
        axes[0].grid(True, alpha=0.3)
        
        info_text = f"""
Berry Phase Analysis
--------------------
Total Phase: {total_phase:.6f} rad
Phase (mod 2π): {mod_2pi:.6f} rad
Winding Number: {winding}

Interpretation:
{"Trivial (γ ≈ 0)" if abs(mod_2pi) < 0.1 else "Non-trivial Z₂ (γ ≈ π)" if abs(abs(mod_2pi) - np.pi) < 0.3 else "Generic topological"}

Checkpoints analyzed: {berry_results.get('num_checkpoints', 0)}
"""
        axes[1].text(0.1, 0.5, info_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[1].axis('off')
        axes[1].set_title('Berry Phase Summary', fontweight='bold')
        
        plt.tight_layout()
        berry_viz_path = os.path.join(output_dir, f"berry_phase_visualization.{self.config.SAVE_FORMAT}")
        plt.savefig(berry_viz_path, dpi=self.config.FIGURE_DPI, format=self.config.SAVE_FORMAT)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Dirac Equation Crystallography Analysis Suite'
    )
    parser.add_argument(
        '--checkpoint_dir', '-c',
        type=str,
        default='checkpoints_dirac',
        help='Directory containing checkpoints'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='crystallography_analysis',
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--grid_size', '-g',
        type=int,
        default=16,
        help='Grid size for network'
    )
    parser.add_argument(
        '--hidden_dim', '-hd',
        type=int,
        default=32,
        help='Hidden dimension'
    )
    parser.add_argument(
        '--expansion_dim', '-ed',
        type=int,
        default=64,
        help='Expansion dimension'
    )
    parser.add_argument(
        '--spectral_layers', '-sl',
        type=int,
        default=2,
        help='Number of spectral layers'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    config = CrystallographySuiteConfig(
        GRID_SIZE=args.grid_size,
        HIDDEN_DIM=args.hidden_dim,
        EXPANSION_DIM=args.expansion_dim,
        NUM_SPECTRAL_LAYERS=args.spectral_layers,
        LOG_LEVEL=args.log_level
    )
    
    suite = DiracCrystallographySuite(config)
    suite.run_analysis(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()