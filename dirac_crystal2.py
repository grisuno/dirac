#!/usr/bin/env python3
"""
dirac_crystal.py

Author: Gris Iscomeback
Email: grisiscomeback@gmail.com
Date of creation: 2026
License: AGPL v3

Description:
Dirac Equation Grokking via Hamiltonian Topological Crystallization.

The Dirac equation: (i * gamma^mu * partial_mu - m) * psi = 0

This implementation extends the Schrodinger crystallization architecture
to handle relativistic quantum mechanics with 4-component spinors and
gamma matrices. The network learns to evolve Dirac spinors under the
full relativistic Hamiltonian including the mass term.

Five-phase protocol:
  Phase 1 - Batch size prospecting
  Phase 2 - Seed mining with decreasing delta criterion
  Phase 3 - Full training of best seed + batch size until grokking
  Phase 4 - Refinement via simulated annealing toward crystal state
  Phase 5 - Quadruple precision (float128) high-pressure crystallization
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from collections import deque
import logging
import math
import copy
import warnings

warnings.filterwarnings('ignore')


@dataclass
class Config:
    GRID_SIZE: int = 16
    HIDDEN_DIM: int = 32
    NUM_SPECTRAL_LAYERS: int = 2
    EXPANSION_DIM: int = 64
    SPINOR_COMPONENTS: int = 4
    POTENTIAL_DEPTH: float = 5.0
    POTENTIAL_WIDTH: float = 0.3
    NUM_EIGENSTATES: int = 8
    ENERGY_SCALE: float = 1.0
    SPINOR_NORM_TARGET: float = 1.0

    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.005
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 5000
    REFINEMENT_EPOCHS: int = 2000
    PHASE5_EPOCHS: int = 3000
    CHECKPOINT_INTERVAL_MINUTES: int = 5
    MAX_CHECKPOINTS: int = 10
    TARGET_ACCURACY: float = 0.95
    TIME_STEPS: int = 2
    DT: float = 0.01
    TRAIN_RATIO: float = 0.7
    NUM_SAMPLES: int = 200
    GRADIENT_CLIP_NORM: float = 1.0
    NOISE_AMPLITUDE: float = 0.01
    NOISE_INTERVAL_EPOCHS: int = 25
    MOMENTUM: float = 0.9
    COSINE_ANNEALING_ETA_MIN_FACTOR: float = 0.01
    MSE_THRESHOLD: float = 0.05
    CYCLIC_LR_BASE_FACTOR: float = 0.01
    CYCLIC_LR_MAX_FACTOR: float = 2.0
    CYCLIC_LR_STEP_SIZE: int = 50

    ENTROPY_BINS: int = 50
    PCA_COMPONENTS: int = 2
    KDE_BANDWIDTH: str = 'scott'
    MIN_VARIANCE_THRESHOLD: float = 1e-8
    ENTROPY_EPS: float = 1e-10
    HBAR: float = 1e-6
    HBAR_PHYSICAL: float = 1.0545718e-34
    POYNTING_THRESHOLD: float = 1.0
    ENERGY_FLOW_SCALE: float = 0.1
    DISCRETIZATION_MARGIN: float = 0.1
    TARGET_SLOTS: int = 7
    KAPPA_MAX_DIM: int = 10000
    EIGENVALUE_TOL: float = 1e-10
    KAPPA_GRADIENT_BATCHES: int = 5
    ALPHA_CRYSTAL_THRESHOLD: float = 7.0
    ALPHA_PERFECT_CRYSTAL_THRESHOLD: float = 10.0
    SPECTRAL_PEAK_LIMIT: int = 10
    SPECTRAL_POWER_LIMIT: int = 100
    PARAM_FLATTEN_LIMIT: int = 1000
    GRADIENT_BUFFER_LIMIT: int = 500
    GRADIENT_BUFFER_WINDOW: int = 10
    LOSS_HISTORY_WINDOW: int = 50
    GRADIENT_BUFFER_MAXLEN: int = 50
    LOSS_HISTORY_MAXLEN: int = 100
    TEMP_HISTORY_MAXLEN: int = 100
    CV_HISTORY_MAXLEN: int = 100
    CV_THRESHOLD: float = 1.0
    WEIGHT_METRIC_DIM_LIMIT: int = 256

    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED: int = 42
    LOG_LEVEL: str = 'INFO'
    RESULTS_DIR: str = 'dirac_results'

    MINING_MAX_ATTEMPTS: int = 200
    MINING_START_SEED: int = 1
    MINING_GLASS_PATIENCE_EPOCHS: int = 50
    MINING_PROSPECT_EPOCHS: int = 40
    MINING_PROSPECT_DELTA_EPOCH_INTERVAL: int = 10
    MINING_TARGET_LC: float = 0.01
    MINING_TARGET_SP: float = 0.01
    MINING_TARGET_KAPPA: float = 1.01
    MINING_TARGET_DELTA: float = 0.001
    MINING_TARGET_TEMP: float = 1e-10
    MINING_TARGET_CV: float = 1e-10

    BATCH_CANDIDATES: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    BATCH_PROSPECT_EPOCHS: int = 30
    BATCH_PROSPECT_SEED: int = 42

    LAMBDA_INITIAL: float = 1.0
    LAMBDA_MAX: float = 1e35
    LAMBDA_GROWTH_FACTOR: float = 10.0
    LAMBDA_GROWTH_INTERVAL_EPOCHS: int = 500
    LAMBDA_PRECISION_DTYPE: str = 'float64'

    ANNEALING_INITIAL_TEMPERATURE: float = 1.0
    ANNEALING_FINAL_TEMPERATURE: float = 1e-6
    ANNEALING_COOLING_RATE: float = 0.995
    ANNEALING_RESTART_THRESHOLD: float = 0.01

    GROKKING_TRAIN_ACC_THRESHOLD: float = 0.99
    GROKKING_VAL_ACC_THRESHOLD: float = 0.95
    GROKKING_PATIENCE: int = 200
    GROKKING_DELTA_SLOPE_WINDOW: int = 50
    GROKKING_DELTA_SLOPE_THRESHOLD: float = -1e-6

    BACKBONE_CHECKPOINT_PATH: str = 'best.pth'
    BACKBONE_ENABLED: bool = True
    BACKBONE_ADAPT_CHANNELS: bool = True

    LOG_INTERVAL_EPOCHS: int = 10

    NORMALIZATION_EPS: float = 1e-8
    POYNTING_OUTER_LIMIT: int = 50

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

    FOURIER_MODE_COUPLING: float = 0.3
    FOURIER_DOMINANT_MODE_THRESHOLD: float = 0.5
    FOURIER_SPECTRAL_CONCENTRATION_THRESHOLD: float = 0.8
    FOURIER_PHASE_COHERENCE_THRESHOLD: float = 0.6

    PHASE5_LAMBDA_INITIAL: float = 1e5
    PHASE5_LAMBDA_MAX: float = 1e35
    PHASE5_LAMBDA_GROWTH_FACTOR: float = 1.5
    PHASE5_LAMBDA_GROWTH_INTERVAL_EPOCHS: int = 200
    PHASE5_PRECISION: str = 'float128'
    PHASE5_THERMAL_INJECTION_SCALE: float = 1e-6
    PHASE5_DELTA_TARGET: float = 0.001
    PHASE5_ALPHA_TARGET: float = 7.0
    PHASE5_ENABLE: bool = True
    PHASE5_CHECKPOINT_LATEST_PATH: str = 'weights/dirac_phase5_latest.pth'
    PHASE5_THERMAL_RESCALING_FACTOR: float = 1e10
    RESUME_PHASE5: bool = False
    RESUME_PHASE4: bool = False
    RESUME_PHASE3: bool = False
    PHASE3_CHECKPOINT_PATH: str = 'checkpoints_dirac_phase3/latest.pth'
    PHASE4_CHECKPOINT_PATH: str = 'checkpoints_dirac_phase4/latest.pth'

    STAGNATION_PATIENCE: int = 100
    STAGNATION_LAMBDA_BOOST: float = 1.5
    STAGNATION_THERMAL_SHOCK_SCALE: float = 1e-5
    STAGNATION_MIN_DELTA_IMPROVEMENT: float = 1e-7
    STAGNATION_WORSENING_THRESHOLD: float = 1e-4

    RICCI_FLOW_ENABLED: bool = True
    RICCI_FLOW_REGULARIZATION_WEIGHT: float = 0.1
    RICCI_FLOW_SMOOTHING_FACTOR: float = 0.01
    RICCI_SURGERY_THRESHOLD: float = 1e15
    RICCI_ADAPTIVE_LR_FACTOR: float = 0.1
    RICCI_ANISOTROPY_TARGET: float = 0.1
    RICCI_METRIC_EPSILON: float = 1e-10
    RICCI_EIGENVALUE_EPSILON: float = 1e-12

    LAMBDA_MAX_SAFE: float = 1e38

    FLOOD_FILL_ENABLED: bool = True
    FLOOD_FILL_INITIAL_LAMBDA: float = 1e3
    FLOOD_FILL_GROWTH_FACTOR: float = 1.2
    FLOOD_FILL_SPEC_GAP_THRESHOLD: float = 0.02
    FLOOD_FILL_DIFFUSION_SCALE: float = 1e-8
    FLOOD_FILL_ANISOTROPY_THRESHOLD: float = 0.3
    FLOOD_FILL_MAX_LAMBDA: float = 1e15
    FLOOD_FILL_PATIENCE: int = 20

    BALLISTIC_RESONANCE_TARGET: float = 0.35
    BALLISTIC_ANISOTROPY_TARGET: float = 0.1

    DELTA_CRYSTAL_THRESHOLD: float = 0.1
    DELTA_OPTICAL_THRESHOLD: float = 0.01
    KAPPA_CRYSTAL_THRESHOLD: float = 1.5
    TEMPERATURE_CRYSTAL_THRESHOLD: float = 1e-9

    GIBBS_T0: float = 1e-3
    GIBBS_C: float = 0.5
    GIBBS_FREE_ENERGY_WEIGHT: float = 0.1

    RICCI_CURVATURE_SAMPLES: int = 100
    RICCI_MAX_DIMENSION: int = 5000

    RICCI_SCALAR_WEIGHT: float = 0.05
    SPECTRAL_GAP_WEIGHT: float = 0.05
    PARTICIPATION_RATIO_WEIGHT: float = 0.05
    MBL_LEVEL_SPACING_WEIGHT: float = 0.05
    BRAGG_PEAK_HARMONIC_RATIO_THRESHOLD: float = 0.1

    DIRAC_MASS: float = 1.0
    DIRAC_C: float = 1.0
    DIRAC_GAMMA_REPRESENTATION: str = 'dirac'


class IPhaseDetector(ABC):
    @abstractmethod
    def detect(self, spectral_field: torch.Tensor) -> Dict[str, Any]:
        pass


class IMetricCalculator(ABC):
    @abstractmethod
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        pass


class SeedManager:
    @staticmethod
    def set_seed(seed: int, device: str = Config.DEVICE):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class LoggerFactory:
    @staticmethod
    def create_logger(name: str, level: str = Config.LOG_LEVEL) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


class GammaMatrices:
    """
    Dirac gamma matrices in various representations.
    Default: Dirac (standard) representation.
    """
    def __init__(self, representation: str = 'dirac', device: str = 'cpu'):
        self.representation = representation
        self.device = device
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
        
        elif self.representation == 'weyl':
            self.gamma0 = torch.tensor([
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0]
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
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=torch.complex64, device=self.device)
        
        else:
            raise ValueError(f"Unknown gamma matrix representation: {self.representation}")
        
        self.gammas = [self.gamma0, self.gamma1, self.gamma2, self.gamma3]

    def to(self, device: str):
        self.device = device
        self._init_matrices()
        return self


class DiracHamiltonianOperator:
    """
    Dirac Hamiltonian operator for 4-component spinors.
    H_Dirac = c * alpha . p + beta * m * c^2
  where alpha_i = gamma0 @ gammai and beta = gamma0.
    In natural units (c=1): H = alpha . p + beta * m
  """
    def __init__(self, config: Config):
        self.config = config
        self.grid_size = config.GRID_SIZE
        self.mass = config.DIRAC_MASS
        self.c = config.DIRAC_C
        self.gamma = GammaMatrices(config.DIRAC_GAMMA_REPRESENTATION, config.DEVICE)
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
        """
        Apply Dirac Hamiltonian to 4-component spinor.
        H_psi = c * (alpha_x * p_x + alpha_y * p_y) @ psi + beta * m * c^2 * psi
        
        Input shape: [4, H, W] or [batch, 4, H, W]
        Output shape: same as input
        """
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

    def time_evolution(self, spinor: torch.Tensor, dt: float = Config.DT) -> torch.Tensor:
        """
        Time evolution of Dirac spinor using split-step method.
        psi(t+dt) = exp(-i * H * dt) * psi(t)
        """
        squeeze_output = False
        if spinor.dim() == 3:
            spinor = spinor.unsqueeze(0)
            squeeze_output = True
        
        evolved = self.apply_dirac_hamiltonian(spinor)
        result = spinor - 1j * dt * evolved
        
        norm_original = torch.norm(spinor.view(spinor.shape[0], -1), dim=1, keepdim=True)
        norm_evolved = torch.norm(result.view(result.shape[0], -1), dim=1, keepdim=True)
        norm_evolved = norm_evolved + Config.NORMALIZATION_EPS
        result = result * (norm_original / norm_evolved).unsqueeze(-1).unsqueeze(-1)
        
        if squeeze_output:
            result = result.squeeze(0)
        
        return result


class SpectralLayer(nn.Module):
    def __init__(self, channels: int, grid_size: int):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
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
            kernel_real_exp,
            size=(freq_h, freq_w),
            mode='bilinear',
            align_corners=False
        )
        kernel_imag_interp = F.interpolate(
            kernel_imag_exp,
            size=(freq_h, freq_w),
            mode='bilinear',
            align_corners=False
        )
        real_part = x_fft.real * kernel_real_interp - x_fft.imag * kernel_imag_interp
        imag_part = x_fft.real * kernel_imag_interp + x_fft.imag * kernel_real_interp
        output_fft = torch.complex(real_part, imag_part)
        output = torch.fft.irfft2(output_fft, s=(self.grid_size, self.grid_size))
        return output


class DiracSpectralNetwork(nn.Module):
    """
    Neural network for learning Dirac equation dynamics.
    Handles 4-component spinors with real and imaginary parts (8 channels total).
    """
    def __init__(
        self,
        grid_size: int = Config.GRID_SIZE,
        hidden_dim: int = Config.HIDDEN_DIM,
        expansion_dim: int = Config.EXPANSION_DIM,
        num_spectral_layers: int = Config.NUM_SPECTRAL_LAYERS,
        spinor_components: int = Config.SPINOR_COMPONENTS,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spinor_components = spinor_components
        self.input_channels = spinor_components * 2
        self.output_channels = spinor_components * 2
        
        self.input_proj = nn.Conv2d(self.input_channels, hidden_dim, kernel_size=1)
        self.expansion_proj = nn.Conv2d(hidden_dim, expansion_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            SpectralLayer(expansion_dim, grid_size)
            for _ in range(num_spectral_layers)
        ])
        self.contraction_proj = nn.Conv2d(expansion_dim, hidden_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(hidden_dim, self.output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = F.gelu(self.input_proj(x))
        x = F.gelu(self.expansion_proj(x))
        for spectral_layer in self.spectral_layers:
            x = F.gelu(spectral_layer(x))
        x = F.gelu(self.contraction_proj(x))
        return self.output_proj(x)


class HamiltonianBackbone(nn.Module):
    def __init__(
        self,
        grid_size: int = Config.GRID_SIZE,
        hidden_dim: int = Config.HIDDEN_DIM,
        num_spectral_layers: int = Config.NUM_SPECTRAL_LAYERS
    ):
        super().__init__()
        self.grid_size = grid_size
        self.input_proj = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            SpectralLayer(hidden_dim, grid_size)
            for _ in range(num_spectral_layers)
        ])
        self.output_proj = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.gelu(self.input_proj(x))
        for spectral_layer in self.spectral_layers:
            x = F.gelu(spectral_layer(x))
        return self.output_proj(x).squeeze(1)


class HamiltonianInferenceEngine:
    def __init__(self, config: Config):
        self.config = config
        self.logger = LoggerFactory.create_logger("HamiltonianInferenceEngine")
        self.backbone = None
        self.fallback_operator = DiracHamiltonianOperator(config)
        self._try_load_backbone()

    def _try_load_backbone(self):
        if not self.config.BACKBONE_ENABLED:
            self.logger.info("Backbone disabled, using analytical Dirac Hamiltonian operator")
            return
        checkpoint_path = self.config.BACKBONE_CHECKPOINT_PATH
        if not os.path.exists(checkpoint_path):
            self.logger.info(
                f"Backbone checkpoint not found at {checkpoint_path}, "
                "using analytical Dirac Hamiltonian operator"
            )
            return
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.config.DEVICE,
                weights_only=False
            )
            self.backbone = HamiltonianBackbone(
                grid_size=self.config.GRID_SIZE,
                hidden_dim=self.config.HIDDEN_DIM,
                num_spectral_layers=self.config.NUM_SPECTRAL_LAYERS
            ).to(self.config.DEVICE)
            if 'model_state_dict' in checkpoint:
                self.backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.backbone.load_state_dict(checkpoint, strict=False)
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.logger.info(f"Backbone loaded from {checkpoint_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load backbone: {e}, using analytical operator")
            self.backbone = None

    def apply_hamiltonian(self, spinor: torch.Tensor) -> torch.Tensor:
        """
        Apply Dirac Hamiltonian to 4-component spinor.
        Always uses analytical operator since backbone is not compatible
        with 4-component Dirac spinors.
        """
        return self.fallback_operator.apply_dirac_hamiltonian(spinor)

    def time_evolve(self, spinor: torch.Tensor, dt: float = Config.DT) -> torch.Tensor:
        return self.fallback_operator.time_evolution(spinor, dt)


class DiracPotentialGenerator:
    """
    Generate potentials for the Dirac equation.
    In relativistic QM, the potential couples differently to particle/antiparticle components.
    """
    def __init__(self, config: Config):
        self.config = config
        self.grid_size = config.GRID_SIZE

    def scalar_potential(self) -> torch.Tensor:
        """
        Scalar potential (couples equally to all components).
        V_s * psi (same coupling for particle and antiparticle).
    """
        x = torch.linspace(0, 2 * np.pi, self.grid_size)
        y = torch.linspace(0, 2 * np.pi, self.grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        cx = np.pi
        cy = np.pi
        return 0.5 * self.config.POTENTIAL_DEPTH * ((X - cx)**2 + (Y - cy)**2) / (np.pi**2)

    def vector_potential(self) -> torch.Tensor:
        """
        Vector potential (time-component of 4-vector).
        V_v * gamma0 * psi (couples with opposite sign to particle/antiparticle).
    """
        x = torch.linspace(0, 2 * np.pi, self.grid_size)
        y = torch.linspace(0, 2 * np.pi, self.grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        cx, cy = np.pi, np.pi
        r = torch.sqrt((X - cx)**2 + (Y - cy)**2) + self.config.POTENTIAL_WIDTH
        return -self.config.POTENTIAL_DEPTH / r

    def magnetic_potential_2d(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Magnetic potential (spatial components of 4-vector).
        A * alpha * psi (couples to spin).
    """
        x = torch.linspace(0, 2 * np.pi, self.grid_size)
        y = torch.linspace(0, 2 * np.pi, self.grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        Ax = self.config.POTENTIAL_DEPTH * 0.1 * torch.sin(Y)
        Ay = self.config.POTENTIAL_DEPTH * 0.1 * torch.sin(X)
        
        return Ax, Ay

    def periodic_lattice_potential(self) -> torch.Tensor:
        x = torch.linspace(0, 2 * np.pi, self.grid_size)
        y = torch.linspace(0, 2 * np.pi, self.grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        return self.config.POTENTIAL_DEPTH * (torch.cos(2 * X) + torch.cos(2 * Y))

    def generate_mixed_potential(self, seed: int) -> Dict[str, torch.Tensor]:
        rng = np.random.RandomState(seed)
        weights = rng.dirichlet([1.0, 1.0, 1.0, 1.0])
        
        scalar = self.scalar_potential()
        vector = self.vector_potential()
        Ax, Ay = self.magnetic_potential_2d()
        lattice = self.periodic_lattice_potential()
        
        scalar_weighted = weights[0] * scalar
        vector_weighted = weights[1] * vector
        lattice_weighted = weights[2] * lattice
        Ax_weighted = weights[3] * Ax
        Ay_weighted = weights[3] * Ay
        
        return {
            'scalar': scalar_weighted,
            'vector': vector_weighted,
            'Ax': Ax_weighted,
            'Ay': Ay_weighted,
            'lattice': lattice_weighted
        }


class DiracDataset(Dataset):
    """
    Dataset for Dirac equation evolution.
    Generates 4-component spinors and their time-evolved targets.
    """
    def __init__(
        self,
        config: Config,
        hamiltonian_engine: HamiltonianInferenceEngine,
        seed: int = Config.RANDOM_SEED,
    ):
        self.config = config
        self.num_samples = config.NUM_SAMPLES
        self.grid_size = config.GRID_SIZE
        self.train_ratio = config.TRAIN_RATIO
        self.hamiltonian_engine = hamiltonian_engine
        self.potential_generator = DiracPotentialGenerator(config)
        self.gamma = GammaMatrices(config.DIRAC_GAMMA_REPRESENTATION, config.DEVICE)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.initial_states = []
        self.target_states = []
        self.potentials = []
        self.energies = []

        for i in range(self.num_samples):
            potential = self.potential_generator.generate_mixed_potential(seed + i)
            spinor, energy = self._generate_initial_spinor(potential, seed + i)
            evolved_spinor = self._time_evolve_spinor(spinor, potential, energy)
            
            initial = self._spinor_to_real_imag(spinor)
            target = self._spinor_to_real_imag(evolved_spinor)
            
            self.initial_states.append(initial)
            self.target_states.append(target)
            self.potentials.append(potential)
            self.energies.append(energy)

        self.initial_states = torch.stack(self.initial_states)
        self.target_states = torch.stack(self.target_states)
        self.energies = torch.tensor(self.energies)

        split_idx = int(self.num_samples * self.train_ratio)
        self.train_states = self.initial_states[:split_idx]
        self.train_targets = self.target_states[:split_idx]
        self.val_states = self.initial_states[split_idx:]
        self.val_targets = self.target_states[split_idx:]

    def _generate_initial_spinor(
        self, potential: Dict[str, torch.Tensor], sample_seed: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate an initial Dirac spinor (4-component).
        The spinor is constructed to be a superposition of positive energy states.
        """
        rng = np.random.RandomState(sample_seed)
        
        spinor = torch.zeros((4, self.grid_size, self.grid_size), dtype=torch.complex64)
        
        for c in range(4):
            real_part = torch.randn(self.grid_size, self.grid_size)
            imag_part = torch.randn(self.grid_size, self.grid_size)
            spinor[c] = torch.complex(real_part, imag_part)
        
        scalar_pot = potential['scalar'].to(self.config.DEVICE)
        for c in range(4):
            spinor[c] = spinor[c] * torch.exp(-scalar_pot / (2 * self.config.POTENTIAL_DEPTH))
        
        norm = torch.sqrt(torch.sum(torch.abs(spinor)**2)) + Config.NORMALIZATION_EPS
        spinor = spinor / norm * self.config.SPINOR_NORM_TARGET
        
        phase = torch.randn(self.grid_size, self.grid_size) * 0.5
        for c in range(4):
            spinor[c] = spinor[c] * torch.exp(1j * phase * (c + 1) * 0.1)
        
        mass_term = self.config.DIRAC_MASS * self.config.DIRAC_C**2
        energy = mass_term + rng.uniform(0, 2 * self.config.POTENTIAL_DEPTH)
        
        return spinor, energy

    def _time_evolve_spinor(
        self,
        spinor: torch.Tensor,
        potential: Dict[str, torch.Tensor],
        energy: float
    ) -> torch.Tensor:
        """
        Time evolve a Dirac spinor under the influence of potentials.
        """
        dt = self.config.DT
        evolved = spinor.clone()
        
        for _ in range(self.config.TIME_STEPS):
            h_spinor = self.hamiltonian_engine.apply_hamiltonian(evolved.unsqueeze(0)).squeeze(0)
            
            scalar_pot = potential['scalar'].to(evolved.device)
            for c in range(4):
                evolved[c] = evolved[c] - 1j * dt * h_spinor[c]
                evolved[c] = evolved[c] - 1j * dt * scalar_pot * evolved[c]
            
            norm = torch.sqrt(torch.sum(torch.abs(evolved)**2)) + Config.NORMALIZATION_EPS
            target_norm = torch.sqrt(torch.sum(torch.abs(spinor)**2)) + Config.NORMALIZATION_EPS
            evolved = evolved / norm * target_norm
        
        return evolved

    def _spinor_to_real_imag(self, spinor: torch.Tensor) -> torch.Tensor:
        """
        Convert 4-component complex spinor to 8-channel real tensor.
        Channels: [Re(psi0), Im(psi0), Re(psi1), Im(psi1), ...]
    """
        channels = []
        for c in range(4):
            channels.append(spinor[c].real)
            channels.append(spinor[c].imag)
        return torch.stack(channels, dim=0)

    def __len__(self):
        return len(self.train_states)

    def __getitem__(self, idx):
        return self.train_states[idx], self.train_targets[idx]

    def get_validation_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.val_states, self.val_targets


class FullFourierAnalyzer:
    def __init__(self, config: Config):
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
        mean_power = power_spectrum.mean(dim=(-2, -1))
        std_power = power_spectrum.std(dim=(-2, -1))
        spectral_concentration = (power_spectrum.max(dim=-1)[0].max(dim=-1)[0]) / (total_power + 1e-10)

        center = H // 2
        low_freq_power = power_spectrum[:, :, center-2:center+3, center-2:center+3].sum(dim=(-2, -1))
        low_freq_ratio = low_freq_power / (total_power + 1e-10)

        high_freq_power = total_power - low_freq_power
        high_freq_ratio = high_freq_power / (total_power + 1e-10)

        phase_flat = phase.view(B, C, -1)
        phase_mean = torch.atan2(
            torch.sin(phase_flat).mean(dim=-1),
            torch.cos(phase_flat).mean(dim=-1)
        )
        phase_coherence = torch.abs(torch.cos(phase_flat - phase_mean.unsqueeze(-1))).mean(dim=-1)

        k_mag = self.K_MAG.to(device)
        if H != self.grid_size or W != self.grid_size:
            k_mag = F.interpolate(
                k_mag.unsqueeze(0).unsqueeze(0).float(),
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)

        k_mag_flat = k_mag.flatten()
        power_flat = power_spectrum.mean(dim=1).view(B, -1)

        radial_profile = torch.zeros(B, H // 2, device=device)
        for b in range(B):
            for r in range(H // 2):
                mask = (k_mag_flat >= r) & (k_mag_flat < r + 1)
                if mask.sum() > 0:
                    radial_profile[b, r] = power_flat[b, mask].mean()

        dominant_k = radial_profile.argmax(dim=-1).float()

        return {
            'fft_2d': fft_shifted,
            'magnitude': magnitude,
            'phase': phase,
            'power_spectrum': power_spectrum,
            'power_normalized': power_normalized,
            'spectral_concentration': spectral_concentration,
            'low_freq_ratio': low_freq_ratio,
            'high_freq_ratio': high_freq_ratio,
            'phase_coherence': phase_coherence,
            'radial_profile': radial_profile,
            'dominant_k': dominant_k,
            'total_power': total_power,
            'mean_power': mean_power,
            'std_power': std_power
        }

    def detect_bragg_peaks(self, power_spectrum: torch.Tensor, threshold_sigma: float = 2.0) -> Dict[str, Any]:
        if power_spectrum.dim() > 2:
            power_spectrum = power_spectrum.mean(dim=tuple(range(power_spectrum.dim() - 2)))

        ps_numpy = power_spectrum.cpu().numpy()
        threshold = np.mean(ps_numpy) + threshold_sigma * np.std(ps_numpy)

        peaks = []
        H, W = ps_numpy.shape
        for i in range(H):
            for j in range(W):
                if ps_numpy[i, j] > threshold:
                    is_local_max = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < H and 0 <= nj < W:
                                if ps_numpy[ni, nj] > ps_numpy[i, j]:
                                    is_local_max = False
                                    break
                        if not is_local_max:
                            break
                    if is_local_max:
                        peaks.append({
                            'i': int(i),
                            'j': int(j),
                            'power': float(ps_numpy[i, j]),
                            'k_mag': float(np.sqrt((i - H//2)**2 + (j - W//2)**2))
                        })

        peaks.sort(key=lambda x: x['power'], reverse=True)

        harmonic_ratios = []
        for i in range(len(peaks)):
            for j in range(i + 1, min(len(peaks), i + 5)):
                k1 = peaks[i]['k_mag']
                k2 = peaks[j]['k_mag']
                if k1 > 1e-6 and k2 > 1e-6:
                    ratio = max(k1, k2) / min(k1, k2)
                    harmonic_ratios.append(ratio)

        harmonic_count = sum(
            1 for r in harmonic_ratios
            if abs(r - round(r)) < self.config.BRAGG_PEAK_HARMONIC_RATIO_THRESHOLD
        )

        return {
            'peaks': peaks[:self.config.SPECTRAL_PEAK_LIMIT],
            'num_peaks': len(peaks),
            'harmonic_ratios': harmonic_ratios[:10],
            'harmonic_count': harmonic_count,
            'is_crystalline': len(peaks) > 3 and harmonic_count > 1
        }

    def compute_resonance_metrics(self, spectral_field: torch.Tensor) -> Dict[str, Any]:
        spectrum = self.compute_full_spectrum(spectral_field)

        spectral_conc = spectrum['spectral_concentration'].mean().item()
        low_freq_ratio = spectrum['low_freq_ratio'].mean().item()
        high_freq_ratio = spectrum['high_freq_ratio'].mean().item()
        phase_coherence = spectrum['phase_coherence'].mean().item()
        dominant_k = spectrum['dominant_k'].mean().item()

        power_2d = spectrum['power_spectrum'].mean(dim=1)
        bragg_analysis = self.detect_bragg_peaks(power_2d[0])

        resonance_score = (
            spectral_conc * 0.3 +
            phase_coherence * 0.3 +
            min(low_freq_ratio * 2, 1.0) * 0.2 +
            (1.0 - min(high_freq_ratio * 2, 1.0)) * 0.2
        )

        is_resonant = (
            spectral_conc > self.config.FOURIER_SPECTRAL_CONCENTRATION_THRESHOLD and
            phase_coherence > self.config.FOURIER_PHASE_COHERENCE_THRESHOLD and
            bragg_analysis['is_crystalline']
        )

        return {
            'spectral_concentration': spectral_conc,
            'low_freq_ratio': low_freq_ratio,
            'high_freq_ratio': high_freq_ratio,
            'phase_coherence': phase_coherence,
            'dominant_k': dominant_k,
            'resonance_score': resonance_score,
            'is_resonant': is_resonant,
            'bragg_peaks': bragg_analysis['peaks'],
            'num_bragg_peaks': bragg_analysis['num_peaks'],
            'harmonic_count': bragg_analysis['harmonic_count'],
            'is_crystalline_spectrum': bragg_analysis['is_crystalline'],
            'radial_profile': spectrum['radial_profile'][0].cpu().tolist()
        }


class FourierMassCenterAnalyzer:
    def __init__(self, config: Config):
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

        dx = kx - R_x.view(-1, 1, 1)
        dy = ky - R_y.view(-1, 1, 1)
        I_xx = (dx**2 * density).sum(dim=(-2, -1)) / total_mass.squeeze(-1).squeeze(-1)
        I_yy = (dy**2 * density).sum(dim=(-2, -1)) / total_mass.squeeze(-1).squeeze(-1)
        I_xy = (dx * dy * density).sum(dim=(-2, -1)) / total_mass.squeeze(-1).squeeze(-1)

        inertia_tensor = torch.stack([
            torch.stack([I_xx, I_xy], dim=-1),
            torch.stack([I_xy, I_yy], dim=-1)
        ], dim=-2)

        eigenvalues = torch.linalg.eigvalsh(inertia_tensor)
        anisotropy = eigenvalues[..., 0] / (eigenvalues[..., 1] + 1e-10)
        left_alignment = (R_x < -0.5).float()

        resonance = self.full_fourier.compute_resonance_metrics(spectral_field)

        return {
            'R_cm': torch.stack([R_x, R_y], dim=-1),
            'inertia_tensor': inertia_tensor,
            'eigenvalues': eigenvalues,
            'anisotropy': anisotropy,
            'left_alignment': left_alignment,
            'localization_index': 1.0 - anisotropy,
            'total_mass': total_mass.squeeze(),
            'resonance_score': resonance['resonance_score'],
            'is_resonant': resonance['is_resonant'],
            'phase_coherence': resonance['phase_coherence'],
            'spectral_concentration': resonance['spectral_concentration'],
            'num_bragg_peaks': resonance['num_bragg_peaks'],
            'harmonic_count': resonance['harmonic_count']
        }


class TopologicalPhaseDetector(IPhaseDetector):
    def __init__(self, config: Config):
        self.config = config
        self.mass_analyzer = FourierMassCenterAnalyzer(config)
        self.phase_state = 0.0
        self.alignment_history = np.zeros(config.TOPO_ALIGNMENT_HISTORY_LEN)
        self.history_ptr = 0

    def detect(self, spectral_field: torch.Tensor) -> Dict[str, Any]:
        mass_analysis = self.mass_analyzer.compute_mass_center(spectral_field)

        R_cm = mass_analysis['R_cm']
        localization = mass_analysis['localization_index']
        alignment = mass_analysis['left_alignment']
        resonance_score = mass_analysis['resonance_score']
        phase_coherence = mass_analysis['phase_coherence']
        spectral_conc = mass_analysis['spectral_concentration']

        alignment_val = alignment.mean().item() if alignment.dim() > 0 else alignment.item()
        self.alignment_history[self.history_ptr] = alignment_val
        self.history_ptr = (self.history_ptr + 1) % len(self.alignment_history)

        alignment_trend = 0.0
        if self.history_ptr > 10:
            recent = self.alignment_history[max(0, self.history_ptr - 10):self.history_ptr]
            if len(recent) > 1:
                alignment_trend = float((recent[-1] - recent[0]) / len(recent))

        loc_mean = localization.mean().item() if localization.dim() > 0 else localization.item()

        is_aligned = float(alignment_val > self.config.TOPO_ALIGNMENT_THRESHOLD)
        is_localized = float(loc_mean > 0.8)
        is_resonant = float(resonance_score > 0.6)

        if self.phase_state < 0.5:
            transition_prob = is_aligned * is_localized * is_resonant
        else:
            threshold_low = self.config.TOPO_ALIGNMENT_THRESHOLD - self.config.TOPO_HYSTERESIS_WIDTH
            transition_prob = float(
                (alignment_val > threshold_low) and
                (loc_mean > 0.7) and
                (resonance_score > 0.4)
            )

        alpha = self.config.TOPO_PHASE_SMOOTHING
        self.phase_state = alpha * self.phase_state + (1 - alpha) * transition_prob

        return {
            'R_cm': R_cm.detach(),
            'R_cm_x': float(R_cm[..., 0].mean().item()),
            'R_cm_y': float(R_cm[..., 1].mean().item()),
            'localization_index': float(loc_mean),
            'alignment_score': float(alignment_val),
            'alignment_trend': alignment_trend,
            'anisotropy': float(mass_analysis['anisotropy'].mean().item()),
            'phase_state': float(self.phase_state),
            'is_crystalline': float(self.phase_state > 0.7),
            'transition_probability': float(transition_prob),
            'inertia_eigenvalues': mass_analysis['eigenvalues'].detach().cpu().tolist()
                if mass_analysis['eigenvalues'].dim() > 0 else [0.0, 0.0],
            'resonance_score': float(resonance_score),
            'is_resonant': mass_analysis['is_resonant'],
            'phase_coherence': float(phase_coherence),
            'spectral_concentration': float(spectral_conc),
            'num_bragg_peaks': int(mass_analysis['num_bragg_peaks']),
            'harmonic_count': int(mass_analysis['harmonic_count'])
        }


class SpectralFieldExtractor:
    @staticmethod
    def extract(model: nn.Module, grid_size: int = 16) -> Optional[torch.Tensor]:
        if hasattr(model, 'spectral_layers'):
            layers = model.spectral_layers
        elif hasattr(model, 'dirac_net') and hasattr(model.dirac_net, 'spectral_layers'):
            layers = model.dirac_net.spectral_layers
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


class TopologicalCrystallizationLoss(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.lambda_current = config.TOPO_LAMBDA_BASE

    def forward(self, phase_info: Dict[str, Any], epoch: int) -> Dict[str, torch.Tensor]:
        device = self.config.DEVICE
        r_cm_x = phase_info.get('R_cm_x', 0.0)
        quadrant_penalty = torch.tensor(max(0.0, r_cm_x), device=device, dtype=torch.float32)

        phase = phase_info.get('phase_state', 0.0)
        loc_target = self.config.TOPO_LOCALIZATION_LIQUID + \
            (self.config.TOPO_LOCALIZATION_CRYSTAL - self.config.TOPO_LOCALIZATION_LIQUID) * phase
        loc_current = phase_info.get('localization_index', 0.0)
        localization_loss = torch.tensor((loc_current - loc_target)**2, device=device, dtype=torch.float32)

        resonance = phase_info.get('resonance_score', 0.0)
        resonance_target = 0.7 * phase
        resonance_loss = torch.tensor((resonance - resonance_target)**2, device=device, dtype=torch.float32)

        if phase > 0.6 and self.lambda_current < self.config.TOPO_LAMBDA_CRITICAL:
            self.lambda_current *= 1.1

        total = (
            0.1 * quadrant_penalty +
            0.3 * localization_loss +
            0.3 * resonance_loss
        )
        return {
            'total': total,
            'quadrant': quadrant_penalty,
            'localization': localization_loss,
            'resonance': resonance_loss,
            'lambda_effective': torch.tensor(self.lambda_current, device=device)
        }


class CrystallizationPressureApplicator:
    def __init__(self, config: Config):
        self.config = config
        self.decay = config.TOPO_CRYSTALLIZATION_PRESSURE_DECAY

    def apply(self, model: nn.Module, phase_info: Dict[str, Any]):
        if phase_info.get('is_crystalline', 0.0) < 0.5:
            return
        pressure = phase_info.get('phase_state', 0.0)
        effective_decay = self.decay ** (1.0 / (1.0 + pressure))
        for param in model.parameters():
            if param.requires_grad:
                param.data *= effective_decay


class TopologicalMetricsCalculator(IMetricCalculator):
    def __init__(self, config: Config):
        self.config = config
        self.phase_detector = TopologicalPhaseDetector(config)
        self.field_extractor = SpectralFieldExtractor()
        self.topo_loss = TopologicalCrystallizationLoss(config)
        self.pressure_applicator = CrystallizationPressureApplicator(config)

    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        if not self.config.TOPO_ENABLED:
            return self._empty_metrics()
        spectral_field = self.field_extractor.extract(model, self.config.GRID_SIZE)
        if spectral_field is None:
            return self._empty_metrics()
        phase_info = self.phase_detector.detect(spectral_field)
        epoch = kwargs.get('epoch', 0)
        topo_loss_info = self.topo_loss(phase_info, epoch)
        return {
            'topo_R_cm_x': phase_info['R_cm_x'],
            'topo_R_cm_y': phase_info['R_cm_y'],
            'topo_localization': phase_info['localization_index'],
            'topo_alignment': phase_info['alignment_score'],
            'topo_alignment_trend': phase_info['alignment_trend'],
            'topo_anisotropy': phase_info['anisotropy'],
            'topo_phase_state': phase_info['phase_state'],
            'topo_is_crystalline': phase_info['is_crystalline'],
            'topo_transition_prob': phase_info['transition_probability'],
            'topo_resonance_score': phase_info['resonance_score'],
            'topo_is_resonant': float(phase_info['is_resonant']),
            'topo_phase_coherence': phase_info['phase_coherence'],
            'topo_spectral_conc': phase_info['spectral_concentration'],
            'topo_num_bragg_peaks': phase_info['num_bragg_peaks'],
            'topo_harmonic_count': phase_info['harmonic_count'],
            'topo_loss_quadrant': float(topo_loss_info['quadrant'].item()),
            'topo_loss_localization': float(topo_loss_info['localization'].item()),
            'topo_loss_resonance': float(topo_loss_info['resonance'].item()),
            'topo_loss_total': float(topo_loss_info['total'].item()),
            'topo_lambda_effective': float(topo_loss_info['lambda_effective'].item()),
            '_phase_info': phase_info
        }

    def apply_crystallization_pressure(self, model: nn.Module, topo_metrics: Dict[str, Any]):
        phase_info = topo_metrics.get('_phase_info')
        if phase_info is not None:
            self.pressure_applicator.apply(model, phase_info)

    @staticmethod
    def _empty_metrics() -> Dict[str, Any]:
        return {
            'topo_R_cm_x': 0.0, 'topo_R_cm_y': 0.0,
            'topo_localization': 0.0, 'topo_alignment': 0.0,
            'topo_alignment_trend': 0.0, 'topo_anisotropy': 1.0,
            'topo_phase_state': 0.0, 'topo_is_crystalline': 0.0,
            'topo_transition_prob': 0.0, 'topo_resonance_score': 0.0,
            'topo_is_resonant': 0.0, 'topo_phase_coherence': 0.0,
            'topo_spectral_conc': 0.0, 'topo_num_bragg_peaks': 0,
            'topo_harmonic_count': 0, 'topo_loss_quadrant': 0.0,
            'topo_loss_localization': 0.0, 'topo_loss_resonance': 0.0,
            'topo_loss_total': 0.0, 'topo_lambda_effective': 0.0,
            '_phase_info': None
        }


class LocalComplexityAnalyzer:
    @staticmethod
    def compute_local_complexity(weights: torch.Tensor, epsilon: float = 1e-6) -> float:
        if weights.numel() == 0:
            return 0.0
        w = weights.detach().flatten()
        w = w / (torch.norm(w) + epsilon)
        w_expanded = w.unsqueeze(0)
        similarities = F.cosine_similarity(w_expanded, w_expanded.unsqueeze(1), dim=2)
        mask = ~torch.eye(similarities.size(0), device=similarities.device, dtype=torch.bool)
        if mask.sum() == 0:
            return 0.0
        avg_similarity = (similarities.abs() * mask).sum() / mask.sum()
        lc = 1.0 - avg_similarity.item()
        return max(0.0, min(1.0, lc))


class SuperpositionAnalyzer:
    @staticmethod
    def compute_superposition(weights: torch.Tensor) -> float:
        w = weights.detach()
        if w.dim() > 2:
            w = w.reshape(w.size(0), -1)
        if w.size(0) < 2 or w.size(1) < 2:
            return 0.0
        try:
            correlation_matrix = torch.corrcoef(w)
            correlation_matrix = correlation_matrix.nan_to_num(nan=0.0)
            n = correlation_matrix.size(0)
            mask = ~torch.eye(n, device=correlation_matrix.device, dtype=torch.bool)
            if mask.sum() == 0:
                return 0.0
            avg_correlation = (correlation_matrix.abs() * mask).sum() / mask.sum()
            return avg_correlation.item()
        except Exception:
            return 0.0


class CrystallographyMetricsCalculator(IMetricCalculator):
    def __init__(self, config: Config):
        self.config = config
        self.logger = LoggerFactory.create_logger("CrystallographyMetrics")

    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        val_x = kwargs.get('val_x')
        val_y = kwargs.get('val_y')
        return self.compute_all_metrics(model, val_x, val_y)

    def compute_kappa(
        self,
        model: nn.Module,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        num_batches: int = None
    ) -> float:
        if num_batches is None:
            num_batches = self.config.KAPPA_GRADIENT_BATCHES
        model.eval()
        grads = []
        for i in range(num_batches):
            try:
                model.zero_grad()
                noise_scale = self.config.NOISE_AMPLITUDE * (i + 1) / num_batches
                val_x_perturbed = val_x + torch.randn_like(val_x) * noise_scale
                outputs = model(val_x_perturbed)
                loss = F.mse_loss(outputs, val_y)
                loss.backward()
                grad_list = []
                for p in model.parameters():
                    if p.grad is not None and p.grad.numel() > 0:
                        grad_list.append(p.grad.flatten())
                if grad_list:
                    grad_vector = torch.cat(grad_list)
                    if torch.isfinite(grad_vector).all():
                        grads.append(grad_vector.detach())
            except Exception as e:
                self.logger.debug(f"Gradient computation failed batch {i}: {e}")
                continue
        if len(grads) < 2:
            return float('inf')
        grads_tensor = torch.stack(grads)
        n_samples, n_dims = grads_tensor.shape
        if n_dims > self.config.KAPPA_MAX_DIM:
            indices = torch.randperm(n_dims, device=grads_tensor.device)[:self.config.KAPPA_MAX_DIM]
            grads_tensor = grads_tensor[:, indices]
            n_dims = self.config.KAPPA_MAX_DIM
        try:
            if n_samples < n_dims:
                gram = torch.mm(grads_tensor, grads_tensor.t()) / max(n_samples - 1, 1)
                eigenvals = torch.linalg.eigvalsh(gram)
            else:
                cov = torch.cov(grads_tensor.t())
                eigenvals = torch.linalg.eigvalsh(cov).real
            eigenvals = eigenvals[eigenvals > self.config.EIGENVALUE_TOL]
            if len(eigenvals) == 0:
                return float('inf')
            return (eigenvals.max() / eigenvals.min()).item()
        except Exception as e:
            self.logger.debug(f"Eigenvalue computation failed: {e}")
            return float('inf')

    def compute_discretization_margin(self, model: nn.Module) -> float:
        margins = []
        for param in model.parameters():
            if param.numel() > 0:
                margin = (param.data - param.data.round()).abs().max().item()
                margins.append(margin)
        return max(margins) if margins else 0.0

    def compute_alpha_purity(self, model: nn.Module) -> float:
        delta = self.compute_discretization_margin(model)
        if delta < self.config.MIN_VARIANCE_THRESHOLD:
            return 20.0
        return -np.log(delta + self.config.ENTROPY_EPS)

    def compute_kappa_quantum(self, model: nn.Module) -> float:
        flat_params = []
        for param in model.parameters():
            if param.numel() > 0:
                flat_params.append(param.data.detach().flatten())
        if not flat_params:
            return 1.0
        W = torch.cat(flat_params)[:self.config.KAPPA_MAX_DIM]
        n = W.numel()
        if n < 2:
            return 1.0
        params_centered = W - W.mean()
        cov_matrix = torch.outer(params_centered, params_centered) / n
        cov_matrix = cov_matrix + self.config.HBAR * torch.eye(n, device=W.device)
        try:
            eigenvals = torch.linalg.eigvalsh(cov_matrix)
            eigenvals = eigenvals[eigenvals > self.config.HBAR]
            return (eigenvals.max() / eigenvals.min()).item() if len(eigenvals) > 0 else 1.0
        except Exception:
            return 1.0

    def compute_poynting_vector(self, model: nn.Module) -> Dict[str, Any]:
        all_params = []
        for param in model.parameters():
            if param is not None and param.numel() > 0:
                all_params.append(param.data.detach().flatten())
        if not all_params:
            return {
                'poynting_magnitude': 0.0,
                'energy_distribution': {},
                'is_radiating': False,
                'field_orthogonality': 0.0
            }
        E = torch.cat(all_params)[:self.config.PARAM_FLATTEN_LIMIT]
        state_dict = model.state_dict()
        spectral_norms = []
        spectral_indices = set()
        for key in state_dict.keys():
            if key.startswith('spectral_layers.'):
                parts = key.split('.')
                if len(parts) >= 2:
                    try:
                        idx = int(parts[1])
                        spectral_indices.add(idx)
                    except ValueError:
                        continue
        for idx in sorted(spectral_indices):
            layer_param_keys = [
                k for k in state_dict.keys()
                if k.startswith(f'spectral_layers.{idx}.')
            ]
            if layer_param_keys:
                layer_params = [state_dict[k] for k in layer_param_keys]
                concatenated = torch.cat([p.flatten() for p in layer_params])
                layer_norm = torch.norm(concatenated)
                spectral_norms.append(layer_norm)
        if len(spectral_norms) > 1:
            differences = []
            for i in range(len(spectral_norms) - 1):
                diff = torch.abs(spectral_norms[i] - spectral_norms[i + 1])
                differences.append(diff)
            H_magnitude = torch.stack(differences).sum()
        else:
            H_magnitude = torch.tensor(0.0, device=E.device)
        poynting_magnitude = torch.norm(E) * H_magnitude * self.config.ENERGY_FLOW_SCALE
        energy_distribution = {
            'total_norm': float(torch.norm(E).item()),
            'spectral_total': float(sum(sn.item() for sn in spectral_norms)) if spectral_norms else 0.0,
            'n_spectral_layers': len(spectral_norms)
        }
        return {
            'poynting_magnitude': float(poynting_magnitude.item()),
            'energy_distribution': energy_distribution,
            'is_radiating': float(poynting_magnitude.item()) > self.config.POYNTING_THRESHOLD,
            'field_orthogonality': float(H_magnitude.item())
        }

    def compute_hbar_effective(self, model: nn.Module, lambda_pressure: float) -> float:
        delta = self.compute_discretization_margin(model)
        if lambda_pressure <= 0:
            return 0.0
        omega = math.sqrt(abs(lambda_pressure))
        if omega < self.config.NORMALIZATION_EPS:
            return 0.0
        return (delta**2 * lambda_pressure) / omega

    def compute_all_metrics(
        self,
        model: nn.Module,
        val_x: torch.Tensor,
        val_y: torch.Tensor
    ) -> Dict[str, Any]:
        try:
            delta = self.compute_discretization_margin(model)
            alpha = self.compute_alpha_purity(model)
        except Exception as e:
            self.logger.warning(f"Basic crystallography failed: {e}")
            delta, alpha = 1.0, 0.0

        def safe_compute(func, *args, default=None, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.debug(f"{func.__name__} failed: {e}")
                return default

        kappa = safe_compute(self.compute_kappa, model, val_x, val_y, default=float('inf'))
        kappa_q = safe_compute(self.compute_kappa_quantum, model, default=1.0)
        poynting = safe_compute(
            self.compute_poynting_vector, model,
            default={
                'poynting_magnitude': 0.0,
                'is_radiating': False,
                'energy_distribution': {},
                'field_orthogonality': 0.0
            }
        )
        metrics = {
            'kappa': kappa,
            'delta': delta,
            'alpha': alpha,
            'kappa_q': kappa_q,
            'poynting': poynting
        }
        metrics['purity_index'] = 1.0 - delta
        metrics['is_crystal'] = alpha > self.config.ALPHA_CRYSTAL_THRESHOLD
        if isinstance(poynting, dict):
            metrics['energy_flow'] = poynting.get('poynting_magnitude', 0.0)
        else:
            metrics['energy_flow'] = 0.0
        return metrics


class ThermodynamicMetricsCalculator(IMetricCalculator):
    def __init__(self, config: Config):
        self.config = config

    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        gradient_buffer = kwargs.get('gradient_buffer', [])
        learning_rate = kwargs.get('learning_rate', self.config.LEARNING_RATE)
        loss_history = kwargs.get('loss_history', [])
        temp_history = kwargs.get('temp_history', [])
        delta = kwargs.get('delta', 1.0)
        alpha = kwargs.get('alpha', 0.0)
        t_eff = kwargs.get('effective_temperature', 1.0)

        temperature = self.compute_effective_temperature(gradient_buffer, learning_rate)
        cv, is_transition = self.compute_specific_heat(loss_history, temp_history)
        gibbs_free_energy = self.compute_gibbs_free_energy(delta, alpha, temperature)
        critical_temp = self.compute_critical_temperature(alpha)
        phase_stability_numeric = 1 if temperature < critical_temp else 0
        phase_stability_str = "stable" if temperature < critical_temp else "unstable"

        return {
            'temperature': temperature,
            'specific_heat': cv,
            'is_phase_transition': is_transition,
            'gibbs_free_energy': gibbs_free_energy,
            'critical_temperature': critical_temp,
            'phase_stability': phase_stability_numeric,
            'phase_stability_str': phase_stability_str
        }

    def compute_effective_temperature(
        self, gradient_buffer: Any, learning_rate: float
    ) -> float:
        buf = list(gradient_buffer) if not isinstance(gradient_buffer, list) else gradient_buffer
        if len(buf) < 2:
            return 0.0
        grads = []
        window = self.config.GRADIENT_BUFFER_WINDOW
        limit = self.config.GRADIENT_BUFFER_LIMIT
        for g in buf[-window:]:
            flat = g.detach().flatten()
            if flat.numel() > 0:
                grads.append(flat[:limit])
        if not grads:
            return 0.0
        try:
            grads_stacked = torch.stack(grads)
            second_moment = torch.mean(torch.norm(grads_stacked, dim=1)**2)
            first_moment_sq = torch.norm(torch.mean(grads_stacked, dim=0))**2
            variance = second_moment - first_moment_sq
            return float((learning_rate / 2.0) * variance)
        except Exception:
            return 0.0

    def compute_specific_heat(
        self, loss_history: Any, temp_history: Any
    ) -> Tuple[float, bool]:
        l_hist = list(loss_history) if not isinstance(loss_history, list) else loss_history
        t_hist = list(temp_history) if not isinstance(temp_history, list) else temp_history
        window = self.config.LOSS_HISTORY_WINDOW
        if len(l_hist) < 2 or len(t_hist) < 2:
            return 0.0, False
        u_var = np.var(l_hist[-window:])
        t_mean = np.mean(t_hist[-window:]) + self.config.NORMALIZATION_EPS
        cv = u_var / (t_mean**2)
        is_latent_crystallization = cv > self.config.CV_THRESHOLD
        return float(cv), is_latent_crystallization

    def compute_gibbs_free_energy(self, delta: float, alpha: float, temperature: float) -> float:
        internal_energy = delta
        entropy_proxy = -alpha
        if temperature > 0:
            return internal_energy - temperature * entropy_proxy
        return internal_energy

    def compute_critical_temperature(self, alpha: float) -> float:
        return self.config.GIBBS_T0 * np.exp(-self.config.GIBBS_C * alpha)


class SpectralGeometryCalculator(IMetricCalculator):
    def __init__(self, config: Config):
        self.config = config

    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        all_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
        all_weights = all_weights[:self.config.PARAM_FLATTEN_LIMIT].cpu().numpy()

        n = len(all_weights)
        if n < 2:
            return {
                'spectral_gap': 0.0, 'effective_dimension': 0,
                'participation_ratio': 0.0, 'level_spacing_ratio': 0.0
            }

        outer_product = np.outer(all_weights, all_weights) / n
        outer_product += np.eye(n) * self.config.EIGENVALUE_TOL

        try:
            eigenvalues = np.linalg.eigvalsh(outer_product)
            eigenvalues = np.sort(eigenvalues)[::-1]

            effective_dim = np.sum(eigenvalues > self.config.EIGENVALUE_TOL)
            spectral_gap = float(eigenvalues[0] - eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
            participation_ratio = float((np.sum(eigenvalues)**2) / (np.sum(eigenvalues**2) + 1e-10))

            level_spacing = np.diff(eigenvalues)
            level_spacing_ratio = self._compute_level_spacing_ratio(level_spacing)

            return {
                'spectral_gap': spectral_gap,
                'effective_dimension': int(effective_dim),
                'participation_ratio': participation_ratio,
                'level_spacing_ratio': level_spacing_ratio
            }
        except Exception:
            return {
                'spectral_gap': 0.0, 'effective_dimension': 0,
                'participation_ratio': 0.0, 'level_spacing_ratio': 0.0
            }

    def _compute_level_spacing_ratio(self, spacings: np.ndarray) -> float:
        if len(spacings) < 2:
            return 0.0
        ratios = []
        for i in range(len(spacings) - 1):
            s1 = abs(spacings[i])
            s2 = abs(spacings[i + 1])
            if s1 > 1e-15 and s2 > 1e-15:
                ratios.append(min(s1, s2) / max(s1, s2))
        return float(np.mean(ratios)) if ratios else 0.0


class RicciCurvatureCalculator(IMetricCalculator):
    def __init__(self, config: Config):
        self.config = config

    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        all_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
        n = min(len(all_weights), self.config.PARAM_FLATTEN_LIMIT)
        w = all_weights[:n].cpu().numpy()

        if n < 2:
            return {'ricci_scalar': 0.0, 'mean_sectional_curvature': 0.0}

        metric_tensor = np.outer(w, w) / n
        metric_tensor += np.eye(n) * self.config.EIGENVALUE_TOL

        ricci_scalar = self._compute_ricci_scalar(metric_tensor)
        sectional_curvatures = self._estimate_sectional_curvatures(metric_tensor)

        return {
            'ricci_scalar': float(ricci_scalar),
            'mean_sectional_curvature': float(np.mean(sectional_curvatures))
        }

    def _compute_ricci_scalar(self, metric: np.ndarray) -> float:
        eigenvalues = np.linalg.eigvalsh(metric)
        eigenvalues = eigenvalues[eigenvalues > self.config.EIGENVALUE_TOL]
        n = len(eigenvalues)
        if n < 2:
            return 0.0
        return float(n * np.sum(1.0 / eigenvalues))

    def _estimate_sectional_curvatures(self, metric: np.ndarray) -> np.ndarray:
        curvatures = []
        n = metric.shape[0]
        samples = min(self.config.RICCI_CURVATURE_SAMPLES, n * (n - 1) // 2)
        for _ in range(samples):
            i, j = np.random.choice(n, 2, replace=False)
            block = metric[np.ix_([i, j], [i, j])]
            det = np.linalg.det(block)
            if det > self.config.EIGENVALUE_TOL:
                curvatures.append(1.0 / det)
        return np.array(curvatures) if curvatures else np.array([0.0])


class PerelmanRicciFlow:
    def __init__(self, config: Config):
        self.config = config
        self.logger = LoggerFactory.create_logger("PerelmanRicciFlow")
        self.curvature_history = deque(maxlen=100)
        self.surgery_count = 0
        self.flow_time = 0.0
        self.last_ricci_scalar = 0.0

    def compute_ricci_scalar_fast(self, model: nn.Module) -> float:
        try:
            all_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
            n = min(len(all_weights), self.config.PARAM_FLATTEN_LIMIT)
            w = all_weights[:n].cpu().numpy()

            if n < 2:
                return 0.0

            w = w / (np.linalg.norm(w) + 1e-10)

            metric = np.outer(w, w) / n

            epsilon = max(self.config.RICCI_METRIC_EPSILON, 1e-8)
            metric += np.eye(n) * epsilon

            metric += np.random.randn(n, n) * epsilon * 0.01
            metric = (metric + metric.T) / 2

            try:
                eigenvalues = np.linalg.eigvalsh(metric)
            except np.linalg.LinAlgError:
                self.logger.warning("Eigenvalue computation failed, using cached Ricci scalar")
                return self.last_ricci_scalar

            min_eigenvalue = max(self.config.RICCI_EIGENVALUE_EPSILON, epsilon)
            eigenvalues = eigenvalues[eigenvalues > min_eigenvalue]

            if len(eigenvalues) < 2:
                return self.last_ricci_scalar

            inv_sum = np.sum(1.0 / np.clip(eigenvalues, min_eigenvalue, 1e10))
            ricci = float(len(eigenvalues) * inv_sum)

            ricci = min(ricci, 1e20)

            self.last_ricci_scalar = ricci
            return ricci

        except Exception as e:
            self.logger.warning(f"Ricci scalar computation error: {e}, using cached value")
            return self.last_ricci_scalar

    def compute_local_curvature(self, param: torch.Tensor) -> torch.Tensor:
        if param.numel() < 2:
            return torch.tensor(0.0, device=param.device)

        flat = param.detach().flatten()
        diff1 = flat[1:] - flat[:-1]
        diff2 = diff1[1:] - diff1[:-1]

        local_curv = torch.abs(diff2).mean()
        return local_curv

    def compute_anisotropy(self, model: nn.Module) -> float:
        try:
            all_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
            n = min(len(all_weights), self.config.PARAM_FLATTEN_LIMIT)
            w = all_weights[:n].cpu().numpy()

            if n < 4:
                return 0.0

            w = w / (np.linalg.norm(w) + 1e-10)

            cov = np.cov(w) if len(w) > 1 else np.array([[1.0]])

            if cov.ndim == 0:
                cov = np.array([[float(cov)]])

            if cov.shape[0] > 0:
                cov += np.eye(cov.shape[0]) * self.config.RICCI_METRIC_EPSILON

            try:
                eigenvalues = np.linalg.eigvalsh(cov)
            except np.linalg.LinAlgError:
                return 0.0

            eigenvalues = eigenvalues[eigenvalues > self.config.RICCI_EIGENVALUE_EPSILON]

            if len(eigenvalues) < 2:
                return 0.0

            anisotropy = float(eigenvalues.min() / (eigenvalues.max() + 1e-10))
            return min(anisotropy, 1.0)
        except Exception as e:
            return 0.0

    def compute_ricci_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        if not self.config.RICCI_FLOW_ENABLED:
            return torch.tensor(0.0, device=self.config.DEVICE)

        ricci_loss = torch.tensor(0.0, device=self.config.DEVICE)
        total_params = 0

        for param in model.parameters():
            if param.numel() < 4:
                continue

            flat = param.flatten()

            diff1 = flat[1:] - flat[:-1]
            diff2 = torch.abs(diff1[1:] - diff1[:-1])

            ricci_loss += diff2.mean() ** 2
            total_params += 1

        if total_params == 0:
            return ricci_loss

        ricci_loss = ricci_loss / total_params
        return ricci_loss * self.config.RICCI_FLOW_REGULARIZATION_WEIGHT

    def apply_ricci_flow_step(self, model: nn.Module, lr: float) -> Dict[str, Any]:
        if not self.config.RICCI_FLOW_ENABLED:
            return {'ricci_flow_applied': False}

        smoothing = self.config.RICCI_FLOW_SMOOTHING_FACTOR
        total_curvature_reduced = 0.0
        params_modified = 0

        with torch.no_grad():
            for param in model.parameters():
                if param.numel() < 4:
                    continue

                original_shape = param.shape

                local_curv = self.compute_local_curvature(param)

                if local_curv > 1e-8:
                    flat = param.flatten()
                    n = flat.numel()

                    smoothed = flat.clone()
                    for i in range(1, n - 1):
                        smoothed[i] = 0.25 * flat[i-1] + 0.5 * flat[i] + 0.25 * flat[i+1]

                    blend = min(smoothing * (1.0 + local_curv.item()), 0.1)
                    new_flat = flat * (1 - blend) + smoothed * blend

                    param.copy_(new_flat.reshape(original_shape))

                    total_curvature_reduced += local_curv.item()
                    params_modified += 1

        self.flow_time += 1.0

        return {
            'ricci_flow_applied': True,
            'curvature_reduced': total_curvature_reduced,
            'params_modified': params_modified,
            'flow_time': self.flow_time
        }

    def perform_perelman_surgery(self, model: nn.Module, ricci_scalar: float) -> Dict[str, Any]:
        if ricci_scalar < self.config.RICCI_SURGERY_THRESHOLD:
            return {'surgery_performed': False, 'reason': 'curvature_below_threshold'}

        self.surgery_count += 1
        self.logger.warning(
            f"PERELMAN SURGERY #{self.surgery_count}: "
            f"Ricci scalar {ricci_scalar:.2e} exceeds threshold {self.config.RICCI_SURGERY_THRESHOLD:.2e}"
        )

        surgeries_performed = 0
        total_cuts = 0

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.numel() < 4:
                    continue

                flat = param.flatten()
                mean_val = flat.mean()
                std_val = flat.std()

                if std_val < 1e-10:
                    continue

                z_scores = torch.abs((flat - mean_val) / (std_val + 1e-10))
                singularity_mask = z_scores > 3.0

                if singularity_mask.sum() > 0:
                    singular_indices = torch.where(singularity_mask)[0]

                    for idx in singular_indices:
                        left_idx = max(0, idx - 2)
                        right_idx = min(len(flat) - 1, idx + 2)
                        neighbors = torch.cat([flat[left_idx:idx], flat[idx+1:right_idx+1]])

                        if len(neighbors) > 0:
                            flat[idx] = neighbors.mean()
                            total_cuts += 1

                    param.copy_(flat.reshape(param.shape))
                    surgeries_performed += 1

        self.logger.info(
            f"  Surgery complete: {surgeries_performed} parameters modified, "
            f"{total_cuts} singularities cut"
        )

        return {
            'surgery_performed': True,
            'surgeries_on_params': surgeries_performed,
            'total_cuts': total_cuts,
            'surgery_count': self.surgery_count
        }

    def compute_adaptive_lr_factor(self, model: nn.Module) -> float:
        if not self.config.RICCI_FLOW_ENABLED:
            return 1.0

        ricci = self.compute_ricci_scalar_fast(model)
        self.curvature_history.append(ricci)

        if len(self.curvature_history) > 10:
            recent = list(self.curvature_history)[-10:]
            mean_curv = np.mean(recent)
            std_curv = np.std(recent) + 1e-10

            relative_curv = (ricci - mean_curv) / std_curv

            factor = max(self.config.RICCI_ADAPTIVE_LR_FACTOR,
                        1.0 - self.config.RICCI_ADAPTIVE_LR_FACTOR * relative_curv)
            return min(factor, 1.0)

        return 1.0

    def get_flow_metrics(self, model: nn.Module) -> Dict[str, Any]:
        ricci_scalar = self.compute_ricci_scalar_fast(model)
        anisotropy = self.compute_anisotropy(model)
        lr_factor = self.compute_adaptive_lr_factor(model)

        return {
            'ricci_scalar': ricci_scalar,
            'anisotropy': anisotropy,
            'lr_factor': lr_factor,
            'flow_time': self.flow_time,
            'surgery_count': self.surgery_count,
            'target_anisotropy': self.config.RICCI_ANISOTROPY_TARGET,
            'distance_to_hypersphere': abs(anisotropy - self.config.RICCI_ANISOTROPY_TARGET)
        }


class SpectroscopyMetricsCalculator(IMetricCalculator):
    def __init__(self, config: Config):
        self.config = config

    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        coeffs = {name: param.data for name, param in model.named_parameters()}
        return self.compute_weight_diffraction(coeffs)

    def compute_weight_diffraction(self, coeffs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        limit = self.config.PARAM_FLATTEN_LIMIT
        W = torch.cat([c.detach().flatten()[:limit] for c in coeffs.values()])
        fft_spectrum = torch.fft.fft(W)
        power_spectrum = torch.abs(fft_spectrum)**2
        peaks = []
        threshold = torch.mean(power_spectrum) + 2 * torch.std(power_spectrum)
        peak_limit = self.config.SPECTRAL_PEAK_LIMIT
        for i, power in enumerate(power_spectrum):
            if power > threshold and len(peaks) < peak_limit:
                peaks.append({'frequency': i, 'intensity': float(power)})
        is_crystalline = len(peaks) > 0 and len(peaks) < len(power_spectrum) // 2
        return {
            'bragg_peaks': peaks,
            'is_crystalline_structure': is_crystalline,
            'spectral_entropy': float(self._compute_spectral_entropy(power_spectrum)),
            'num_peaks': len(peaks)
        }

    @staticmethod
    def _compute_spectral_entropy(power_spectrum: torch.Tensor) -> float:
        ps_normalized = power_spectrum / (torch.sum(power_spectrum) + 1e-10)
        ps_normalized = ps_normalized[ps_normalized > 1e-10]
        if len(ps_normalized) == 0:
            return 0.0
        entropy = -torch.sum(ps_normalized * torch.log(ps_normalized + 1e-10))
        return float(entropy)


class LambdaPressureScheduler:
    def __init__(self, config: Config):
        self.config = config
        self._lambda = np.float64(config.LAMBDA_INITIAL)
        self._lambda_max = np.float64(config.LAMBDA_MAX)
        self._growth_factor = np.float64(config.LAMBDA_GROWTH_FACTOR)
        self._growth_interval = config.LAMBDA_GROWTH_INTERVAL_EPOCHS
        self.logger = LoggerFactory.create_logger("LambdaPressureScheduler")

    @property
    def current_lambda(self) -> np.float64:
        return self._lambda

    def step(self, epoch: int):
        if epoch > 0 and epoch % self._growth_interval == 0:
            new_lambda = self._lambda * self._growth_factor
            if new_lambda <= self._lambda_max:
                self._lambda = np.float64(new_lambda)
                self.logger.info(
                    f"Lambda pressure increased to {self._lambda:.6e} at epoch {epoch}"
                )

    def compute_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=self.config.DEVICE, dtype=torch.float64)
        total_params = 0
        for param in model.parameters():
            if param.numel() > 0:
                deviation = param.double() - param.double().round()
                reg_loss = reg_loss + (deviation ** 2).sum()
                total_params += param.numel()

        if total_params > 0:
            reg_loss = reg_loss / total_params

        return reg_loss * float(self._lambda)

    def set_lambda(self, value: float):
        self._lambda = np.float64(value)


class AdaptiveLambdaScheduler(LambdaPressureScheduler):
    def __init__(self, config: Config):
        super().__init__(config)
        self._base_growth_factor = np.float64(config.LAMBDA_GROWTH_FACTOR)
        self._accelerated_growth_factor = np.float64(config.LAMBDA_GROWTH_FACTOR * 2.0)

    def step_adaptive(self, epoch: int, topo_phase_state: float = 0.0):
        if epoch > 0 and epoch % self._growth_interval == 0:
            if topo_phase_state > 0.5:
                growth = self._accelerated_growth_factor
            else:
                growth = self._base_growth_factor
            new_lambda = self._lambda * growth
            if new_lambda <= self._lambda_max:
                self._lambda = np.float64(new_lambda)
                self.logger.info(
                    f"Lambda pressure {'(ACCELERATED) ' if topo_phase_state > 0.5 else ''}"
                    f"increased to {self._lambda:.6e} at epoch {epoch}"
                )


class QuadruplePrecisionLambdaScheduler:
    def __init__(self, config: Config):
        self.config = config
        self._lambda = np.longdouble(str(config.PHASE5_LAMBDA_INITIAL))
        self._lambda_max = np.longdouble(str(config.PHASE5_LAMBDA_MAX))
        self._growth_factor = np.longdouble(str(config.PHASE5_LAMBDA_GROWTH_FACTOR))
        self._growth_interval = config.PHASE5_LAMBDA_GROWTH_INTERVAL_EPOCHS
        self.logger = LoggerFactory.create_logger("QuadruplePrecisionLambdaScheduler")

    @property
    def current_lambda(self) -> np.longdouble:
        return self._lambda

    def step(self, epoch: int, improvement: bool = True):
        if epoch > 0 and epoch % self._growth_interval == 0 and improvement:
            new_lambda = self._lambda * self._growth_factor
            if new_lambda <= self._lambda_max:
                self._lambda = new_lambda
                self.logger.info(
                    f"Phase 5 Lambda (float128) increased to {float(self._lambda):.6e} at epoch {epoch}"
                )

    def compute_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=self.config.DEVICE, dtype=torch.float64)
        total_params = 0
        for param in model.parameters():
            if param.numel() > 0:
                deviation = param.double() - param.double().round()
                reg_loss = reg_loss + (deviation ** 2).sum()
                total_params += param.numel()

        if total_params > 0:
            reg_loss = reg_loss / total_params

        return reg_loss * float(self._lambda)

    def set_lambda(self, value: float):
        self._lambda = np.longdouble(str(value))


class AnnealingScheduler:
    def __init__(self, config: Config):
        self.config = config
        self._temperature = config.ANNEALING_INITIAL_TEMPERATURE
        self._cooling_rate = config.ANNEALING_COOLING_RATE
        self._final_temp = config.ANNEALING_FINAL_TEMPERATURE
        self.logger = LoggerFactory.create_logger("AnnealingScheduler")

    @property
    def temperature(self) -> float:
        return self._temperature

    def step(self):
        self._temperature = max(
            self._temperature * self._cooling_rate,
            self._final_temp
        )

    def accept_perturbation(self, delta_loss: float) -> bool:
        if delta_loss < 0:
            return True
        if self._temperature < self.config.NORMALIZATION_EPS:
            return False
        probability = math.exp(-delta_loss / self._temperature)
        return np.random.random() < probability

    def should_restart(self, current_delta: float, best_delta: float) -> bool:
        return (current_delta - best_delta) > self.config.ANNEALING_RESTART_THRESHOLD


class TopologicalAnnealingScheduler(AnnealingScheduler):
    def __init__(self, config: Config):
        super().__init__(config)
        self._base_cooling_rate = config.ANNEALING_COOLING_RATE

    def step_adaptive(self, alignment_trend: float = 0.0, resonance_score: float = 0.0):
        combined_signal = alignment_trend + (resonance_score - 0.5) * 0.1
        if combined_signal > 0:
            effective_rate = self._base_cooling_rate * 0.99
        elif combined_signal < -0.01:
            effective_rate = min(self._base_cooling_rate * 1.01, 0.9999)
        else:
            effective_rate = self._base_cooling_rate
        self._temperature = max(
            self._temperature * effective_rate,
            self._final_temp
        )


class TrainingMetricsMonitor:
    def __init__(self, config: Config):
        self.config = config
        self.metrics_history = {
            'epoch': [], 'loss': [], 'val_loss': [], 'val_acc': [],
            'train_acc': [], 'lc': [], 'sp': [], 'alpha': [],
            'kappa': [], 'kappa_q': [], 'delta': [], 'temperature': [],
            'specific_heat': [], 'poynting_magnitude': [],
            'energy_flow': [], 'purity_index': [], 'is_crystal': [],
            'lambda_pressure': [], 'hbar_effective': [],
            'spectral_entropy': [], 'num_bragg_peaks': [],
            'learning_rate': [], 'annealing_temp': [],
            'delta_slope': [], 'norm_conservation_error': [],
            'gibbs_free_energy': [], 'critical_temperature': [],
            'spectral_gap': [], 'participation_ratio': [],
            'level_spacing_ratio': [], 'ricci_scalar': [],
            'topo_R_cm_x': [], 'topo_R_cm_y': [],
            'topo_localization': [], 'topo_alignment': [],
            'topo_alignment_trend': [], 'topo_anisotropy': [],
            'topo_phase_state': [], 'topo_is_crystalline': [],
            'topo_transition_prob': [], 'topo_resonance_score': [],
            'topo_is_resonant': [], 'topo_phase_coherence': [],
            'topo_spectral_conc': [], 'topo_num_bragg_peaks': [],
            'topo_harmonic_count': [], 'topo_loss_total': [],
            'topo_lambda_effective': [], 'phase_stability': []
        }
        self.gradient_buffer = deque(maxlen=config.GRADIENT_BUFFER_MAXLEN)
        self.loss_history = deque(maxlen=config.LOSS_HISTORY_MAXLEN)
        self.temp_history = deque(maxlen=config.TEMP_HISTORY_MAXLEN)
        self.cv_history = deque(maxlen=config.CV_HISTORY_MAXLEN)

    def update_metrics(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        if 'loss' in kwargs:
            self.loss_history.append(kwargs['loss'])
        if 'temperature' in kwargs:
            self.temp_history.append(kwargs['temperature'])
        if 'specific_heat' in kwargs:
            self.cv_history.append(kwargs['specific_heat'])

    def compute_delta_slope(self) -> float:
        window = self.config.GROKKING_DELTA_SLOPE_WINDOW
        deltas = self.metrics_history.get('delta', [])
        if len(deltas) < window:
            return 0.0
        recent = deltas[-window:]
        x = np.arange(len(recent), dtype=np.float64)
        y = np.array(recent, dtype=np.float64)
        if np.std(x) < 1e-15:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def format_progress_bar(self, epoch: int, total_epochs: int, phase: str) -> str:
        h = self.metrics_history
        idx = -1

        def safe_get(key):
            vals = h.get(key, [])
            return vals[idx] if vals else 0.0

        loss = safe_get('loss')
        val_loss = safe_get('val_loss')
        val_acc = safe_get('val_acc')
        train_acc = safe_get('train_acc')
        lc = safe_get('lc')
        sp = safe_get('sp')
        alpha = safe_get('alpha')
        kappa = safe_get('kappa')
        kappa_q = safe_get('kappa_q')
        delta = safe_get('delta')
        temp = safe_get('temperature')
        cv = safe_get('specific_heat')
        poynting = safe_get('poynting_magnitude')
        lam = safe_get('lambda_pressure')
        hbar = safe_get('hbar_effective')
        s_ent = safe_get('spectral_entropy')
        n_peaks = safe_get('num_bragg_peaks')
        lr = safe_get('learning_rate')
        a_temp = safe_get('annealing_temp')
        d_slope = safe_get('delta_slope')
        norm_err = safe_get('norm_conservation_error')
        purity = safe_get('purity_index')
        is_cryst = safe_get('is_crystal')
        gibbs = safe_get('gibbs_free_energy')
        crit_temp = safe_get('critical_temperature')
        spec_gap = safe_get('spectral_gap')
        part_ratio = safe_get('participation_ratio')
        lvl_spacing = safe_get('level_spacing_ratio')
        ricci = safe_get('ricci_scalar')
        phase_stab = safe_get('phase_stability')

        topo_rcm_x = safe_get('topo_R_cm_x')
        topo_loc = safe_get('topo_localization')
        topo_align = safe_get('topo_alignment')
        topo_phase = safe_get('topo_phase_state')
        topo_cryst = safe_get('topo_is_crystalline')
        topo_aniso = safe_get('topo_anisotropy')
        topo_res = safe_get('topo_resonance_score')
        topo_phase_coh = safe_get('topo_phase_coherence')
        topo_spec_conc = safe_get('topo_spectral_conc')
        topo_harm = safe_get('topo_harmonic_count')

        kappa_str = f"{kappa:.2e}" if kappa != float('inf') and not np.isinf(kappa) else "inf"

        lines = []
        lines.append(f"[{phase}] Epoch {epoch}/{total_epochs}")
        lines.append(
            f"Loss={loss:.6f} ValLoss={val_loss:.6f} "
            f"TrainAcc={train_acc:.4f} ValAcc={val_acc:.4f}"
        )
        lines.append(
            f"LC={lc:.4f} SP={sp:.4f} Alpha={alpha:.4f} "
            f"Kappa={kappa_str} Kappa_q={kappa_q:.2e}"
        )
        lines.append(
            f"Delta={delta:.6f} Purity={purity:.4f} Crystal={int(is_cryst)}"
        )
        lines.append(
            f"T_eff={temp:.2e} C_v={cv:.2e} Poynting={poynting:.2e} "
            f"E_flow={safe_get('energy_flow'):.2e}"
        )
        lines.append(
            f"Lambda={lam:.2e} hbar_eff={hbar:.2e} "
            f"S_spectral={s_ent:.4f} Bragg={int(n_peaks)}"
        )
        lines.append(
            f"LR={lr:.2e} AnnealT={a_temp:.2e} "
            f"dDelta/dt={d_slope:.2e} NormErr={norm_err:.2e}"
        )
        lines.append(
            f"Gibbs={gibbs:.4e} T_crit={crit_temp:.2e} "
            f"SpecGap={spec_gap:.4e} PartRatio={part_ratio:.4f}"
        )
        stability_str = "S" if phase_stab >= 0.5 else "U"
        lines.append(
            f"LvlSpacing={lvl_spacing:.4f} Ricci={ricci:.2e} "
            f"Stability={stability_str}"
        )
        lines.append(
            f"Topo[CM_x={topo_rcm_x:.4f} Loc={topo_loc:.4f} "
            f"Align={topo_align:.2f} Aniso={topo_aniso:.4f} "
            f"Phase={topo_phase:.4f} Cryst={int(topo_cryst)}]"
        )
        lines.append(
            f"Resonance={topo_res:.4f} PhaseCoh={topo_phase_coh:.4f} "
            f"SpecConc={topo_spec_conc:.4f} Harm={int(topo_harm)}"
        )
        return "\n".join(lines)


class CheckpointManager:
    def __init__(self, config: Config, checkpoint_dir: str = "checkpoints_dirac"):
        self.config = config
        self.interval_minutes = config.CHECKPOINT_INTERVAL_MINUTES
        self.max_checkpoints = config.MAX_CHECKPOINTS
        self.last_checkpoint_time = time.time()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_files = []

    def should_save_checkpoint(self) -> bool:
        current_time = time.time()
        elapsed_minutes = (current_time - self.last_checkpoint_time) / 60
        return elapsed_minutes >= self.interval_minutes

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, Any],
        phase: str = "training",
        lambda_value: float = 0.0,
        config_snapshot: Dict[str, Any] = None
    ) -> str:
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'lambda_pressure': float(lambda_value),
            'config': config_snapshot if config_snapshot else asdict(self.config),
            'timestamp': datetime.now().isoformat()
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{phase}_epoch_{epoch}_{timestamp}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_files.append(checkpoint_path)
        if len(self.checkpoint_files) > self.max_checkpoints:
            oldest_file = self.checkpoint_files.pop(0)
            if os.path.exists(oldest_file):
                os.remove(oldest_file)
        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        torch.save(checkpoint, latest_path)
        self.last_checkpoint_time = time.time()
        return checkpoint_path

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        if os.path.exists(latest_path):
            return torch.load(latest_path, map_location=self.config.DEVICE, weights_only=False)
        return None


class Phase5CheckpointManager:
    def __init__(self, config: Config):
        self.config = config
        self.checkpoint_dir = "weights"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.latest_path = config.PHASE5_CHECKPOINT_LATEST_PATH
        self.best_delta = float('inf')
        self.best_alpha = 0.0
        self.best_acc = 0.0
        self.logger = LoggerFactory.create_logger("Phase5CheckpointManager")
        self._load_best_metrics()

    def _load_best_metrics(self):
        if os.path.exists(self.latest_path):
            try:
                checkpoint = torch.load(
                    self.latest_path,
                    map_location='cpu',
                    weights_only=False
                )
                metrics = checkpoint.get('metrics', {})
                self.best_delta = metrics.get('delta', float('inf'))
                self.best_alpha = metrics.get('alpha', 0.0)
                self.best_acc = metrics.get('val_acc', 0.0)
                self.logger.info(
                    f"Loaded Phase 5 best metrics: delta={self.best_delta:.6f}, "
                    f"alpha={self.best_alpha:.4f}, acc={self.best_acc:.4f}"
                )
            except Exception as e:
                self.logger.warning(f"Could not load Phase 5 checkpoint: {e}")

    def should_save(self, current_delta: float, current_alpha: float, current_acc: float) -> bool:
        is_better_delta = current_delta < self.best_delta
        is_better_alpha = current_alpha > self.best_alpha
        acc_not_collapsed = current_acc >= self.best_acc * 0.9

        if is_better_delta and acc_not_collapsed:
            return True
        if is_better_alpha and is_better_delta * 0.5 and acc_not_collapsed:
            return True
        return False

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, Any],
        lambda_value: np.longdouble
    ) -> bool:
        current_delta = metrics.get('delta', float('inf'))
        current_alpha = metrics.get('alpha', 0.0)
        current_acc = metrics.get('val_acc', 0.0)

        if not self.should_save(current_delta, current_alpha, current_acc):
            self.logger.info(
                f"Phase 5 checkpoint NOT saved: current metrics not better "
                f"(delta={current_delta:.6f} vs {self.best_delta:.6f}, "
                f"acc={current_acc:.4f} vs {self.best_acc:.4f})"
            )
            return False

        checkpoint = {
            'epoch': epoch,
            'phase': 'phase5',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'lambda_pressure': float(lambda_value),
            'lambda_precision': 'float128',
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, self.latest_path)
        self.best_delta = current_delta
        self.best_alpha = current_alpha
        self.best_acc = current_acc

        self.logger.info(
            f"Phase 5 checkpoint SAVED: epoch={epoch}, "
            f"delta={current_delta:.6f}, alpha={current_alpha:.4f}, "
            f"acc={current_acc:.4f}"
        )
        return True

    def load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer) -> Optional[int]:
        if not os.path.exists(self.latest_path):
            return None
        try:
            checkpoint = torch.load(
                self.latest_path,
                map_location=self.config.DEVICE,
                weights_only=False
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint.get('epoch', 0)
            self.logger.info(f"Loaded Phase 5 checkpoint from epoch {epoch}")
            return epoch
        except Exception as e:
            self.logger.warning(f"Failed to load Phase 5 checkpoint: {e}")
            return None


class GlassStateDetector:
    def __init__(self, config: Config):
        self.config = config
        self.patience_epochs = config.MINING_GLASS_PATIENCE_EPOCHS
        self.metrics_buffer = deque(maxlen=self.patience_epochs)
        self.logger = LoggerFactory.create_logger("GlassStateDetector")

    def should_stop(
        self, epoch: int, lc: float, sp: float,
        kappa: float, delta: float, temp: float, cv: float
    ) -> bool:
        self.metrics_buffer.append({
            'epoch': epoch, 'lc': lc, 'sp': sp,
            'kappa': kappa, 'delta': delta, 'temp': temp, 'cv': cv
        })
        if epoch > self.patience_epochs:
            recent = list(self.metrics_buffer)[-self.patience_epochs:]
            avg_lc = np.mean([m['lc'] for m in recent])
            avg_sp = np.mean([m['sp'] for m in recent])
            avg_kappa = np.mean([m['kappa'] for m in recent])
            avg_delta = np.mean([m['delta'] for m in recent])
            avg_temp = np.mean([m['temp'] for m in recent])
            avg_cv = np.mean([m['cv'] for m in recent])
            is_glass = (
                avg_lc > self.config.MINING_TARGET_LC or
                avg_sp > self.config.MINING_TARGET_SP or
                avg_kappa > self.config.MINING_TARGET_KAPPA or
                avg_delta > self.config.MINING_TARGET_DELTA or
                avg_temp > self.config.MINING_TARGET_TEMP or
                avg_cv > self.config.MINING_TARGET_CV
            )
            return is_glass
        if epoch == self.patience_epochs:
            if (lc > self.config.MINING_TARGET_LC or
                sp > self.config.MINING_TARGET_SP or
                kappa > self.config.MINING_TARGET_KAPPA or
                delta > self.config.MINING_TARGET_DELTA or
                temp > self.config.MINING_TARGET_TEMP or
                cv > self.config.MINING_TARGET_CV):
                return True
        return False

    def is_crystal_formed(
        self, lc: float, sp: float, kappa: float,
        delta: float, temp: float, cv: float
    ) -> bool:
        return (
            lc < self.config.MINING_TARGET_LC and
            sp < self.config.MINING_TARGET_SP and
            kappa < self.config.MINING_TARGET_KAPPA and
            delta < self.config.MINING_TARGET_DELTA and
            temp < self.config.MINING_TARGET_TEMP and
            cv < self.config.MINING_TARGET_CV
        )


class WeightIntegrityChecker:
    @staticmethod
    def check(model: nn.Module) -> Dict[str, Any]:
        has_nan = False
        has_inf = False
        total_params = 0
        nan_count = 0
        inf_count = 0
        for name, param in model.named_parameters():
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


class TrainingEngine:
    def __init__(self, config: Config):
        self.config = config
        self.logger = LoggerFactory.create_logger("TrainingEngine")
        self.criterion = nn.MSELoss()
        self.lc_analyzer = LocalComplexityAnalyzer()
        self.sp_analyzer = SuperpositionAnalyzer()
        self.crystal_calc = CrystallographyMetricsCalculator(config)
        self.thermo_calc = ThermodynamicMetricsCalculator(config)
        self.spectro_calc = SpectroscopyMetricsCalculator(config)
        self.spectral_geom_calc = SpectralGeometryCalculator(config)
        self.ricci_calc = RicciCurvatureCalculator(config)
        self.topo_calc = TopologicalMetricsCalculator(config)

    def compute_weight_metrics(self, model: nn.Module) -> Tuple[float, float]:
        lc_values = []
        sp_values = []
        limit = self.config.WEIGHT_METRIC_DIM_LIMIT
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                w = param[:min(param.size(0), limit), :min(param.size(1), limit)]
                lc = self.lc_analyzer.compute_local_complexity(w)
                sp = self.sp_analyzer.compute_superposition(w)
                lc_values.append(lc)
                sp_values.append(sp)
        lc = np.mean(lc_values) if lc_values else 0.0
        sp = np.mean(sp_values) if sp_values else 0.0
        return lc, sp

    def compute_norm_conservation_error(
        self, model: nn.Module, val_x: torch.Tensor
    ) -> float:
        model.eval()
        with torch.no_grad():
            outputs = model(val_x)
            input_norms = torch.norm(val_x.view(val_x.size(0), -1), dim=1)
            output_norms = torch.norm(outputs.view(outputs.size(0), -1), dim=1)
            relative_error = torch.abs(output_norms - input_norms) / (
                input_norms + self.config.NORMALIZATION_EPS
            )
            return relative_error.mean().item()

    def train_single_epoch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        dataloader: DataLoader,
        epoch: int,
        lambda_scheduler: Optional[Union[LambdaPressureScheduler, QuadruplePrecisionLambdaScheduler]] = None,
        ricci_flow: Optional[Any] = None
    ) -> Tuple[float, float]:
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_reg = 0.0
        total_ricci = 0.0
        total_samples = 0
        correct = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.config.DEVICE)
            batch_y = batch_y.to(self.config.DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            mse_loss = self.criterion(outputs, batch_y)

            loss = mse_loss

            if lambda_scheduler is not None:
                reg_loss = lambda_scheduler.compute_regularization_loss(model)
                loss = loss + reg_loss
                total_reg += reg_loss.item() * batch_x.size(0)

            if ricci_flow is not None:
                ricci_loss = ricci_flow.compute_ricci_regularization_loss(model)
                loss = loss + ricci_loss
                total_ricci += ricci_loss.item() * batch_x.size(0)

            loss.backward()
            if epoch % self.config.NOISE_INTERVAL_EPOCHS == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.randn_like(param.grad) * self.config.NOISE_AMPLITUDE
                        param.grad.add_(noise)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=self.config.GRADIENT_CLIP_NORM
            )
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            total_mse += mse_loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
            with torch.no_grad():
                per_sample_mse = ((outputs - batch_y)**2).mean(dim=(1, 2, 3))
                correct += (per_sample_mse < self.config.MSE_THRESHOLD).sum().item()
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_acc = correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, train_acc

    def validate(
        self, model: nn.Module, val_x: torch.Tensor, val_y: torch.Tensor
    ) -> Tuple[float, float]:
        model.eval()
        val_x = val_x.to(self.config.DEVICE)
        val_y = val_y.to(self.config.DEVICE)
        with torch.no_grad():
            outputs = model(val_x)
            val_loss = self.criterion(outputs, val_y).item()
            per_sample_mse = ((outputs - val_y)**2).mean(dim=(1, 2, 3))
            val_acc = (per_sample_mse < self.config.MSE_THRESHOLD).float().mean().item()
        return val_loss, val_acc

    def collect_all_metrics(
        self,
        model: nn.Module,
        monitor: TrainingMetricsMonitor,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        lambda_scheduler: Optional[Union[LambdaPressureScheduler, QuadruplePrecisionLambdaScheduler]] = None,
        annealing_scheduler: Optional[AnnealingScheduler] = None,
        current_lr: float = 0.0,
        epoch: int = 0
    ) -> Dict[str, Any]:
        lc, sp = self.compute_weight_metrics(model)
        crystal_metrics = self.crystal_calc.compute_all_metrics(model, val_x, val_y)
        alpha = crystal_metrics.get('alpha', 0.0)
        kappa = crystal_metrics.get('kappa', float('inf'))
        kappa_q = crystal_metrics.get('kappa_q', 1.0)
        delta = crystal_metrics.get('delta', 1.0)
        poynting = crystal_metrics.get('energy_flow', 0.0)
        purity_index = crystal_metrics.get('purity_index', 0.0)
        is_crystal = crystal_metrics.get('is_crystal', False)

        effective_lr = current_lr if current_lr > 0 else self.config.LEARNING_RATE
        thermo_metrics = self.thermo_calc.compute(
            model,
            gradient_buffer=monitor.gradient_buffer,
            learning_rate=effective_lr,
            loss_history=monitor.loss_history,
            temp_history=monitor.temp_history,
            delta=delta,
            alpha=alpha,
            effective_temperature=0.0
        )
        temp = thermo_metrics.get('temperature', 0.0)
        cv = thermo_metrics.get('specific_heat', 0.0)
        gibbs = thermo_metrics.get('gibbs_free_energy', 0.0)
        crit_temp = thermo_metrics.get('critical_temperature', 0.0)
        phase_stab = thermo_metrics.get('phase_stability', 0.0)

        spectro = self.spectro_calc.compute(model)
        spectral_entropy = spectro.get('spectral_entropy', 0.0)
        num_peaks = spectro.get('num_peaks', 0)

        spectral_geom = self.spectral_geom_calc.compute(model)
        spectral_gap = spectral_geom.get('spectral_gap', 0.0)
        part_ratio = spectral_geom.get('participation_ratio', 0.0)
        lvl_spacing = spectral_geom.get('level_spacing_ratio', 0.0)

        ricci_metrics = self.ricci_calc.compute(model)
        ricci_scalar = ricci_metrics.get('ricci_scalar', 0.0)

        lambda_val = float(lambda_scheduler.current_lambda) if lambda_scheduler else 0.0
        hbar_eff = self.crystal_calc.compute_hbar_effective(model, lambda_val)
        annealing_temp = annealing_scheduler.temperature if annealing_scheduler else 0.0
        delta_slope = monitor.compute_delta_slope()
        norm_err = self.compute_norm_conservation_error(model, val_x)

        topo_metrics = self.topo_calc.compute(model, epoch=epoch)
        self.topo_calc.apply_crystallization_pressure(model, topo_metrics)
        topo_public = {k: v for k, v in topo_metrics.items() if not k.startswith('_')}

        result = {
            'lc': lc, 'sp': sp, 'alpha': alpha,
            'kappa': kappa, 'kappa_q': kappa_q, 'delta': delta,
            'temperature': temp, 'specific_heat': cv,
            'poynting_magnitude': poynting, 'energy_flow': poynting,
            'purity_index': purity_index, 'is_crystal': is_crystal,
            'lambda_pressure': lambda_val, 'hbar_effective': hbar_eff,
            'spectral_entropy': spectral_entropy,
            'num_bragg_peaks': num_peaks,
            'learning_rate': current_lr,
            'annealing_temp': annealing_temp,
            'delta_slope': delta_slope,
            'norm_conservation_error': norm_err,
            'gibbs_free_energy': gibbs,
            'critical_temperature': crit_temp,
            'spectral_gap': spectral_gap,
            'participation_ratio': part_ratio,
            'level_spacing_ratio': lvl_spacing,
            'ricci_scalar': ricci_scalar,
            'phase_stability': phase_stab
        }
        result.update(topo_public)
        return result


class BatchSizeProspector:
    def __init__(self, config: Config, hamiltonian_engine: HamiltonianInferenceEngine):
        self.config = config
        self.hamiltonian_engine = hamiltonian_engine
        self.logger = LoggerFactory.create_logger("BatchSizeProspector")

    def prospect(self) -> int:
        self.logger.info("Phase 1: Batch size prospecting started")
        candidates = self.config.BATCH_CANDIDATES
        seed = self.config.BATCH_PROSPECT_SEED
        epochs = self.config.BATCH_PROSPECT_EPOCHS
        results = {}
        for bs in candidates:
            self.logger.info(f"  Testing batch_size={bs}")
            SeedManager.set_seed(seed, self.config.DEVICE)
            dataset = DiracDataset(self.config, self.hamiltonian_engine, seed=seed)
            loader = DataLoader(dataset, batch_size=bs, shuffle=True)
            val_x, val_y = dataset.get_validation_batch()
            val_x = val_x.to(self.config.DEVICE)
            val_y = val_y.to(self.config.DEVICE)
            model = DiracSpectralNetwork(
                grid_size=self.config.GRID_SIZE,
                hidden_dim=self.config.HIDDEN_DIM,
                expansion_dim=self.config.EXPANSION_DIM,
                num_spectral_layers=self.config.NUM_SPECTRAL_LAYERS
            ).to(self.config.DEVICE)
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                momentum=self.config.MOMENTUM
            )
            engine = TrainingEngine(self.config)
            crystal_calc = CrystallographyMetricsCalculator(self.config)
            for ep in range(1, epochs + 1):
                engine.train_single_epoch(model, optimizer, loader, ep)
            val_loss, val_acc = engine.validate(model, val_x, val_y)
            delta = crystal_calc.compute_discretization_margin(model)
            kappa = crystal_calc.compute_kappa(model, val_x, val_y)
            results[bs] = {
                'delta': delta,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'kappa': kappa
            }
            self.logger.info(
                f"    batch_size={bs}: delta={delta:.6f}, "
                f"val_acc={val_acc:.4f}, kappa={kappa:.2e}"
            )

        KAPPA_CRYSTAL_THRESHOLD = 1.5

        crystal_candidates = {
            bs: r for bs, r in results.items()
            if r['kappa'] < KAPPA_CRYSTAL_THRESHOLD and r['kappa'] != float('inf')
        }

        if crystal_candidates:
            best_bs = min(crystal_candidates.keys(), key=lambda k: crystal_candidates[k]['delta'])
            reason = f"kappa={crystal_candidates[best_bs]['kappa']:.2e} (CRYSTALLINE)"
        else:
            best_bs = min(results.keys(), key=lambda k: results[k]['delta'])
            reason = f"no kappa~1 found, using delta fallback"

        self.logger.info(
            f"Phase 1 complete: Best batch_size={best_bs} "
            f"(delta={results[best_bs]['delta']:.6f}, kappa={results[best_bs]['kappa']:.2e}) - {reason}"
        )
        return best_bs


class SeedMiner:
    def __init__(
        self,
        config: Config,
        hamiltonian_engine: HamiltonianInferenceEngine,
        batch_size: int
    ):
        self.config = config
        self.hamiltonian_engine = hamiltonian_engine
        self.batch_size = batch_size
        self.logger = LoggerFactory.create_logger("SeedMiner")

    def mine(self) -> Optional[int]:
        self.logger.info("Phase 2: Seed mining started")
        self.logger.info("  Using kappa~1 priority + v_delta (delta velocity) criterion")
        prospect_epochs = self.config.MINING_PROSPECT_EPOCHS
        interval = self.config.MINING_PROSPECT_DELTA_EPOCH_INTERVAL
        max_attempts = self.config.MINING_MAX_ATTEMPTS
        start_seed = self.config.MINING_START_SEED
        seed_results = {}

        KAPPA_CRYSTAL_THRESHOLD = 1.5

        for i in range(start_seed, start_seed + max_attempts):
            current_seed = i
            SeedManager.set_seed(current_seed, self.config.DEVICE)
            dataset = DiracDataset(
                self.config, self.hamiltonian_engine, seed=current_seed
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            val_x, val_y = dataset.get_validation_batch()
            val_x = val_x.to(self.config.DEVICE)
            val_y = val_y.to(self.config.DEVICE)
            model = DiracSpectralNetwork(
                grid_size=self.config.GRID_SIZE,
                hidden_dim=self.config.HIDDEN_DIM,
                expansion_dim=self.config.EXPANSION_DIM,
                num_spectral_layers=self.config.NUM_SPECTRAL_LAYERS
            ).to(self.config.DEVICE)
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                momentum=self.config.MOMENTUM
            )
            engine = TrainingEngine(self.config)
            crystal_calc = CrystallographyMetricsCalculator(self.config)
            delta_trajectory = []
            kappa_trajectory = []

            for ep in range(1, prospect_epochs + 1):
                engine.train_single_epoch(model, optimizer, loader, ep)
                if ep % interval == 0:
                    delta = crystal_calc.compute_discretization_margin(model)
                    kappa = crystal_calc.compute_kappa(model, val_x, val_y)
                    delta_trajectory.append((ep, delta))
                    kappa_trajectory.append((ep, kappa))

            if len(delta_trajectory) >= 2:
                first_delta = delta_trajectory[0][1]
                last_delta = delta_trajectory[-1][1]
                delta_change = last_delta - first_delta
                epochs_span = delta_trajectory[-1][0] - delta_trajectory[0][0]
                delta_velocity = delta_change / max(epochs_span, 1)
                is_cooling = delta_velocity < 0
            else:
                first_delta = delta_trajectory[0][1] if delta_trajectory else 1.0
                last_delta = first_delta
                delta_change = 0.0
                delta_velocity = 0.0
                is_cooling = False

            final_kappa = kappa_trajectory[-1][1] if kappa_trajectory else float('inf')
            mean_kappa = np.mean([k[1] for k in kappa_trajectory]) if kappa_trajectory else float('inf')
            is_kappa_crystalline = final_kappa < KAPPA_CRYSTAL_THRESHOLD and final_kappa != float('inf')

            seed_results[current_seed] = {
                'first_delta': first_delta,
                'last_delta': last_delta,
                'delta_change': delta_change,
                'delta_velocity': delta_velocity,
                'is_cooling': is_cooling,
                'final_kappa': final_kappa,
                'mean_kappa': mean_kappa,
                'is_kappa_crystalline': is_kappa_crystalline,
                'trajectory': delta_trajectory
            }

            kappa_status = "kappa~1" if is_kappa_crystalline else f"kappa={final_kappa:.1e}"
            temp_status = "COOLING" if is_cooling else "warming"
            self.logger.info(
                f"  Seed {current_seed:>4} ({i - start_seed + 1}/{max_attempts}): "
                f"delta={last_delta:.6f} v_delta={delta_velocity:+.6f} [{temp_status}] "
                f"{kappa_status}"
            )

        crystalline_seeds = {
            s: r for s, r in seed_results.items() if r['is_kappa_crystalline']
        }
        cooling_seeds = {
            s: r for s, r in seed_results.items() if r['is_cooling']
        }

        crystalline_and_cooling = {
            s: r for s, r in crystalline_seeds.items() if r['is_cooling']
        }

        if crystalline_and_cooling:
            best_seed = min(
                crystalline_and_cooling.keys(),
                key=lambda s: crystalline_and_cooling[s]['last_delta']
            )
            reason = "kappa~1 + COOLING + lowest delta (OPTIMAL)"
        elif crystalline_seeds:
            best_seed = min(
                crystalline_seeds.keys(),
                key=lambda s: crystalline_seeds[s]['last_delta']
            )
            reason = "kappa~1 + lowest delta (crystalline kappa, not cooling)"
        elif cooling_seeds:
            best_seed = min(
                cooling_seeds.keys(),
                key=lambda s: cooling_seeds[s]['last_delta']
            )
            reason = "COOLING + lowest delta (no kappa~1)"
        else:
            best_seed = min(
                seed_results.keys(),
                key=lambda s: seed_results[s]['last_delta']
            )
            reason = "lowest delta fallback (no kappa~1, no cooling)"

        result = seed_results[best_seed]
        self.logger.info(
            f"Phase 2 complete: Best seed={best_seed} "
            f"(delta={result['last_delta']:.6f}, v_delta={result['delta_velocity']:+.6f}, "
            f"kappa={result['final_kappa']:.2e}) - {reason}"
        )
        return best_seed


class FullTrainingOrchestrator:
    def __init__(
        self,
        config: Config,
        hamiltonian_engine: HamiltonianInferenceEngine,
        seed: int,
        batch_size: int
    ):
        self.config = config
        self.hamiltonian_engine = hamiltonian_engine
        self.seed = seed
        self.batch_size = batch_size
        self.logger = LoggerFactory.create_logger("FullTrainingOrchestrator")

    def run_phase3_training(self, start_epoch: int = 0, model: Optional[nn.Module] = None) -> Tuple[nn.Module, optim.Optimizer, TrainingMetricsMonitor]:
        self.logger.info(
            f"Phase 3: Full training started (seed={self.seed}, "
            f"batch_size={self.batch_size}, epochs={self.config.EPOCHS}, start_epoch={start_epoch})"
        )
        SeedManager.set_seed(self.seed, self.config.DEVICE)
        dataset = DiracDataset(
            self.config, self.hamiltonian_engine, seed=self.seed
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        val_x, val_y = dataset.get_validation_batch()
        val_x = val_x.to(self.config.DEVICE)
        val_y = val_y.to(self.config.DEVICE)
        if model is None:
            model = DiracSpectralNetwork(
                grid_size=self.config.GRID_SIZE,
                hidden_dim=self.config.HIDDEN_DIM,
                expansion_dim=self.config.EXPANSION_DIM,
                num_spectral_layers=self.config.NUM_SPECTRAL_LAYERS
            ).to(self.config.DEVICE)
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            momentum=self.config.MOMENTUM
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.EPOCHS,
            eta_min=self.config.LEARNING_RATE * self.config.COSINE_ANNEALING_ETA_MIN_FACTOR
        )
        lambda_scheduler = AdaptiveLambdaScheduler(self.config)
        engine = TrainingEngine(self.config)
        monitor = TrainingMetricsMonitor(self.config)
        checkpoint_mgr = CheckpointManager(self.config, checkpoint_dir="checkpoints_dirac_phase3")
        grokking_detected = False
        grokking_epoch = None
        best_delta = float('inf')
        patience_counter = 0
        start_time = time.time()
        for epoch in range(start_epoch + 1, self.config.EPOCHS + 1):
            topo_phase = 0.0
            if monitor.metrics_history['topo_phase_state']:
                topo_phase = monitor.metrics_history['topo_phase_state'][-1]
            lambda_scheduler.step_adaptive(epoch, topo_phase)
            train_loss, train_acc = engine.train_single_epoch(
                model, optimizer, loader, epoch, lambda_scheduler
            )
            scheduler.step()
            val_loss, val_acc = engine.validate(model, val_x, val_y)
            current_lr = optimizer.param_groups[0]['lr']
            all_metrics = engine.collect_all_metrics(
                model, monitor, val_x, val_y,
                lambda_scheduler=lambda_scheduler,
                current_lr=current_lr,
                epoch=epoch
            )
            monitor.update_metrics(
                epoch=epoch, loss=train_loss, val_loss=val_loss,
                val_acc=val_acc, train_acc=train_acc, **all_metrics
            )
            for param in model.parameters():
                if param.grad is not None:
                    monitor.gradient_buffer.append(param.grad.detach().clone().flatten()[:500])
                    break
            delta = all_metrics['delta']
            if delta < best_delta:
                best_delta = delta
                patience_counter = 0
            else:
                patience_counter += 1
            if (train_acc >= self.config.GROKKING_TRAIN_ACC_THRESHOLD and
                val_acc >= self.config.GROKKING_VAL_ACC_THRESHOLD and
                not grokking_detected):
                delta_slope = monitor.compute_delta_slope()
                if delta_slope < self.config.GROKKING_DELTA_SLOPE_THRESHOLD:
                    grokking_detected = True
                    grokking_epoch = epoch
                    self.logger.info(
                        f"  GROKKING detected at epoch {epoch}: "
                        f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
                        f"delta={delta:.6f}, delta_slope={delta_slope:.2e}"
                    )
            if epoch % self.config.LOG_INTERVAL_EPOCHS == 0:
                bar = monitor.format_progress_bar(epoch, self.config.EPOCHS, "P3-TRAIN")
                self.logger.info(bar)
            if checkpoint_mgr.should_save_checkpoint():
                metrics_snapshot = {
                    'epoch': epoch, 'train_loss': train_loss,
                    'val_loss': val_loss, 'val_acc': val_acc,
                    'train_acc': train_acc, **all_metrics
                }
                path = checkpoint_mgr.save_checkpoint(
                    model, optimizer, epoch, metrics_snapshot,
                    phase="phase3_training",
                    lambda_value=float(lambda_scheduler.current_lambda)
                )
                self.logger.info(f"  Checkpoint saved: {path}")
            integrity = WeightIntegrityChecker.check(model)
            if not integrity['is_valid']:
                self.logger.warning(
                    f"  Weight integrity compromised at epoch {epoch}: "
                    f"NaN={integrity['nan_count']}, Inf={integrity['inf_count']}"
                )
            if grokking_detected and patience_counter > self.config.GROKKING_PATIENCE:
                self.logger.info(
                    f"  Stopping Phase 3 at epoch {epoch}: "
                    f"grokking achieved and patience exceeded"
                )
                break
        elapsed = time.time() - start_time
        self.logger.info(
            f"Phase 3 complete in {elapsed:.1f}s. "
            f"Grokking={'YES at epoch ' + str(grokking_epoch) if grokking_detected else 'NO'}. "
            f"Best delta={best_delta:.6f}"
        )
        return model, optimizer, monitor


class RefinementOrchestrator:
    def __init__(
        self,
        config: Config,
        hamiltonian_engine: HamiltonianInferenceEngine,
        model: nn.Module,
        optimizer: optim.Optimizer,
        monitor: TrainingMetricsMonitor,
        seed: int,
        batch_size: int
    ):
        self.config = config
        self.hamiltonian_engine = hamiltonian_engine
        self.model = model
        self.optimizer = optimizer
        self.monitor = monitor
        self.seed = seed
        self.batch_size = batch_size
        self.logger = LoggerFactory.create_logger("RefinementOrchestrator")

    def run_phase4_refinement(self, start_epoch: int = 0) -> nn.Module:
        self.logger.info(
            f"Phase 4: Refinement via simulated annealing "
            f"(epochs={self.config.REFINEMENT_EPOCHS}, start_epoch={start_epoch})"
        )
        dataset = DiracDataset(
            self.config, self.hamiltonian_engine, seed=self.seed
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        val_x, val_y = dataset.get_validation_batch()
        val_x = val_x.to(self.config.DEVICE)
        val_y = val_y.to(self.config.DEVICE)
        lambda_scheduler = AdaptiveLambdaScheduler(self.config)
        lambda_scheduler.set_lambda(self.config.LAMBDA_MAX * 0.1)
        annealing = TopologicalAnnealingScheduler(self.config)
        engine = TrainingEngine(self.config)
        checkpoint_mgr = CheckpointManager(
            self.config, checkpoint_dir="checkpoints_dirac_phase4"
        )
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_delta = CrystallographyMetricsCalculator(self.config).compute_discretization_margin(
            self.model
        )
        self.logger.info(f"  Starting refinement from delta={best_delta:.6f}")
        refinement_lr = self.config.LEARNING_RATE * 0.1
        for pg in self.optimizer.param_groups:
            pg['lr'] = refinement_lr
        start_time = time.time()
        for epoch in range(start_epoch + 1, self.config.REFINEMENT_EPOCHS + 1):
            topo_phase = 0.0
            topo_alignment_trend = 0.0
            topo_resonance = 0.0
            if self.monitor.metrics_history['topo_phase_state']:
                topo_phase = self.monitor.metrics_history['topo_phase_state'][-1]
            if self.monitor.metrics_history['topo_alignment_trend']:
                topo_alignment_trend = self.monitor.metrics_history['topo_alignment_trend'][-1]
            if self.monitor.metrics_history['topo_resonance_score']:
                topo_resonance = self.monitor.metrics_history['topo_resonance_score'][-1]
            lambda_scheduler.step_adaptive(epoch, topo_phase)
            annealing.step_adaptive(topo_alignment_trend, topo_resonance)
            previous_state = copy.deepcopy(self.model.state_dict())
            train_loss, train_acc = engine.train_single_epoch(
                self.model, self.optimizer, loader, epoch, lambda_scheduler
            )
            val_loss, val_acc = engine.validate(self.model, val_x, val_y)
            current_delta = CrystallographyMetricsCalculator(
                self.config
            ).compute_discretization_margin(self.model)
            delta_diff = current_delta - best_delta
            if not annealing.accept_perturbation(delta_diff):
                self.model.load_state_dict(previous_state)
                current_delta = best_delta
            else:
                if current_delta < best_delta:
                    best_delta = current_delta
                    best_model_state = copy.deepcopy(self.model.state_dict())
            if annealing.should_restart(current_delta, best_delta):
                self.model.load_state_dict(best_model_state)
                self.logger.info(
                    f"  Annealing restart at epoch {epoch}, "
                    f"reverting to best_delta={best_delta:.6f}"
                )
            current_lr = self.optimizer.param_groups[0]['lr']
            all_metrics = engine.collect_all_metrics(
                self.model, self.monitor, val_x, val_y,
                lambda_scheduler=lambda_scheduler,
                annealing_scheduler=annealing,
                current_lr=current_lr,
                epoch=epoch
            )
            self.monitor.update_metrics(
                epoch=epoch, loss=train_loss, val_loss=val_loss,
                val_acc=val_acc, train_acc=train_acc, **all_metrics
            )
            if epoch % self.config.LOG_INTERVAL_EPOCHS == 0:
                bar = self.monitor.format_progress_bar(
                    epoch, self.config.REFINEMENT_EPOCHS, "P4-REFINE"
                )
                self.logger.info(bar)
                self.logger.info(
                    f"  Annealing T={annealing.temperature:.2e}, "
                    f"best_delta={best_delta:.6f}, "
                    f"current_delta={current_delta:.6f}, "
                    f"topo_phase={topo_phase:.4f}"
                )
            if checkpoint_mgr.should_save_checkpoint():
                metrics_snapshot = {
                    'epoch': epoch, 'train_loss': train_loss,
                    'val_loss': val_loss, 'val_acc': val_acc,
                    'train_acc': train_acc,
                    'best_delta': best_delta,
                    'annealing_temperature': annealing.temperature,
                    **all_metrics
                }
                path = checkpoint_mgr.save_checkpoint(
                    self.model, self.optimizer, epoch, metrics_snapshot,
                    phase="phase4_refinement",
                    lambda_value=float(lambda_scheduler.current_lambda)
                )
                self.logger.info(f"  Checkpoint saved: {path}")
        self.model.load_state_dict(best_model_state)
        elapsed = time.time() - start_time
        self.logger.info(
            f"Phase 4 complete in {elapsed:.1f}s. "
            f"Final best_delta={best_delta:.6f}"
        )
        return self.model


class Phase5Orchestrator:
    def __init__(
        self,
        config: Config,
        hamiltonian_engine: HamiltonianInferenceEngine,
        model: nn.Module,
        monitor: TrainingMetricsMonitor,
        seed: int,
        batch_size: int
    ):
        self.config = config
        self.hamiltonian_engine = hamiltonian_engine
        self.model = model
        self.monitor = monitor
        self.seed = seed
        self.batch_size = batch_size
        self.logger = LoggerFactory.create_logger("Phase5Orchestrator")
        self.checkpoint_mgr = Phase5CheckpointManager(config)
        self.ricci_flow = PerelmanRicciFlow(config)
        self.stagnation_counter = 0
        self.last_best_delta = float('inf')
        self.stagnation_shocks_applied = 0
        self.delta_history = deque(maxlen=50)
        self.worsening_counter = 0
        self.flood_fill_level = 0
        self.flood_fill_patience_counter = 0
        self.spec_gap_history = deque(maxlen=30)
        self.resonance_history = deque(maxlen=30)
        self.flood_fill_lambda = self.config.FLOOD_FILL_INITIAL_LAMBDA

    def _detect_blocked_labyrinth(self, spec_gap: float, anisotropy: float, resonance: float) -> bool:
        self.spec_gap_history.append(spec_gap)
        self.resonance_history.append(resonance)
        if len(self.spec_gap_history) < 10:
            return False
        recent_gaps = list(self.spec_gap_history)[-10:]
        gap_closing = all(recent_gaps[i] >= recent_gaps[i+1] for i in range(len(recent_gaps)-1))
        stuck_against_wall = anisotropy > self.config.FLOOD_FILL_ANISOTROPY_THRESHOLD
        recent_resonance = list(self.resonance_history)[-10:]
        resonance_stagnant = (max(recent_resonance) - min(recent_resonance)) < 0.01
        return gap_closing and stuck_against_wall and resonance_stagnant

    def _apply_flood_fill_pressure(self, lambda_scheduler, epoch: int, spec_gap: float, anisotropy: float):
        self.flood_fill_patience_counter += 1
        current_lambda = float(lambda_scheduler.current_lambda)
        self.flood_fill_lambda = current_lambda
        blocked = self._detect_blocked_labyrinth(spec_gap, anisotropy, 0.27)
        if self.flood_fill_patience_counter >= self.config.FLOOD_FILL_PATIENCE or blocked:
            old_lambda = current_lambda
            growth = self.config.FLOOD_FILL_GROWTH_FACTOR
            new_lambda = min(
                current_lambda * growth,
                self.config.FLOOD_FILL_MAX_LAMBDA
            )
            self.flood_fill_lambda = new_lambda
            lambda_scheduler.set_lambda(new_lambda)
            self._inject_diffusion_energy()
            self.flood_fill_level += 1
            self.flood_fill_patience_counter = 0
            self.logger.warning(
                f"  FLOOD FILL Level {self.flood_fill_level}: "
                f"Lambda {old_lambda:.2e} -> {new_lambda:.2e}, "
                f"SpecGap={spec_gap:.4f}, Aniso={anisotropy:.4f}, "
                f"Blocked={'YES' if blocked else 'NO'}"
            )
            return True
        return False

    def _inject_diffusion_energy(self):
        if not self.config.FLOOD_FILL_ENABLED:
            return
        scale = self.config.FLOOD_FILL_DIFFUSION_SCALE
        lambda_factor = 1.0 / (1.0 + np.log10(self.flood_fill_lambda + 1))
        with torch.no_grad():
            for param in self.model.parameters():
                noise_scale = scale * lambda_factor
                diffusion = torch.randn_like(param) * noise_scale
                param.add_(diffusion)

    def _find_ballistic_trajectory(self, resonance: float, anisotropy: float) -> Dict[str, Any]:
        resonance_progress = resonance / self.config.BALLISTIC_RESONANCE_TARGET
        anisotropy_progress = 1.0 - (anisotropy / self.config.FLOOD_FILL_ANISOTROPY_THRESHOLD)
        ballistic_score = (resonance_progress + anisotropy_progress) / 2.0
        return {
            'resonance_progress': min(resonance_progress, 1.0),
            'anisotropy_progress': max(anisotropy_progress, 0.0),
            'ballistic_score': ballistic_score,
            'on_ballistic_path': ballistic_score > 0.5
        }

    def _load_phase5_checkpoint(self, optimizer: optim.Optimizer, lambda_scheduler) -> int:
        checkpoint_path = self.config.PHASE5_CHECKPOINT_LATEST_PATH
        if not os.path.exists(checkpoint_path):
            return 0
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.config.DEVICE,
                weights_only=False
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            saved_lambda = checkpoint.get('lambda_pressure', self.config.FLOOD_FILL_INITIAL_LAMBDA)
            if saved_lambda > self.config.FLOOD_FILL_MAX_LAMBDA:
                self.logger.warning(
                    f"Saved Lambda {saved_lambda:.2e} is too high! "
                    f"Resetting to Flood Fill initial: {self.config.FLOOD_FILL_INITIAL_LAMBDA:.2e}"
                )
                saved_lambda = self.config.FLOOD_FILL_INITIAL_LAMBDA
            lambda_scheduler.set_lambda(saved_lambda)
            self.flood_fill_lambda = saved_lambda
            epoch = checkpoint.get('epoch', 0)
            self.last_best_delta = checkpoint.get('metrics', {}).get('delta', float('inf'))
            self.logger.info(f"Loaded Phase 5 checkpoint: epoch={epoch}, delta={self.last_best_delta:.6f}, lambda={saved_lambda:.2e}")
            return epoch
        except Exception as e:
            self.logger.warning(f"Failed to load Phase 5 checkpoint: {e}")
            return 0

    def _apply_perelman_surgery(self, lambda_scheduler, epoch: int, ricci_scalar: float):
        surgery_result = self.ricci_flow.perform_perelman_surgery(self.model, ricci_scalar)
        old_lambda = float(lambda_scheduler.current_lambda)
        new_lambda = old_lambda * self.config.STAGNATION_LAMBDA_BOOST
        if new_lambda > self.config.LAMBDA_MAX_SAFE:
            new_lambda = self.config.LAMBDA_MAX_SAFE
            self.logger.warning(f"  Lambda capped at safe maximum: {new_lambda:.2e}")
        lambda_scheduler.set_lambda(new_lambda)
        self.stagnation_shocks_applied += 1
        self.stagnation_counter = 0
        if surgery_result['surgery_performed']:
            self.logger.warning(
                f"  PERELMAN SURGERY at epoch {epoch}! "
                f"Cut {surgery_result['total_cuts']} singularities, "
                f"lambda {old_lambda:.2e} -> {new_lambda:.2e}"
            )
        else:
            shock_scale = self.config.STAGNATION_THERMAL_SHOCK_SCALE * 0.1
            with torch.no_grad():
                for param in self.model.parameters():
                    shock = torch.randn_like(param) * shock_scale
                    param.add_(shock)
            self.logger.warning(
                f"  GENTLE THERMAL NUDGE at epoch {epoch}! "
                f"lambda {old_lambda:.2e} -> {new_lambda:.2e}"
            )

    def run_phase5_crystallization(self, start_epoch: int = 0) -> nn.Module:
        self.logger.info("=" * 80)
        self.logger.info("PHASE 5: QUADRUPLE PRECISION (float128) HIGH-PRESSURE CRYSTALLIZATION")
        self.logger.info("Enhanced with PERELMAN RICCI FLOW for hypersphere convergence")
        self.logger.info("=" * 80)
        self.logger.info(f"Precision: {self.config.PHASE5_PRECISION}")
        self.logger.info(f"Lambda initial: {self.config.PHASE5_LAMBDA_INITIAL:.2e}")
        self.logger.info(f"Lambda max: {self.config.PHASE5_LAMBDA_MAX:.2e}")
        self.logger.info(f"Target delta: {self.config.PHASE5_DELTA_TARGET}")
        self.logger.info(f"Target alpha: {self.config.PHASE5_ALPHA_TARGET}")
        self.logger.info(f"Ricci Flow: {'ENABLED' if self.config.RICCI_FLOW_ENABLED else 'DISABLED'}")
        self.logger.info(f"Target anisotropy (hypersphere): {self.config.RICCI_ANISOTROPY_TARGET}")
        self.logger.info(f"Stagnation patience: {self.config.STAGNATION_PATIENCE} epochs")

        dataset = DiracDataset(
            self.config, self.hamiltonian_engine, seed=self.seed
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        val_x, val_y = dataset.get_validation_batch()
        val_x = val_x.to(self.config.DEVICE)
        val_y = val_y.to(self.config.DEVICE)

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE * 0.01,
            weight_decay=self.config.WEIGHT_DECAY,
            momentum=self.config.MOMENTUM
        )

        lambda_scheduler = QuadruplePrecisionLambdaScheduler(self.config)
        engine = TrainingEngine(self.config)
        crystal_calc = CrystallographyMetricsCalculator(self.config)

        best_delta = crystal_calc.compute_discretization_margin(self.model)
        best_alpha = crystal_calc.compute_alpha_purity(self.model)
        best_val_acc = 0.0

        if start_epoch == 0:
            loaded_epoch = self._load_phase5_checkpoint(optimizer, lambda_scheduler)
            if loaded_epoch > 0:
                start_epoch = loaded_epoch
                best_delta = self.last_best_delta
                self.logger.info(f"  Resuming Phase 5 from epoch {start_epoch}, delta={best_delta:.6f}")
            else:
                self.logger.info(f"  Starting Phase 5 from delta={best_delta:.6f}, alpha={best_alpha:.4f}")
        else:
            self.logger.info(f"  Resuming Phase 5 from epoch {start_epoch}, delta={best_delta:.6f}")

        self.last_best_delta = best_delta

        start_time = time.time()
        collapsed = False

        for epoch in range(start_epoch + 1, self.config.PHASE5_EPOCHS + 1):
            previous_state = copy.deepcopy(self.model.state_dict())

            if self.config.RICCI_FLOW_ENABLED:
                lr_factor = self.ricci_flow.compute_adaptive_lr_factor(self.model)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.config.LEARNING_RATE * 0.01 * lr_factor

            train_loss, train_acc = engine.train_single_epoch(
                self.model, optimizer, loader, epoch, lambda_scheduler,
                ricci_flow=self.ricci_flow if self.config.RICCI_FLOW_ENABLED else None
            )

            thermal_scale = self.config.PHASE5_THERMAL_INJECTION_SCALE * (
                1.0 + epoch / self.config.PHASE5_EPOCHS
            )
            with torch.no_grad():
                for param in self.model.parameters():
                    thermal_noise = torch.randn_like(param) * thermal_scale
                    param.add_(thermal_noise)

            ricci_flow_result = {'ricci_flow_applied': False}
            if self.config.RICCI_FLOW_ENABLED:
                ricci_flow_result = self.ricci_flow.apply_ricci_flow_step(
                    self.model, optimizer.param_groups[0]['lr']
                )

            val_loss, val_acc = engine.validate(self.model, val_x, val_y)

            current_delta = crystal_calc.compute_discretization_margin(self.model)
            current_alpha = crystal_calc.compute_alpha_purity(self.model)

            ricci_metrics = self.ricci_flow.get_flow_metrics(self.model) if self.config.RICCI_FLOW_ENABLED else {}

            all_metrics = engine.collect_all_metrics(
                self.model, self.monitor, val_x, val_y,
                lambda_scheduler=lambda_scheduler,
                current_lr=optimizer.param_groups[0]['lr'],
                epoch=epoch
            )

            spec_gap = all_metrics.get('spectral_gap', 0.05)
            resonance = ricci_metrics.get('resonance_score', 0.27) if ricci_metrics else 0.27
            anisotropy = ricci_metrics.get('anisotropy', 0.67) if ricci_metrics else 0.67

            integrity = WeightIntegrityChecker.check(self.model)
            if not integrity['is_valid']:
                self.logger.warning(
                    f"  Model collapsed at epoch {epoch}: "
                    f"NaN={integrity['nan_count']}, Inf={integrity['inf_count']}. "
                    f"Reverting to previous state."
                )
                self.model.load_state_dict(previous_state)
                collapsed = True
                continue

            is_improvement = (
                current_delta < best_delta - self.config.STAGNATION_MIN_DELTA_IMPROVEMENT or
                (current_alpha > best_alpha and val_acc >= best_val_acc * 0.95)
            )

            self.delta_history.append(current_delta)

            is_worsening = False
            if len(self.delta_history) >= 10:
                recent = list(self.delta_history)[-10:]
                if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                    is_worsening = True
                    self.worsening_counter += 1
                else:
                    self.worsening_counter = max(0, self.worsening_counter - 1)

            if current_delta < best_delta:
                best_delta = current_delta
                self.stagnation_counter = 0
                self.worsening_counter = 0
                self.flood_fill_patience_counter = 0
            else:
                self.stagnation_counter += 1

            if current_alpha > best_alpha:
                best_alpha = current_alpha
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            flood_applied = False
            if self.config.FLOOD_FILL_ENABLED:
                flood_applied = self._apply_flood_fill_pressure(
                    lambda_scheduler, epoch, spec_gap, anisotropy
                )
                if flood_applied:
                    current_delta = crystal_calc.compute_discretization_margin(self.model)
                    current_alpha = crystal_calc.compute_alpha_purity(self.model)

            needs_surgery = (
                self.stagnation_counter >= self.config.STAGNATION_PATIENCE or
                self.worsening_counter >= 10
            )

            if needs_surgery and not flood_applied:
                ricci_scalar = ricci_metrics.get('ricci_scalar', 0)
                self._apply_perelman_surgery(lambda_scheduler, epoch, ricci_scalar)
                current_delta = crystal_calc.compute_discretization_margin(self.model)
                current_alpha = crystal_calc.compute_alpha_purity(self.model)
                ricci_metrics = self.ricci_flow.get_flow_metrics(self.model) if self.config.RICCI_FLOW_ENABLED else {}
                if current_delta < best_delta:
                    best_delta = current_delta
                self.delta_history.clear()
                self.worsening_counter = 0

            lambda_scheduler.step(epoch, improvement=is_improvement and not collapsed)
            collapsed = False

            all_metrics['val_acc'] = val_acc
            all_metrics['train_acc'] = train_acc
            all_metrics['stagnation_counter'] = self.stagnation_counter
            all_metrics['stagnation_shocks'] = self.stagnation_shocks_applied
            all_metrics['flood_fill_level'] = self.flood_fill_level

            all_metrics.update(ricci_metrics)
            all_metrics['ricci_flow_applied'] = ricci_flow_result.get('ricci_flow_applied', False)

            if self.config.FLOOD_FILL_ENABLED:
                ballistic = self._find_ballistic_trajectory(resonance, anisotropy)
                all_metrics['ballistic_score'] = ballistic['ballistic_score']
                all_metrics['on_ballistic_path'] = ballistic['on_ballistic_path']

            self.monitor.update_metrics(
                epoch=epoch, loss=train_loss, val_loss=val_loss,
                **all_metrics
            )

            if epoch % self.config.LOG_INTERVAL_EPOCHS == 0:
                bar = self.monitor.format_progress_bar(
                    epoch, self.config.PHASE5_EPOCHS, "P5-CRYSTAL"
                )
                self.logger.info(bar)

                flood_info = f"flood_level={self.flood_fill_level}"
                if self.config.FLOOD_FILL_ENABLED:
                    ballistic = self._find_ballistic_trajectory(resonance, anisotropy)
                    flood_info += f", ballistic={ballistic['ballistic_score']:.2f}"

                stagnation_info = f"stagnation={self.stagnation_counter}/{self.config.STAGNATION_PATIENCE}"
                if self.stagnation_shocks_applied > 0:
                    stagnation_info += f", surgeries={self.stagnation_shocks_applied}"

                ricci_info = ""
                if self.config.RICCI_FLOW_ENABLED:
                    ricci_scalar = ricci_metrics.get('ricci_scalar', 0)
                    ricci_info = f", anisotropy={anisotropy:.4f}, Ricci={ricci_scalar:.2e}, SpecGap={spec_gap:.4f}"

                self.logger.info(
                    f"  Lambda={float(lambda_scheduler.current_lambda):.2e}, "
                    f"delta={current_delta:.6f}, alpha={current_alpha:.4f}, "
                    f"val_acc={val_acc:.4f}"
                )
                self.logger.info(
                    f"  [{flood_info}, {stagnation_info}]{ricci_info}"
                )

            self.checkpoint_mgr.save_checkpoint(
                self.model, optimizer, epoch, all_metrics,
                lambda_scheduler.current_lambda
            )

            if (current_delta < self.config.PHASE5_DELTA_TARGET and
                current_alpha > self.config.PHASE5_ALPHA_TARGET):
                self.logger.info(
                    f"  CRYSTAL ACHIEVED at epoch {epoch}: "
                    f"delta={current_delta:.6f}, alpha={current_alpha:.4f}"
                )
                break

        elapsed = time.time() - start_time
        final_ricci = self.ricci_flow.get_flow_metrics(self.model) if self.config.RICCI_FLOW_ENABLED else {}
        self.logger.info(
            f"Phase 5 complete in {elapsed:.1f}s. "
            f"Final delta={best_delta:.6f}, alpha={best_alpha:.4f}, "
            f"surgeries={self.stagnation_shocks_applied}"
        )
        if self.config.RICCI_FLOW_ENABLED:
            self.logger.info(
                f"  Final anisotropy={final_ricci.get('anisotropy', 0):.4f} "
                f"(target={self.config.RICCI_ANISOTROPY_TARGET}), "
            )
        return self.model


def main():
    parser = argparse.ArgumentParser(
        description="Dirac Equation Grokking via Hamiltonian Topological Crystallization"
    )
    parser.add_argument(
        '--phase', type=str, default='all',
        choices=['all', '1', '2', '3', '4', '5'],
        help='Training phase to run (default: all)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Override batch size (skip Phase 1)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Override random seed (skip Phase 2)'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override training epochs'
    )
    parser.add_argument(
        '--hidden-dim', type=int, default=Config.HIDDEN_DIM,
        help='Hidden dimension'
    )
    parser.add_argument(
        '--expansion-dim', type=int, default=Config.EXPANSION_DIM,
        help='Expansion dimension'
    )
    parser.add_argument(
        '--num-spectral-layers', type=int, default=Config.NUM_SPECTRAL_LAYERS,
        help='Number of spectral layers'
    )
    parser.add_argument(
        '--grid-size', type=int, default=Config.GRID_SIZE,
        help='Grid size for spatial discretization'
    )
    parser.add_argument(
        '--dirac-mass', type=float, default=Config.DIRAC_MASS,
        help='Dirac mass parameter'
    )
    parser.add_argument(
        '--backbone', type=str, default=Config.BACKBONE_CHECKPOINT_PATH,
        help='Path to backbone checkpoint'
    )
    parser.add_argument(
        '--no-backbone', action='store_true',
        help='Disable backbone loading'
    )
    parser.add_argument(
        '--resume-phase5', action='store_true',
        help='Resume from Phase 5 checkpoint'
    )
    parser.add_argument(
        '--resume-phase4', action='store_true',
        help='Resume from Phase 4 checkpoint'
    )
    parser.add_argument(
        '--resume-phase3', action='store_true',
        help='Resume from Phase 3 checkpoint'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to JSON config file'
    )
    args = parser.parse_args()

    config = Config()

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    if args.hidden_dim:
        config.HIDDEN_DIM = args.hidden_dim
    if args.expansion_dim:
        config.EXPANSION_DIM = args.expansion_dim
    if args.num_spectral_layers:
        config.NUM_SPECTRAL_LAYERS = args.num_spectral_layers
    if args.grid_size:
        config.GRID_SIZE = args.grid_size
    if args.dirac_mass:
        config.DIRAC_MASS = args.dirac_mass
    if args.backbone:
        config.BACKBONE_CHECKPOINT_PATH = args.backbone
        config.BACKBONE_ENABLED = True
    if args.no_backbone:
        config.BACKBONE_ENABLED = False
    if args.resume_phase5:
        config.RESUME_PHASE5 = True
    if args.resume_phase4:
        config.RESUME_PHASE4 = True
    if args.resume_phase3:
        config.RESUME_PHASE3 = True
    if args.epochs:
        config.EPOCHS = args.epochs

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs('checkpoints_dirac_phase3', exist_ok=True)
    os.makedirs('checkpoints_dirac_phase4', exist_ok=True)
    os.makedirs('weights', exist_ok=True)

    logger = LoggerFactory.create_logger("Main")
    logger.info("=" * 80)
    logger.info("DIRAC EQUATION GROKKING VIA HAMILTONIAN TOPOLOGICAL CRYSTALLIZATION")
    logger.info("=" * 80)
    logger.info(f"Grid size: {config.GRID_SIZE}")
    logger.info(f"Hidden dim: {config.HIDDEN_DIM}")
    logger.info(f"Expansion dim: {config.EXPANSION_DIM}")
    logger.info(f"Spectral layers: {config.NUM_SPECTRAL_LAYERS}")
    logger.info(f"Dirac mass: {config.DIRAC_MASS}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Backbone: {config.BACKBONE_CHECKPOINT_PATH if config.BACKBONE_ENABLED else 'DISABLED'}")
    if config.RESUME_PHASE5:
        logger.info("RESUME MODE: Phase 5")
    elif config.RESUME_PHASE4:
        logger.info("RESUME MODE: Phase 4")
    elif config.RESUME_PHASE3:
        logger.info("RESUME MODE: Phase 3")
    logger.info("=" * 80)

    SeedManager.set_seed(config.RANDOM_SEED, config.DEVICE)

    hamiltonian_engine = HamiltonianInferenceEngine(config)

    batch_size = args.batch_size if args.batch_size else config.BATCH_SIZE
    seed = args.seed if args.seed else config.RANDOM_SEED
    model = None
    optimizer = None
    monitor = None
    start_phase = 1

    def load_latest_checkpoint(model: nn.Module, checkpoint_paths: List[str]) -> Tuple[nn.Module, int, Dict]:
        for path in checkpoint_paths:
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location=config.DEVICE, weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    epoch = checkpoint.get('epoch', 0)
                    metrics = checkpoint.get('metrics', {})
                    logger.info(f"Loaded checkpoint from {path}, epoch={epoch}")
                    return model, epoch, metrics
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
        return model, 0, {}

    if config.RESUME_PHASE5:
        model = DiracSpectralNetwork(
            grid_size=config.GRID_SIZE,
            hidden_dim=config.HIDDEN_DIM,
            expansion_dim=config.EXPANSION_DIM,
            num_spectral_layers=config.NUM_SPECTRAL_LAYERS,
            spinor_components=config.SPINOR_COMPONENTS
        ).to(config.DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        monitor = TrainingMetricsMonitor(config)
        checkpoint_paths = [
            config.PHASE5_CHECKPOINT_LATEST_PATH,
            config.PHASE4_CHECKPOINT_PATH,
            config.PHASE3_CHECKPOINT_PATH
        ]
        model, start_epoch, loaded_metrics = load_latest_checkpoint(model, checkpoint_paths)
        if start_epoch > 0:
            logger.info(f"Resuming Phase 5 from epoch {start_epoch}")
        phase5 = Phase5Orchestrator(
            config, hamiltonian_engine, model, monitor, seed, batch_size
        )
        model = phase5.run_phase5_crystallization(start_epoch=start_epoch)

    elif config.RESUME_PHASE4:
        model = DiracSpectralNetwork(
            grid_size=config.GRID_SIZE,
            hidden_dim=config.HIDDEN_DIM,
            expansion_dim=config.EXPANSION_DIM,
            num_spectral_layers=config.NUM_SPECTRAL_LAYERS,
            spinor_components=config.SPINOR_COMPONENTS
        ).to(config.DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        monitor = TrainingMetricsMonitor(config)
        checkpoint_paths = [config.PHASE4_CHECKPOINT_PATH, config.PHASE3_CHECKPOINT_PATH]
        model, start_epoch, loaded_metrics = load_latest_checkpoint(model, checkpoint_paths)
        if start_epoch > 0:
            logger.info(f"Resuming Phase 4 from epoch {start_epoch}")
        refiner = RefinementOrchestrator(
            config, hamiltonian_engine, model, optimizer, monitor, seed, batch_size
        )
        model = refiner.run_phase4_refinement(start_epoch=start_epoch)
        if config.PHASE5_ENABLE:
            phase5 = Phase5Orchestrator(
                config, hamiltonian_engine, model, monitor, seed, batch_size
            )
            model = phase5.run_phase5_crystallization()

    elif config.RESUME_PHASE3:
        model = DiracSpectralNetwork(
            grid_size=config.GRID_SIZE,
            hidden_dim=config.HIDDEN_DIM,
            expansion_dim=config.EXPANSION_DIM,
            num_spectral_layers=config.NUM_SPECTRAL_LAYERS,
            spinor_components=config.SPINOR_COMPONENTS
        ).to(config.DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        monitor = TrainingMetricsMonitor(config)
        checkpoint_paths = [config.PHASE3_CHECKPOINT_PATH]
        model, start_epoch, loaded_metrics = load_latest_checkpoint(model, checkpoint_paths)
        if start_epoch > 0:
            logger.info(f"Resuming Phase 3 from epoch {start_epoch}")
        orchestrator = FullTrainingOrchestrator(
            config, hamiltonian_engine, seed, batch_size
        )
        model, optimizer, monitor = orchestrator.run_phase3_training(start_epoch=start_epoch, model=model)
        refiner = RefinementOrchestrator(
            config, hamiltonian_engine, model, optimizer, monitor, seed, batch_size
        )
        model = refiner.run_phase4_refinement()
        if config.PHASE5_ENABLE:
            phase5 = Phase5Orchestrator(
                config, hamiltonian_engine, model, monitor, seed, batch_size
            )
            model = phase5.run_phase5_crystallization()

    else:
        if args.phase in ['all', '1'] and batch_size is None:
            prospector = BatchSizeProspector(config, hamiltonian_engine)
            batch_size = prospector.prospect()
        elif batch_size is None:
            batch_size = config.BATCH_SIZE
            logger.info(f"Skipping Phase 1, using default batch_size={batch_size}")

        if args.phase in ['all', '2'] and seed is None:
            miner = SeedMiner(config, hamiltonian_engine, batch_size)
            seed = miner.mine()
        elif seed is None:
            seed = config.RANDOM_SEED
            logger.info(f"Skipping Phase 2, using default seed={seed}")

        if args.phase in ['all', '3']:
            orchestrator = FullTrainingOrchestrator(
                config, hamiltonian_engine, seed, batch_size
            )
            model, optimizer, monitor = orchestrator.run_phase3_training()
        else:
            logger.info("Skipping Phase 3")

        if args.phase in ['all', '4'] and model is not None:
            refiner = RefinementOrchestrator(
                config, hamiltonian_engine, model, optimizer, monitor, seed, batch_size
            )
            model = refiner.run_phase4_refinement()
        else:
            logger.info("Skipping Phase 4")

        if args.phase in ['all', '5'] and model is not None:
            phase5 = Phase5Orchestrator(
                config, hamiltonian_engine, model, monitor, seed, batch_size
            )
            model = phase5.run_phase5_crystallization()
        else:
            logger.info("Skipping Phase 5")

    logger.info("=" * 80)
    logger.info("DIRAC CRYSTALLIZATION COMPLETE")
    logger.info("=" * 80)

    results_path = os.path.join(config.RESULTS_DIR, 'final_results.json')
    if monitor is not None:
        final_metrics = {
            key: vals[-1] if vals else 0.0
            for key, vals in monitor.metrics_history.items()
            if isinstance(vals, list) and len(vals) > 0
        }
        with open(results_path, 'w') as f:
            json.dump(final_metrics, f, indent=2, default=str)
        logger.info(f"Final results saved to {results_path}")

    return 0


if __name__ == "__main__":
    exit(main())