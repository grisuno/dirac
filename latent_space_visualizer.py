#!/usr/bin/env python3
import sys
import os
import csv
import time
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sklearn.decomposition
from sklearn.preprocessing import StandardScaler

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QGroupBox, QScrollArea, QSplitter, QTabWidget,
    QStatusBar, QMenuBar, QMenu, QAction, QMessageBox,
    QProgressBar, QFrame, QCheckBox, QPlainTextEdit
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPalette, QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent / 'upload'))
from dirac_crystal2 import (
    Config as DiracConfig,
    DiracSpectralNetwork,
    DiracDataset,
    HamiltonianInferenceEngine,
    SeedManager,
    LoggerFactory,
    CrystallographyMetricsCalculator,
    SpectralGeometryCalculator,
    RicciCurvatureCalculator,
    TopologicalMetricsCalculator,
    SpectroscopyMetricsCalculator,
    LocalComplexityAnalyzer,
    SuperpositionAnalyzer,
)


@dataclass
class VisualizerConfig:
    PCA_COMPONENTS: int = 3
    UPDATE_INTERVAL_MS: int = 100
    HISTORY_LENGTH: int = 500
    WEIGHT_SAMPLE_LIMIT: int = 2000
    CSV_BUFFER_SIZE: int = 50
    OUTPUT_DIR: str = 'download/visualizer_output'
    CSV_FILENAME: str = 'training_metrics_realtime.csv'
    TRAINING_EPOCHS: int = 100
    TRAINING_BATCH_SIZE: int = 32
    TRAINING_SEED: int = 42
    TRAINING_LR: float = 0.005
    DELTA_THRESHOLD_CRYSTAL: float = 0.1
    KAPPA_THRESHOLD_CRYSTAL: float = 1.5
    ALPHA_THRESHOLD_CRYSTAL: float = 7.0
    FIGURE_DPI: int = 100
    MAX_3D_POINTS: int = 5000
    TEXTURE_RESOLUTION: int = 64


class CSVLogger:
    def __init__(self, config: VisualizerConfig):
        self._config = config
        self._output_dir = Path(config.OUTPUT_DIR)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._output_dir / config.CSV_FILENAME
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_size = config.CSV_BUFFER_SIZE
        self._fieldnames: List[str] = []
        self._file_handle = None
        self._writer = None
        self._mutex = threading.Lock()
        self._initialized = False
    
    def log(self, metrics: Dict[str, Any]) -> None:
        with self._mutex:
            flat = self._flatten_dict(metrics)
            if not self._initialized:
                self._fieldnames = list(flat.keys())
                self._file_handle = open(self._csv_path, 'w', newline='')
                self._writer = csv.DictWriter(self._file_handle, fieldnames=self._fieldnames)
                self._writer.writeheader()
                self._file_handle.flush()
                self._initialized = True
            self._buffer.append(flat)
            if len(self._buffer) >= self._buffer_size:
                self._flush()
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (int, float)):
                for i, item in enumerate(v[:10]):
                    items.append((f"{new_key}_{i}", item))
            elif isinstance(v, (int, float, str, bool)):
                items.append((new_key, v))
            elif v is None:
                items.append((new_key, ''))
        return dict(items)
    
    def _flush(self) -> None:
        if self._writer and self._buffer:
            for row in self._buffer:
                self._writer.writerow(row)
            self._file_handle.flush()
            self._buffer.clear()
    
    def close(self) -> None:
        with self._mutex:
            self._flush()
            if self._file_handle:
                self._file_handle.close()
    
    def get_csv_path(self) -> Path:
        return self._csv_path


class LatentSpaceWidget(FigureCanvas):
    def __init__(self, config: VisualizerConfig, parent=None):
        self._fig = Figure(figsize=(5, 4), dpi=config.FIGURE_DPI)
        super().__init__(self._fig)
        self.setParent(parent)
        self._config = config
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._history: deque = deque(maxlen=config.HISTORY_LENGTH)
        self._colors: deque = deque(maxlen=config.HISTORY_LENGTH)
        self._ax.set_xlabel('PC1')
        self._ax.set_ylabel('PC2')
        self._ax.set_zlabel('PC3')
        self._ax.set_title('Latent Space (PCA)')
        self._fig.tight_layout()
    
    def update_data(self, weights: np.ndarray, metric_value: float = 0.0) -> None:
        if weights.shape[0] < 3:
            return
        try:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(weights)
            pca = sklearn.decomposition.PCA(n_components=3)
            proj = pca.fit_transform(scaled)
            
            self._history.append(proj.copy())
            self._colors.append(metric_value)
            
            self._ax.clear()
            self._ax.set_xlabel('PC1')
            self._ax.set_ylabel('PC2')
            self._ax.set_zlabel('PC3')
            
            all_proj = np.vstack(list(self._history))
            colors = np.array(list(self._colors))
            
            if len(colors) > 0:
                norm = Normalize(vmin=colors.min(), vmax=colors.max() + 1e-10)
            else:
                norm = Normalize(vmin=0, vmax=1)
            
            if all_proj.shape[0] > self._config.MAX_3D_POINTS:
                idx = np.random.choice(all_proj.shape[0], self._config.MAX_3D_POINTS, replace=False)
                all_proj = all_proj[idx]
            
            self._ax.scatter(all_proj[:, 0], all_proj[:, 1], all_proj[:, 2],
                           c=np.repeat(colors, all_proj.shape[0] // len(colors) + 1)[:all_proj.shape[0]],
                           cmap='viridis', s=5, alpha=0.6, norm=norm)
            
            self._ax.set_title(f'Latent Space (PC: {pca.explained_variance_ratio_[0]*100:.1f}%)')
            self.draw()
        except Exception:
            pass
    
    def clear(self) -> None:
        self._history.clear()
        self._colors.clear()
        self._ax.clear()
        self._ax.set_xlabel('PC1')
        self._ax.set_ylabel('PC2')
        self._ax.set_zlabel('PC3')
        self._ax.set_title('Latent Space (PCA)')
        self.draw()


class MetricsWidget(FigureCanvas):
    def __init__(self, config: VisualizerConfig, parent=None):
        self._fig = Figure(figsize=(6, 5), dpi=config.FIGURE_DPI)
        super().__init__(self._fig)
        self.setParent(parent)
        self._config = config
        self._axes = self._fig.subplots(2, 2)
        self._history: Dict[str, deque] = {k: deque(maxlen=config.HISTORY_LENGTH) 
                                           for k in ['loss', 'delta', 'kappa', 'alpha', 'topo_phase_state']}
        self._epochs: deque = deque(maxlen=config.HISTORY_LENGTH)
        self._setup_axes()
    
    def _setup_axes(self) -> None:
        self._axes[0, 0].set_title('Loss')
        self._axes[0, 1].set_title('Delta (Discretization)')
        self._axes[1, 0].set_title('Kappa (Cond. Number)')
        self._axes[1, 1].set_title('Topo Phase State')
        for ax in self._axes.flat:
            ax.set_xlabel('Epoch', fontsize=8)
        self._fig.tight_layout()
    
    def update_data(self, metrics: Dict[str, Any]) -> None:
        self._epochs.append(metrics.get('epoch', 0))
        for k in self._history:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, (int, float)):
                    self._history[k].append(v if not np.isinf(v) else np.nan)
        
        for ax in self._axes.flat:
            ax.clear()
        
        epochs = list(self._epochs)
        
        if self._history['loss']:
            self._axes[0, 0].semilogy(epochs, list(self._history['loss']), 'b-')
        self._axes[0, 0].set_title('Loss')
        self._axes[0, 0].axhline(y=0.05, color='g', linestyle='--', alpha=0.5)
        
        if self._history['delta']:
            self._axes[0, 1].semilogy(epochs, list(self._history['delta']), 'r-')
        self._axes[0, 1].set_title('Delta')
        self._axes[0, 1].axhline(y=self._config.DELTA_THRESHOLD_CRYSTAL, color='g', linestyle='--', alpha=0.5)
        
        if self._history['kappa']:
            kappa = [k if k < 1e6 else 1e6 for k in self._history['kappa']]
            self._axes[1, 0].semilogy(epochs, kappa, 'b-')
        self._axes[1, 0].set_title('Kappa')
        self._axes[1, 0].axhline(y=self._config.KAPPA_THRESHOLD_CRYSTAL, color='g', linestyle='--', alpha=0.5)
        
        if self._history['topo_phase_state']:
            self._axes[1, 1].plot(epochs, list(self._history['topo_phase_state']), 'b-')
            self._axes[1, 1].fill_between(epochs, 0, list(self._history['topo_phase_state']), alpha=0.3)
        self._axes[1, 1].set_title('Topo Phase State')
        self._axes[1, 1].axhline(y=0.7, color='g', linestyle='--', alpha=0.5)
        self._axes[1, 1].set_ylim(0, 1.1)
        
        for ax in self._axes.flat:
            ax.set_xlabel('Epoch', fontsize=8)
        
        self._fig.tight_layout()
        self.draw()
    
    def clear(self) -> None:
        for k in self._history:
            self._history[k].clear()
        self._epochs.clear()
        for ax in self._axes.flat:
            ax.clear()
        self._setup_axes()
        self.draw()


class WeightTextureWidget(FigureCanvas):
    def __init__(self, config: VisualizerConfig, parent=None):
        self._fig = Figure(figsize=(6, 3), dpi=config.FIGURE_DPI)
        super().__init__(self._fig)
        self.setParent(parent)
        self._config = config
        self._axes = self._fig.subplots(1, 2)
        self._axes[0].set_title('Weights')
        self._axes[1].set_title('Gradients')
        self._fig.tight_layout()
    
    def update_data(self, weights: np.ndarray, gradients: np.ndarray) -> None:
        for ax in self._axes:
            ax.clear()
        
        w = self._reshape(weights)
        g = self._reshape(gradients)
        
        self._axes[0].imshow(w, cmap='RdBu_r', aspect='auto')
        self._axes[0].set_title(f'Weights ({weights.mean():.4f})')
        
        self._axes[1].imshow(g, cmap='RdBu_r', aspect='auto')
        self._axes[1].set_title(f'Gradients ({gradients.mean():.6f})')
        
        self._fig.tight_layout()
        self.draw()
    
    def _reshape(self, arr: np.ndarray) -> np.ndarray:
        flat = arr.flatten()
        r = self._config.TEXTURE_RESOLUTION
        size = r * r
        if flat.size > size:
            flat = flat[:size]
        else:
            flat = np.pad(flat, (0, size - flat.size))
        return flat.reshape(r, r)
    
    def clear(self) -> None:
        for ax in self._axes:
            ax.clear()
        self._axes[0].set_title('Weights')
        self._axes[1].set_title('Gradients')
        self.draw()


class TrainingWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(dict)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, config: VisualizerConfig, dirac_config: DiracConfig):
        super().__init__()
        self._config = config
        self._dirac_config = dirac_config
        self._is_running = False
        self._is_paused = False
        self._model = None
        self._dataloader = None
        self._val_x = None
        self._val_y = None
        self._optimizer = None
        self._criterion = nn.MSELoss()
        self._crystal_calc = None
        self._spectral_calc = None
        self._ricci_calc = None
        self._topo_calc = None
        self._spectro_calc = None
    
    def setup(self):
        self.log_message.emit("Initializing model...")
        
        SeedManager.set_seed(self._config.TRAINING_SEED, self._dirac_config.DEVICE)
        
        self._model = DiracSpectralNetwork(
            grid_size=self._dirac_config.GRID_SIZE,
            hidden_dim=self._dirac_config.HIDDEN_DIM,
            expansion_dim=self._dirac_config.EXPANSION_DIM,
            num_spectral_layers=self._dirac_config.NUM_SPECTRAL_LAYERS
        ).to(self._dirac_config.DEVICE)
        
        n_params = sum(p.numel() for p in self._model.parameters())
        self.log_message.emit(f"Model: {n_params} parameters")
        
        engine = HamiltonianInferenceEngine(self._dirac_config)
        dataset = DiracDataset(self._dirac_config, engine, seed=self._config.TRAINING_SEED)
        
        self._dataloader = DataLoader(dataset, batch_size=self._config.TRAINING_BATCH_SIZE, shuffle=True)
        self._val_x, self._val_y = dataset.get_validation_batch()
        self._val_x = self._val_x.to(self._dirac_config.DEVICE)
        self._val_y = self._val_y.to(self._dirac_config.DEVICE)
        
        self.log_message.emit(f"Dataset: {len(dataset)} samples")
        
        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self._config.TRAINING_LR,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        self._crystal_calc = CrystallographyMetricsCalculator(self._dirac_config)
        self._spectral_calc = SpectralGeometryCalculator(self._dirac_config)
        self._ricci_calc = RicciCurvatureCalculator(self._dirac_config)
        self._topo_calc = TopologicalMetricsCalculator(self._dirac_config)
        self._spectro_calc = SpectroscopyMetricsCalculator(self._dirac_config)
        
        self.log_message.emit("Ready to train")
    
    def run(self):
        self._is_running = True
        self.setup()
        
        for epoch in range(1, self._config.TRAINING_EPOCHS + 1):
            if not self._is_running:
                break
            
            while self._is_paused:
                time.sleep(0.1)
            
            try:
                train_loss = self._train_epoch()
                val_loss, val_acc = self._validate()
                
                metrics = self._compute_metrics(epoch, train_loss, val_loss, val_acc)
                self.progress.emit(metrics)
                
                self.log_message.emit(
                    f"Epoch {epoch}: loss={train_loss:.6f}, delta={metrics['delta']:.6f}, "
                    f"kappa={metrics['kappa']:.2e}, phase={metrics['topo_phase_state']:.4f}"
                )
                
            except Exception as e:
                self.error.emit(f"Error at epoch {epoch}: {str(e)}")
                import traceback
                self.log_message.emit(traceback.format_exc())
                break
        
        self._is_running = False
        self.finished.emit()
    
    def _train_epoch(self) -> float:
        self._model.train()
        total_loss = 0.0
        n = 0
        for bx, by in self._dataloader:
            bx = bx.to(self._dirac_config.DEVICE)
            by = by.to(self._dirac_config.DEVICE)
            
            self._optimizer.zero_grad()
            out = self._model(bx)
            loss = self._criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            self._optimizer.step()
            
            total_loss += loss.item() * bx.size(0)
            n += bx.size(0)
        
        return total_loss / n if n > 0 else 0.0
    
    def _validate(self) -> Tuple[float, float]:
        self._model.eval()
        with torch.no_grad():
            out = self._model(self._val_x)
            loss = self._criterion(out, self._val_y).item()
            mse = ((out - self._val_y)**2).mean(dim=(1, 2, 3))
            acc = (mse < 0.05).float().mean().item()
        return loss, acc
    
    def _compute_metrics(self, epoch: int, train_loss: float, val_loss: float, val_acc: float) -> Dict[str, Any]:
        metrics = {
            'epoch': epoch,
            'loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            c = self._crystal_calc.compute_all_metrics(self._model, self._val_x, self._val_y)
            metrics['kappa'] = c.get('kappa', float('inf'))
            metrics['delta'] = c.get('delta', 1.0)
            metrics['alpha'] = c.get('alpha', 0.0)
            metrics['is_crystal'] = c.get('is_crystal', False)
        except:
            metrics['kappa'] = float('inf')
            metrics['delta'] = 1.0
            metrics['alpha'] = 0.0
            metrics['is_crystal'] = False
        
        try:
            s = self._spectral_calc.compute(self._model)
            metrics['spectral_gap'] = s.get('spectral_gap', 0.0)
            metrics['participation_ratio'] = s.get('participation_ratio', 0.0)
        except:
            metrics['spectral_gap'] = 0.0
            metrics['participation_ratio'] = 0.0
        
        try:
            r = self._ricci_calc.compute(self._model)
            metrics['ricci_scalar'] = r.get('ricci_scalar', 0.0)
        except:
            metrics['ricci_scalar'] = 0.0
        
        try:
            t = self._topo_calc.compute(self._model, epoch=epoch)
            metrics['topo_phase_state'] = t.get('topo_phase_state', 0.0)
            metrics['topo_localization'] = t.get('topo_localization', 0.0)
        except:
            metrics['topo_phase_state'] = 0.0
            metrics['topo_localization'] = 0.0
        
        try:
            sp = self._spectro_calc.compute(self._model)
            metrics['spectral_entropy'] = sp.get('spectral_entropy', 0.0)
        except:
            metrics['spectral_entropy'] = 0.0
        
        weights = self._extract_weights()
        gradients = self._extract_gradients()
        metrics['_weights'] = weights
        metrics['_gradients'] = gradients
        
        return metrics
    
    def _extract_weights(self) -> np.ndarray:
        ws = []
        for p in self._model.parameters():
            if p.dim() >= 1:
                ws.append(p.detach().cpu().flatten()[:2000])
        if not ws:
            return np.zeros((1, 2000))
        min_len = min(w.shape[0] for w in ws)
        return torch.stack([w[:min_len] for w in ws]).numpy()
    
    def _extract_gradients(self) -> np.ndarray:
        gs = []
        for p in self._model.parameters():
            if p.grad is not None:
                gs.append(p.grad.detach().cpu().flatten()[:2000])
        if not gs:
            return np.zeros((1, 2000))
        min_len = min(g.shape[0] for g in gs)
        return torch.stack([g[:min_len] for g in gs]).numpy()
    
    def stop(self):
        self._is_running = False
    
    def pause(self):
        self._is_paused = True
    
    def resume(self):
        self._is_paused = False


class MainWindow(QMainWindow):
    def __init__(self, config: VisualizerConfig):
        super().__init__()
        self._config = config
        self._dirac_config = DiracConfig()
        self._dirac_config.DEVICE = 'cpu'
        
        self._worker = None
        self._worker_thread = None
        self._csv_logger = CSVLogger(config)
        
        self._setup_ui()
    
    def _setup_ui(self):
        self.setWindowTitle("Dirac Crystal - Latent Space Visualizer")
        self.setMinimumSize(1200, 800)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        left = QWidget()
        left_layout = QVBoxLayout(left)
        
        ctrl = QGroupBox("Training Controls")
        ctrl_layout = QVBoxLayout(ctrl)
        
        params = QGridLayout()
        params.addWidget(QLabel("Epochs:"), 0, 0)
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 10000)
        self._epochs_spin.setValue(self._config.TRAINING_EPOCHS)
        params.addWidget(self._epochs_spin, 0, 1)
        
        params.addWidget(QLabel("Batch Size:"), 1, 0)
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 256)
        self._batch_spin.setValue(self._config.TRAINING_BATCH_SIZE)
        params.addWidget(self._batch_spin, 1, 1)
        
        params.addWidget(QLabel("Seed:"), 2, 0)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 99999)
        self._seed_spin.setValue(self._config.TRAINING_SEED)
        params.addWidget(self._seed_spin, 2, 1)
        
        params.addWidget(QLabel("Learning Rate:"), 3, 0)
        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(1e-6, 1.0)
        self._lr_spin.setDecimals(6)
        self._lr_spin.setValue(self._config.TRAINING_LR)
        params.addWidget(self._lr_spin, 3, 1)
        
        ctrl_layout.addLayout(params)
        
        btns = QHBoxLayout()
        self._start_btn = QPushButton("Start")
        self._pause_btn = QPushButton("Pause")
        self._stop_btn = QPushButton("Stop")
        self._clear_btn = QPushButton("Clear")
        
        self._start_btn.clicked.connect(self._start)
        self._pause_btn.clicked.connect(self._pause)
        self._stop_btn.clicked.connect(self._stop)
        self._clear_btn.clicked.connect(self._clear)
        
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        
        for b in [self._start_btn, self._pause_btn, self._stop_btn, self._clear_btn]:
            btns.addWidget(b)
        
        ctrl_layout.addLayout(btns)
        left_layout.addWidget(ctrl)
        
        log_grp = QGroupBox("Log")
        log_layout = QVBoxLayout(log_grp)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(150)
        log_layout.addWidget(self._log)
        left_layout.addWidget(log_grp)
        
        metrics_grp = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout(metrics_grp)
        self._metrics_label = QLabel("Waiting...")
        self._metrics_label.setFont(QFont('Monospace', 9))
        self._metrics_label.setAlignment(Qt.AlignTop)
        self._metrics_label.setWordWrap(True)
        scroll = QScrollArea()
        scroll.setWidget(self._metrics_label)
        scroll.setWidgetResizable(True)
        metrics_layout.addWidget(scroll)
        left_layout.addWidget(metrics_grp)
        
        csv_grp = QGroupBox("CSV")
        csv_layout = QVBoxLayout(csv_grp)
        self._csv_label = QLabel(f"Path: {self._csv_logger.get_csv_path()}")
        self._csv_label.setWordWrap(True)
        csv_layout.addWidget(self._csv_label)
        left_layout.addWidget(csv_grp)
        
        left.setMaximumWidth(280)
        main_layout.addWidget(left)
        
        right = QSplitter(Qt.Vertical)
        
        top = QSplitter(Qt.Horizontal)
        self._latent_widget = LatentSpaceWidget(self._config)
        self._metrics_widget = MetricsWidget(self._config)
        top.addWidget(self._latent_widget)
        top.addWidget(self._metrics_widget)
        
        self._texture_widget = WeightTextureWidget(self._config)
        
        tabs = QTabWidget()
        tabs.addTab(self._texture_widget, "Textures")
        
        right.addWidget(top)
        right.addWidget(tabs)
        right.setSizes([500, 200])
        
        main_layout.addWidget(right, stretch=1)
        
        self._status = QLabel("Ready")
        self.statusBar().addWidget(self._status)
    
    def _log_msg(self, msg: str):
        self._log.appendPlainText(msg)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())
    
    def _start(self):
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker.resume()
            self._status.setText("Resumed")
            return
        
        self._config.TRAINING_EPOCHS = self._epochs_spin.value()
        self._config.TRAINING_BATCH_SIZE = self._batch_spin.value()
        self._config.TRAINING_SEED = self._seed_spin.value()
        self._config.TRAINING_LR = self._lr_spin.value()
        
        self._worker = TrainingWorker(self._config, self._dirac_config)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        
        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(lambda e: QMessageBox.warning(self, "Error", e))
        self._worker.log_message.connect(self._log_msg)
        
        self._worker_thread.start()
        
        self._start_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._status.setText("Training...")
    
    def _pause(self):
        if self._worker:
            self._worker.pause()
            self._status.setText("Paused")
    
    def _stop(self):
        if self._worker:
            self._worker.stop()
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait()
        self._on_finished()
    
    def _clear(self):
        self._latent_widget.clear()
        self._metrics_widget.clear()
        self._texture_widget.clear()
        self._metrics_label.setText("Cleared")
        self._status.setText("Cleared")
    
    def _on_progress(self, metrics: Dict[str, Any]):
        weights = metrics.pop('_weights', None)
        grads = metrics.pop('_gradients', None)
        
        self._csv_logger.log(metrics)
        
        if weights is not None:
            self._latent_widget.update_data(weights, metrics.get('delta', 0.5))
            self._texture_widget.update_data(weights, grads if grads is not None else weights)
        
        self._metrics_widget.update_data(metrics)
        
        delta = metrics.get('delta', 1.0)
        kappa = metrics.get('kappa', float('inf'))
        phase = metrics.get('topo_phase_state', 0.0)
        
        k_str = f"{kappa:.2e}" if kappa != float('inf') else "inf"
        
        self._status.setText(f"Epoch {metrics.get('epoch', 0)} | delta={delta:.6f} | kappa={k_str}")
        
        txt = [
            f"Epoch: {metrics.get('epoch', 0)}",
            f"Loss: {metrics.get('loss', 0):.6f}",
            f"Val Loss: {metrics.get('val_loss', 0):.6f}",
            f"Val Acc: {metrics.get('val_acc', 0):.4f}",
            "",
            "--- Crystal ---",
            f"Delta: {metrics.get('delta', 1):.6f}",
            f"Kappa: {metrics.get('kappa', float('inf')):.2e}",
            f"Alpha: {metrics.get('alpha', 0):.4f}",
            "",
            "--- Topo ---",
            f"Phase: {metrics.get('topo_phase_state', 0):.4f}",
            f"Loc: {metrics.get('topo_localization', 0):.4f}",
        ]
        self._metrics_label.setText("\n".join(txt))
    
    def _on_finished(self):
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._status.setText("Finished")
        self._csv_logger.close()
        
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait()
    
    def closeEvent(self, e):
        self._stop()
        self._csv_logger.close()
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    app.setPalette(palette)
    
    config = VisualizerConfig()
    win = MainWindow(config)
    win.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
