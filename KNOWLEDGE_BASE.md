# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis.

**Total Files Parsed:** 10 | **Total Symbols Extracted:** 532 | **Total Imports:** 153

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray: 5 5,color:#aaa;
    dirac_crystallography_suite_py["dirac_crystallography_suite.py (py)"]
    class dirac_crystallography_suite_py mod;
    dirac_crystallography_suite_py_CrystallographySuiteConfig["CrystallographySuiteConfig"]
    class dirac_crystallography_suite_py_CrystallographySuiteConfig cls;
    dirac_crystallography_suite_py --> dirac_crystallography_suite_py_CrystallographySuiteConfig
    dirac_crystallography_suite_py_LoggerFactory["LoggerFactory"]
    class dirac_crystallography_suite_py_LoggerFactory cls;
    dirac_crystallography_suite_py --> dirac_crystallography_suite_py_LoggerFactory
    dirac_crystallography_suite_py_IMetricCalculator["IMetricCalculator"]
    class dirac_crystallography_suite_py_IMetricCalculator cls;
    dirac_crystallography_suite_py --> dirac_crystallography_suite_py_IMetricCalculator
    dirac_crystallography_suite_py_IPhaseDetector["IPhaseDetector"]
    class dirac_crystallography_suite_py_IPhaseDetector cls;
    dirac_crystallography_suite_py --> dirac_crystallography_suite_py_IPhaseDetector
    dirac_crystallography_suite_py_GammaMatrices["GammaMatrices"]
    class dirac_crystallography_suite_py_GammaMatrices cls;
    dirac_crystallography_suite_py --> dirac_crystallography_suite_py_GammaMatrices
    weight_space_lidar_py["weight_space_lidar.py (py)"]
    class weight_space_lidar_py mod;
    weight_space_lidar_py_WeightSpaceLiDARConfig["WeightSpaceLiDARConfig"]
    class weight_space_lidar_py_WeightSpaceLiDARConfig cls;
    weight_space_lidar_py --> weight_space_lidar_py_WeightSpaceLiDARConfig
    weight_space_lidar_py_ILogger["ILogger"]
    class weight_space_lidar_py_ILogger cls;
    weight_space_lidar_py --> weight_space_lidar_py_ILogger
    weight_space_lidar_py_LoggerFactory["LoggerFactory"]
    class weight_space_lidar_py_LoggerFactory cls;
    weight_space_lidar_py --> weight_space_lidar_py_LoggerFactory
    weight_space_lidar_py_IWeightExtractor["IWeightExtractor"]
    class weight_space_lidar_py_IWeightExtractor cls;
    weight_space_lidar_py --> weight_space_lidar_py_IWeightExtractor
    weight_space_lidar_py_IDimensionalityReducer["IDimensionalityReducer"]
    class weight_space_lidar_py_IDimensionalityReducer cls;
    weight_space_lidar_py --> weight_space_lidar_py_IDimensionalityReducer
    latent_space_visualizer_py["latent_space_visualizer.py (py)"]
    class latent_space_visualizer_py mod;
    latent_space_visualizer_py_VisualizerConfig["VisualizerConfig"]
    class latent_space_visualizer_py_VisualizerConfig cls;
    latent_space_visualizer_py --> latent_space_visualizer_py_VisualizerConfig
    latent_space_visualizer_py_CSVLogger["CSVLogger"]
    class latent_space_visualizer_py_CSVLogger cls;
    latent_space_visualizer_py --> latent_space_visualizer_py_CSVLogger
    latent_space_visualizer_py_LatentSpaceWidget["LatentSpaceWidget"]
    class latent_space_visualizer_py_LatentSpaceWidget cls;
    latent_space_visualizer_py --> latent_space_visualizer_py_LatentSpaceWidget
    latent_space_visualizer_py_MetricsWidget["MetricsWidget"]
    class latent_space_visualizer_py_MetricsWidget cls;
    latent_space_visualizer_py --> latent_space_visualizer_py_MetricsWidget
    latent_space_visualizer_py_WeightTextureWidget["WeightTextureWidget"]
    class latent_space_visualizer_py_WeightTextureWidget cls;
    latent_space_visualizer_py --> latent_space_visualizer_py_WeightTextureWidget
    relativistic_hydrogen_py["relativistic_hydrogen.py (py)"]
    class relativistic_hydrogen_py mod;
    relativistic_hydrogen_py_Config["Config"]
    class relativistic_hydrogen_py_Config cls;
    relativistic_hydrogen_py --> relativistic_hydrogen_py_Config
    relativistic_hydrogen_py_LoggerFactory["LoggerFactory"]
    class relativistic_hydrogen_py_LoggerFactory cls;
    relativistic_hydrogen_py --> relativistic_hydrogen_py_LoggerFactory
    relativistic_hydrogen_py_GammaMatrices["GammaMatrices"]
    class relativistic_hydrogen_py_GammaMatrices cls;
    relativistic_hydrogen_py --> relativistic_hydrogen_py_GammaMatrices
    relativistic_hydrogen_py_DiracHamiltonianOperator["DiracHamiltonianOperator"]
    class relativistic_hydrogen_py_DiracHamiltonianOperator cls;
    relativistic_hydrogen_py --> relativistic_hydrogen_py_DiracHamiltonianOperator
    relativistic_hydrogen_py_SpectralLayer["SpectralLayer"]
    class relativistic_hydrogen_py_SpectralLayer cls;
    relativistic_hydrogen_py --> relativistic_hydrogen_py_SpectralLayer
    dirac_crystal2_py["dirac_crystal2.py (py)"]
    class dirac_crystal2_py mod;
    dirac_crystal2_py_Config["Config"]
    class dirac_crystal2_py_Config cls;
    dirac_crystal2_py --> dirac_crystal2_py_Config
    dirac_crystal2_py_IPhaseDetector["IPhaseDetector"]
    class dirac_crystal2_py_IPhaseDetector cls;
    dirac_crystal2_py --> dirac_crystal2_py_IPhaseDetector
    dirac_crystal2_py_IMetricCalculator["IMetricCalculator"]
    class dirac_crystal2_py_IMetricCalculator cls;
    dirac_crystal2_py --> dirac_crystal2_py_IMetricCalculator
    dirac_crystal2_py_SeedManager["SeedManager"]
    class dirac_crystal2_py_SeedManager cls;
    dirac_crystal2_py --> dirac_crystal2_py_SeedManager
    dirac_crystal2_py_LoggerFactory["LoggerFactory"]
    class dirac_crystal2_py_LoggerFactory cls;
    dirac_crystal2_py --> dirac_crystal2_py_LoggerFactory
    weight_3d_standard_py["weight_3d_standard.py (py)"]
    class weight_3d_standard_py mod;
    weight_3d_standard_py_StandardWeightVisualizer["StandardWeightVisualizer"]
    class weight_3d_standard_py_StandardWeightVisualizer cls;
    weight_3d_standard_py --> weight_3d_standard_py_StandardWeightVisualizer
    weight_3d_standard_py_generate_standard_html["generate_standard_html"]
    class weight_3d_standard_py_generate_standard_html fn;
    weight_3d_standard_py --> weight_3d_standard_py_generate_standard_html
    weight_3d_standard_py_generate_continuous_html["generate_continuous_html"]
    class weight_3d_standard_py_generate_continuous_html fn;
    weight_3d_standard_py --> weight_3d_standard_py_generate_continuous_html
    weight_3d_standard_py_main["main"]
    class weight_3d_standard_py_main fn;
    weight_3d_standard_py --> weight_3d_standard_py_main
    weight_3d_standard_py___init__["__init__"]
    class weight_3d_standard_py___init__ fn;
    weight_3d_standard_py --> weight_3d_standard_py___init__
    lidar_interactive_viewer_py["lidar_interactive_viewer.py (py)"]
    class lidar_interactive_viewer_py mod;
    lidar_interactive_viewer_py_load_csv_point_cloud["load_csv_point_cloud"]
    class lidar_interactive_viewer_py_load_csv_point_cloud fn;
    lidar_interactive_viewer_py --> lidar_interactive_viewer_py_load_csv_point_cloud
    lidar_interactive_viewer_py_generate_interactive_html["generate_interactive_html"]
    class lidar_interactive_viewer_py_generate_interactive_html fn;
    lidar_interactive_viewer_py --> lidar_interactive_viewer_py_generate_interactive_html
    lidar_interactive_viewer_py_generate_plotly_html["generate_plotly_html"]
    class lidar_interactive_viewer_py_generate_plotly_html fn;
    lidar_interactive_viewer_py --> lidar_interactive_viewer_py_generate_plotly_html
    lidar_interactive_viewer_py_main["main"]
    class lidar_interactive_viewer_py_main fn;
    lidar_interactive_viewer_py --> lidar_interactive_viewer_py_main
    visualize_lidar_csv2_py["visualize_lidar_csv2.py (py)"]
    class visualize_lidar_csv2_py mod;
    visualize_lidar_csv2_py_visualize_csv["visualize_csv"]
    class visualize_lidar_csv2_py_visualize_csv fn;
    visualize_lidar_csv2_py --> visualize_lidar_csv2_py_visualize_csv
    visualize_lidar_csv2_py_main["main"]
    class visualize_lidar_csv2_py_main fn;
    visualize_lidar_csv2_py --> visualize_lidar_csv2_py_main
    app_py["app.py (py)"]
    class app_py mod;
    install_sh["install.sh (sh)"]
    class install_sh mod;
    ext_argparse["argparse"]
    class ext_argparse ext;
    dirac_crystal2_py -.->|imports| ext_argparse
    ext_torch["torch"]
    class ext_torch ext;
    dirac_crystal2_py -.->|imports| ext_torch
    ext_torch_nn["torch.nn"]
    class ext_torch_nn ext;
    dirac_crystal2_py -.->|imports| ext_torch_nn
    ext_torch_nn_functional["torch.nn.functional"]
    class ext_torch_nn_functional ext;
    dirac_crystal2_py -.->|imports| ext_torch_nn_functional
    ext_torch_optim["torch.optim"]
    class ext_torch_optim ext;
    dirac_crystal2_py -.->|imports| ext_torch_optim
    ext_torch_utils_data["torch.utils.data"]
    class ext_torch_utils_data ext;
    dirac_crystal2_py -.->|imports| ext_torch_utils_data
    ext_numpy["numpy"]
    class ext_numpy ext;
    dirac_crystal2_py -.->|imports| ext_numpy
    ext_os["os"]
    class ext_os ext;
    dirac_crystal2_py -.->|imports| ext_os
    ext_time["time"]
    class ext_time ext;
    dirac_crystal2_py -.->|imports| ext_time
    ext_json["json"]
    class ext_json ext;
    dirac_crystal2_py -.->|imports| ext_json
    ext_datetime["datetime"]
    class ext_datetime ext;
    dirac_crystal2_py -.->|imports| ext_datetime
    ext_typing["typing"]
    class ext_typing ext;
    dirac_crystal2_py -.->|imports| ext_typing
    ext_abc["abc"]
    class ext_abc ext;
    dirac_crystal2_py -.->|imports| ext_abc
    ext_dataclasses["dataclasses"]
    class ext_dataclasses ext;
    dirac_crystal2_py -.->|imports| ext_dataclasses
    ext_collections["collections"]
    class ext_collections ext;
    dirac_crystal2_py -.->|imports| ext_collections
    ext_logging["logging"]
    class ext_logging ext;
    dirac_crystal2_py -.->|imports| ext_logging
    ext_math["math"]
    class ext_math ext;
    dirac_crystal2_py -.->|imports| ext_math
    ext_copy["copy"]
    class ext_copy ext;
    dirac_crystal2_py -.->|imports| ext_copy
    ext_warnings["warnings"]
    class ext_warnings ext;
    dirac_crystal2_py -.->|imports| ext_warnings
    dirac_crystallography_suite_py -.->|imports| ext_argparse
    dirac_crystallography_suite_py -.->|imports| ext_copy
    ext_glob["glob"]
    class ext_glob ext;
    dirac_crystallography_suite_py -.->|imports| ext_glob
    dirac_crystallography_suite_py -.->|imports| ext_json
    dirac_crystallography_suite_py -.->|imports| ext_logging
    dirac_crystallography_suite_py -.->|imports| ext_math
    dirac_crystallography_suite_py -.->|imports| ext_os
    ext_re["re"]
    class ext_re ext;
    dirac_crystallography_suite_py -.->|imports| ext_re
    dirac_crystallography_suite_py -.->|imports| ext_time
    dirac_crystallography_suite_py -.->|imports| ext_warnings
    dirac_crystallography_suite_py -.->|imports| ext_abc
    dirac_crystallography_suite_py -.->|imports| ext_collections
    dirac_crystallography_suite_py -.->|imports| ext_dataclasses
    dirac_crystallography_suite_py -.->|imports| ext_datetime
    ext_pathlib["pathlib"]
    class ext_pathlib ext;
    dirac_crystallography_suite_py -.->|imports| ext_pathlib
    dirac_crystallography_suite_py -.->|imports| ext_typing
    ext_matplotlib["matplotlib"]
    class ext_matplotlib ext;
    dirac_crystallography_suite_py -.->|imports| ext_matplotlib
    ext_matplotlib_pyplot["matplotlib.pyplot"]
    class ext_matplotlib_pyplot ext;
    dirac_crystallography_suite_py -.->|imports| ext_matplotlib_pyplot
    ext_matplotlib_gridspec["matplotlib.gridspec"]
    class ext_matplotlib_gridspec ext;
    dirac_crystallography_suite_py -.->|imports| ext_matplotlib_gridspec
    ext_seaborn["seaborn"]
    class ext_seaborn ext;
    dirac_crystallography_suite_py -.->|imports| ext_seaborn
    dirac_crystallography_suite_py -.->|imports| ext_numpy
    dirac_crystallography_suite_py -.->|imports| ext_torch
    dirac_crystallography_suite_py -.->|imports| ext_torch_nn
    dirac_crystallography_suite_py -.->|imports| ext_torch_nn_functional
    dirac_crystallography_suite_py -.->|imports| ext_torch_optim
    dirac_crystallography_suite_py -.->|imports| ext_torch_utils_data
    ext_scipy["scipy"]
    class ext_scipy ext;
    dirac_crystallography_suite_py -.->|imports| ext_scipy
    ext_scipy_stats["scipy.stats"]
    class ext_scipy_stats ext;
    dirac_crystallography_suite_py -.->|imports| ext_scipy_stats
    ext_scipy_linalg["scipy.linalg"]
    class ext_scipy_linalg ext;
    dirac_crystallography_suite_py -.->|imports| ext_scipy_linalg
    ext_scipy_optimize["scipy.optimize"]
    class ext_scipy_optimize ext;
    dirac_crystallography_suite_py -.->|imports| ext_scipy_optimize
    ext_scipy_sparse["scipy.sparse"]
    class ext_scipy_sparse ext;
    dirac_crystallography_suite_py -.->|imports| ext_scipy_sparse
    ext_scipy_sparse_linalg["scipy.sparse.linalg"]
    class ext_scipy_sparse_linalg ext;
    dirac_crystallography_suite_py -.->|imports| ext_scipy_sparse_linalg
    ext_sklearn_decomposition["sklearn.decomposition"]
    class ext_sklearn_decomposition ext;
    dirac_crystallography_suite_py -.->|imports| ext_sklearn_decomposition
    ext_traceback["traceback"]
    class ext_traceback ext;
    dirac_crystallography_suite_py -.->|imports| ext_traceback
    ext_sys["sys"]
    class ext_sys ext;
    latent_space_visualizer_py -.->|imports| ext_sys
    latent_space_visualizer_py -.->|imports| ext_os
    ext_csv["csv"]
    class ext_csv ext;
    latent_space_visualizer_py -.->|imports| ext_csv
    latent_space_visualizer_py -.->|imports| ext_time
    ext_threading["threading"]
    class ext_threading ext;
    latent_space_visualizer_py -.->|imports| ext_threading
    latent_space_visualizer_py -.->|imports| ext_pathlib
    latent_space_visualizer_py -.->|imports| ext_dataclasses
    latent_space_visualizer_py -.->|imports| ext_typing
    latent_space_visualizer_py -.->|imports| ext_collections
    latent_space_visualizer_py -.->|imports| ext_datetime
    latent_space_visualizer_py -.->|imports| ext_numpy
    latent_space_visualizer_py -.->|imports| ext_torch
    latent_space_visualizer_py -.->|imports| ext_torch_nn
    latent_space_visualizer_py -.->|imports| ext_torch_utils_data
    latent_space_visualizer_py -.->|imports| ext_sklearn_decomposition
    ext_sklearn_preprocessing["sklearn.preprocessing"]
    class ext_sklearn_preprocessing ext;
    latent_space_visualizer_py -.->|imports| ext_sklearn_preprocessing
    ext_PyQt5_QtWidgets["PyQt5.QtWidgets"]
    class ext_PyQt5_QtWidgets ext;
    latent_space_visualizer_py -.->|imports| ext_PyQt5_QtWidgets
    ext_PyQt5_QtCore["PyQt5.QtCore"]
    class ext_PyQt5_QtCore ext;
    latent_space_visualizer_py -.->|imports| ext_PyQt5_QtCore
    ext_PyQt5_QtGui["PyQt5.QtGui"]
    class ext_PyQt5_QtGui ext;
    latent_space_visualizer_py -.->|imports| ext_PyQt5_QtGui
    latent_space_visualizer_py -.->|imports| ext_matplotlib
    ext_matplotlib_backends_backend_qt5agg["matplotlib.backends.backend_qt5agg"]
    class ext_matplotlib_backends_backend_qt5agg ext;
    latent_space_visualizer_py -.->|imports| ext_matplotlib_backends_backend_qt5agg
    ext_matplotlib_figure["matplotlib.figure"]
    class ext_matplotlib_figure ext;
    latent_space_visualizer_py -.->|imports| ext_matplotlib_figure
    ext_matplotlib_colors["matplotlib.colors"]
    class ext_matplotlib_colors ext;
    latent_space_visualizer_py -.->|imports| ext_matplotlib_colors
    latent_space_visualizer_py -.->|imports| ext_matplotlib_pyplot
    ext_mpl_toolkits_mplot3d["mpl_toolkits.mplot3d"]
    class ext_mpl_toolkits_mplot3d ext;
    latent_space_visualizer_py -.->|imports| ext_mpl_toolkits_mplot3d
    ext_dirac_crystal2["dirac_crystal2"]
    class ext_dirac_crystal2 ext;
    latent_space_visualizer_py -.->|imports| ext_dirac_crystal2
    latent_space_visualizer_py -.->|imports| ext_traceback
    lidar_interactive_viewer_py -.->|imports| ext_argparse
    lidar_interactive_viewer_py -.->|imports| ext_csv
    lidar_interactive_viewer_py -.->|imports| ext_json
    lidar_interactive_viewer_py -.->|imports| ext_sys
    lidar_interactive_viewer_py -.->|imports| ext_pathlib
    lidar_interactive_viewer_py -.->|imports| ext_typing
    lidar_interactive_viewer_py -.->|imports| ext_numpy
    relativistic_hydrogen_py -.->|imports| ext_numpy
    ext_scipy_special["scipy.special"]
    class ext_scipy_special ext;
    relativistic_hydrogen_py -.->|imports| ext_scipy_special
    relativistic_hydrogen_py -.->|imports| ext_scipy
    relativistic_hydrogen_py -.->|imports| ext_matplotlib_pyplot
    relativistic_hydrogen_py -.->|imports| ext_matplotlib
    relativistic_hydrogen_py -.->|imports| ext_matplotlib_colors
    relativistic_hydrogen_py -.->|imports| ext_torch
    relativistic_hydrogen_py -.->|imports| ext_torch_nn
    relativistic_hydrogen_py -.->|imports| ext_torch_nn_functional
    relativistic_hydrogen_py -.->|imports| ext_os
    relativistic_hydrogen_py -.->|imports| ext_sys
    relativistic_hydrogen_py -.->|imports| ext_warnings
    relativistic_hydrogen_py -.->|imports| ext_json
    relativistic_hydrogen_py -.->|imports| ext_typing
    relativistic_hydrogen_py -.->|imports| ext_dataclasses
    relativistic_hydrogen_py -.->|imports| ext_abc
    relativistic_hydrogen_py -.->|imports| ext_logging
    relativistic_hydrogen_py -.->|imports| ext_math
    relativistic_hydrogen_py -.->|imports| ext_glob
    relativistic_hydrogen_py -.->|imports| ext_traceback
    visualize_lidar_csv2_py -.->|imports| ext_argparse
    visualize_lidar_csv2_py -.->|imports| ext_sys
    visualize_lidar_csv2_py -.->|imports| ext_csv
    visualize_lidar_csv2_py -.->|imports| ext_numpy
    visualize_lidar_csv2_py -.->|imports| ext_pathlib
    visualize_lidar_csv2_py -.->|imports| ext_matplotlib_pyplot
    visualize_lidar_csv2_py -.->|imports| ext_mpl_toolkits_mplot3d
    weight_3d_standard_py -.->|imports| ext_argparse
    weight_3d_standard_py -.->|imports| ext_csv
    weight_3d_standard_py -.->|imports| ext_json
    weight_3d_standard_py -.->|imports| ext_sys
    weight_3d_standard_py -.->|imports| ext_pathlib
    weight_3d_standard_py -.->|imports| ext_typing
    weight_3d_standard_py -.->|imports| ext_numpy
    weight_3d_standard_py -.->|imports| ext_torch
    weight_3d_standard_py -.->|imports| ext_sklearn_decomposition
    ext_sklearn_manifold["sklearn.manifold"]
    class ext_sklearn_manifold ext;
    weight_3d_standard_py -.->|imports| ext_sklearn_manifold
    weight_3d_standard_py -.->|imports| ext_sklearn_preprocessing
    ext___future__["__future__"]
    class ext___future__ ext;
    weight_space_lidar_py -.->|imports| ext___future__
    weight_space_lidar_py -.->|imports| ext_argparse
    weight_space_lidar_py -.->|imports| ext_json
    weight_space_lidar_py -.->|imports| ext_logging
    weight_space_lidar_py -.->|imports| ext_math
    weight_space_lidar_py -.->|imports| ext_os
    weight_space_lidar_py -.->|imports| ext_sys
    weight_space_lidar_py -.->|imports| ext_warnings
    weight_space_lidar_py -.->|imports| ext_abc
    weight_space_lidar_py -.->|imports| ext_dataclasses
    weight_space_lidar_py -.->|imports| ext_datetime
    weight_space_lidar_py -.->|imports| ext_pathlib
    weight_space_lidar_py -.->|imports| ext_typing
    weight_space_lidar_py -.->|imports| ext_numpy
    weight_space_lidar_py -.->|imports| ext_scipy
    weight_space_lidar_py -.->|imports| ext_scipy_linalg
    ext_scipy_spatial_distance["scipy.spatial.distance"]
    class ext_scipy_spatial_distance ext;
    weight_space_lidar_py -.->|imports| ext_scipy_spatial_distance
    weight_space_lidar_py -.->|imports| ext_torch
    weight_space_lidar_py -.->|imports| ext_torch_nn
    weight_space_lidar_py -.->|imports| ext_torch
    weight_space_lidar_py -.->|imports| ext_sklearn_decomposition
    weight_space_lidar_py -.->|imports| ext_sklearn_manifold
    weight_space_lidar_py -.->|imports| ext_re
    weight_space_lidar_py -.->|imports| ext_csv
    weight_space_lidar_py -.->|imports| ext_matplotlib_pyplot
    weight_space_lidar_py -.->|imports| ext_mpl_toolkits_mplot3d
    weight_space_lidar_py -.->|imports| ext_matplotlib_pyplot
    ext_laspy["laspy"]
    class ext_laspy ext;
    weight_space_lidar_py -.->|imports| ext_laspy
```

---

## Architecture Reference

### PY (9 files)

#### `app.py`
**Path:** `app.py`

*No symbols extracted*

#### `dirac_crystal2.py`
**Path:** `dirac_crystal2.py`

**Classs:**
- `Config` (line 52)
- `IPhaseDetector` (line 250)
- `IMetricCalculator` (line 256)
- `SeedManager` (line 262)
- `LoggerFactory` (line 274)
- `GammaMatrices` (line 289) - *Dirac gamma matrices in various representations.
Default: Dirac (standard) representation.*
- `DiracHamiltonianOperator` (line 383) - *  Dirac Hamiltonian operator for 4-component spinors.
  H_Dirac = c * alpha . p + beta * m * c^2
where alpha_i = gamma0 @ gammai and beta = gamma0.
  In natural units (c=1): H = alpha . p + beta * m*
- `SpectralLayer` (line 481)
- `DiracSpectralNetwork` (line 519) - *Neural network for learning Dirac equation dynamics.
Handles 4-component spinors with real and imaginary parts (8 channels total).*
- `HamiltonianBackbone` (line 558)
- `HamiltonianInferenceEngine` (line 585)
- `DiracPotentialGenerator` (line 639) - *Generate potentials for the Dirac equation.
In relativistic QM, the potential couples differently to particle/antiparticle components.*
- `DiracDataset` (line 716) - *Dataset for Dirac equation evolution.
Generates 4-component spinors and their time-evolved targets.*
- `FullFourierAnalyzer` (line 845)
- `FourierMassCenterAnalyzer` (line 1021)
- `TopologicalPhaseDetector` (line 1092)
- `SpectralFieldExtractor` (line 1161)
- `TopologicalCrystallizationLoss` (line 1183)
- `CrystallizationPressureApplicator` (line 1221)
- `TopologicalMetricsCalculator` (line 1236)
- `LocalComplexityAnalyzer` (line 1299)
- `SuperpositionAnalyzer` (line 1316)
- `CrystallographyMetricsCalculator` (line 1337)
- `ThermodynamicMetricsCalculator` (line 1547)
- `SpectralGeometryCalculator` (line 1626)
- `RicciCurvatureCalculator` (line 1679)
- `PerelmanRicciFlow` (line 1723)
- `SpectroscopyMetricsCalculator` (line 1978)
- `LambdaPressureScheduler` (line 2015)
- `AdaptiveLambdaScheduler` (line 2055)
- `QuadruplePrecisionLambdaScheduler` (line 2076)
- `AnnealingScheduler` (line 2116)
- `TopologicalAnnealingScheduler` (line 2146)
- `TrainingMetricsMonitor` (line 2165)
- `CheckpointManager` (line 2318)
- `Phase5CheckpointManager` (line 2376)
- `GlassStateDetector` (line 2481)
- `WeightIntegrityChecker` (line 2537)
- `TrainingEngine` (line 2569)
- `BatchSizeProspector` (line 2764)
- `SeedMiner` (line 2835)
- `FullTrainingOrchestrator` (line 2976)
- `RefinementOrchestrator` (line 3109)
- `Phase5Orchestrator` (line 3238)

**Functions:**
- `main` (line 3626)
- `detect` (line 252)
- `compute` (line 258)
- `set_seed` (line 264)
- `create_logger` (line 276)
- `__init__` (line 294)
- `_init_matrices` (line 299)
- `to` (line 377)
- `__init__` (line 390)
- `_precompute_operators` (line 398)
- `apply_dirac_hamiltonian` (line 414) - *Apply Dirac Hamiltonian to 4-component spinor.
H_psi = c * (alpha_x * p_x + alpha_y * p_y) @ psi + beta * m * c^2 * psi

Input shape: [4, H, W] or [batch, 4, H, W]
Output shape: same as input*
- `time_evolution` (line 457) - *Time evolution of Dirac spinor using split-step method.
psi(t+dt) = exp(-i * H * dt) * psi(t)*
- `__init__` (line 482)
- `forward` (line 493)
- `__init__` (line 524)
- `forward` (line 547)
- `__init__` (line 559)
- `forward` (line 574)
- `__init__` (line 586)
- `_try_load_backbone` (line 593)
- `apply_hamiltonian` (line 627) - *Apply Dirac Hamiltonian to 4-component spinor.
Always uses analytical operator since backbone is not compatible
with 4-component Dirac spinors.*
- `time_evolve` (line 635)
- `__init__` (line 644)
- `scalar_potential` (line 648) - *Scalar potential (couples equally to all components).
V_s * psi (same coupling for particle and antiparticle).*
- `vector_potential` (line 660) - *Vector potential (time-component of 4-vector).
V_v * gamma0 * psi (couples with opposite sign to particle/antiparticle).*
- `magnetic_potential_2d` (line 672) - *Magnetic potential (spatial components of 4-vector).
A * alpha * psi (couples to spin).*
- `periodic_lattice_potential` (line 686)
- `generate_mixed_potential` (line 692)
- `__init__` (line 721)
- `_generate_initial_spinor` (line 766) - *Generate an initial Dirac spinor (4-component).
The spinor is constructed to be a superposition of positive energy states.*
- `_time_evolve_spinor` (line 798) - *Time evolve a Dirac spinor under the influence of potentials.*
- `_spinor_to_real_imag` (line 824) - *Convert 4-component complex spinor to 8-channel real tensor.
Channels: [Re(psi0), Im(psi0), Re(psi1), Im(psi1), ...]*
- `__len__` (line 835)
- `__getitem__` (line 838)
- `get_validation_batch` (line 841)
- `__init__` (line 846)
- `compute_full_spectrum` (line 854)
- `detect_bragg_peaks` (line 926)
- `compute_resonance_metrics` (line 980)
- `__init__` (line 1022)
- `compute_mass_center` (line 1030)
- `__init__` (line 1093)
- `detect` (line 1100)
- `extract` (line 1163)
- `__init__` (line 1184)
- `forward` (line 1189)
- `__init__` (line 1222)
- `apply` (line 1226)
- `__init__` (line 1237)
- `compute` (line 1244)
- `apply_crystallization_pressure` (line 1277)
- `_empty_metrics` (line 1283)
- `compute_local_complexity` (line 1301)
- `compute_superposition` (line 1318)
- `__init__` (line 1338)
- `compute` (line 1342)
- `compute_kappa` (line 1347)
- `compute_discretization_margin` (line 1400)
- `compute_alpha_purity` (line 1408)
- `compute_kappa_quantum` (line 1414)
- `compute_poynting_vector` (line 1435)
- `compute_hbar_effective` (line 1491)
- `compute_all_metrics` (line 1500)
- `__init__` (line 1548)
- `compute` (line 1551)
- `compute_effective_temperature` (line 1577)
- `compute_specific_heat` (line 1601)
- `compute_gibbs_free_energy` (line 1615)
- `compute_critical_temperature` (line 1622)
- `__init__` (line 1627)
- `compute` (line 1630)
- `_compute_level_spacing_ratio` (line 1667)
- `__init__` (line 1680)
- `compute` (line 1683)
- `_compute_ricci_scalar` (line 1702)
- `_estimate_sectional_curvatures` (line 1710)
- `__init__` (line 1724)
- `compute_ricci_scalar_fast` (line 1732)
- `compute_local_curvature` (line 1775)
- `compute_anisotropy` (line 1786)
- `compute_ricci_regularization_loss` (line 1820)
- `apply_ricci_flow_step` (line 1845)
- `perform_perelman_surgery` (line 1887)
- `compute_adaptive_lr_factor` (line 1942)
- `get_flow_metrics` (line 1962)
- `__init__` (line 1979)
- `compute` (line 1982)
- `compute_weight_diffraction` (line 1986)
- `_compute_spectral_entropy` (line 2006)
- `__init__` (line 2016)
- `current_lambda` (line 2025)
- `step` (line 2028)
- `compute_regularization_loss` (line 2037)
- `set_lambda` (line 2051)
- `__init__` (line 2056)
- `step_adaptive` (line 2061)
- `__init__` (line 2077)
- `current_lambda` (line 2086)
- `step` (line 2089)
- `compute_regularization_loss` (line 2098)
- `set_lambda` (line 2112)
- `__init__` (line 2117)
- `temperature` (line 2125)
- `step` (line 2128)
- `accept_perturbation` (line 2134)
- `should_restart` (line 2142)
- `__init__` (line 2147)
- `step_adaptive` (line 2151)
- `__init__` (line 2166)
- `update_metrics` (line 2196)
- `compute_delta_slope` (line 2207)
- `format_progress_bar` (line 2220)
- `__init__` (line 2319)
- `should_save_checkpoint` (line 2328)
- `save_checkpoint` (line 2333)
- `load_latest_checkpoint` (line 2369)
- `__init__` (line 2377)
- `_load_best_metrics` (line 2388)
- `should_save` (line 2407)
- `save_checkpoint` (line 2418)
- `load_checkpoint` (line 2462)
- `__init__` (line 2482)
- `should_stop` (line 2488)
- `is_crystal_formed` (line 2523)
- `check` (line 2539)
- `__init__` (line 2570)
- `compute_weight_metrics` (line 2583)
- `compute_norm_conservation_error` (line 2598)
- `train_single_epoch` (line 2611)
- `validate` (line 2666)
- `collect_all_metrics` (line 2679)
- `__init__` (line 2765)
- `prospect` (line 2770)
- `__init__` (line 2836)
- `mine` (line 2847)
- `__init__` (line 2977)
- `run_phase3_training` (line 2990)
- `__init__` (line 3110)
- `run_phase4_refinement` (line 3129)
- `__init__` (line 3239)
- `_detect_blocked_labyrinth` (line 3268)
- `_apply_flood_fill_pressure` (line 3280)
- `_inject_diffusion_energy` (line 3306)
- `_find_ballistic_trajectory` (line 3317)
- `_load_phase5_checkpoint` (line 3328)
- `_apply_perelman_surgery` (line 3357)
- `run_phase5_crystallization` (line 3384)
- `load_latest_checkpoint` (line 3761)
- `safe_compute` (line 1513)
- `safe_get` (line 2224)

#### `dirac_crystallography_suite.py`
**Path:** `dirac_crystallography_suite.py`

**Classs:**
- `CrystallographySuiteConfig` (line 55) - *Master configuration for the complete crystallography suite.*
- `LoggerFactory` (line 178) - *Factory for creating configured logger instances.*
- `IMetricCalculator` (line 196) - *Protocol for metric calculation strategies.*
- `IPhaseDetector` (line 204) - *Protocol for phase detection strategies.*
- `GammaMatrices` (line 212) - *Dirac gamma matrices in various representations.*
- `DiracHamiltonianOperator` (line 263) - *Dirac Hamiltonian operator for 4-component spinors.*
- `SpectralLayer` (line 328) - *Spectral convolution layer operating in Fourier space.*
- `DiracSpectralNetwork` (line 363) - *Neural network for learning Dirac equation dynamics.*
- `WeightIntegrityCalculator` (line 394) - *Calculator for weight integrity metrics.*
- `DiscretizationCalculator` (line 433) - *Calculator for discretization margin and alpha purity metrics.*
- `SpectralGeometryCalculator` (line 481) - *Calculator for spectral geometry metrics including MBL level spacing.*
- `RicciCurvatureCalculator` (line 536) - *Calculator for Ricci curvature estimation in weight space.*
- `BerryPhaseCalculator` (line 581) - *Calculates Berry phase from training checkpoint trajectory.*
- `ControlSystemAnalyzer` (line 689) - *Control theory analysis for neural network dynamics.*
- `ThermodynamicCalculator` (line 781) - *Calculator for thermodynamic potentials.*
- `FullFourierAnalyzer` (line 826) - *Complete Fourier analysis for spectral fields.*
- `FourierMassCenterAnalyzer` (line 882) - *Analyzer for center of mass in Fourier space.*
- `TopologicalPhaseDetector` (line 933) - *Detects topological phases from spectral field analysis.*
- `SpectralFieldExtractor` (line 971) - *Extracts spectral fields from neural network layers.*
- `TopologicalMetricsCalculator` (line 993) - *Calculates topological metrics from model spectral fields.*
- `GradientDynamicsCalculator` (line 1034) - *Calculator for gradient-based metrics.*
- `SchrodingerAnalyzer` (line 1127) - *Quantum mechanical analysis of network parameters.*
- `ComprehensiveVisualizer` (line 1189) - *Generates comprehensive visualizations for all metrics.*
- `CheckpointAnalyzer` (line 1451) - *Main analyzer that orchestrates all metric calculations.*
- `BatchProcessor` (line 1576) - *Processes multiple checkpoints in batch mode.*
- `DiracCrystallographySuite` (line 1723) - *Main entry point for the crystallography analysis suite.*

**Functions:**
- `main` (line 1806)
- `create_logger` (line 182)
- `compute` (line 199) - *Compute metrics for the given model.*
- `detect` (line 207) - *Detect phase from spectral field.*
- `__init__` (line 215)
- `_init_matrices` (line 221)
- `__init__` (line 266)
- `_precompute_operators` (line 274)
- `apply_dirac_hamiltonian` (line 290) - *Apply Dirac Hamiltonian to 4-component spinor.*
- `__init__` (line 331)
- `forward` (line 343)
- `__init__` (line 366)
- `forward` (line 383)
- `__init__` (line 397)
- `compute` (line 400)
- `__init__` (line 436)
- `compute` (line 439)
- `_compute_spectral_entropy` (line 466)
- `__init__` (line 484)
- `compute` (line 487)
- `_compute_level_spacing_ratio` (line 524)
- `__init__` (line 539)
- `compute` (line 542)
- `_compute_ricci_scalar` (line 559)
- `_estimate_sectional_curvatures` (line 568)
- `__init__` (line 584)
- `load_checkpoints` (line 588)
- `_extract_epoch` (line 607)
- `flatten_kernel_params` (line 611)
- `compute_berry_connection_discrete` (line 635)
- `calculate_berry_phase` (line 650)
- `__init__` (line 692)
- `extract_state_space` (line 695)
- `analyze_stability` (line 755)
- `compute` (line 772)
- `__init__` (line 784)
- `compute` (line 787)
- `_classify_phase` (line 812)
- `__init__` (line 829)
- `compute_full_spectrum` (line 837)
- `compute_resonance_metrics` (line 867)
- `__init__` (line 885)
- `compute_mass_center` (line 893)
- `__init__` (line 936)
- `detect` (line 943)
- `extract` (line 975)
- `__init__` (line 996)
- `compute` (line 1001)
- `_empty_metrics` (line 1022)
- `__init__` (line 1037)
- `compute` (line 1040)
- `__init__` (line 1130)
- `extract_compressed_wavefunction` (line 1135)
- `_compress_johnson_lindenstrauss` (line 1156)
- `compute` (line 1168)
- `__init__` (line 1192)
- `visualize_checkpoint_analysis` (line 1195)
- `_plot_weight_distribution` (line 1222)
- `_plot_spectral_analysis` (line 1233)
- `_plot_phase_diagram` (line 1247)
- `_plot_curvature_distribution` (line 1264)
- `_plot_level_spacing` (line 1278)
- `_plot_eigenvalue_spectrum` (line 1292)
- `_plot_thermodynamic_potentials` (line 1306)
- `_plot_topological_metrics` (line 1320)
- `_plot_berry_phase` (line 1334)
- `_plot_control_stability` (line 1354)
- `_plot_quantum_metrics` (line 1366)
- `_plot_summary_table` (line 1380)
- `_plot_layer_deltas` (line 1403)
- `_plot_resonance_metrics` (line 1415)
- `_plot_spectral_concentration` (line 1428)
- `_plot_health_score` (line 1439)
- `__init__` (line 1454)
- `analyze_checkpoint` (line 1470)
- `_compute_health_score` (line 1546)
- `__init__` (line 1579)
- `process_directory` (line 1585)
- `_generate_summary` (line 1631)
- `_generate_evolution_plots` (line 1682)
- `__init__` (line 1726)
- `run_analysis` (line 1732)
- `_generate_berry_phase_visualization` (line 1756)

#### `latent_space_visualizer.py`
**Path:** `latent_space_visualizer.py`

**Classs:**
- `VisualizerConfig` (line 59)
- `CSVLogger` (line 79)
- `LatentSpaceWidget` (line 139)
- `MetricsWidget` (line 203)
- `WeightTextureWidget` (line 276)
- `TrainingWorker` (line 321)
- `MainWindow` (line 533)

**Functions:**
- `main` (line 766)
- `__init__` (line 80)
- `log` (line 93)
- `_flatten_dict` (line 107)
- `_flush` (line 122)
- `close` (line 129)
- `get_csv_path` (line 135)
- `__init__` (line 140)
- `update_data` (line 154)
- `clear` (line 192)
- `__init__` (line 204)
- `_setup_axes` (line 215)
- `update_data` (line 224)
- `clear` (line 266)
- `__init__` (line 277)
- `update_data` (line 287)
- `_reshape` (line 303)
- `clear` (line 313)
- `__init__` (line 327)
- `setup` (line 345)
- `run` (line 385)
- `_train_epoch` (line 417)
- `_validate` (line 437)
- `_compute_metrics` (line 446)
- `_extract_weights` (line 503)
- `_extract_gradients` (line 513)
- `stop` (line 523)
- `pause` (line 526)
- `resume` (line 529)
- `__init__` (line 534)
- `_setup_ui` (line 546)
- `_log_msg` (line 660)
- `_start` (line 664)
- `_pause` (line 692)
- `_stop` (line 697)
- `_clear` (line 705)
- `_on_progress` (line 712)
- `_on_finished` (line 749)
- `closeEvent` (line 760)

#### `lidar_interactive_viewer.py`
**Path:** `lidar_interactive_viewer.py`

**Functions:**
- `load_csv_point_cloud` (line 22) - *Load point cloud from CSV file.

Returns:
    points: Nx3 array of coordinates
    attributes: dict of additional attributes (intensity, range, etc.)*
- `generate_interactive_html` (line 64) - *Generate interactive HTML using Three.js for 3D navigation.*
- `generate_plotly_html` (line 581) - *Generate interactive HTML using Plotly.js (alternative viewer).
Sometimes more compatible with large point clouds.*
- `main` (line 675)

#### `relativistic_hydrogen.py`
**Path:** `relativistic_hydrogen.py`

**Classs:**
- `Config` (line 41)
- `LoggerFactory` (line 95)
- `GammaMatrices` (line 113) - *Dirac gamma matrices in Dirac (standard) representation.
gamma^0 = beta, gamma^i = beta * alpha_i*
- `DiracHamiltonianOperator` (line 199) - *Dirac Hamiltonian operator for relativistic quantum mechanics.
H_Dirac = c * alpha . p + beta * m * c^2 + V(r)

In atomic units (c = 1/alpha ~ 137):
H = c * alpha . p + beta * m * c^2 + V*
- `SpectralLayer` (line 310)
- `DiracSpectralNetwork` (line 351) - *Neural network for learning Dirac equation dynamics.
Handles 4-component spinors with real and imaginary parts (8 channels total).*
- `DiracModelWrapper` (line 393) - *Wrapper to load and use the trained Dirac model.*
- `DiracHydrogenAtom` (line 516) - *Relativistic hydrogen atom with Dirac equation.
Computes energy levels including fine structure.*
- `ZitterbewegungSimulator` (line 640) - *Simulates the Zitterbewegung (trembling motion) of a relativistic electron.

In Dirac theory, the position operator has a term oscillating with frequency
~ 2mc^2/hbar, which is the interference between positive and negative energy states.

<x(t)> = <x(0)> + (p/m) * t + oscillating term
The oscillating term has amplitude ~ hbar/(2mc) ~ 10^-12 m*
- `DiracWavefunctionCalculator` (line 833) - *Calculate relativistic hydrogen wavefunctions.*
- `DiracMonteCarloSampler` (line 956) - *Monte Carlo sampling for relativistic orbital visualization.*
- `DiracVisualizer` (line 1075) - *Visualization suite for Dirac equation results.*
- `DiracValidationSuite` (line 1370) - *Complete validation suite for Dirac equation grokking.*

**Functions:**
- `main` (line 1613)
- `create_logger` (line 97)
- `__init__` (line 118)
- `_init_matrices` (line 122)
- `__init__` (line 207)
- `_precompute_operators` (line 215)
- `apply_dirac_hamiltonian` (line 223) - *Apply Dirac Hamiltonian to 4-component spinor.

Args:
    spinor: Shape [4, H, W] or [batch, 4, H, W] - 4-component spinor
    potential: Optional scalar potential V(r)

Returns:
    H * psi with same shape as input*
- `time_evolution` (line 282) - *Time evolution of Dirac spinor using first-order split-step.
psi(t+dt) = exp(-i * H * dt) * psi(t) ~ (1 - i*H*dt) * psi*
- `__init__` (line 311)
- `forward` (line 322)
- `__init__` (line 356)
- `forward` (line 379)
- `__init__` (line 397)
- `_find_best_checkpoint` (line 406)
- `_load_model` (line 452)
- `apply_hamiltonian` (line 498) - *Apply Hamiltonian using analytical operator.
The NN model learns spinor evolution, but the Hamiltonian operator
is applied analytically for physical validation.*
- `evolve_spinor` (line 506) - *Evolve spinor in time using the analytical Dirac operator.*
- `__init__` (line 521)
- `energy_level_dirac` (line 526) - *Exact Dirac energy level for hydrogen-like atom.

E = m*c^2 / sqrt(1 + (alpha*Z)^2 / (n - |kappa| + sqrt(kappa^2 - (alpha*Z)^2))^2)

For hydrogen (Z=1):
E = mc^2 * [1 + (alpha^2 / (n - |kappa| + sqrt(kappa^2 - alpha^2)))^2]^(-1/2)

Args:
    n: Principal quantum number
    kappa: Relativistic quantum number (kappa = -(l+1) for j=l+1/2, kappa = l for j=l-1/2)

Returns:
    Energy in atomic units (relative to m*c^2)*
- `fine_structure_splitting` (line 556) - *Calculate fine structure splitting for given n, l.

Fine structure includes:
1. Relativistic correction to kinetic energy
2. Spin-orbit coupling
3. Darwin term (for l=0)

Returns energies for j = l+1/2 and j = l-1/2*
- `energy_spectrum` (line 597) - *Generate relativistic energy spectrum up to n_max.*
- `__init__` (line 650)
- `create_gaussian_wave_packet` (line 657) - *Create a Gaussian wave packet for a free particle.

For Dirac, we need a 4-component spinor that's a superposition
of positive energy states.*
- `compute_position_expectation` (line 702) - *Compute expectation value of position operator.
<x> = <psi| x |psi>*
- `compute_velocity_expectation` (line 724) - *Compute expectation value of velocity operator.
In Dirac theory, v = c * alpha

<v_x> = c * <psi| alpha_x |psi>*
- `simulate` (line 750) - *Run Zitterbewegung simulation.

Returns time evolution of position and velocity showing the
oscillatory ZBW term.*
- `__init__` (line 837)
- `radial_wavefunction_schrodinger` (line 843) - *Non-relativistic radial wavefunction for comparison.*
- `radial_wavefunction_dirac` (line 853) - *Relativistic radial wavefunctions for hydrogen.

Returns (f, g) - small and large components.
For bound states, the Dirac radial functions are:
f(r) = sqrt((E+mc^2)/(2E)) * G(r)
g(r) = sqrt((E-mc^2)/(2E)) * F(r)

Simplified version using Sommerfeld fine-structure formula.*
- `spherical_harmonic_real` (line 900) - *Real spherical harmonics.*
- `spin_angular_function` (line 910) - *Spin-angular functions Omega_{kappa,m_j}(theta, phi).

These couple the orbital and spin degrees of freedom.*
- `__init__` (line 960)
- `sample_orbital` (line 966) - *Sample points from a relativistic hydrogen orbital.*
- `__init__` (line 1079)
- `visualize_orbital` (line 1082) - *Visualize relativistic orbital.*
- `visualize_energy_spectrum` (line 1213) - *Visualize relativistic energy spectrum with fine structure.*
- `visualize_zitterbewegung` (line 1296) - *Visualize Zitterbewegung oscillation.*
- `__init__` (line 1374)
- `print_header` (line 1398)
- `validate_fine_structure` (line 1419) - *Validate fine structure energy corrections.*
- `validate_zitterbewegung` (line 1477) - *Validate Zitterbewegung simulation.*
- `validate_energy_spectrum` (line 1509) - *Validate complete energy spectrum.*
- `validate_orbital` (line 1524) - *Validate single orbital visualization.*
- `run_full_validation` (line 1541) - *Run complete validation suite.*
- `interactive_mode` (line 1575) - *Run in interactive mode.*

#### `visualize_lidar_csv2.py`
**Path:** `visualize_lidar_csv2.py`

**Functions:**
- `visualize_csv` (line 14) - *Visualize a point cloud CSV file.

CSV must have columns: x, y, z (and optionally: intensity, range)*
- `main` (line 108)

#### `weight_3d_standard.py`
**Path:** `weight_3d_standard.py`

**Classs:**
- `StandardWeightVisualizer` (line 41) - *Standard 3D weight visualization without LiDAR physics.

Each point represents:
- Per-layer mode: one layer (weights flattened)
- Per-neuron mode: one neuron/filter
- Sliding window mode: consecutive weight chunks*

**Functions:**
- `generate_standard_html` (line 264) - *Generate interactive 3D visualization using Plotly.
Standard approach - simple and effective.*
- `generate_continuous_html` (line 415) - *Generate visualization with continuous color scale.*
- `main` (line 492)
- `__init__` (line 51)
- `load_checkpoint` (line 60) - *Load PyTorch checkpoint.*
- `extract_weights_per_layer` (line 66) - *Extract weights organized by layer.

Returns:
    weights: List of flattened weight arrays per layer
    names: Layer names
    stats: Statistics per layer*
- `extract_weights_per_neuron` (line 115) - *Extract weights organized per neuron/filter.

Each row = one neuron's incoming weights.*
- `extract_weights_sliding_window` (line 181) - *Extract weights using sliding window approach.

Each point = consecutive chunk of weights.*
- `reduce_dimensions` (line 221) - *Apply dimensionality reduction.*
- `_simple_projection` (line 254) - *Fallback projection without sklearn.*

#### `weight_space_lidar.py`
**Path:** `weight_space_lidar.py`

**Classs:**
- `WeightSpaceLiDARConfig` (line 72) - *Master configuration for Weight Space LiDAR system.
All parameters are immutable and type-safe.*
- `ILogger` (line 167) - *Protocol for logger implementations.*
- `LoggerFactory` (line 176) - *Factory for creating configured logger instances.*
- `IWeightExtractor` (line 194) - *Protocol for weight extraction strategies.*
- `IDimensionalityReducer` (line 207) - *Protocol for dimensionality reduction strategies.*
- `IRangeCalculator` (line 220) - *Protocol for range calculation strategies.*
- `ITransmissionCalculator` (line 229) - *Protocol for transmission calculation strategies.*
- `IPointCloudGenerator` (line 238) - *Protocol for point cloud generation strategies.*
- `DefaultWeightExtractor` (line 251) - *Default implementation for weight extraction from PyTorch checkpoints.*
- `PCAReducer` (line 304) - *PCA-based dimensionality reduction for weight space.*
- `TSNEReducer` (line 340) - *t-SNE based dimensionality reduction for weight space visualization.*
- `FrobeniusRangeCalculator` (line 374) - *Calculate range using Frobenius norm in weight space.*
- `BeerLambertTransmissionCalculator` (line 411) - *Calculate transmission using Beer-Lambert law adapted for weight space.*
- `HessianCurvatureEstimator` (line 442) - *Estimate local curvature (backscatter coefficient) using Hessian approximation.*
- `LiDARPhysicsEngine` (line 517) - *Core LiDAR physics engine adapted for weight space analysis.

Implements the full LiDAR equation:
P(r) = (E_L * c / 2) * A * [β_a * P_a + β_m * P_m] * exp(-2∫σ(r')dr') / r² + M(r) + b

Adapted for weight space where:
- E_L: laser energy -> probing intensity
- β: backscatter coefficient -> Hessian curvature
- σ: extinction coefficient -> gradient magnitude
- r: range -> weight space distance*
- `TemporalCheckpointScanner` (line 690) - *Scanner for temporal evolution of checkpoints.
Provides 4D visualization (3D space + time) of weight space dynamics.*
- `PointCloudGenerator` (line 884) - *Generate point cloud representations of weight space.*
- `WeightSpaceNavigator` (line 1066) - *Main navigation interface for weight space LiDAR.
Provides high-level API for exploring neural network checkpoints.*
- `WeightSpaceLiDARCLI` (line 1447) - *Command-line interface for Weight Space LiDAR.*

**Functions:**
- `main` (line 1707) - *Entry point for Weight Space LiDAR CLI.*
- `debug` (line 170)
- `info` (line 171)
- `warning` (line 172)
- `error` (line 173)
- `create` (line 180)
- `extract` (line 197) - *Extract weight vector from checkpoint.*
- `get_layer_names` (line 201) - *Get list of layer names from checkpoint.*
- `fit_transform` (line 210) - *Fit and transform data to lower dimensions.*
- `transform` (line 214) - *Transform new data using fitted model.*
- `calculate` (line 223) - *Calculate range between two points in weight space.*
- `calculate` (line 232) - *Calculate transmission along a path.*
- `generate` (line 241) - *Generate point cloud from weight data.*
- `__init__` (line 254)
- `extract` (line 258)
- `get_layer_names` (line 272)
- `_resolve_state_dict` (line 279)
- `_is_weight_tensor` (line 286)
- `_flatten_and_sample` (line 292)
- `__init__` (line 307)
- `fit_transform` (line 312)
- `transform` (line 329)
- `get_explained_variance` (line 334)
- `__init__` (line 343)
- `fit_transform` (line 348)
- `transform` (line 370)
- `__init__` (line 377)
- `calculate` (line 383)
- `calculate_batch` (line 395)
- `__init__` (line 414)
- `calculate` (line 418)
- `calculate_optical_depth` (line 424)
- `__init__` (line 445)
- `estimate` (line 450)
- `_estimate_hessian_diagonal` (line 473)
- `_numerical_hessian_diag` (line 483)
- `_empirical_curvature_estimate` (line 505)
- `__init__` (line 531)
- `compute_return_signal` (line 538)
- `compute_point_cloud` (line 577)
- `_compute_backscatter` (line 613)
- `_compute_geometric_factor` (line 626)
- `_compute_received_power` (line 633)
- `_compute_intensity` (line 652)
- `_compute_intensity_field` (line 668)
- `__init__` (line 696)
- `scan_directory` (line 702)
- `_find_checkpoint_files` (line 735)
- `_sort_checkpoints` (line 741)
- `_extract_epoch` (line 755)
- `_extract_temporal_weights` (line 765)
- `_load_checkpoint` (line 797)
- `_compute_temporal_signals` (line 803)
- `_compute_trajectories` (line 832)
- `_simple_trajectory` (line 862)
- `__init__` (line 887)
- `generate_from_checkpoint` (line 892)
- `generate_from_weights` (line 906)
- `_load_checkpoint` (line 937)
- `_extract_layer_weights` (line 943)
- `_create_weight_vectors` (line 957)
- `_create_synthetic_points` (line 982)
- `_apply_reduction` (line 998)
- `_simple_projection` (line 1015)
- `_post_process` (line 1025)
- `_remove_outliers` (line 1035)
- `_normalize_coordinates` (line 1052)
- `__init__` (line 1072)
- `scan_checkpoints` (line 1081)
- `generate_point_cloud` (line 1093)
- `compute_range_map` (line 1105)
- `temporal_evolution` (line 1136)
- `export_point_cloud` (line 1158)
- `visualize_3d` (line 1179)
- `visualize_temporal` (line 1219)
- `_load_checkpoint` (line 1276)
- `_compute_layer_ranges` (line 1282)
- `_compute_evolution_metrics` (line 1308)
- `_export_las` (line 1343)
- `_export_ply` (line 1367)
- `_export_csv` (line 1394)
- `_export_json` (line 1419)
- `__init__` (line 1450)
- `_create_parser` (line 1453)
- `run` (line 1558)
- `_handle_scan` (line 1585)
- `_handle_cloud` (line 1617)
- `_handle_range` (line 1647)
- `_handle_evolution` (line 1668)

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
