# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 10 | **Total Symbols Extracted:** 532 | **Total Imports:** 153

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
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

**Classes:**
- `Config` (line 52) `class Config`
- `IPhaseDetector` (line 250) `class IPhaseDetector(ABC)`
- `IMetricCalculator` (line 256) `class IMetricCalculator(ABC)`
- `SeedManager` (line 262) `class SeedManager`
- `LoggerFactory` (line 274) `class LoggerFactory`
- `GammaMatrices` (line 289) `class GammaMatrices` - *Dirac gamma matrices in various representations.
Default: Dirac (standard) representation.*
- `DiracHamiltonianOperator` (line 383) `class DiracHamiltonianOperator` - *  Dirac Hamiltonian operator for 4-component spinors.
  H_Dirac = c * alpha . p + beta * m * c^2
where alpha_i = gamma0 @ gammai and beta = gamma0.
  In natural units (c=1): H = alpha . p + beta * m*
- `SpectralLayer` (line 481) `class SpectralLayer`
- `DiracSpectralNetwork` (line 519) `class DiracSpectralNetwork` - *Neural network for learning Dirac equation dynamics.
Handles 4-component spinors with real and imaginary parts (8 channels total).*
- `HamiltonianBackbone` (line 558) `class HamiltonianBackbone`
- `HamiltonianInferenceEngine` (line 585) `class HamiltonianInferenceEngine`
- `DiracPotentialGenerator` (line 639) `class DiracPotentialGenerator` - *Generate potentials for the Dirac equation.
In relativistic QM, the potential couples differently to particle/antiparticle components.*
- `DiracDataset` (line 716) `class DiracDataset(Dataset)` - *Dataset for Dirac equation evolution.
Generates 4-component spinors and their time-evolved targets.*
- `FullFourierAnalyzer` (line 845) `class FullFourierAnalyzer`
- `FourierMassCenterAnalyzer` (line 1021) `class FourierMassCenterAnalyzer`
- `TopologicalPhaseDetector` (line 1092) `class TopologicalPhaseDetector(IPhaseDetector)`
- `SpectralFieldExtractor` (line 1161) `class SpectralFieldExtractor`
- `TopologicalCrystallizationLoss` (line 1183) `class TopologicalCrystallizationLoss`
- `CrystallizationPressureApplicator` (line 1221) `class CrystallizationPressureApplicator`
- `TopologicalMetricsCalculator` (line 1236) `class TopologicalMetricsCalculator(IMetricCalculator)`
- `LocalComplexityAnalyzer` (line 1299) `class LocalComplexityAnalyzer`
- `SuperpositionAnalyzer` (line 1316) `class SuperpositionAnalyzer`
- `CrystallographyMetricsCalculator` (line 1337) `class CrystallographyMetricsCalculator(IMetricCalculator)`
- `ThermodynamicMetricsCalculator` (line 1547) `class ThermodynamicMetricsCalculator(IMetricCalculator)`
- `SpectralGeometryCalculator` (line 1626) `class SpectralGeometryCalculator(IMetricCalculator)`
- `RicciCurvatureCalculator` (line 1679) `class RicciCurvatureCalculator(IMetricCalculator)`
- `PerelmanRicciFlow` (line 1723) `class PerelmanRicciFlow`
- `SpectroscopyMetricsCalculator` (line 1978) `class SpectroscopyMetricsCalculator(IMetricCalculator)`
- `LambdaPressureScheduler` (line 2015) `class LambdaPressureScheduler`
- `AdaptiveLambdaScheduler` (line 2055) `class AdaptiveLambdaScheduler(LambdaPressureScheduler)`
- `QuadruplePrecisionLambdaScheduler` (line 2076) `class QuadruplePrecisionLambdaScheduler`
- `AnnealingScheduler` (line 2116) `class AnnealingScheduler`
- `TopologicalAnnealingScheduler` (line 2146) `class TopologicalAnnealingScheduler(AnnealingScheduler)`
- `TrainingMetricsMonitor` (line 2165) `class TrainingMetricsMonitor`
- `CheckpointManager` (line 2318) `class CheckpointManager`
- `Phase5CheckpointManager` (line 2376) `class Phase5CheckpointManager`
- `GlassStateDetector` (line 2481) `class GlassStateDetector`
- `WeightIntegrityChecker` (line 2537) `class WeightIntegrityChecker`
- `TrainingEngine` (line 2569) `class TrainingEngine`
- `BatchSizeProspector` (line 2764) `class BatchSizeProspector`
- `SeedMiner` (line 2835) `class SeedMiner`
- `FullTrainingOrchestrator` (line 2976) `class FullTrainingOrchestrator`
- `RefinementOrchestrator` (line 3109) `class RefinementOrchestrator`
- `Phase5Orchestrator` (line 3238) `class Phase5Orchestrator`

**Functions:**
- `main` (line 3626) `def main()`
- `detect` (line 252) `def detect(self, spectral_field)`
- `compute` (line 258) `def compute(self, model)`
- `set_seed` (line 264) `def set_seed(seed, device)`
- `create_logger` (line 276) `def create_logger(name, level)`
- `__init__` (line 294) `def __init__(self, representation, device)`
- `_init_matrices` (line 299) `def _init_matrices(self)`
- `to` (line 377) `def to(self, device)`
- `__init__` (line 390) `def __init__(self, config)`
- `_precompute_operators` (line 398) `def _precompute_operators(self)`
- `apply_dirac_hamiltonian` (line 414) `def apply_dirac_hamiltonian(self, spinor)` - *Apply Dirac Hamiltonian to 4-component spinor.
H_psi = c * (alpha_x * p_x + alpha_y * p_y) @ psi + beta * m * c^2 * psi

Input shape: [4, H, W] or [batch, 4, H, W]
Output shape: same as input*
- `time_evolution` (line 457) `def time_evolution(self, spinor, dt)` - *Time evolution of Dirac spinor using split-step method.
psi(t+dt) = exp(-i * H * dt) * psi(t)*
- `__init__` (line 482) `def __init__(self, channels, grid_size)`
- `forward` (line 493) `def forward(self, x)`
- `__init__` (line 524) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers, spinor_components)`
- `forward` (line 547) `def forward(self, x)`
- `__init__` (line 559) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 574) `def forward(self, x)`
- `__init__` (line 586) `def __init__(self, config)`
- `_try_load_backbone` (line 593) `def _try_load_backbone(self)`
- `apply_hamiltonian` (line 627) `def apply_hamiltonian(self, spinor)` - *Apply Dirac Hamiltonian to 4-component spinor.
Always uses analytical operator since backbone is not compatible
with 4-component Dirac spinors.*
- `time_evolve` (line 635) `def time_evolve(self, spinor, dt)`
- `__init__` (line 644) `def __init__(self, config)`
- `scalar_potential` (line 648) `def scalar_potential(self)` - *Scalar potential (couples equally to all components).
V_s * psi (same coupling for particle and antiparticle).*
- `vector_potential` (line 660) `def vector_potential(self)` - *Vector potential (time-component of 4-vector).
V_v * gamma0 * psi (couples with opposite sign to particle/antiparticle).*
- `magnetic_potential_2d` (line 672) `def magnetic_potential_2d(self)` - *Magnetic potential (spatial components of 4-vector).
A * alpha * psi (couples to spin).*
- `periodic_lattice_potential` (line 686) `def periodic_lattice_potential(self)`
- `generate_mixed_potential` (line 692) `def generate_mixed_potential(self, seed)`
- `__init__` (line 721) `def __init__(self, config, hamiltonian_engine, seed)`
- `_generate_initial_spinor` (line 766) `def _generate_initial_spinor(self, potential, sample_seed)` - *Generate an initial Dirac spinor (4-component).
The spinor is constructed to be a superposition of positive energy states.*
- `_time_evolve_spinor` (line 798) `def _time_evolve_spinor(self, spinor, potential, energy)` - *Time evolve a Dirac spinor under the influence of potentials.*
- `_spinor_to_real_imag` (line 824) `def _spinor_to_real_imag(self, spinor)` - *Convert 4-component complex spinor to 8-channel real tensor.
Channels: [Re(psi0), Im(psi0), Re(psi1), Im(psi1), ...]*
- `__len__` (line 835) `def __len__(self)`
- `__getitem__` (line 838) `def __getitem__(self, idx)`
- `get_validation_batch` (line 841) `def get_validation_batch(self)`
- `__init__` (line 846) `def __init__(self, config)`
- `compute_full_spectrum` (line 854) `def compute_full_spectrum(self, spectral_field)`
- `detect_bragg_peaks` (line 926) `def detect_bragg_peaks(self, power_spectrum, threshold_sigma)`
- `compute_resonance_metrics` (line 980) `def compute_resonance_metrics(self, spectral_field)`
- `__init__` (line 1022) `def __init__(self, config)`
- `compute_mass_center` (line 1030) `def compute_mass_center(self, spectral_field)`
- `__init__` (line 1093) `def __init__(self, config)`
- `detect` (line 1100) `def detect(self, spectral_field)`
- `extract` (line 1163) `def extract(model, grid_size)`
- `__init__` (line 1184) `def __init__(self, config)`
- `forward` (line 1189) `def forward(self, phase_info, epoch)`
- `__init__` (line 1222) `def __init__(self, config)`
- `apply` (line 1226) `def apply(self, model, phase_info)`
- `__init__` (line 1237) `def __init__(self, config)`
- `compute` (line 1244) `def compute(self, model)`
- `apply_crystallization_pressure` (line 1277) `def apply_crystallization_pressure(self, model, topo_metrics)`
- `_empty_metrics` (line 1283) `def _empty_metrics()`
- `compute_local_complexity` (line 1301) `def compute_local_complexity(weights, epsilon)`
- `compute_superposition` (line 1318) `def compute_superposition(weights)`
- `__init__` (line 1338) `def __init__(self, config)`
- `compute` (line 1342) `def compute(self, model)`
- `compute_kappa` (line 1347) `def compute_kappa(self, model, val_x, val_y, num_batches)`
- `compute_discretization_margin` (line 1400) `def compute_discretization_margin(self, model)`
- `compute_alpha_purity` (line 1408) `def compute_alpha_purity(self, model)`
- `compute_kappa_quantum` (line 1414) `def compute_kappa_quantum(self, model)`
- `compute_poynting_vector` (line 1435) `def compute_poynting_vector(self, model)`
- `compute_hbar_effective` (line 1491) `def compute_hbar_effective(self, model, lambda_pressure)`
- `compute_all_metrics` (line 1500) `def compute_all_metrics(self, model, val_x, val_y)`
- `__init__` (line 1548) `def __init__(self, config)`
- `compute` (line 1551) `def compute(self, model)`
- `compute_effective_temperature` (line 1577) `def compute_effective_temperature(self, gradient_buffer, learning_rate)`
- `compute_specific_heat` (line 1601) `def compute_specific_heat(self, loss_history, temp_history)`
- `compute_gibbs_free_energy` (line 1615) `def compute_gibbs_free_energy(self, delta, alpha, temperature)`
- `compute_critical_temperature` (line 1622) `def compute_critical_temperature(self, alpha)`
- `__init__` (line 1627) `def __init__(self, config)`
- `compute` (line 1630) `def compute(self, model)`
- `_compute_level_spacing_ratio` (line 1667) `def _compute_level_spacing_ratio(self, spacings)`
- `__init__` (line 1680) `def __init__(self, config)`
- `compute` (line 1683) `def compute(self, model)`
- `_compute_ricci_scalar` (line 1702) `def _compute_ricci_scalar(self, metric)`
- `_estimate_sectional_curvatures` (line 1710) `def _estimate_sectional_curvatures(self, metric)`
- `__init__` (line 1724) `def __init__(self, config)`
- `compute_ricci_scalar_fast` (line 1732) `def compute_ricci_scalar_fast(self, model)`
- `compute_local_curvature` (line 1775) `def compute_local_curvature(self, param)`
- `compute_anisotropy` (line 1786) `def compute_anisotropy(self, model)`
- `compute_ricci_regularization_loss` (line 1820) `def compute_ricci_regularization_loss(self, model)`
- `apply_ricci_flow_step` (line 1845) `def apply_ricci_flow_step(self, model, lr)`
- `perform_perelman_surgery` (line 1887) `def perform_perelman_surgery(self, model, ricci_scalar)`
- `compute_adaptive_lr_factor` (line 1942) `def compute_adaptive_lr_factor(self, model)`
- `get_flow_metrics` (line 1962) `def get_flow_metrics(self, model)`
- `__init__` (line 1979) `def __init__(self, config)`
- `compute` (line 1982) `def compute(self, model)`
- `compute_weight_diffraction` (line 1986) `def compute_weight_diffraction(self, coeffs)`
- `_compute_spectral_entropy` (line 2006) `def _compute_spectral_entropy(power_spectrum)`
- `__init__` (line 2016) `def __init__(self, config)`
- `current_lambda` (line 2025) `def current_lambda(self)`
- `step` (line 2028) `def step(self, epoch)`
- `compute_regularization_loss` (line 2037) `def compute_regularization_loss(self, model)`
- `set_lambda` (line 2051) `def set_lambda(self, value)`
- `__init__` (line 2056) `def __init__(self, config)`
- `step_adaptive` (line 2061) `def step_adaptive(self, epoch, topo_phase_state)`
- `__init__` (line 2077) `def __init__(self, config)`
- `current_lambda` (line 2086) `def current_lambda(self)`
- `step` (line 2089) `def step(self, epoch, improvement)`
- `compute_regularization_loss` (line 2098) `def compute_regularization_loss(self, model)`
- `set_lambda` (line 2112) `def set_lambda(self, value)`
- `__init__` (line 2117) `def __init__(self, config)`
- `temperature` (line 2125) `def temperature(self)`
- `step` (line 2128) `def step(self)`
- `accept_perturbation` (line 2134) `def accept_perturbation(self, delta_loss)`
- `should_restart` (line 2142) `def should_restart(self, current_delta, best_delta)`
- `__init__` (line 2147) `def __init__(self, config)`
- `step_adaptive` (line 2151) `def step_adaptive(self, alignment_trend, resonance_score)`
- `__init__` (line 2166) `def __init__(self, config)`
- `update_metrics` (line 2196) `def update_metrics(self)`
- `compute_delta_slope` (line 2207) `def compute_delta_slope(self)`
- `format_progress_bar` (line 2220) `def format_progress_bar(self, epoch, total_epochs, phase)`
- `__init__` (line 2319) `def __init__(self, config, checkpoint_dir)`
- `should_save_checkpoint` (line 2328) `def should_save_checkpoint(self)`
- `save_checkpoint` (line 2333) `def save_checkpoint(self, model, optimizer, epoch, metrics, phase, lambda_value, config_snapshot)`
- `load_latest_checkpoint` (line 2369) `def load_latest_checkpoint(self)`
- `__init__` (line 2377) `def __init__(self, config)`
- `_load_best_metrics` (line 2388) `def _load_best_metrics(self)`
- `should_save` (line 2407) `def should_save(self, current_delta, current_alpha, current_acc)`
- `save_checkpoint` (line 2418) `def save_checkpoint(self, model, optimizer, epoch, metrics, lambda_value)`
- `load_checkpoint` (line 2462) `def load_checkpoint(self, model, optimizer)`
- `__init__` (line 2482) `def __init__(self, config)`
- `should_stop` (line 2488) `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- `is_crystal_formed` (line 2523) `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)`
- `check` (line 2539) `def check(model)`
- `__init__` (line 2570) `def __init__(self, config)`
- `compute_weight_metrics` (line 2583) `def compute_weight_metrics(self, model)`
- `compute_norm_conservation_error` (line 2598) `def compute_norm_conservation_error(self, model, val_x)`
- `train_single_epoch` (line 2611) `def train_single_epoch(self, model, optimizer, dataloader, epoch, lambda_scheduler, ricci_flow)`
- `validate` (line 2666) `def validate(self, model, val_x, val_y)`
- `collect_all_metrics` (line 2679) `def collect_all_metrics(self, model, monitor, val_x, val_y, lambda_scheduler, annealing_scheduler, current_lr, epoch)`
- `__init__` (line 2765) `def __init__(self, config, hamiltonian_engine)`
- `prospect` (line 2770) `def prospect(self)`
- `__init__` (line 2836) `def __init__(self, config, hamiltonian_engine, batch_size)`
- `mine` (line 2847) `def mine(self)`
- `__init__` (line 2977) `def __init__(self, config, hamiltonian_engine, seed, batch_size)`
- `run_phase3_training` (line 2990) `def run_phase3_training(self, start_epoch, model)`
- `__init__` (line 3110) `def __init__(self, config, hamiltonian_engine, model, optimizer, monitor, seed, batch_size)`
- `run_phase4_refinement` (line 3129) `def run_phase4_refinement(self, start_epoch)`
- `__init__` (line 3239) `def __init__(self, config, hamiltonian_engine, model, monitor, seed, batch_size)`
- `_detect_blocked_labyrinth` (line 3268) `def _detect_blocked_labyrinth(self, spec_gap, anisotropy, resonance)`
- `_apply_flood_fill_pressure` (line 3280) `def _apply_flood_fill_pressure(self, lambda_scheduler, epoch, spec_gap, anisotropy)`
- `_inject_diffusion_energy` (line 3306) `def _inject_diffusion_energy(self)`
- `_find_ballistic_trajectory` (line 3317) `def _find_ballistic_trajectory(self, resonance, anisotropy)`
- `_load_phase5_checkpoint` (line 3328) `def _load_phase5_checkpoint(self, optimizer, lambda_scheduler)`
- `_apply_perelman_surgery` (line 3357) `def _apply_perelman_surgery(self, lambda_scheduler, epoch, ricci_scalar)`
- `run_phase5_crystallization` (line 3384) `def run_phase5_crystallization(self, start_epoch)`
- `load_latest_checkpoint` (line 3761) `def load_latest_checkpoint(model, checkpoint_paths)`
- `safe_compute` (line 1513) `def safe_compute(func)`
- `safe_get` (line 2224) `def safe_get(key)`

#### `dirac_crystallography_suite.py`
**Path:** `dirac_crystallography_suite.py`

**Classes:**
- `CrystallographySuiteConfig` (line 55) `class CrystallographySuiteConfig` - *Master configuration for the complete crystallography suite.*
- `LoggerFactory` (line 178) `class LoggerFactory` - *Factory for creating configured logger instances.*
- `IMetricCalculator` (line 196) `class IMetricCalculator(Protocol)` - *Protocol for metric calculation strategies.*
- `IPhaseDetector` (line 204) `class IPhaseDetector(Protocol)` - *Protocol for phase detection strategies.*
- `GammaMatrices` (line 212) `class GammaMatrices` - *Dirac gamma matrices in various representations.*
- `DiracHamiltonianOperator` (line 263) `class DiracHamiltonianOperator` - *Dirac Hamiltonian operator for 4-component spinors.*
- `SpectralLayer` (line 328) `class SpectralLayer` - *Spectral convolution layer operating in Fourier space.*
- `DiracSpectralNetwork` (line 363) `class DiracSpectralNetwork` - *Neural network for learning Dirac equation dynamics.*
- `WeightIntegrityCalculator` (line 394) `class WeightIntegrityCalculator` - *Calculator for weight integrity metrics.*
- `DiscretizationCalculator` (line 433) `class DiscretizationCalculator` - *Calculator for discretization margin and alpha purity metrics.*
- `SpectralGeometryCalculator` (line 481) `class SpectralGeometryCalculator` - *Calculator for spectral geometry metrics including MBL level spacing.*
- `RicciCurvatureCalculator` (line 536) `class RicciCurvatureCalculator` - *Calculator for Ricci curvature estimation in weight space.*
- `BerryPhaseCalculator` (line 581) `class BerryPhaseCalculator` - *Calculates Berry phase from training checkpoint trajectory.*
- `ControlSystemAnalyzer` (line 689) `class ControlSystemAnalyzer` - *Control theory analysis for neural network dynamics.*
- `ThermodynamicCalculator` (line 781) `class ThermodynamicCalculator` - *Calculator for thermodynamic potentials.*
- `FullFourierAnalyzer` (line 826) `class FullFourierAnalyzer` - *Complete Fourier analysis for spectral fields.*
- `FourierMassCenterAnalyzer` (line 882) `class FourierMassCenterAnalyzer` - *Analyzer for center of mass in Fourier space.*
- `TopologicalPhaseDetector` (line 933) `class TopologicalPhaseDetector` - *Detects topological phases from spectral field analysis.*
- `SpectralFieldExtractor` (line 971) `class SpectralFieldExtractor` - *Extracts spectral fields from neural network layers.*
- `TopologicalMetricsCalculator` (line 993) `class TopologicalMetricsCalculator` - *Calculates topological metrics from model spectral fields.*
- `GradientDynamicsCalculator` (line 1034) `class GradientDynamicsCalculator` - *Calculator for gradient-based metrics.*
- `SchrodingerAnalyzer` (line 1127) `class SchrodingerAnalyzer` - *Quantum mechanical analysis of network parameters.*
- `ComprehensiveVisualizer` (line 1189) `class ComprehensiveVisualizer` - *Generates comprehensive visualizations for all metrics.*
- `CheckpointAnalyzer` (line 1451) `class CheckpointAnalyzer` - *Main analyzer that orchestrates all metric calculations.*
- `BatchProcessor` (line 1576) `class BatchProcessor` - *Processes multiple checkpoints in batch mode.*
- `DiracCrystallographySuite` (line 1723) `class DiracCrystallographySuite` - *Main entry point for the crystallography analysis suite.*

**Functions:**
- `main` (line 1806) `def main()`
- `create_logger` (line 182) `def create_logger(name, level, config)`
- `compute` (line 199) `def compute(self, model)` - *Compute metrics for the given model.*
- `detect` (line 207) `def detect(self, spectral_field)` - *Detect phase from spectral field.*
- `__init__` (line 215) `def __init__(self, representation, device, config)`
- `_init_matrices` (line 221) `def _init_matrices(self)`
- `__init__` (line 266) `def __init__(self, config)`
- `_precompute_operators` (line 274) `def _precompute_operators(self)`
- `apply_dirac_hamiltonian` (line 290) `def apply_dirac_hamiltonian(self, spinor)` - *Apply Dirac Hamiltonian to 4-component spinor.*
- `__init__` (line 331) `def __init__(self, channels, grid_size, config)`
- `forward` (line 343) `def forward(self, x)`
- `__init__` (line 366) `def __init__(self, config)`
- `forward` (line 383) `def forward(self, x)`
- `__init__` (line 397) `def __init__(self, config)`
- `compute` (line 400) `def compute(self, model)`
- `__init__` (line 436) `def __init__(self, config)`
- `compute` (line 439) `def compute(self, model)`
- `_compute_spectral_entropy` (line 466) `def _compute_spectral_entropy(self, weights)`
- `__init__` (line 484) `def __init__(self, config)`
- `compute` (line 487) `def compute(self, model)`
- `_compute_level_spacing_ratio` (line 524) `def _compute_level_spacing_ratio(self, spacings)`
- `__init__` (line 539) `def __init__(self, config)`
- `compute` (line 542) `def compute(self, model)`
- `_compute_ricci_scalar` (line 559) `def _compute_ricci_scalar(self, metric)`
- `_estimate_sectional_curvatures` (line 568) `def _estimate_sectional_curvatures(self, metric, samples)`
- `__init__` (line 584) `def __init__(self, config)`
- `load_checkpoints` (line 588) `def load_checkpoints(self, checkpoint_dir)`
- `_extract_epoch` (line 607) `def _extract_epoch(self, filepath)`
- `flatten_kernel_params` (line 611) `def flatten_kernel_params(self, state_dict)`
- `compute_berry_connection_discrete` (line 635) `def compute_berry_connection_discrete(self, theta_prev, theta_curr)`
- `calculate_berry_phase` (line 650) `def calculate_berry_phase(self, checkpoint_dir)`
- `__init__` (line 692) `def __init__(self, config)`
- `extract_state_space` (line 695) `def extract_state_space(self, model)`
- `analyze_stability` (line 755) `def analyze_stability(self, A)`
- `compute` (line 772) `def compute(self, model)`
- `__init__` (line 784) `def __init__(self, config)`
- `compute` (line 787) `def compute(self, model)`
- `_classify_phase` (line 812) `def _classify_phase(self, delta, kappa, temp, alpha)`
- `__init__` (line 829) `def __init__(self, config)`
- `compute_full_spectrum` (line 837) `def compute_full_spectrum(self, spectral_field)`
- `compute_resonance_metrics` (line 867) `def compute_resonance_metrics(self, spectral_field)`
- `__init__` (line 885) `def __init__(self, config)`
- `compute_mass_center` (line 893) `def compute_mass_center(self, spectral_field)`
- `__init__` (line 936) `def __init__(self, config)`
- `detect` (line 943) `def detect(self, spectral_field)`
- `extract` (line 975) `def extract(model, grid_size)`
- `__init__` (line 996) `def __init__(self, config)`
- `compute` (line 1001) `def compute(self, model)`
- `_empty_metrics` (line 1022) `def _empty_metrics()`
- `__init__` (line 1037) `def __init__(self, config)`
- `compute` (line 1040) `def compute(self, model)`
- `__init__` (line 1130) `def __init__(self, config)`
- `extract_compressed_wavefunction` (line 1135) `def extract_compressed_wavefunction(self, model)`
- `_compress_johnson_lindenstrauss` (line 1156) `def _compress_johnson_lindenstrauss(self, vector)`
- `compute` (line 1168) `def compute(self, model)`
- `__init__` (line 1192) `def __init__(self, config)`
- `visualize_checkpoint_analysis` (line 1195) `def visualize_checkpoint_analysis(self, results, output_path)`
- `_plot_weight_distribution` (line 1222) `def _plot_weight_distribution(self, results, ax)`
- `_plot_spectral_analysis` (line 1233) `def _plot_spectral_analysis(self, results, ax)`
- `_plot_phase_diagram` (line 1247) `def _plot_phase_diagram(self, results, ax)`
- `_plot_curvature_distribution` (line 1264) `def _plot_curvature_distribution(self, results, ax)`
- `_plot_level_spacing` (line 1278) `def _plot_level_spacing(self, results, ax)`
- `_plot_eigenvalue_spectrum` (line 1292) `def _plot_eigenvalue_spectrum(self, results, ax)`
- `_plot_thermodynamic_potentials` (line 1306) `def _plot_thermodynamic_potentials(self, results, ax)`
- `_plot_topological_metrics` (line 1320) `def _plot_topological_metrics(self, results, ax)`
- `_plot_berry_phase` (line 1334) `def _plot_berry_phase(self, results, ax)`
- `_plot_control_stability` (line 1354) `def _plot_control_stability(self, results, ax)`
- `_plot_quantum_metrics` (line 1366) `def _plot_quantum_metrics(self, results, ax)`
- `_plot_summary_table` (line 1380) `def _plot_summary_table(self, results, ax)`
- `_plot_layer_deltas` (line 1403) `def _plot_layer_deltas(self, results, ax)`
- `_plot_resonance_metrics` (line 1415) `def _plot_resonance_metrics(self, results, ax)`
- `_plot_spectral_concentration` (line 1428) `def _plot_spectral_concentration(self, results, ax)`
- `_plot_health_score` (line 1439) `def _plot_health_score(self, results, ax)`
- `__init__` (line 1454) `def __init__(self, config)`
- `analyze_checkpoint` (line 1470) `def analyze_checkpoint(self, checkpoint_path, val_data)`
- `_compute_health_score` (line 1546) `def _compute_health_score(self, results)`
- `__init__` (line 1579) `def __init__(self, config)`
- `process_directory` (line 1585) `def process_directory(self, checkpoint_dir, output_dir, val_data)`
- `_generate_summary` (line 1631) `def _generate_summary(self, all_results)`
- `_generate_evolution_plots` (line 1682) `def _generate_evolution_plots(self, all_results, output_dir)`
- `__init__` (line 1726) `def __init__(self, config)`
- `run_analysis` (line 1732) `def run_analysis(self, checkpoint_dir, output_dir)`
- `_generate_berry_phase_visualization` (line 1756) `def _generate_berry_phase_visualization(self, berry_results, output_dir)`

#### `latent_space_visualizer.py`
**Path:** `latent_space_visualizer.py`

**Classes:**
- `VisualizerConfig` (line 59) `class VisualizerConfig`
- `CSVLogger` (line 79) `class CSVLogger`
- `LatentSpaceWidget` (line 139) `class LatentSpaceWidget(FigureCanvas)`
- `MetricsWidget` (line 203) `class MetricsWidget(FigureCanvas)`
- `WeightTextureWidget` (line 276) `class WeightTextureWidget(FigureCanvas)`
- `TrainingWorker` (line 321) `class TrainingWorker(QObject)`
- `MainWindow` (line 533) `class MainWindow(QMainWindow)`

**Functions:**
- `main` (line 766) `def main()`
- `__init__` (line 80) `def __init__(self, config)`
- `log` (line 93) `def log(self, metrics)`
- `_flatten_dict` (line 107) `def _flatten_dict(self, d, parent_key, sep)`
- `_flush` (line 122) `def _flush(self)`
- `close` (line 129) `def close(self)`
- `get_csv_path` (line 135) `def get_csv_path(self)`
- `__init__` (line 140) `def __init__(self, config, parent)`
- `update_data` (line 154) `def update_data(self, weights, metric_value)`
- `clear` (line 192) `def clear(self)`
- `__init__` (line 204) `def __init__(self, config, parent)`
- `_setup_axes` (line 215) `def _setup_axes(self)`
- `update_data` (line 224) `def update_data(self, metrics)`
- `clear` (line 266) `def clear(self)`
- `__init__` (line 277) `def __init__(self, config, parent)`
- `update_data` (line 287) `def update_data(self, weights, gradients)`
- `_reshape` (line 303) `def _reshape(self, arr)`
- `clear` (line 313) `def clear(self)`
- `__init__` (line 327) `def __init__(self, config, dirac_config)`
- `setup` (line 345) `def setup(self)`
- `run` (line 385) `def run(self)`
- `_train_epoch` (line 417) `def _train_epoch(self)`
- `_validate` (line 437) `def _validate(self)`
- `_compute_metrics` (line 446) `def _compute_metrics(self, epoch, train_loss, val_loss, val_acc)`
- `_extract_weights` (line 503) `def _extract_weights(self)`
- `_extract_gradients` (line 513) `def _extract_gradients(self)`
- `stop` (line 523) `def stop(self)`
- `pause` (line 526) `def pause(self)`
- `resume` (line 529) `def resume(self)`
- `__init__` (line 534) `def __init__(self, config)`
- `_setup_ui` (line 546) `def _setup_ui(self)`
- `_log_msg` (line 660) `def _log_msg(self, msg)`
- `_start` (line 664) `def _start(self)`
- `_pause` (line 692) `def _pause(self)`
- `_stop` (line 697) `def _stop(self)`
- `_clear` (line 705) `def _clear(self)`
- `_on_progress` (line 712) `def _on_progress(self, metrics)`
- `_on_finished` (line 749) `def _on_finished(self)`
- `closeEvent` (line 760) `def closeEvent(self, e)`

#### `lidar_interactive_viewer.py`
**Path:** `lidar_interactive_viewer.py`

**Functions:**
- `load_csv_point_cloud` (line 22) `def load_csv_point_cloud(csv_path)` - *Load point cloud from CSV file.

Returns:
    points: Nx3 array of coordinates
    attributes: dict of additional attributes (intensity, range, etc.)*
- `generate_interactive_html` (line 64) `def generate_interactive_html(points, attributes, output_path, title, point_size, colormap, intensity_col)` - *Generate interactive HTML using Three.js for 3D navigation.*
- `generate_plotly_html` (line 581) `def generate_plotly_html(points, attributes, output_path, title)` - *Generate interactive HTML using Plotly.js (alternative viewer).
Sometimes more compatible with large point clouds.*
- `main` (line 675) `def main()`

#### `relativistic_hydrogen.py`
**Path:** `relativistic_hydrogen.py`

**Classes:**
- `Config` (line 41) `class Config`
- `LoggerFactory` (line 95) `class LoggerFactory`
- `GammaMatrices` (line 113) `class GammaMatrices` - *Dirac gamma matrices in Dirac (standard) representation.
gamma^0 = beta, gamma^i = beta * alpha_i*
- `DiracHamiltonianOperator` (line 199) `class DiracHamiltonianOperator` - *Dirac Hamiltonian operator for relativistic quantum mechanics.
H_Dirac = c * alpha . p + beta * m * c^2 + V(r)

In atomic units (c = 1/alpha ~ 137):
H = c * alpha . p + beta * m * c^2 + V*
- `SpectralLayer` (line 310) `class SpectralLayer`
- `DiracSpectralNetwork` (line 351) `class DiracSpectralNetwork` - *Neural network for learning Dirac equation dynamics.
Handles 4-component spinors with real and imaginary parts (8 channels total).*
- `DiracModelWrapper` (line 393) `class DiracModelWrapper` - *Wrapper to load and use the trained Dirac model.*
- `DiracHydrogenAtom` (line 516) `class DiracHydrogenAtom` - *Relativistic hydrogen atom with Dirac equation.
Computes energy levels including fine structure.*
- `ZitterbewegungSimulator` (line 640) `class ZitterbewegungSimulator` - *Simulates the Zitterbewegung (trembling motion) of a relativistic electron.

In Dirac theory, the position operator has a term oscillating with frequency
~ 2mc^2/hbar, which is the interference between positive and negative energy states.

<x(t)> = <x(0)> + (p/m) * t + oscillating term
The oscillating term has amplitude ~ hbar/(2mc) ~ 10^-12 m*
- `DiracWavefunctionCalculator` (line 833) `class DiracWavefunctionCalculator` - *Calculate relativistic hydrogen wavefunctions.*
- `DiracMonteCarloSampler` (line 956) `class DiracMonteCarloSampler` - *Monte Carlo sampling for relativistic orbital visualization.*
- `DiracVisualizer` (line 1075) `class DiracVisualizer` - *Visualization suite for Dirac equation results.*
- `DiracValidationSuite` (line 1370) `class DiracValidationSuite` - *Complete validation suite for Dirac equation grokking.*

**Functions:**
- `main` (line 1613) `def main()`
- `create_logger` (line 97) `def create_logger(name, level)`
- `__init__` (line 118) `def __init__(self, device)`
- `_init_matrices` (line 122) `def _init_matrices(self)`
- `__init__` (line 207) `def __init__(self, config)`
- `_precompute_operators` (line 215) `def _precompute_operators(self)`
- `apply_dirac_hamiltonian` (line 223) `def apply_dirac_hamiltonian(self, spinor, potential)` - *Apply Dirac Hamiltonian to 4-component spinor.

Args:
    spinor: Shape [4, H, W] or [batch, 4, H, W] - 4-component spinor
    potential: Optional scalar potential V(r)

Returns:
    H * psi with same shape as input*
- `time_evolution` (line 282) `def time_evolution(self, spinor, dt, potential)` - *Time evolution of Dirac spinor using first-order split-step.
psi(t+dt) = exp(-i * H * dt) * psi(t) ~ (1 - i*H*dt) * psi*
- `__init__` (line 311) `def __init__(self, channels, grid_size)`
- `forward` (line 322) `def forward(self, x)`
- `__init__` (line 356) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers, spinor_components)`
- `forward` (line 379) `def forward(self, x)`
- `__init__` (line 397) `def __init__(self, config)`
- `_find_best_checkpoint` (line 406) `def _find_best_checkpoint(self)`
- `_load_model` (line 452) `def _load_model(self)`
- `apply_hamiltonian` (line 498) `def apply_hamiltonian(self, spinor, potential)` - *Apply Hamiltonian using analytical operator.
The NN model learns spinor evolution, but the Hamiltonian operator
is applied analytically for physical validation.*
- `evolve_spinor` (line 506) `def evolve_spinor(self, spinor, dt, potential)` - *Evolve spinor in time using the analytical Dirac operator.*
- `__init__` (line 521) `def __init__(self, config)`
- `energy_level_dirac` (line 526) `def energy_level_dirac(self, n, kappa)` - *Exact Dirac energy level for hydrogen-like atom.

E = m*c^2 / sqrt(1 + (alpha*Z)^2 / (n - |kappa| + sqrt(kappa^2 - (alpha*Z)^2))^2)

For hydrogen (Z=1):
E = mc^2 * [1 + (alpha^2 / (n - |kappa| + sqrt(kappa^2 - alpha^2)))^2]^(-1/2)

Args:
    n: Principal quantum number
    kappa: Relativistic quantum number (kappa = -(l+1) for j=l+1/2, kappa = l for j=l-1/2)

Returns:
    Energy in atomic units (relative to m*c^2)*
- `fine_structure_splitting` (line 556) `def fine_structure_splitting(self, n, l)` - *Calculate fine structure splitting for given n, l.

Fine structure includes:
1. Relativistic correction to kinetic energy
2. Spin-orbit coupling
3. Darwin term (for l=0)

Returns energies for j = l+1/2 and j = l-1/2*
- `energy_spectrum` (line 597) `def energy_spectrum(self, n_max)` - *Generate relativistic energy spectrum up to n_max.*
- `__init__` (line 650) `def __init__(self, config, model_wrapper)`
- `create_gaussian_wave_packet` (line 657) `def create_gaussian_wave_packet(self, sigma, momentum)` - *Create a Gaussian wave packet for a free particle.

For Dirac, we need a 4-component spinor that's a superposition
of positive energy states.*
- `compute_position_expectation` (line 702) `def compute_position_expectation(self, spinor)` - *Compute expectation value of position operator.
<x> = <psi| x |psi>*
- `compute_velocity_expectation` (line 724) `def compute_velocity_expectation(self, spinor)` - *Compute expectation value of velocity operator.
In Dirac theory, v = c * alpha

<v_x> = c * <psi| alpha_x |psi>*
- `simulate` (line 750) `def simulate(self, duration, dt, sigma)` - *Run Zitterbewegung simulation.

Returns time evolution of position and velocity showing the
oscillatory ZBW term.*
- `__init__` (line 837) `def __init__(self, config)`
- `radial_wavefunction_schrodinger` (line 843) `def radial_wavefunction_schrodinger(n, l, r)` - *Non-relativistic radial wavefunction for comparison.*
- `radial_wavefunction_dirac` (line 853) `def radial_wavefunction_dirac(self, n, kappa, r, Z)` - *Relativistic radial wavefunctions for hydrogen.

Returns (f, g) - small and large components.
For bound states, the Dirac radial functions are:
f(r) = sqrt((E+mc^2)/(2E)) * G(r)
g(r) = sqrt((E-mc^2)/(2E)) * F(r)

Simplified version using Sommerfeld fine-structure formula.*
- `spherical_harmonic_real` (line 900) `def spherical_harmonic_real(self, l, m, theta, phi)` - *Real spherical harmonics.*
- `spin_angular_function` (line 910) `def spin_angular_function(self, kappa, m_j, theta, phi)` - *Spin-angular functions Omega_{kappa,m_j}(theta, phi).

These couple the orbital and spin degrees of freedom.*
- `__init__` (line 960) `def __init__(self, config, model_wrapper)`
- `sample_orbital` (line 966) `def sample_orbital(self, n, l, j, num_samples)` - *Sample points from a relativistic hydrogen orbital.*
- `__init__` (line 1079) `def __init__(self, config)`
- `visualize_orbital` (line 1082) `def visualize_orbital(self, data, save_path)` - *Visualize relativistic orbital.*
- `visualize_energy_spectrum` (line 1213) `def visualize_energy_spectrum(self, spectrum, save_path)` - *Visualize relativistic energy spectrum with fine structure.*
- `visualize_zitterbewegung` (line 1296) `def visualize_zitterbewegung(self, zbw_data, save_path)` - *Visualize Zitterbewegung oscillation.*
- `__init__` (line 1374) `def __init__(self, config)`
- `print_header` (line 1398) `def print_header(self)`
- `validate_fine_structure` (line 1419) `def validate_fine_structure(self)` - *Validate fine structure energy corrections.*
- `validate_zitterbewegung` (line 1477) `def validate_zitterbewegung(self)` - *Validate Zitterbewegung simulation.*
- `validate_energy_spectrum` (line 1509) `def validate_energy_spectrum(self)` - *Validate complete energy spectrum.*
- `validate_orbital` (line 1524) `def validate_orbital(self, orbital_name, num_samples)` - *Validate single orbital visualization.*
- `run_full_validation` (line 1541) `def run_full_validation(self)` - *Run complete validation suite.*
- `interactive_mode` (line 1575) `def interactive_mode(self)` - *Run in interactive mode.*

#### `visualize_lidar_csv2.py`
**Path:** `visualize_lidar_csv2.py`

**Functions:**
- `visualize_csv` (line 14) `def visualize_csv(csv_path, output_path, colormap)` - *Visualize a point cloud CSV file.

CSV must have columns: x, y, z (and optionally: intensity, range)*
- `main` (line 108) `def main()`

#### `weight_3d_standard.py`
**Path:** `weight_3d_standard.py`

**Classes:**
- `StandardWeightVisualizer` (line 41) `class StandardWeightVisualizer` - *Standard 3D weight visualization without LiDAR physics.

Each point represents:
- Per-layer mode: one layer (weights flattened)
- Per-neuron mode: one neuron/filter
- Sliding window mode: consecutive weight chunks*

**Functions:**
- `generate_standard_html` (line 264) `def generate_standard_html(coordinates, colors, labels, output_path, title, hover_data)` - *Generate interactive 3D visualization using Plotly.
Standard approach - simple and effective.*
- `generate_continuous_html` (line 415) `def generate_continuous_html(coordinates, color_values, output_path, title, colorbar_title)` - *Generate visualization with continuous color scale.*
- `main` (line 492) `def main()`
- `__init__` (line 51) `def __init__(self, max_samples, random_seed)`
- `load_checkpoint` (line 60) `def load_checkpoint(self, path)` - *Load PyTorch checkpoint.*
- `extract_weights_per_layer` (line 66) `def extract_weights_per_layer(self, checkpoint)` - *Extract weights organized by layer.

Returns:
    weights: List of flattened weight arrays per layer
    names: Layer names
    stats: Statistics per layer*
- `extract_weights_per_neuron` (line 115) `def extract_weights_per_neuron(self, checkpoint, max_neurons)` - *Extract weights organized per neuron/filter.

Each row = one neuron's incoming weights.*
- `extract_weights_sliding_window` (line 181) `def extract_weights_sliding_window(self, checkpoint, window_size, num_windows)` - *Extract weights using sliding window approach.

Each point = consecutive chunk of weights.*
- `reduce_dimensions` (line 221) `def reduce_dimensions(self, data, method, n_components)` - *Apply dimensionality reduction.*
- `_simple_projection` (line 254) `def _simple_projection(self, data)` - *Fallback projection without sklearn.*

#### `weight_space_lidar.py`
**Path:** `weight_space_lidar.py`

**Classes:**
- `WeightSpaceLiDARConfig` (line 72) `class WeightSpaceLiDARConfig` - *Master configuration for Weight Space LiDAR system.
All parameters are immutable and type-safe.*
- `ILogger` (line 167) `class ILogger(Protocol)` - *Protocol for logger implementations.*
- `LoggerFactory` (line 176) `class LoggerFactory` - *Factory for creating configured logger instances.*
- `IWeightExtractor` (line 194) `class IWeightExtractor(Protocol)` - *Protocol for weight extraction strategies.*
- `IDimensionalityReducer` (line 207) `class IDimensionalityReducer(Protocol)` - *Protocol for dimensionality reduction strategies.*
- `IRangeCalculator` (line 220) `class IRangeCalculator(Protocol)` - *Protocol for range calculation strategies.*
- `ITransmissionCalculator` (line 229) `class ITransmissionCalculator(Protocol)` - *Protocol for transmission calculation strategies.*
- `IPointCloudGenerator` (line 238) `class IPointCloudGenerator(Protocol)` - *Protocol for point cloud generation strategies.*
- `DefaultWeightExtractor` (line 251) `class DefaultWeightExtractor` - *Default implementation for weight extraction from PyTorch checkpoints.*
- `PCAReducer` (line 304) `class PCAReducer` - *PCA-based dimensionality reduction for weight space.*
- `TSNEReducer` (line 340) `class TSNEReducer` - *t-SNE based dimensionality reduction for weight space visualization.*
- `FrobeniusRangeCalculator` (line 374) `class FrobeniusRangeCalculator` - *Calculate range using Frobenius norm in weight space.*
- `BeerLambertTransmissionCalculator` (line 411) `class BeerLambertTransmissionCalculator` - *Calculate transmission using Beer-Lambert law adapted for weight space.*
- `HessianCurvatureEstimator` (line 442) `class HessianCurvatureEstimator` - *Estimate local curvature (backscatter coefficient) using Hessian approximation.*
- `LiDARPhysicsEngine` (line 517) `class LiDARPhysicsEngine` - *Core LiDAR physics engine adapted for weight space analysis.

Implements the full LiDAR equation:
P(r) = (E_L * c / 2) * A * [β_a * P_a + β_m * P_m] * exp(-2∫σ(r')dr') / r² + M(r) + b

Adapted for weight space where:
- E_L: laser energy -> probing intensity
- β: backscatter coefficient -> Hessian curvature
- σ: extinction coefficient -> gradient magnitude
- r: range -> weight space distance*
- `TemporalCheckpointScanner` (line 690) `class TemporalCheckpointScanner` - *Scanner for temporal evolution of checkpoints.
Provides 4D visualization (3D space + time) of weight space dynamics.*
- `PointCloudGenerator` (line 884) `class PointCloudGenerator` - *Generate point cloud representations of weight space.*
- `WeightSpaceNavigator` (line 1066) `class WeightSpaceNavigator` - *Main navigation interface for weight space LiDAR.
Provides high-level API for exploring neural network checkpoints.*
- `WeightSpaceLiDARCLI` (line 1447) `class WeightSpaceLiDARCLI` - *Command-line interface for Weight Space LiDAR.*

**Functions:**
- `main` (line 1707) `def main()` - *Entry point for Weight Space LiDAR CLI.*
- `debug` (line 170) `def debug(self, msg)`
- `info` (line 171) `def info(self, msg)`
- `warning` (line 172) `def warning(self, msg)`
- `error` (line 173) `def error(self, msg)`
- `create` (line 180) `def create(name, level)`
- `extract` (line 197) `def extract(self, checkpoint)` - *Extract weight vector from checkpoint.*
- `get_layer_names` (line 201) `def get_layer_names(self, checkpoint)` - *Get list of layer names from checkpoint.*
- `fit_transform` (line 210) `def fit_transform(self, data)` - *Fit and transform data to lower dimensions.*
- `transform` (line 214) `def transform(self, data)` - *Transform new data using fitted model.*
- `calculate` (line 223) `def calculate(self, origin, target)` - *Calculate range between two points in weight space.*
- `calculate` (line 232) `def calculate(self, path_integral, extinction)` - *Calculate transmission along a path.*
- `generate` (line 241) `def generate(self, weights, intensities, ranges)` - *Generate point cloud from weight data.*
- `__init__` (line 254) `def __init__(self, config)`
- `extract` (line 258) `def extract(self, checkpoint)`
- `get_layer_names` (line 272) `def get_layer_names(self, checkpoint)`
- `_resolve_state_dict` (line 279) `def _resolve_state_dict(self, checkpoint)`
- `_is_weight_tensor` (line 286) `def _is_weight_tensor(self, name, tensor)`
- `_flatten_and_sample` (line 292) `def _flatten_and_sample(self, tensor)`
- `__init__` (line 307) `def __init__(self, config)`
- `fit_transform` (line 312) `def fit_transform(self, data)`
- `transform` (line 329) `def transform(self, data)`
- `get_explained_variance` (line 334) `def get_explained_variance(self)`
- `__init__` (line 343) `def __init__(self, config)`
- `fit_transform` (line 348) `def fit_transform(self, data)`
- `transform` (line 370) `def transform(self, data)`
- `__init__` (line 377) `def __init__(self, config)`
- `calculate` (line 383) `def calculate(self, origin, target)`
- `calculate_batch` (line 395) `def calculate_batch(self, origin, targets)`
- `__init__` (line 414) `def __init__(self, config)`
- `calculate` (line 418) `def calculate(self, path_integral, extinction)`
- `calculate_optical_depth` (line 424) `def calculate_optical_depth(self, gradients, weights)`
- `__init__` (line 445) `def __init__(self, config)`
- `estimate` (line 450) `def estimate(self, weights, loss_fn)`
- `_estimate_hessian_diagonal` (line 473) `def _estimate_hessian_diagonal(self, weights, loss_fn)`
- `_numerical_hessian_diag` (line 483) `def _numerical_hessian_diag(self, weights, loss_fn)`
- `_empirical_curvature_estimate` (line 505) `def _empirical_curvature_estimate(self, weights)`
- `__init__` (line 531) `def __init__(self, config)`
- `compute_return_signal` (line 538) `def compute_return_signal(self, origin_weights, target_weights, hessian_estimate, gradient_integral)`
- `compute_point_cloud` (line 577) `def compute_point_cloud(self, origin_weights, weight_matrix, reduction_result)`
- `_compute_backscatter` (line 613) `def _compute_backscatter(self, hessian_estimate, range_value)`
- `_compute_geometric_factor` (line 626) `def _compute_geometric_factor(self, range_value)`
- `_compute_received_power` (line 633) `def _compute_received_power(self, backscatter, transmission, geometric_factor, range_value)`
- `_compute_intensity` (line 652) `def _compute_intensity(self, power_received, range_value)`
- `_compute_intensity_field` (line 668) `def _compute_intensity_field(self, weight_matrix, ranges, origin)`
- `__init__` (line 696) `def __init__(self, config)`
- `scan_directory` (line 702) `def scan_directory(self, checkpoint_dir, sort_by)`
- `_find_checkpoint_files` (line 735) `def _find_checkpoint_files(self, directory)`
- `_sort_checkpoints` (line 741) `def _sort_checkpoints(self, files, method)`
- `_extract_epoch` (line 755) `def _extract_epoch(self, filepath)`
- `_extract_temporal_weights` (line 765) `def _extract_temporal_weights(self, checkpoint_files)`
- `_load_checkpoint` (line 797) `def _load_checkpoint(self, filepath)`
- `_compute_temporal_signals` (line 803) `def _compute_temporal_signals(self, temporal_data)`
- `_compute_trajectories` (line 832) `def _compute_trajectories(self, temporal_data)`
- `_simple_trajectory` (line 862) `def _simple_trajectory(self, weights)`
- `__init__` (line 887) `def __init__(self, config)`
- `generate_from_checkpoint` (line 892) `def generate_from_checkpoint(self, checkpoint_path, reduction_method)`
- `generate_from_weights` (line 906) `def generate_from_weights(self, flat_weights, layer_weights, reduction_method)`
- `_load_checkpoint` (line 937) `def _load_checkpoint(self, path)`
- `_extract_layer_weights` (line 943) `def _extract_layer_weights(self, checkpoint)`
- `_create_weight_vectors` (line 957) `def _create_weight_vectors(self, flat_weights, layer_weights)`
- `_create_synthetic_points` (line 982) `def _create_synthetic_points(self, weights)`
- `_apply_reduction` (line 998) `def _apply_reduction(self, weight_vectors, method)`
- `_simple_projection` (line 1015) `def _simple_projection(self, vectors)`
- `_post_process` (line 1025) `def _post_process(self, point_cloud)`
- `_remove_outliers` (line 1035) `def _remove_outliers(self, point_cloud)`
- `_normalize_coordinates` (line 1052) `def _normalize_coordinates(self, point_cloud)`
- `__init__` (line 1072) `def __init__(self, config)`
- `scan_checkpoints` (line 1081) `def scan_checkpoints(self, checkpoint_dir, sort_by)`
- `generate_point_cloud` (line 1093) `def generate_point_cloud(self, checkpoint_path, reduction_method)`
- `compute_range_map` (line 1105) `def compute_range_map(self, checkpoint_path, reference_path)`
- `temporal_evolution` (line 1136) `def temporal_evolution(self, checkpoint_dir)`
- `export_point_cloud` (line 1158) `def export_point_cloud(self, point_cloud, output_path, format)`
- `visualize_3d` (line 1179) `def visualize_3d(self, point_cloud, title, save_path)`
- `visualize_temporal` (line 1219) `def visualize_temporal(self, evolution_data, title, save_path)`
- `_load_checkpoint` (line 1276) `def _load_checkpoint(self, path)`
- `_compute_layer_ranges` (line 1282) `def _compute_layer_ranges(self, checkpoint, origin)`
- `_compute_evolution_metrics` (line 1308) `def _compute_evolution_metrics(self, trajectories, signals, epochs)`
- `_export_las` (line 1343) `def _export_las(self, point_cloud, output_path)`
- `_export_ply` (line 1367) `def _export_ply(self, point_cloud, output_path)`
- `_export_csv` (line 1394) `def _export_csv(self, point_cloud, output_path)`
- `_export_json` (line 1419) `def _export_json(self, point_cloud, output_path)`
- `__init__` (line 1450) `def __init__(self)`
- `_create_parser` (line 1453) `def _create_parser(self)`
- `run` (line 1558) `def run(self, args)`
- `_handle_scan` (line 1585) `def _handle_scan(self, navigator, args)`
- `_handle_cloud` (line 1617) `def _handle_cloud(self, navigator, args)`
- `_handle_range` (line 1647) `def _handle_range(self, navigator, args)`
- `_handle_evolution` (line 1668) `def _handle_evolution(self, navigator, args)`

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
