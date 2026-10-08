# root

*Community 0 | 2 files | cohesion 1.00*

## Definition

This community groups 2 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `AdaptiveLambdaScheduler`, `AnnealingScheduler`, `BatchSizeProspector`, `CSVLogger`, `CheckpointManager`, `Config`, `CrystallizationPressureApplicator`, `CrystallographyMetricsCalculator`. Core file: `dirac_crystal2.py` (194 symbols). Documented purpose: Author: Gris Iscomeback Email: grisiscomeback@gmail.com Date of creation: 2026 License: AGPL v3  Description: Dirac Equation Grokking via Hamiltonian Topologica.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `dirac_crystal2.py` | py | utility | 194 | yes |
| `latent_space_visualizer.py` | py | utility | 46 | no |

## Key Symbols

- `Config` (class, `dirac_crystal2.py:52`) `class Config`
- `IPhaseDetector` (class, `dirac_crystal2.py:250`) `class IPhaseDetector(ABC)`
- `detect` (method, `dirac_crystal2.py:252`) `def detect(self, spectral_field)`
- `IMetricCalculator` (class, `dirac_crystal2.py:256`) `class IMetricCalculator(ABC)`
- `compute` (method, `dirac_crystal2.py:258`) `def compute(self, model)`
- `SeedManager` (class, `dirac_crystal2.py:262`) `class SeedManager`
- `set_seed` (method, `dirac_crystal2.py:264`) `def set_seed(seed, device)`
- `LoggerFactory` (class, `dirac_crystal2.py:274`) `class LoggerFactory`
- `create_logger` (method, `dirac_crystal2.py:276`) `def create_logger(name, level)`
- `GammaMatrices` (class, `dirac_crystal2.py:289`) `class GammaMatrices` - Dirac gamma matrices in various representations.
- `__init__` (method, `dirac_crystal2.py:294`) `def __init__(self, representation, device)`
- `_init_matrices` (method, `dirac_crystal2.py:299`) `def _init_matrices(self)`
- `to` (method, `dirac_crystal2.py:377`) `def to(self, device)`
- `DiracHamiltonianOperator` (class, `dirac_crystal2.py:383`) `class DiracHamiltonianOperator` - Dirac Hamiltonian operator for 4-component spinors.
- `__init__` (method, `dirac_crystal2.py:390`) `def __init__(self, config)`
- `_precompute_operators` (method, `dirac_crystal2.py:398`) `def _precompute_operators(self)`
- `apply_dirac_hamiltonian` (method, `dirac_crystal2.py:414`) `def apply_dirac_hamiltonian(self, spinor)` - Apply Dirac Hamiltonian to 4-component spinor.
- `time_evolution` (method, `dirac_crystal2.py:457`) `def time_evolution(self, spinor, dt)` - Time evolution of Dirac spinor using split-step method.
- `SpectralLayer` (class, `dirac_crystal2.py:481`) `class SpectralLayer(Module)`
- `__init__` (method, `dirac_crystal2.py:482`) `def __init__(self, channels, grid_size)`
- `forward` (method, `dirac_crystal2.py:493`) `def forward(self, x)`
- `DiracSpectralNetwork` (class, `dirac_crystal2.py:519`) `class DiracSpectralNetwork(Module)` - Neural network for learning Dirac equation dynamics.
- `__init__` (method, `dirac_crystal2.py:524`) `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers, sp`
- `forward` (method, `dirac_crystal2.py:547`) `def forward(self, x)`
- `HamiltonianBackbone` (class, `dirac_crystal2.py:558`) `class HamiltonianBackbone(Module)`
- `__init__` (method, `dirac_crystal2.py:559`) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (method, `dirac_crystal2.py:574`) `def forward(self, x)`
- `HamiltonianInferenceEngine` (class, `dirac_crystal2.py:585`) `class HamiltonianInferenceEngine`
- `__init__` (method, `dirac_crystal2.py:586`) `def __init__(self, config)`
- `_try_load_backbone` (method, `dirac_crystal2.py:593`) `def _try_load_backbone(self)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 1
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 1 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (root) and community 1 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `latent_space_visualizer.py`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `dirac_crystal2.py`
- `latent_space_visualizer.py`
