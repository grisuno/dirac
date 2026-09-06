# API

## dirac_crystal2.py

### main `def main()`
- Defined: `dirac_crystal2.py:3626`
- Imported by: `latent_space_visualizer.py`

### detect `def detect(self, spectral_field)`
- Defined: `dirac_crystal2.py:252`
- Imported by: `latent_space_visualizer.py`

### compute `def compute(self, model)`
- Defined: `dirac_crystal2.py:258`
- Imported by: `latent_space_visualizer.py`

### set_seed `def set_seed(seed, device)`
- Defined: `dirac_crystal2.py:264`
- Imported by: `latent_space_visualizer.py`

### create_logger `def create_logger(name, level)`
- Defined: `dirac_crystal2.py:276`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, representation, device)`
- Defined: `dirac_crystal2.py:294`
- Imported by: `latent_space_visualizer.py`

### _init_matrices `def _init_matrices(self)`
- Defined: `dirac_crystal2.py:299`
- Imported by: `latent_space_visualizer.py`

### to `def to(self, device)`
- Defined: `dirac_crystal2.py:377`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:390`
- Imported by: `latent_space_visualizer.py`

### _precompute_operators `def _precompute_operators(self)`
- Defined: `dirac_crystal2.py:398`
- Imported by: `latent_space_visualizer.py`

### apply_dirac_hamiltonian `def apply_dirac_hamiltonian(self, spinor)`
- Defined: `dirac_crystal2.py:414`
- Doc: Apply Dirac Hamiltonian to 4-component spinor.
- Imported by: `latent_space_visualizer.py`

### time_evolution `def time_evolution(self, spinor, dt)`
- Defined: `dirac_crystal2.py:457`
- Doc: Time evolution of Dirac spinor using split-step method.
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, channels, grid_size)`
- Defined: `dirac_crystal2.py:482`
- Imported by: `latent_space_visualizer.py`

### forward `def forward(self, x)`
- Defined: `dirac_crystal2.py:493`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers, spinor_components)`
- Defined: `dirac_crystal2.py:524`
- Imported by: `latent_space_visualizer.py`

### forward `def forward(self, x)`
- Defined: `dirac_crystal2.py:547`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- Defined: `dirac_crystal2.py:559`
- Imported by: `latent_space_visualizer.py`

### forward `def forward(self, x)`
- Defined: `dirac_crystal2.py:574`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:586`
- Imported by: `latent_space_visualizer.py`

### _try_load_backbone `def _try_load_backbone(self)`
- Defined: `dirac_crystal2.py:593`
- Imported by: `latent_space_visualizer.py`

### apply_hamiltonian `def apply_hamiltonian(self, spinor)`
- Defined: `dirac_crystal2.py:627`
- Doc: Apply Dirac Hamiltonian to 4-component spinor.
- Imported by: `latent_space_visualizer.py`

### time_evolve `def time_evolve(self, spinor, dt)`
- Defined: `dirac_crystal2.py:635`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:644`
- Imported by: `latent_space_visualizer.py`

### scalar_potential `def scalar_potential(self)`
- Defined: `dirac_crystal2.py:648`
- Doc: Scalar potential (couples equally to all components).
- Imported by: `latent_space_visualizer.py`

### vector_potential `def vector_potential(self)`
- Defined: `dirac_crystal2.py:660`
- Doc: Vector potential (time-component of 4-vector).
- Imported by: `latent_space_visualizer.py`

### magnetic_potential_2d `def magnetic_potential_2d(self)`
- Defined: `dirac_crystal2.py:672`
- Doc: Magnetic potential (spatial components of 4-vector).
- Imported by: `latent_space_visualizer.py`

### periodic_lattice_potential `def periodic_lattice_potential(self)`
- Defined: `dirac_crystal2.py:686`
- Imported by: `latent_space_visualizer.py`

### generate_mixed_potential `def generate_mixed_potential(self, seed)`
- Defined: `dirac_crystal2.py:692`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config, hamiltonian_engine, seed)`
- Defined: `dirac_crystal2.py:721`
- Imported by: `latent_space_visualizer.py`

### _generate_initial_spinor `def _generate_initial_spinor(self, potential, sample_seed)`
- Defined: `dirac_crystal2.py:766`
- Doc: Generate an initial Dirac spinor (4-component).
- Imported by: `latent_space_visualizer.py`

### _time_evolve_spinor `def _time_evolve_spinor(self, spinor, potential, energy)`
- Defined: `dirac_crystal2.py:798`
- Doc: Time evolve a Dirac spinor under the influence of potentials.
- Imported by: `latent_space_visualizer.py`

### _spinor_to_real_imag `def _spinor_to_real_imag(self, spinor)`
- Defined: `dirac_crystal2.py:824`
- Doc: Convert 4-component complex spinor to 8-channel real tensor.
- Imported by: `latent_space_visualizer.py`

### __len__ `def __len__(self)`
- Defined: `dirac_crystal2.py:835`
- Imported by: `latent_space_visualizer.py`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `dirac_crystal2.py:838`
- Imported by: `latent_space_visualizer.py`

### get_validation_batch `def get_validation_batch(self)`
- Defined: `dirac_crystal2.py:841`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:846`
- Imported by: `latent_space_visualizer.py`

### compute_full_spectrum `def compute_full_spectrum(self, spectral_field)`
- Defined: `dirac_crystal2.py:854`
- Imported by: `latent_space_visualizer.py`

### detect_bragg_peaks `def detect_bragg_peaks(self, power_spectrum, threshold_sigma)`
- Defined: `dirac_crystal2.py:926`
- Imported by: `latent_space_visualizer.py`

### compute_resonance_metrics `def compute_resonance_metrics(self, spectral_field)`
- Defined: `dirac_crystal2.py:980`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:1022`
- Imported by: `latent_space_visualizer.py`

### compute_mass_center `def compute_mass_center(self, spectral_field)`
- Defined: `dirac_crystal2.py:1030`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:1093`
- Imported by: `latent_space_visualizer.py`

### detect `def detect(self, spectral_field)`
- Defined: `dirac_crystal2.py:1100`
- Imported by: `latent_space_visualizer.py`

### extract `def extract(model, grid_size)`
- Defined: `dirac_crystal2.py:1163`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:1184`
- Imported by: `latent_space_visualizer.py`

### forward `def forward(self, phase_info, epoch)`
- Defined: `dirac_crystal2.py:1189`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:1222`
- Imported by: `latent_space_visualizer.py`

### apply `def apply(self, model, phase_info)`
- Defined: `dirac_crystal2.py:1226`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:1237`
- Imported by: `latent_space_visualizer.py`

### compute `def compute(self, model)`
- Defined: `dirac_crystal2.py:1244`
- Imported by: `latent_space_visualizer.py`

### apply_crystallization_pressure `def apply_crystallization_pressure(self, model, topo_metrics)`
- Defined: `dirac_crystal2.py:1277`
- Imported by: `latent_space_visualizer.py`

### _empty_metrics `def _empty_metrics()`
- Defined: `dirac_crystal2.py:1283`
- Imported by: `latent_space_visualizer.py`

### compute_local_complexity `def compute_local_complexity(weights, epsilon)`
- Defined: `dirac_crystal2.py:1301`
- Imported by: `latent_space_visualizer.py`

### compute_superposition `def compute_superposition(weights)`
- Defined: `dirac_crystal2.py:1318`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:1338`
- Imported by: `latent_space_visualizer.py`

### compute `def compute(self, model)`
- Defined: `dirac_crystal2.py:1342`
- Imported by: `latent_space_visualizer.py`

### compute_kappa `def compute_kappa(self, model, val_x, val_y, num_batches)`
- Defined: `dirac_crystal2.py:1347`
- Imported by: `latent_space_visualizer.py`

### compute_discretization_margin `def compute_discretization_margin(self, model)`
- Defined: `dirac_crystal2.py:1400`
- Imported by: `latent_space_visualizer.py`

### compute_alpha_purity `def compute_alpha_purity(self, model)`
- Defined: `dirac_crystal2.py:1408`
- Imported by: `latent_space_visualizer.py`

### compute_kappa_quantum `def compute_kappa_quantum(self, model)`
- Defined: `dirac_crystal2.py:1414`
- Imported by: `latent_space_visualizer.py`

### compute_poynting_vector `def compute_poynting_vector(self, model)`
- Defined: `dirac_crystal2.py:1435`
- Imported by: `latent_space_visualizer.py`

### compute_hbar_effective `def compute_hbar_effective(self, model, lambda_pressure)`
- Defined: `dirac_crystal2.py:1491`
- Imported by: `latent_space_visualizer.py`

### compute_all_metrics `def compute_all_metrics(self, model, val_x, val_y)`
- Defined: `dirac_crystal2.py:1500`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:1548`
- Imported by: `latent_space_visualizer.py`

### compute `def compute(self, model)`
- Defined: `dirac_crystal2.py:1551`
- Imported by: `latent_space_visualizer.py`

### compute_effective_temperature `def compute_effective_temperature(self, gradient_buffer, learning_rate)`
- Defined: `dirac_crystal2.py:1577`
- Imported by: `latent_space_visualizer.py`

### compute_specific_heat `def compute_specific_heat(self, loss_history, temp_history)`
- Defined: `dirac_crystal2.py:1601`
- Imported by: `latent_space_visualizer.py`

### compute_gibbs_free_energy `def compute_gibbs_free_energy(self, delta, alpha, temperature)`
- Defined: `dirac_crystal2.py:1615`
- Imported by: `latent_space_visualizer.py`

### compute_critical_temperature `def compute_critical_temperature(self, alpha)`
- Defined: `dirac_crystal2.py:1622`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:1627`
- Imported by: `latent_space_visualizer.py`

### compute `def compute(self, model)`
- Defined: `dirac_crystal2.py:1630`
- Imported by: `latent_space_visualizer.py`

### _compute_level_spacing_ratio `def _compute_level_spacing_ratio(self, spacings)`
- Defined: `dirac_crystal2.py:1667`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:1680`
- Imported by: `latent_space_visualizer.py`

### compute `def compute(self, model)`
- Defined: `dirac_crystal2.py:1683`
- Imported by: `latent_space_visualizer.py`

### _compute_ricci_scalar `def _compute_ricci_scalar(self, metric)`
- Defined: `dirac_crystal2.py:1702`
- Imported by: `latent_space_visualizer.py`

### _estimate_sectional_curvatures `def _estimate_sectional_curvatures(self, metric)`
- Defined: `dirac_crystal2.py:1710`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:1724`
- Imported by: `latent_space_visualizer.py`

### compute_ricci_scalar_fast `def compute_ricci_scalar_fast(self, model)`
- Defined: `dirac_crystal2.py:1732`
- Imported by: `latent_space_visualizer.py`

### compute_local_curvature `def compute_local_curvature(self, param)`
- Defined: `dirac_crystal2.py:1775`
- Imported by: `latent_space_visualizer.py`

### compute_anisotropy `def compute_anisotropy(self, model)`
- Defined: `dirac_crystal2.py:1786`
- Imported by: `latent_space_visualizer.py`

### compute_ricci_regularization_loss `def compute_ricci_regularization_loss(self, model)`
- Defined: `dirac_crystal2.py:1820`
- Imported by: `latent_space_visualizer.py`

### apply_ricci_flow_step `def apply_ricci_flow_step(self, model, lr)`
- Defined: `dirac_crystal2.py:1845`
- Imported by: `latent_space_visualizer.py`

### perform_perelman_surgery `def perform_perelman_surgery(self, model, ricci_scalar)`
- Defined: `dirac_crystal2.py:1887`
- Imported by: `latent_space_visualizer.py`

### compute_adaptive_lr_factor `def compute_adaptive_lr_factor(self, model)`
- Defined: `dirac_crystal2.py:1942`
- Imported by: `latent_space_visualizer.py`

### get_flow_metrics `def get_flow_metrics(self, model)`
- Defined: `dirac_crystal2.py:1962`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:1979`
- Imported by: `latent_space_visualizer.py`

### compute `def compute(self, model)`
- Defined: `dirac_crystal2.py:1982`
- Imported by: `latent_space_visualizer.py`

### compute_weight_diffraction `def compute_weight_diffraction(self, coeffs)`
- Defined: `dirac_crystal2.py:1986`
- Imported by: `latent_space_visualizer.py`

### _compute_spectral_entropy `def _compute_spectral_entropy(power_spectrum)`
- Defined: `dirac_crystal2.py:2006`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:2016`
- Imported by: `latent_space_visualizer.py`

### current_lambda `def current_lambda(self)`
- Defined: `dirac_crystal2.py:2025`
- Imported by: `latent_space_visualizer.py`

### step `def step(self, epoch)`
- Defined: `dirac_crystal2.py:2028`
- Imported by: `latent_space_visualizer.py`

### compute_regularization_loss `def compute_regularization_loss(self, model)`
- Defined: `dirac_crystal2.py:2037`
- Imported by: `latent_space_visualizer.py`

### set_lambda `def set_lambda(self, value)`
- Defined: `dirac_crystal2.py:2051`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:2056`
- Imported by: `latent_space_visualizer.py`

### step_adaptive `def step_adaptive(self, epoch, topo_phase_state)`
- Defined: `dirac_crystal2.py:2061`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:2077`
- Imported by: `latent_space_visualizer.py`

### current_lambda `def current_lambda(self)`
- Defined: `dirac_crystal2.py:2086`
- Imported by: `latent_space_visualizer.py`

### step `def step(self, epoch, improvement)`
- Defined: `dirac_crystal2.py:2089`
- Imported by: `latent_space_visualizer.py`

### compute_regularization_loss `def compute_regularization_loss(self, model)`
- Defined: `dirac_crystal2.py:2098`
- Imported by: `latent_space_visualizer.py`

### set_lambda `def set_lambda(self, value)`
- Defined: `dirac_crystal2.py:2112`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:2117`
- Imported by: `latent_space_visualizer.py`

### temperature `def temperature(self)`
- Defined: `dirac_crystal2.py:2125`
- Imported by: `latent_space_visualizer.py`

### step `def step(self)`
- Defined: `dirac_crystal2.py:2128`
- Imported by: `latent_space_visualizer.py`

### accept_perturbation `def accept_perturbation(self, delta_loss)`
- Defined: `dirac_crystal2.py:2134`
- Imported by: `latent_space_visualizer.py`

### should_restart `def should_restart(self, current_delta, best_delta)`
- Defined: `dirac_crystal2.py:2142`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:2147`
- Imported by: `latent_space_visualizer.py`

### step_adaptive `def step_adaptive(self, alignment_trend, resonance_score)`
- Defined: `dirac_crystal2.py:2151`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:2166`
- Imported by: `latent_space_visualizer.py`

### update_metrics `def update_metrics(self)`
- Defined: `dirac_crystal2.py:2196`
- Imported by: `latent_space_visualizer.py`

### compute_delta_slope `def compute_delta_slope(self)`
- Defined: `dirac_crystal2.py:2207`
- Imported by: `latent_space_visualizer.py`

### format_progress_bar `def format_progress_bar(self, epoch, total_epochs, phase)`
- Defined: `dirac_crystal2.py:2220`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config, checkpoint_dir)`
- Defined: `dirac_crystal2.py:2319`
- Imported by: `latent_space_visualizer.py`

### should_save_checkpoint `def should_save_checkpoint(self)`
- Defined: `dirac_crystal2.py:2328`
- Imported by: `latent_space_visualizer.py`

### save_checkpoint `def save_checkpoint(self, model, optimizer, epoch, metrics, phase, lambda_value, config_snapshot)`
- Defined: `dirac_crystal2.py:2333`
- Imported by: `latent_space_visualizer.py`

### load_latest_checkpoint `def load_latest_checkpoint(self)`
- Defined: `dirac_crystal2.py:2369`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:2377`
- Imported by: `latent_space_visualizer.py`

### _load_best_metrics `def _load_best_metrics(self)`
- Defined: `dirac_crystal2.py:2388`
- Imported by: `latent_space_visualizer.py`

### should_save `def should_save(self, current_delta, current_alpha, current_acc)`
- Defined: `dirac_crystal2.py:2407`
- Imported by: `latent_space_visualizer.py`

### save_checkpoint `def save_checkpoint(self, model, optimizer, epoch, metrics, lambda_value)`
- Defined: `dirac_crystal2.py:2418`
- Imported by: `latent_space_visualizer.py`

### load_checkpoint `def load_checkpoint(self, model, optimizer)`
- Defined: `dirac_crystal2.py:2462`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:2482`
- Imported by: `latent_space_visualizer.py`

### should_stop `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- Defined: `dirac_crystal2.py:2488`
- Imported by: `latent_space_visualizer.py`

### is_crystal_formed `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)`
- Defined: `dirac_crystal2.py:2523`
- Imported by: `latent_space_visualizer.py`

### check `def check(model)`
- Defined: `dirac_crystal2.py:2539`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystal2.py:2570`
- Imported by: `latent_space_visualizer.py`

### compute_weight_metrics `def compute_weight_metrics(self, model)`
- Defined: `dirac_crystal2.py:2583`
- Imported by: `latent_space_visualizer.py`

### compute_norm_conservation_error `def compute_norm_conservation_error(self, model, val_x)`
- Defined: `dirac_crystal2.py:2598`
- Imported by: `latent_space_visualizer.py`

### train_single_epoch `def train_single_epoch(self, model, optimizer, dataloader, epoch, lambda_scheduler, ricci_flow)`
- Defined: `dirac_crystal2.py:2611`
- Imported by: `latent_space_visualizer.py`

### validate `def validate(self, model, val_x, val_y)`
- Defined: `dirac_crystal2.py:2666`
- Imported by: `latent_space_visualizer.py`

### collect_all_metrics `def collect_all_metrics(self, model, monitor, val_x, val_y, lambda_scheduler, annealing_scheduler, current_lr, epoch)`
- Defined: `dirac_crystal2.py:2679`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config, hamiltonian_engine)`
- Defined: `dirac_crystal2.py:2765`
- Imported by: `latent_space_visualizer.py`

### prospect `def prospect(self)`
- Defined: `dirac_crystal2.py:2770`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config, hamiltonian_engine, batch_size)`
- Defined: `dirac_crystal2.py:2836`
- Imported by: `latent_space_visualizer.py`

### mine `def mine(self)`
- Defined: `dirac_crystal2.py:2847`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config, hamiltonian_engine, seed, batch_size)`
- Defined: `dirac_crystal2.py:2977`
- Imported by: `latent_space_visualizer.py`

### run_phase3_training `def run_phase3_training(self, start_epoch, model)`
- Defined: `dirac_crystal2.py:2990`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config, hamiltonian_engine, model, optimizer, monitor, seed, batch_size)`
- Defined: `dirac_crystal2.py:3110`
- Imported by: `latent_space_visualizer.py`

### run_phase4_refinement `def run_phase4_refinement(self, start_epoch)`
- Defined: `dirac_crystal2.py:3129`
- Imported by: `latent_space_visualizer.py`

### __init__ `def __init__(self, config, hamiltonian_engine, model, monitor, seed, batch_size)`
- Defined: `dirac_crystal2.py:3239`
- Imported by: `latent_space_visualizer.py`

### _detect_blocked_labyrinth `def _detect_blocked_labyrinth(self, spec_gap, anisotropy, resonance)`
- Defined: `dirac_crystal2.py:3268`
- Imported by: `latent_space_visualizer.py`

### _apply_flood_fill_pressure `def _apply_flood_fill_pressure(self, lambda_scheduler, epoch, spec_gap, anisotropy)`
- Defined: `dirac_crystal2.py:3280`
- Imported by: `latent_space_visualizer.py`

### _inject_diffusion_energy `def _inject_diffusion_energy(self)`
- Defined: `dirac_crystal2.py:3306`
- Imported by: `latent_space_visualizer.py`

### _find_ballistic_trajectory `def _find_ballistic_trajectory(self, resonance, anisotropy)`
- Defined: `dirac_crystal2.py:3317`
- Imported by: `latent_space_visualizer.py`

### _load_phase5_checkpoint `def _load_phase5_checkpoint(self, optimizer, lambda_scheduler)`
- Defined: `dirac_crystal2.py:3328`
- Imported by: `latent_space_visualizer.py`

### _apply_perelman_surgery `def _apply_perelman_surgery(self, lambda_scheduler, epoch, ricci_scalar)`
- Defined: `dirac_crystal2.py:3357`
- Imported by: `latent_space_visualizer.py`

### run_phase5_crystallization `def run_phase5_crystallization(self, start_epoch)`
- Defined: `dirac_crystal2.py:3384`
- Imported by: `latent_space_visualizer.py`

### load_latest_checkpoint `def load_latest_checkpoint(model, checkpoint_paths)`
- Defined: `dirac_crystal2.py:3761`
- Imported by: `latent_space_visualizer.py`

### safe_compute `def safe_compute(func)`
- Defined: `dirac_crystal2.py:1513`
- Imported by: `latent_space_visualizer.py`

### safe_get `def safe_get(key)`
- Defined: `dirac_crystal2.py:2224`
- Imported by: `latent_space_visualizer.py`

## dirac_crystallography_suite.py

### main `def main()`
- Defined: `dirac_crystallography_suite.py:1806`

### create_logger `def create_logger(name, level, config)`
- Defined: `dirac_crystallography_suite.py:182`

### compute `def compute(self, model)`
- Defined: `dirac_crystallography_suite.py:199`
- Doc: Compute metrics for the given model.

### detect `def detect(self, spectral_field)`
- Defined: `dirac_crystallography_suite.py:207`
- Doc: Detect phase from spectral field.

### __init__ `def __init__(self, representation, device, config)`
- Defined: `dirac_crystallography_suite.py:215`

### _init_matrices `def _init_matrices(self)`
- Defined: `dirac_crystallography_suite.py:221`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:266`

### _precompute_operators `def _precompute_operators(self)`
- Defined: `dirac_crystallography_suite.py:274`

### apply_dirac_hamiltonian `def apply_dirac_hamiltonian(self, spinor)`
- Defined: `dirac_crystallography_suite.py:290`
- Doc: Apply Dirac Hamiltonian to 4-component spinor.

### __init__ `def __init__(self, channels, grid_size, config)`
- Defined: `dirac_crystallography_suite.py:331`

### forward `def forward(self, x)`
- Defined: `dirac_crystallography_suite.py:343`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:366`

### forward `def forward(self, x)`
- Defined: `dirac_crystallography_suite.py:383`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:397`

### compute `def compute(self, model)`
- Defined: `dirac_crystallography_suite.py:400`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:436`

### compute `def compute(self, model)`
- Defined: `dirac_crystallography_suite.py:439`

### _compute_spectral_entropy `def _compute_spectral_entropy(self, weights)`
- Defined: `dirac_crystallography_suite.py:466`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:484`

### compute `def compute(self, model)`
- Defined: `dirac_crystallography_suite.py:487`

### _compute_level_spacing_ratio `def _compute_level_spacing_ratio(self, spacings)`
- Defined: `dirac_crystallography_suite.py:524`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:539`

### compute `def compute(self, model)`
- Defined: `dirac_crystallography_suite.py:542`

### _compute_ricci_scalar `def _compute_ricci_scalar(self, metric)`
- Defined: `dirac_crystallography_suite.py:559`

### _estimate_sectional_curvatures `def _estimate_sectional_curvatures(self, metric, samples)`
- Defined: `dirac_crystallography_suite.py:568`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:584`

### load_checkpoints `def load_checkpoints(self, checkpoint_dir)`
- Defined: `dirac_crystallography_suite.py:588`

### _extract_epoch `def _extract_epoch(self, filepath)`
- Defined: `dirac_crystallography_suite.py:607`

### flatten_kernel_params `def flatten_kernel_params(self, state_dict)`
- Defined: `dirac_crystallography_suite.py:611`

### compute_berry_connection_discrete `def compute_berry_connection_discrete(self, theta_prev, theta_curr)`
- Defined: `dirac_crystallography_suite.py:635`

### calculate_berry_phase `def calculate_berry_phase(self, checkpoint_dir)`
- Defined: `dirac_crystallography_suite.py:650`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:692`

### extract_state_space `def extract_state_space(self, model)`
- Defined: `dirac_crystallography_suite.py:695`

### analyze_stability `def analyze_stability(self, A)`
- Defined: `dirac_crystallography_suite.py:755`

### compute `def compute(self, model)`
- Defined: `dirac_crystallography_suite.py:772`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:784`

### compute `def compute(self, model)`
- Defined: `dirac_crystallography_suite.py:787`

### _classify_phase `def _classify_phase(self, delta, kappa, temp, alpha)`
- Defined: `dirac_crystallography_suite.py:812`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:829`

### compute_full_spectrum `def compute_full_spectrum(self, spectral_field)`
- Defined: `dirac_crystallography_suite.py:837`

### compute_resonance_metrics `def compute_resonance_metrics(self, spectral_field)`
- Defined: `dirac_crystallography_suite.py:867`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:885`

### compute_mass_center `def compute_mass_center(self, spectral_field)`
- Defined: `dirac_crystallography_suite.py:893`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:936`

### detect `def detect(self, spectral_field)`
- Defined: `dirac_crystallography_suite.py:943`

### extract `def extract(model, grid_size)`
- Defined: `dirac_crystallography_suite.py:975`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:996`

### compute `def compute(self, model)`
- Defined: `dirac_crystallography_suite.py:1001`

### _empty_metrics `def _empty_metrics()`
- Defined: `dirac_crystallography_suite.py:1022`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:1037`

### compute `def compute(self, model)`
- Defined: `dirac_crystallography_suite.py:1040`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:1130`

### extract_compressed_wavefunction `def extract_compressed_wavefunction(self, model)`
- Defined: `dirac_crystallography_suite.py:1135`

### _compress_johnson_lindenstrauss `def _compress_johnson_lindenstrauss(self, vector)`
- Defined: `dirac_crystallography_suite.py:1156`

### compute `def compute(self, model)`
- Defined: `dirac_crystallography_suite.py:1168`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:1192`

### visualize_checkpoint_analysis `def visualize_checkpoint_analysis(self, results, output_path)`
- Defined: `dirac_crystallography_suite.py:1195`

### _plot_weight_distribution `def _plot_weight_distribution(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1222`

### _plot_spectral_analysis `def _plot_spectral_analysis(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1233`

### _plot_phase_diagram `def _plot_phase_diagram(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1247`

### _plot_curvature_distribution `def _plot_curvature_distribution(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1264`

### _plot_level_spacing `def _plot_level_spacing(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1278`

### _plot_eigenvalue_spectrum `def _plot_eigenvalue_spectrum(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1292`

### _plot_thermodynamic_potentials `def _plot_thermodynamic_potentials(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1306`

### _plot_topological_metrics `def _plot_topological_metrics(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1320`

### _plot_berry_phase `def _plot_berry_phase(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1334`

### _plot_control_stability `def _plot_control_stability(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1354`

### _plot_quantum_metrics `def _plot_quantum_metrics(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1366`

### _plot_summary_table `def _plot_summary_table(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1380`

### _plot_layer_deltas `def _plot_layer_deltas(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1403`

### _plot_resonance_metrics `def _plot_resonance_metrics(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1415`

### _plot_spectral_concentration `def _plot_spectral_concentration(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1428`

### _plot_health_score `def _plot_health_score(self, results, ax)`
- Defined: `dirac_crystallography_suite.py:1439`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:1454`

### analyze_checkpoint `def analyze_checkpoint(self, checkpoint_path, val_data)`
- Defined: `dirac_crystallography_suite.py:1470`

### _compute_health_score `def _compute_health_score(self, results)`
- Defined: `dirac_crystallography_suite.py:1546`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:1579`

### process_directory `def process_directory(self, checkpoint_dir, output_dir, val_data)`
- Defined: `dirac_crystallography_suite.py:1585`

### _generate_summary `def _generate_summary(self, all_results)`
- Defined: `dirac_crystallography_suite.py:1631`

### _generate_evolution_plots `def _generate_evolution_plots(self, all_results, output_dir)`
- Defined: `dirac_crystallography_suite.py:1682`

### __init__ `def __init__(self, config)`
- Defined: `dirac_crystallography_suite.py:1726`

### run_analysis `def run_analysis(self, checkpoint_dir, output_dir)`
- Defined: `dirac_crystallography_suite.py:1732`

### _generate_berry_phase_visualization `def _generate_berry_phase_visualization(self, berry_results, output_dir)`
- Defined: `dirac_crystallography_suite.py:1756`

## latent_space_visualizer.py

### main `def main()`
- Defined: `latent_space_visualizer.py:766`
- Depends on: `dirac_crystal2.py`

### __init__ `def __init__(self, config)`
- Defined: `latent_space_visualizer.py:80`
- Depends on: `dirac_crystal2.py`

### log `def log(self, metrics)`
- Defined: `latent_space_visualizer.py:93`
- Depends on: `dirac_crystal2.py`

### _flatten_dict `def _flatten_dict(self, d, parent_key, sep)`
- Defined: `latent_space_visualizer.py:107`
- Depends on: `dirac_crystal2.py`

### _flush `def _flush(self)`
- Defined: `latent_space_visualizer.py:122`
- Depends on: `dirac_crystal2.py`

### close `def close(self)`
- Defined: `latent_space_visualizer.py:129`
- Depends on: `dirac_crystal2.py`

### get_csv_path `def get_csv_path(self)`
- Defined: `latent_space_visualizer.py:135`
- Depends on: `dirac_crystal2.py`

### __init__ `def __init__(self, config, parent)`
- Defined: `latent_space_visualizer.py:140`
- Depends on: `dirac_crystal2.py`

### update_data `def update_data(self, weights, metric_value)`
- Defined: `latent_space_visualizer.py:154`
- Depends on: `dirac_crystal2.py`

### clear `def clear(self)`
- Defined: `latent_space_visualizer.py:192`
- Depends on: `dirac_crystal2.py`

### __init__ `def __init__(self, config, parent)`
- Defined: `latent_space_visualizer.py:204`
- Depends on: `dirac_crystal2.py`

### _setup_axes `def _setup_axes(self)`
- Defined: `latent_space_visualizer.py:215`
- Depends on: `dirac_crystal2.py`

### update_data `def update_data(self, metrics)`
- Defined: `latent_space_visualizer.py:224`
- Depends on: `dirac_crystal2.py`

### clear `def clear(self)`
- Defined: `latent_space_visualizer.py:266`
- Depends on: `dirac_crystal2.py`

### __init__ `def __init__(self, config, parent)`
- Defined: `latent_space_visualizer.py:277`
- Depends on: `dirac_crystal2.py`

### update_data `def update_data(self, weights, gradients)`
- Defined: `latent_space_visualizer.py:287`
- Depends on: `dirac_crystal2.py`

### _reshape `def _reshape(self, arr)`
- Defined: `latent_space_visualizer.py:303`
- Depends on: `dirac_crystal2.py`

### clear `def clear(self)`
- Defined: `latent_space_visualizer.py:313`
- Depends on: `dirac_crystal2.py`

### __init__ `def __init__(self, config, dirac_config)`
- Defined: `latent_space_visualizer.py:327`
- Depends on: `dirac_crystal2.py`

### setup `def setup(self)`
- Defined: `latent_space_visualizer.py:345`
- Depends on: `dirac_crystal2.py`

### run `def run(self)`
- Defined: `latent_space_visualizer.py:385`
- Depends on: `dirac_crystal2.py`

### _train_epoch `def _train_epoch(self)`
- Defined: `latent_space_visualizer.py:417`
- Depends on: `dirac_crystal2.py`

### _validate `def _validate(self)`
- Defined: `latent_space_visualizer.py:437`
- Depends on: `dirac_crystal2.py`

### _compute_metrics `def _compute_metrics(self, epoch, train_loss, val_loss, val_acc)`
- Defined: `latent_space_visualizer.py:446`
- Depends on: `dirac_crystal2.py`

### _extract_weights `def _extract_weights(self)`
- Defined: `latent_space_visualizer.py:503`
- Depends on: `dirac_crystal2.py`

### _extract_gradients `def _extract_gradients(self)`
- Defined: `latent_space_visualizer.py:513`
- Depends on: `dirac_crystal2.py`

### stop `def stop(self)`
- Defined: `latent_space_visualizer.py:523`
- Depends on: `dirac_crystal2.py`

### pause `def pause(self)`
- Defined: `latent_space_visualizer.py:526`
- Depends on: `dirac_crystal2.py`

### resume `def resume(self)`
- Defined: `latent_space_visualizer.py:529`
- Depends on: `dirac_crystal2.py`

### __init__ `def __init__(self, config)`
- Defined: `latent_space_visualizer.py:534`
- Depends on: `dirac_crystal2.py`

### _setup_ui `def _setup_ui(self)`
- Defined: `latent_space_visualizer.py:546`
- Depends on: `dirac_crystal2.py`

### _log_msg `def _log_msg(self, msg)`
- Defined: `latent_space_visualizer.py:660`
- Depends on: `dirac_crystal2.py`

### _start `def _start(self)`
- Defined: `latent_space_visualizer.py:664`
- Depends on: `dirac_crystal2.py`

### _pause `def _pause(self)`
- Defined: `latent_space_visualizer.py:692`
- Depends on: `dirac_crystal2.py`

### _stop `def _stop(self)`
- Defined: `latent_space_visualizer.py:697`
- Depends on: `dirac_crystal2.py`

### _clear `def _clear(self)`
- Defined: `latent_space_visualizer.py:705`
- Depends on: `dirac_crystal2.py`

### _on_progress `def _on_progress(self, metrics)`
- Defined: `latent_space_visualizer.py:712`
- Depends on: `dirac_crystal2.py`

### _on_finished `def _on_finished(self)`
- Defined: `latent_space_visualizer.py:749`
- Depends on: `dirac_crystal2.py`

### closeEvent `def closeEvent(self, e)`
- Defined: `latent_space_visualizer.py:760`
- Depends on: `dirac_crystal2.py`

## lidar_interactive_viewer.py

### load_csv_point_cloud `def load_csv_point_cloud(csv_path)`
- Defined: `lidar_interactive_viewer.py:22`
- Doc: Load point cloud from CSV file.

### generate_interactive_html `def generate_interactive_html(points, attributes, output_path, title, point_size, colormap, intensity_col)`
- Defined: `lidar_interactive_viewer.py:64`
- Doc: Generate interactive HTML using Three.js for 3D navigation.

### generate_plotly_html `def generate_plotly_html(points, attributes, output_path, title)`
- Defined: `lidar_interactive_viewer.py:581`
- Doc: Generate interactive HTML using Plotly.js (alternative viewer).

### main `def main()`
- Defined: `lidar_interactive_viewer.py:675`

## relativistic_hydrogen.py

### main `def main()`
- Defined: `relativistic_hydrogen.py:1613`

### create_logger `def create_logger(name, level)`
- Defined: `relativistic_hydrogen.py:97`

### __init__ `def __init__(self, device)`
- Defined: `relativistic_hydrogen.py:118`

### _init_matrices `def _init_matrices(self)`
- Defined: `relativistic_hydrogen.py:122`

### __init__ `def __init__(self, config)`
- Defined: `relativistic_hydrogen.py:207`

### _precompute_operators `def _precompute_operators(self)`
- Defined: `relativistic_hydrogen.py:215`

### apply_dirac_hamiltonian `def apply_dirac_hamiltonian(self, spinor, potential)`
- Defined: `relativistic_hydrogen.py:223`
- Doc: Apply Dirac Hamiltonian to 4-component spinor.

### time_evolution `def time_evolution(self, spinor, dt, potential)`
- Defined: `relativistic_hydrogen.py:282`
- Doc: Time evolution of Dirac spinor using first-order split-step.

### __init__ `def __init__(self, channels, grid_size)`
- Defined: `relativistic_hydrogen.py:311`

### forward `def forward(self, x)`
- Defined: `relativistic_hydrogen.py:322`

### __init__ `def __init__(self, grid_size, hidden_dim, expansion_dim, num_spectral_layers, spinor_components)`
- Defined: `relativistic_hydrogen.py:356`

### forward `def forward(self, x)`
- Defined: `relativistic_hydrogen.py:379`

### __init__ `def __init__(self, config)`
- Defined: `relativistic_hydrogen.py:397`

### _find_best_checkpoint `def _find_best_checkpoint(self)`
- Defined: `relativistic_hydrogen.py:406`

### _load_model `def _load_model(self)`
- Defined: `relativistic_hydrogen.py:452`

### apply_hamiltonian `def apply_hamiltonian(self, spinor, potential)`
- Defined: `relativistic_hydrogen.py:498`
- Doc: Apply Hamiltonian using analytical operator.

### evolve_spinor `def evolve_spinor(self, spinor, dt, potential)`
- Defined: `relativistic_hydrogen.py:506`
- Doc: Evolve spinor in time using the analytical Dirac operator.

### __init__ `def __init__(self, config)`
- Defined: `relativistic_hydrogen.py:521`

### energy_level_dirac `def energy_level_dirac(self, n, kappa)`
- Defined: `relativistic_hydrogen.py:526`
- Doc: Exact Dirac energy level for hydrogen-like atom.

### fine_structure_splitting `def fine_structure_splitting(self, n, l)`
- Defined: `relativistic_hydrogen.py:556`
- Doc: Calculate fine structure splitting for given n, l.

### energy_spectrum `def energy_spectrum(self, n_max)`
- Defined: `relativistic_hydrogen.py:597`
- Doc: Generate relativistic energy spectrum up to n_max.

### __init__ `def __init__(self, config, model_wrapper)`
- Defined: `relativistic_hydrogen.py:650`

### create_gaussian_wave_packet `def create_gaussian_wave_packet(self, sigma, momentum)`
- Defined: `relativistic_hydrogen.py:657`
- Doc: Create a Gaussian wave packet for a free particle.

### compute_position_expectation `def compute_position_expectation(self, spinor)`
- Defined: `relativistic_hydrogen.py:702`
- Doc: Compute expectation value of position operator.

### compute_velocity_expectation `def compute_velocity_expectation(self, spinor)`
- Defined: `relativistic_hydrogen.py:724`
- Doc: Compute expectation value of velocity operator.

### simulate `def simulate(self, duration, dt, sigma)`
- Defined: `relativistic_hydrogen.py:750`
- Doc: Run Zitterbewegung simulation.

### __init__ `def __init__(self, config)`
- Defined: `relativistic_hydrogen.py:837`

### radial_wavefunction_schrodinger `def radial_wavefunction_schrodinger(n, l, r)`
- Defined: `relativistic_hydrogen.py:843`
- Doc: Non-relativistic radial wavefunction for comparison.

### radial_wavefunction_dirac `def radial_wavefunction_dirac(self, n, kappa, r, Z)`
- Defined: `relativistic_hydrogen.py:853`
- Doc: Relativistic radial wavefunctions for hydrogen.

### spherical_harmonic_real `def spherical_harmonic_real(self, l, m, theta, phi)`
- Defined: `relativistic_hydrogen.py:900`
- Doc: Real spherical harmonics.

### spin_angular_function `def spin_angular_function(self, kappa, m_j, theta, phi)`
- Defined: `relativistic_hydrogen.py:910`
- Doc: Spin-angular functions Omega_{kappa,m_j}(theta, phi).

### __init__ `def __init__(self, config, model_wrapper)`
- Defined: `relativistic_hydrogen.py:960`

### sample_orbital `def sample_orbital(self, n, l, j, num_samples)`
- Defined: `relativistic_hydrogen.py:966`
- Doc: Sample points from a relativistic hydrogen orbital.

### __init__ `def __init__(self, config)`
- Defined: `relativistic_hydrogen.py:1079`

### visualize_orbital `def visualize_orbital(self, data, save_path)`
- Defined: `relativistic_hydrogen.py:1082`
- Doc: Visualize relativistic orbital.

### visualize_energy_spectrum `def visualize_energy_spectrum(self, spectrum, save_path)`
- Defined: `relativistic_hydrogen.py:1213`
- Doc: Visualize relativistic energy spectrum with fine structure.

### visualize_zitterbewegung `def visualize_zitterbewegung(self, zbw_data, save_path)`
- Defined: `relativistic_hydrogen.py:1296`
- Doc: Visualize Zitterbewegung oscillation.

### __init__ `def __init__(self, config)`
- Defined: `relativistic_hydrogen.py:1374`

### print_header `def print_header(self)`
- Defined: `relativistic_hydrogen.py:1398`

### validate_fine_structure `def validate_fine_structure(self)`
- Defined: `relativistic_hydrogen.py:1419`
- Doc: Validate fine structure energy corrections.

### validate_zitterbewegung `def validate_zitterbewegung(self)`
- Defined: `relativistic_hydrogen.py:1477`
- Doc: Validate Zitterbewegung simulation.

### validate_energy_spectrum `def validate_energy_spectrum(self)`
- Defined: `relativistic_hydrogen.py:1509`
- Doc: Validate complete energy spectrum.

### validate_orbital `def validate_orbital(self, orbital_name, num_samples)`
- Defined: `relativistic_hydrogen.py:1524`
- Doc: Validate single orbital visualization.

### run_full_validation `def run_full_validation(self)`
- Defined: `relativistic_hydrogen.py:1541`
- Doc: Run complete validation suite.

### interactive_mode `def interactive_mode(self)`
- Defined: `relativistic_hydrogen.py:1575`
- Doc: Run in interactive mode.

## visualize_lidar_csv2.py

### visualize_csv `def visualize_csv(csv_path, output_path, colormap)`
- Defined: `visualize_lidar_csv2.py:14`
- Doc: Visualize a point cloud CSV file.

### main `def main()`
- Defined: `visualize_lidar_csv2.py:108`

## weight_3d_standard.py

### generate_standard_html `def generate_standard_html(coordinates, colors, labels, output_path, title, hover_data)`
- Defined: `weight_3d_standard.py:264`
- Doc: Generate interactive 3D visualization using Plotly.

### generate_continuous_html `def generate_continuous_html(coordinates, color_values, output_path, title, colorbar_title)`
- Defined: `weight_3d_standard.py:415`
- Doc: Generate visualization with continuous color scale.

### main `def main()`
- Defined: `weight_3d_standard.py:492`

### __init__ `def __init__(self, max_samples, random_seed)`
- Defined: `weight_3d_standard.py:51`

### load_checkpoint `def load_checkpoint(self, path)`
- Defined: `weight_3d_standard.py:60`
- Doc: Load PyTorch checkpoint.

### extract_weights_per_layer `def extract_weights_per_layer(self, checkpoint)`
- Defined: `weight_3d_standard.py:66`
- Doc: Extract weights organized by layer.

### extract_weights_per_neuron `def extract_weights_per_neuron(self, checkpoint, max_neurons)`
- Defined: `weight_3d_standard.py:115`
- Doc: Extract weights organized per neuron/filter.

### extract_weights_sliding_window `def extract_weights_sliding_window(self, checkpoint, window_size, num_windows)`
- Defined: `weight_3d_standard.py:181`
- Doc: Extract weights using sliding window approach.

### reduce_dimensions `def reduce_dimensions(self, data, method, n_components)`
- Defined: `weight_3d_standard.py:221`
- Doc: Apply dimensionality reduction.

### _simple_projection `def _simple_projection(self, data)`
- Defined: `weight_3d_standard.py:254`
- Doc: Fallback projection without sklearn.

## weight_space_lidar.py

### main `def main()`
- Defined: `weight_space_lidar.py:1707`
- Doc: Entry point for Weight Space LiDAR CLI.

### debug `def debug(self, msg)`
- Defined: `weight_space_lidar.py:170`

### info `def info(self, msg)`
- Defined: `weight_space_lidar.py:171`

### warning `def warning(self, msg)`
- Defined: `weight_space_lidar.py:172`

### error `def error(self, msg)`
- Defined: `weight_space_lidar.py:173`

### create `def create(name, level)`
- Defined: `weight_space_lidar.py:180`

### extract `def extract(self, checkpoint)`
- Defined: `weight_space_lidar.py:197`
- Doc: Extract weight vector from checkpoint.

### get_layer_names `def get_layer_names(self, checkpoint)`
- Defined: `weight_space_lidar.py:201`
- Doc: Get list of layer names from checkpoint.

### fit_transform `def fit_transform(self, data)`
- Defined: `weight_space_lidar.py:210`
- Doc: Fit and transform data to lower dimensions.

### transform `def transform(self, data)`
- Defined: `weight_space_lidar.py:214`
- Doc: Transform new data using fitted model.

### calculate `def calculate(self, origin, target)`
- Defined: `weight_space_lidar.py:223`
- Doc: Calculate range between two points in weight space.

### calculate `def calculate(self, path_integral, extinction)`
- Defined: `weight_space_lidar.py:232`
- Doc: Calculate transmission along a path.

### generate `def generate(self, weights, intensities, ranges)`
- Defined: `weight_space_lidar.py:241`
- Doc: Generate point cloud from weight data.

### __init__ `def __init__(self, config)`
- Defined: `weight_space_lidar.py:254`

### extract `def extract(self, checkpoint)`
- Defined: `weight_space_lidar.py:258`

### get_layer_names `def get_layer_names(self, checkpoint)`
- Defined: `weight_space_lidar.py:272`

### _resolve_state_dict `def _resolve_state_dict(self, checkpoint)`
- Defined: `weight_space_lidar.py:279`

### _is_weight_tensor `def _is_weight_tensor(self, name, tensor)`
- Defined: `weight_space_lidar.py:286`

### _flatten_and_sample `def _flatten_and_sample(self, tensor)`
- Defined: `weight_space_lidar.py:292`

### __init__ `def __init__(self, config)`
- Defined: `weight_space_lidar.py:307`

### fit_transform `def fit_transform(self, data)`
- Defined: `weight_space_lidar.py:312`

### transform `def transform(self, data)`
- Defined: `weight_space_lidar.py:329`

### get_explained_variance `def get_explained_variance(self)`
- Defined: `weight_space_lidar.py:334`

### __init__ `def __init__(self, config)`
- Defined: `weight_space_lidar.py:343`

### fit_transform `def fit_transform(self, data)`
- Defined: `weight_space_lidar.py:348`

### transform `def transform(self, data)`
- Defined: `weight_space_lidar.py:370`

### __init__ `def __init__(self, config)`
- Defined: `weight_space_lidar.py:377`

### calculate `def calculate(self, origin, target)`
- Defined: `weight_space_lidar.py:383`

### calculate_batch `def calculate_batch(self, origin, targets)`
- Defined: `weight_space_lidar.py:395`

### __init__ `def __init__(self, config)`
- Defined: `weight_space_lidar.py:414`

### calculate `def calculate(self, path_integral, extinction)`
- Defined: `weight_space_lidar.py:418`

### calculate_optical_depth `def calculate_optical_depth(self, gradients, weights)`
- Defined: `weight_space_lidar.py:424`

### __init__ `def __init__(self, config)`
- Defined: `weight_space_lidar.py:445`

### estimate `def estimate(self, weights, loss_fn)`
- Defined: `weight_space_lidar.py:450`

### _estimate_hessian_diagonal `def _estimate_hessian_diagonal(self, weights, loss_fn)`
- Defined: `weight_space_lidar.py:473`

### _numerical_hessian_diag `def _numerical_hessian_diag(self, weights, loss_fn)`
- Defined: `weight_space_lidar.py:483`

### _empirical_curvature_estimate `def _empirical_curvature_estimate(self, weights)`
- Defined: `weight_space_lidar.py:505`

### __init__ `def __init__(self, config)`
- Defined: `weight_space_lidar.py:531`

### compute_return_signal `def compute_return_signal(self, origin_weights, target_weights, hessian_estimate, gradient_integral)`
- Defined: `weight_space_lidar.py:538`

### compute_point_cloud `def compute_point_cloud(self, origin_weights, weight_matrix, reduction_result)`
- Defined: `weight_space_lidar.py:577`

### _compute_backscatter `def _compute_backscatter(self, hessian_estimate, range_value)`
- Defined: `weight_space_lidar.py:613`

### _compute_geometric_factor `def _compute_geometric_factor(self, range_value)`
- Defined: `weight_space_lidar.py:626`

### _compute_received_power `def _compute_received_power(self, backscatter, transmission, geometric_factor, range_value)`
- Defined: `weight_space_lidar.py:633`

### _compute_intensity `def _compute_intensity(self, power_received, range_value)`
- Defined: `weight_space_lidar.py:652`

### _compute_intensity_field `def _compute_intensity_field(self, weight_matrix, ranges, origin)`
- Defined: `weight_space_lidar.py:668`

### __init__ `def __init__(self, config)`
- Defined: `weight_space_lidar.py:696`

### scan_directory `def scan_directory(self, checkpoint_dir, sort_by)`
- Defined: `weight_space_lidar.py:702`

### _find_checkpoint_files `def _find_checkpoint_files(self, directory)`
- Defined: `weight_space_lidar.py:735`

### _sort_checkpoints `def _sort_checkpoints(self, files, method)`
- Defined: `weight_space_lidar.py:741`

### _extract_epoch `def _extract_epoch(self, filepath)`
- Defined: `weight_space_lidar.py:755`

### _extract_temporal_weights `def _extract_temporal_weights(self, checkpoint_files)`
- Defined: `weight_space_lidar.py:765`

### _load_checkpoint `def _load_checkpoint(self, filepath)`
- Defined: `weight_space_lidar.py:797`

### _compute_temporal_signals `def _compute_temporal_signals(self, temporal_data)`
- Defined: `weight_space_lidar.py:803`

### _compute_trajectories `def _compute_trajectories(self, temporal_data)`
- Defined: `weight_space_lidar.py:832`

### _simple_trajectory `def _simple_trajectory(self, weights)`
- Defined: `weight_space_lidar.py:862`

### __init__ `def __init__(self, config)`
- Defined: `weight_space_lidar.py:887`

### generate_from_checkpoint `def generate_from_checkpoint(self, checkpoint_path, reduction_method)`
- Defined: `weight_space_lidar.py:892`

### generate_from_weights `def generate_from_weights(self, flat_weights, layer_weights, reduction_method)`
- Defined: `weight_space_lidar.py:906`

### _load_checkpoint `def _load_checkpoint(self, path)`
- Defined: `weight_space_lidar.py:937`

### _extract_layer_weights `def _extract_layer_weights(self, checkpoint)`
- Defined: `weight_space_lidar.py:943`

### _create_weight_vectors `def _create_weight_vectors(self, flat_weights, layer_weights)`
- Defined: `weight_space_lidar.py:957`

### _create_synthetic_points `def _create_synthetic_points(self, weights)`
- Defined: `weight_space_lidar.py:982`

### _apply_reduction `def _apply_reduction(self, weight_vectors, method)`
- Defined: `weight_space_lidar.py:998`

### _simple_projection `def _simple_projection(self, vectors)`
- Defined: `weight_space_lidar.py:1015`

### _post_process `def _post_process(self, point_cloud)`
- Defined: `weight_space_lidar.py:1025`

### _remove_outliers `def _remove_outliers(self, point_cloud)`
- Defined: `weight_space_lidar.py:1035`

### _normalize_coordinates `def _normalize_coordinates(self, point_cloud)`
- Defined: `weight_space_lidar.py:1052`

### __init__ `def __init__(self, config)`
- Defined: `weight_space_lidar.py:1072`

### scan_checkpoints `def scan_checkpoints(self, checkpoint_dir, sort_by)`
- Defined: `weight_space_lidar.py:1081`

### generate_point_cloud `def generate_point_cloud(self, checkpoint_path, reduction_method)`
- Defined: `weight_space_lidar.py:1093`

### compute_range_map `def compute_range_map(self, checkpoint_path, reference_path)`
- Defined: `weight_space_lidar.py:1105`

### temporal_evolution `def temporal_evolution(self, checkpoint_dir)`
- Defined: `weight_space_lidar.py:1136`

### export_point_cloud `def export_point_cloud(self, point_cloud, output_path, format)`
- Defined: `weight_space_lidar.py:1158`

### visualize_3d `def visualize_3d(self, point_cloud, title, save_path)`
- Defined: `weight_space_lidar.py:1179`

### visualize_temporal `def visualize_temporal(self, evolution_data, title, save_path)`
- Defined: `weight_space_lidar.py:1219`

### _load_checkpoint `def _load_checkpoint(self, path)`
- Defined: `weight_space_lidar.py:1276`

### _compute_layer_ranges `def _compute_layer_ranges(self, checkpoint, origin)`
- Defined: `weight_space_lidar.py:1282`

### _compute_evolution_metrics `def _compute_evolution_metrics(self, trajectories, signals, epochs)`
- Defined: `weight_space_lidar.py:1308`

### _export_las `def _export_las(self, point_cloud, output_path)`
- Defined: `weight_space_lidar.py:1343`

### _export_ply `def _export_ply(self, point_cloud, output_path)`
- Defined: `weight_space_lidar.py:1367`

### _export_csv `def _export_csv(self, point_cloud, output_path)`
- Defined: `weight_space_lidar.py:1394`

### _export_json `def _export_json(self, point_cloud, output_path)`
- Defined: `weight_space_lidar.py:1419`

### __init__ `def __init__(self)`
- Defined: `weight_space_lidar.py:1450`

### _create_parser `def _create_parser(self)`
- Defined: `weight_space_lidar.py:1453`

### run `def run(self, args)`
- Defined: `weight_space_lidar.py:1558`

### _handle_scan `def _handle_scan(self, navigator, args)`
- Defined: `weight_space_lidar.py:1585`

### _handle_cloud `def _handle_cloud(self, navigator, args)`
- Defined: `weight_space_lidar.py:1617`

### _handle_range `def _handle_range(self, navigator, args)`
- Defined: `weight_space_lidar.py:1647`

### _handle_evolution `def _handle_evolution(self, navigator, args)`
- Defined: `weight_space_lidar.py:1668`
