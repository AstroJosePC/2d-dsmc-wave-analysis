
"""
Figure 4, left and center panels.
Two-panel Vertical Normalized Density Perturbations and Phase Delays vs. Altitude for 1D and S6 Simulations
"""

# %%
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns

from optimized_loader_zenodo import OptimizedDSMCDataProcessor
from helper_funcs_zenodo import detect_wave_period

sns.set_theme(context="paper", font_scale=1.5, style="ticks")

# %% [markdown]
# ## 1. Physical Constants and Dispersion Parameters

# %%
# ============================================================================
# Physical constants and dispersion parameters
# ============================================================================
omega = 9.5e-3  # rad/s (wave angular frequency)
gamma = 5 / 3  # adiabatic index for monatomic gas
R_gas = 8.314  # J/mol/K
T_atm = 270  # K (representative temperature)
M_amu = 16  # g/mol (atomic oxygen)
R_prime = R_gas / M_amu * 1e3  # J/kg/K
g_mars = 3.51  # m/s² (Mars surface gravity)

# Derived quantities
c_sound = np.sqrt(gamma * R_prime * T_atm)  # sound speed (m/s)
omega_a = (gamma / 2) * (g_mars / c_sound)  # acoustic cutoff frequency (rad/s)
H_scale = R_prime * T_atm / g_mars / 1e3  # scale height (km)
wave_period = 2 * np.pi / omega  # seconds

# 1D dispersion limit
dispersion_factor = 1 - (omega_a / omega) ** 2
kz_1D = np.sqrt((omega / c_sound) ** 2 * dispersion_factor) if dispersion_factor > 0 else 0
lambda_z_1D = 2 * np.pi / kz_1D / 1e3 if kz_1D > 0 else np.inf  # km

# %% [markdown]
# ## 2. Load 1D Simulation Data

# %%
# ============================================================================
# Load 1D Simulation (single azimuthal cell)
# ============================================================================
import os


SIM_PATH_1D = "Results_20250326_133319.916"

if os.name == "posix":
    SIM_PATH_1D = SIM_PATH_1D.replace("I:\\", "/Volumes/")
    # fix path slashes
    SIM_PATH_1D = SIM_PATH_1D.replace("\\", "/")

processor_1d = OptimizedDSMCDataProcessor(SIM_PATH_1D)
processor_1d.load_parameters()
processor_1d.load_data()

print(f"1D Data loaded: density shape = {processor_1d.density_data.shape}")
print(f"Number of azimuthal cells: {processor_1d.density_data.shape[2]}")

# For 1D simulation, we use the single azimuthal column (index 0)
azi_col_1d = 0
print(f"Using azimuthal column: {azi_col_1d}")

# Get wave timing and detect period
time_1d = processor_1d.params.time
wave_start_1d = processor_1d.params.start_step_wave * processor_1d.params.timestep
wave_end_1d = processor_1d.params.final_step_wave * processor_1d.params.timestep

# Get background profiles
bg_1d = processor_1d.get_background_profiles()

# Detect wave period from a mid-altitude time series
target_alt_idx = np.argmin(np.abs(processor_1d.params.radial_cells - 230))
wave_mask = (time_1d >= wave_start_1d) & (time_1d <= wave_end_1d)
wave_times = time_1d[wave_mask]
wave_indices = np.where(wave_mask)[0]

density_ts = processor_1d.density_data[wave_indices, target_alt_idx, azi_col_1d]
perturbation = (density_ts - bg_1d["density"][target_alt_idx]) / bg_1d["density"][target_alt_idx] * 100

period_1d, freq, _, _ = detect_wave_period(wave_times, perturbation)

print(f"\n1D Simulation:")
print(f"  Detected wave period: {period_1d:.1f} s")
print(f"  Wave start: {wave_start_1d:.1f} s, Wave end: {wave_end_1d:.1f} s")
print(f"  Total wave injection duration: {(wave_end_1d - wave_start_1d) / period_1d:.1f} periods")

# %% [markdown]
# ## 3. Load 2D Simulation Data

# %%
# ============================================================================
# Load 2D Simulation (multiple azimuthal cells)
# ============================================================================
SIM_PATH_2D = "S2_1024_azi"

processor_2d = OptimizedDSMCDataProcessor(SIM_PATH_2D)
processor_2d.load_parameters()
processor_2d.load_data()

print(f"\n2D Data loaded: density shape = {processor_2d.density_data.shape}")
print(f"Number of azimuthal cells: {processor_2d.density_data.shape[2]}")

# Get center azimuthal column
osc_cols = processor_2d.compute_osc_cols()
center_col_2d = int((osc_cols[0] + osc_cols[-1]) / 2)
print(f"Oscillating columns: {osc_cols[0]} to {osc_cols[-1]}")
print(f"Using center column: {center_col_2d}")

# Get wave timing and detect period
time_2d = processor_2d.params.time
wave_start_2d = processor_2d.params.start_step_wave * processor_2d.params.timestep
wave_end_2d = processor_2d.params.final_step_wave * processor_2d.params.timestep

# Get background profiles
bg_2d = processor_2d.get_background_profiles()

# Detect wave period
target_alt_idx = np.argmin(np.abs(processor_2d.params.radial_cells - 230))
wave_mask = (time_2d >= wave_start_2d) & (time_2d <= wave_end_2d)
wave_times = time_2d[wave_mask]
wave_indices = np.where(wave_mask)[0]

density_ts = processor_2d.density_data[wave_indices, target_alt_idx, center_col_2d]
perturbation = (density_ts - bg_2d["density"][target_alt_idx]) / bg_2d["density"][target_alt_idx] * 100

period_2d, freq, _, _ = detect_wave_period(wave_times, perturbation)

print(f"\n2D Simulation:")
print(f"  Detected wave period: {period_2d:.1f} s")
print(f"  Wave start: {wave_start_2d:.1f} s, Wave end: {wave_end_2d:.1f} s")
print(f"  Total wave injection duration: {(wave_end_2d - wave_start_2d) / period_2d:.1f} periods")

# %% [markdown]
# ## 4. Define Altitude Grid and Helper Functions

# %%
# ============================================================================
# Define altitude grid for analysis - use ALL available altitudes in range
# ============================================================================
# Get 1D simulation altitudes in the target range
mask_1d = (processor_1d.params.radial_cells >= 150) & (processor_1d.params.radial_cells <= 420)
alt_indices_1d = np.where(mask_1d)[0]
actual_alts_1d = processor_1d.params.radial_cells[alt_indices_1d]

# Get 2D simulation altitudes in the target range
mask_2d = (processor_2d.params.radial_cells >= 150) & (processor_2d.params.radial_cells <= 420)
alt_indices_2d = np.where(mask_2d)[0]
actual_alts_2d = processor_2d.params.radial_cells[alt_indices_2d]

print(f"1D Simulation: {len(actual_alts_1d)} altitudes from {actual_alts_1d[0]:.1f} to {actual_alts_1d[-1]:.1f} km")
print(f"2D Simulation: {len(actual_alts_2d)} altitudes from {actual_alts_2d[0]:.1f} to {actual_alts_2d[-1]:.1f} km")

# %%
# ============================================================================
# Helper function: Cross-correlation phase lag
# ============================================================================
def get_phase_lag_xcorr(wf_ref, wf_target, time_seconds, period):
    """Compute phase lag using cross-correlation. Returns lag in wave periods."""
    # Normalize waveforms
    wf_ref_norm = (wf_ref - np.mean(wf_ref)) / np.std(wf_ref)
    wf_target_norm = (wf_target - np.mean(wf_target)) / np.std(wf_target)

    # Cross-correlation
    xcorr = np.correlate(wf_target_norm, wf_ref_norm, mode="full")
    lags = np.arange(-len(wf_ref) + 1, len(wf_ref))

    # Find peak near zero lag (within ±1.5 periods)
    dt_step = time_seconds[1] - time_seconds[0]
    max_lag_samples = int(1.5 * period / dt_step)
    center = len(wf_ref) - 1
    search_start = max(0, center - max_lag_samples)
    search_end = min(len(xcorr), center + max_lag_samples)

    peak_idx = search_start + np.argmax(xcorr[search_start:search_end])
    lag_samples = lags[peak_idx]
    lag_seconds = lag_samples * dt_step
    lag_periods = lag_seconds / period

    return lag_periods, lag_seconds

# %%
# ============================================================================
# Helper function: Find zero crossings
# ============================================================================
def find_positive_zero_crossings(signal, times):
    """Find times where signal crosses zero going from negative to positive."""
    crossings = []
    for i in range(1, len(signal)):
        if signal[i - 1] < 0 and signal[i] >= 0:
            # Linear interpolation to find exact crossing time
            frac = -signal[i - 1] / (signal[i] - signal[i - 1])
            t_cross = times[i - 1] + frac * (times[i] - times[i - 1])
            crossings.append(t_cross)
    return np.array(crossings)

# %%
# ============================================================================
# Helper functions: RMS optimization for masked waveforms
# ============================================================================
def compute_rms_for_shift_masked(shift_periods, wf_ref, wf_target, time_periods):
    """
    Compute RMS error between reference and phase-shifted target.
    Handles masked (NaN-padded) waveforms from altitude-dependent windows.
    """
    dt_step = time_periods[1] - time_periods[0]
    lag_samples = int(round(shift_periods / dt_step))

    # Normalize only valid (non-NaN) portions
    ref_valid = ~np.isnan(wf_ref)
    target_valid = ~np.isnan(wf_target)

    if np.sum(ref_valid) < 10 or np.sum(target_valid) < 10:
        return np.inf  # Not enough data

    wf_ref_norm = np.full_like(wf_ref, np.nan)
    wf_target_norm = np.full_like(wf_target, np.nan)

    wf_ref_norm[ref_valid] = (wf_ref[ref_valid] - np.mean(wf_ref[ref_valid])) / np.std(wf_ref[ref_valid])
    wf_target_norm[target_valid] = (wf_target[target_valid] - np.mean(wf_target[target_valid])) / np.std(
        wf_target[target_valid]
    )

    # Shift target waveform
    if lag_samples > 0:
        if lag_samples >= len(wf_target_norm):
            return np.inf
        wf_shifted = np.concatenate([wf_target_norm[lag_samples:], np.full(lag_samples, np.nan)])
    elif lag_samples < 0:
        if -lag_samples >= len(wf_target_norm):
            return np.inf
        wf_shifted = np.concatenate([np.full(-lag_samples, np.nan), wf_target_norm[:lag_samples]])
    else:
        wf_shifted = wf_target_norm.copy()

    # Compute RMS where both signals are valid
    both_valid = (~np.isnan(wf_ref_norm)) & (~np.isnan(wf_shifted))

    if np.sum(both_valid) < 10:
        return np.inf  # Not enough overlap

    residual = wf_ref_norm[both_valid] - wf_shifted[both_valid]
    rms = np.sqrt(np.mean(residual**2))

    return rms


def find_optimal_phase_lag_constrained(wf_ref, wf_target, time_periods, initial_lag, search_width=0.2, n_points=100):
    """
    Find phase lag that minimizes RMS alignment error, constrained near initial estimate.
    """
    # Define constrained search range
    search_min = max(0.0, initial_lag - search_width)
    search_max = initial_lag + search_width

    # Grid search
    test_lags = np.linspace(search_min, search_max, n_points)
    rms_values = np.array([compute_rms_for_shift_masked(lag, wf_ref, wf_target, time_periods) for lag in test_lags])

    # Find minimum
    min_idx = np.argmin(rms_values)
    coarse_optimal = test_lags[min_idx]

    # Refine with bounded optimization
    refine_bounds = (max(search_min, coarse_optimal - 0.05), min(search_max, coarse_optimal + 0.05))

    result = minimize_scalar(
        lambda x: compute_rms_for_shift_masked(x, wf_ref, wf_target, time_periods),
        bounds=refine_bounds,
        method="bounded",
    )

    optimal_lag = result.x
    min_rms = result.fun

    return optimal_lag, min_rms

# %% [markdown]
# ## 5. Process 1D Simulation with Hybrid Method

# %%
# ============================================================================
# FIRST PASS (1D): Use generous window for phase delay computation
# ============================================================================
generous_start_1d = wave_start_1d + 1 * period_1d  # Start at cycle 2
generous_end_1d = min(wave_start_1d + 7 * period_1d, wave_end_1d)  # End at cycle 7
generous_mask_1d = (time_1d >= generous_start_1d) & (time_1d <= generous_end_1d)
generous_indices_1d = np.where(generous_mask_1d)[0]
generous_times_1d = time_1d[generous_mask_1d]

print(f"Generous window: cycles 2-7")
print(f"Time range: {generous_start_1d:.1f} s to {generous_end_1d:.1f} s")
print(f"Window size: {len(generous_indices_1d)} time steps")

print(f"\nUsing {len(actual_alts_1d)} altitude levels from {actual_alts_1d[0]:.1f} to {actual_alts_1d[-1]:.1f} km")

# Extract waveforms using generous window
waveforms_generous_1d = []
for idx in alt_indices_1d:
    density_ts = processor_1d.density_data[generous_indices_1d, idx, azi_col_1d]
    pert = (density_ts - bg_1d["density"][idx]) / bg_1d["density"][idx] * 100
    waveforms_generous_1d.append(pert)

waveforms_generous_1d = np.array(waveforms_generous_1d)

# Compute phase delays relative to LOWEST altitude
ref_idx = 0
phase_delays_1d = np.zeros(len(alt_indices_1d))
phase_delays_seconds_1d = np.zeros(len(alt_indices_1d))

for i in range(len(alt_indices_1d)):
    if i == ref_idx:
        phase_delays_1d[i] = 0.0
        phase_delays_seconds_1d[i] = 0.0
    else:
        lag_periods, lag_seconds = get_phase_lag_xcorr(
            waveforms_generous_1d[ref_idx], waveforms_generous_1d[i], generous_times_1d, period_1d
        )
        phase_delays_1d[i] = lag_periods
        phase_delays_seconds_1d[i] = lag_seconds

# Ensure monotonicity
for i in range(1, len(phase_delays_1d)):
    if phase_delays_1d[i] < phase_delays_1d[i - 1]:
        phase_delays_1d[i] = phase_delays_1d[i - 1]
        phase_delays_seconds_1d[i] = phase_delays_1d[i] * period_1d

print("\nPhase delays relative to lowest altitude:")
print("-" * 45)
for alt, lag_p, lag_s in zip(actual_alts_1d, phase_delays_1d, phase_delays_seconds_1d):
    print(f"  {alt:6.0f} km: Δφ = {lag_p:+.3f} T = {lag_s:+.1f} s")

# %%
# ============================================================================
# SECOND PASS (1D): Apply altitude-dependent windows for exact cycles
# ============================================================================

# Find zero crossings at reference altitude
ref_alt_idx_1d = alt_indices_1d[0]
# CRITICAL: Use actual wave start time from generous window (cycle 2 onwards) to avoid spurious early crossings
wave_mask_full_1d = (time_1d >= generous_start_1d) & (time_1d <= wave_end_1d)
wave_times_full_1d = time_1d[wave_mask_full_1d]
wave_indices_full_1d = np.where(wave_mask_full_1d)[0]

ref_density_1d = processor_1d.density_data[wave_indices_full_1d, ref_alt_idx_1d, azi_col_1d]
ref_perturbation_1d = (ref_density_1d - bg_1d["density"][ref_alt_idx_1d]) / bg_1d["density"][ref_alt_idx_1d] * 100

zero_crossings_1d = find_positive_zero_crossings(ref_perturbation_1d, wave_times_full_1d)
available_cycles_1d = len(zero_crossings_1d)
max_phase_delay_1d = phase_delays_1d.max()

print(f"Found {available_cycles_1d} positive-going zero crossings")
print(f"Maximum phase delay: {max_phase_delay_1d:.3f} periods = {max_phase_delay_1d * period_1d:.1f} s")

# Choose cycles 4-5 if available
# Note: Since we start searching from cycle 2, zero_crossings[0] is cycle 2
# So cycle 4 is at zero_crossings[2], cycle 5 is at zero_crossings[3], cycle 6 is at zero_crossings[4]
if available_cycles_1d >= 3:  # Need at least 3 crossings (cycles 2,3,4) to get cycle 4
    target_cycle_num_1d = 4
    print(f"Using cycles 4 and 5")
else:
    target_cycle_num_1d = max(2, available_cycles_1d)  # Fallback to last available
    print(f"Using cycles {target_cycle_num_1d} and {target_cycle_num_1d + 1}")

# Adjust indices: since zero_crossings[0] is cycle 2, cycle N is at zero_crossings[N-2]
cycle_start_time_ref_1d = zero_crossings_1d[target_cycle_num_1d - 2]
cycle_end_time_ref_1d = (
    zero_crossings_1d[target_cycle_num_1d]
    if target_cycle_num_1d < len(zero_crossings_1d)
    else cycle_start_time_ref_1d + 2 * period_1d
)

# Apply phase delays for each altitude
alt_window_starts_1d = cycle_start_time_ref_1d + phase_delays_1d * period_1d
alt_window_ends_1d = cycle_end_time_ref_1d + phase_delays_1d * period_1d

# Define global extraction window
global_start_1d = alt_window_starts_1d.min()
global_end_1d = min(alt_window_ends_1d.max(), time_1d[-1])

global_mask_1d = (time_1d >= global_start_1d) & (time_1d <= global_end_1d)
global_indices_1d = np.where(global_mask_1d)[0]
global_times_1d = time_1d[global_mask_1d]
time_in_periods_1d = (global_times_1d - wave_start_1d) / period_1d

print(f"Global window: {global_start_1d:.1f} s to {global_end_1d:.1f} s")
print(f"Span: {(global_end_1d - global_start_1d) / period_1d:.2f} wave periods")
print(
    f"Reference cycle start: {cycle_start_time_ref_1d:.1f} s ({(cycle_start_time_ref_1d - wave_start_1d) / period_1d:.2f} periods from wave start)"
)
print(
    f"Reference cycle end: {cycle_end_time_ref_1d:.1f} s ({(cycle_end_time_ref_1d - wave_start_1d) / period_1d:.2f} periods from wave start)"
)

# Extract waveforms with altitude-specific masking
waveforms_masked_1d = []
for i, (idx, t_start, t_end) in enumerate(zip(alt_indices_1d, alt_window_starts_1d, alt_window_ends_1d)):
    density_ts = processor_1d.density_data[global_indices_1d, idx, azi_col_1d]
    pert = (density_ts - bg_1d["density"][idx]) / bg_1d["density"][idx] * 100
    local_mask = (global_times_1d >= t_start) & (global_times_1d <= t_end)
    pert_masked = np.where(local_mask, pert, np.nan)
    waveforms_masked_1d.append(pert_masked)

waveforms_masked_1d = np.array(waveforms_masked_1d)
print(f"Masked waveform array shape: {waveforms_masked_1d.shape}")

# %%
# ============================================================================
# THIRD PASS (1D): Apply constrained RMS optimization
# ============================================================================

rms_optimal_lags_1d = np.zeros(len(actual_alts_1d))
rms_optimal_errors_1d = np.zeros(len(actual_alts_1d))

print(f"{'Alt (km)':<10} {'XCorr φ':<12} {'RMS-Opt φ':<15} {'Min RMS':<10} {'Correction':<12}")
print("-" * 70)

for i, alt in enumerate(actual_alts_1d):
    if i == 0:
        rms_optimal_lags_1d[i] = 0.0
        rms_optimal_errors_1d[i] = 0.0
        print(f"{alt:<10.0f} {phase_delays_1d[i]:<12.3f} {0.0:<15.3f} {0.0:<10.3f} {'[reference]':<12}")
    else:
        initial_lag = phase_delays_1d[i]
        opt_lag, min_rms = find_optimal_phase_lag_constrained(
            waveforms_masked_1d[0],
            waveforms_masked_1d[i],
            time_in_periods_1d,
            initial_lag=initial_lag,
            search_width=0.2,
            n_points=100,
        )
        rms_optimal_lags_1d[i] = opt_lag
        rms_optimal_errors_1d[i] = min_rms
        delta_lag = opt_lag - initial_lag
        correction = f"{delta_lag:+.3f} T" if abs(delta_lag) > 0.001 else "negligible"
        print(f"{alt:<10.0f} {initial_lag:<12.3f} {opt_lag:<15.3f} {min_rms:<10.3f} {correction:<12}")

print()
print(f"Average correction: {np.mean(np.abs(rms_optimal_lags_1d[1:] - phase_delays_1d[1:])):.4f} wave periods")
print(f"Maximum correction: {np.max(np.abs(rms_optimal_lags_1d[1:] - phase_delays_1d[1:])):.4f} wave periods")
print(f"RMS errors range: {rms_optimal_errors_1d[1:].min():.4f} to {rms_optimal_errors_1d[1:].max():.4f}")

# %% [markdown]
# ## 6. Process 2D Simulation with Hybrid Method

# %%
# ============================================================================
# FIRST PASS (2D): Use generous window for phase delay computation
# ============================================================================
generous_start_2d = wave_start_2d + 1 * period_2d  # Start at cycle 2
generous_end_2d = min(wave_start_2d + 7 * period_2d, wave_end_2d)  # End at cycle 7
generous_mask_2d = (time_2d >= generous_start_2d) & (time_2d <= generous_end_2d)
generous_indices_2d = np.where(generous_mask_2d)[0]
generous_times_2d = time_2d[generous_mask_2d]

print(f"Generous window: cycles 2-7")
print(f"Time range: {generous_start_2d:.1f} s to {generous_end_2d:.1f} s")
print(f"Window size: {len(generous_indices_2d)} time steps")

print(f"\nUsing {len(actual_alts_2d)} altitude levels from {actual_alts_2d[0]:.1f} to {actual_alts_2d[-1]:.1f} km")

# Extract waveforms using generous window
waveforms_generous_2d = []
for idx in alt_indices_2d:
    density_ts = processor_2d.density_data[generous_indices_2d, idx, center_col_2d]
    pert = (density_ts - bg_2d["density"][idx]) / bg_2d["density"][idx] * 100
    waveforms_generous_2d.append(pert)

waveforms_generous_2d = np.array(waveforms_generous_2d)

# Compute phase delays relative to LOWEST altitude
ref_idx = 0
phase_delays_2d = np.zeros(len(alt_indices_2d))
phase_delays_seconds_2d = np.zeros(len(alt_indices_2d))

for i in range(len(alt_indices_2d)):
    if i == ref_idx:
        phase_delays_2d[i] = 0.0
        phase_delays_seconds_2d[i] = 0.0
    else:
        lag_periods, lag_seconds = get_phase_lag_xcorr(
            waveforms_generous_2d[ref_idx], waveforms_generous_2d[i], generous_times_2d, period_2d
        )
        phase_delays_2d[i] = lag_periods
        phase_delays_seconds_2d[i] = lag_seconds

# Ensure monotonicity
for i in range(1, len(phase_delays_2d)):
    if phase_delays_2d[i] < phase_delays_2d[i - 1]:
        phase_delays_2d[i] = phase_delays_2d[i - 1]
        phase_delays_seconds_2d[i] = phase_delays_2d[i] * period_2d

print("\nPhase delays relative to lowest altitude:")
print("-" * 45)
for alt, lag_p, lag_s in zip(actual_alts_2d, phase_delays_2d, phase_delays_seconds_2d):
    print(f"  {alt:6.0f} km: Δφ = {lag_p:+.3f} T = {lag_s:+.1f} s")

# %%
# ============================================================================
# SECOND PASS (2D): Apply altitude-dependent windows for exact cycles
# ============================================================================
print("\n" + "=" * 80)
print("2D SIMULATION: SECOND PASS - ALTITUDE-DEPENDENT WINDOWS")
print("=" * 80)

# Find zero crossings at reference altitude
ref_alt_idx_2d = alt_indices_2d[0]
# CRITICAL: Use actual wave start time from generous window (cycle 2 onwards) to avoid spurious early crossings
wave_mask_full_2d = (time_2d >= generous_start_2d) & (time_2d <= wave_end_2d)
wave_times_full_2d = time_2d[wave_mask_full_2d]
wave_indices_full_2d = np.where(wave_mask_full_2d)[0]

ref_density_2d = processor_2d.density_data[wave_indices_full_2d, ref_alt_idx_2d, center_col_2d]
ref_perturbation_2d = (ref_density_2d - bg_2d["density"][ref_alt_idx_2d]) / bg_2d["density"][ref_alt_idx_2d] * 100

zero_crossings_2d = find_positive_zero_crossings(ref_perturbation_2d, wave_times_full_2d)
available_cycles_2d = len(zero_crossings_2d)
max_phase_delay_2d = phase_delays_2d.max()

print(f"Found {available_cycles_2d} positive-going zero crossings")
print(f"Maximum phase delay: {max_phase_delay_2d:.3f} periods = {max_phase_delay_2d * period_2d:.1f} s")

# Choose cycles 4-5 if available
# Note: Since we start searching from cycle 2, zero_crossings[0] is cycle 2
# So cycle 4 is at zero_crossings[2], cycle 5 is at zero_crossings[3], cycle 6 is at zero_crossings[4]
if available_cycles_2d >= 3:  # Need at least 3 crossings (cycles 2,3,4) to get cycle 4
    target_cycle_num_2d = 4
    print(f"Using cycles 4 and 5")
else:
    target_cycle_num_2d = max(2, available_cycles_2d)  # Fallback to last available
    print(f"Using cycles {target_cycle_num_2d} and {target_cycle_num_2d + 1}")

# Adjust indices: since zero_crossings[0] is cycle 2, cycle N is at zero_crossings[N-2]
cycle_start_time_ref_2d = zero_crossings_2d[target_cycle_num_2d - 2]
cycle_end_time_ref_2d = (
    zero_crossings_2d[target_cycle_num_2d]
    if target_cycle_num_2d < len(zero_crossings_2d)
    else cycle_start_time_ref_2d + 2 * period_2d
)

# Apply phase delays for each altitude
alt_window_starts_2d = cycle_start_time_ref_2d + phase_delays_2d * period_2d
alt_window_ends_2d = cycle_end_time_ref_2d + phase_delays_2d * period_2d

# Define global extraction window
global_start_2d = alt_window_starts_2d.min()
global_end_2d = min(alt_window_ends_2d.max(), time_2d[-1])

global_mask_2d = (time_2d >= global_start_2d) & (time_2d <= global_end_2d)
global_indices_2d = np.where(global_mask_2d)[0]
global_times_2d = time_2d[global_mask_2d]
time_in_periods_2d = (global_times_2d - wave_start_2d) / period_2d

print(f"Global window: {global_start_2d:.1f} s to {global_end_2d:.1f} s")
print(f"Span: {(global_end_2d - global_start_2d) / period_2d:.2f} wave periods")
print(
    f"Reference cycle start: {cycle_start_time_ref_2d:.1f} s ({(cycle_start_time_ref_2d - wave_start_2d) / period_2d:.2f} periods from wave start)"
)
print(
    f"Reference cycle end: {cycle_end_time_ref_2d:.1f} s ({(cycle_end_time_ref_2d - wave_start_2d) / period_2d:.2f} periods from wave start)"
)

# Extract waveforms with altitude-specific masking
waveforms_masked_2d = []
for i, (idx, t_start, t_end) in enumerate(zip(alt_indices_2d, alt_window_starts_2d, alt_window_ends_2d)):
    density_ts = processor_2d.density_data[global_indices_2d, idx, center_col_2d]
    pert = (density_ts - bg_2d["density"][idx]) / bg_2d["density"][idx] * 100
    local_mask = (global_times_2d >= t_start) & (global_times_2d <= t_end)
    pert_masked = np.where(local_mask, pert, np.nan)
    waveforms_masked_2d.append(pert_masked)

waveforms_masked_2d = np.array(waveforms_masked_2d)
print(f"Masked waveform array shape: {waveforms_masked_2d.shape}")

# %%
# ============================================================================
# THIRD PASS (2D): Apply constrained RMS optimization
# ============================================================================
print("\n" + "=" * 80)
print("2D SIMULATION: THIRD PASS - CONSTRAINED RMS OPTIMIZATION")
print("=" * 80)

rms_optimal_lags_2d = np.zeros(len(actual_alts_2d))
rms_optimal_errors_2d = np.zeros(len(actual_alts_2d))

print(f"{'Alt (km)':<10} {'XCorr φ':<12} {'RMS-Opt φ':<15} {'Min RMS':<10} {'Correction':<12}")
print("-" * 70)

for i, alt in enumerate(actual_alts_2d):
    if i == 0:
        rms_optimal_lags_2d[i] = 0.0
        rms_optimal_errors_2d[i] = 0.0
        print(f"{alt:<10.0f} {phase_delays_2d[i]:<12.3f} {0.0:<15.3f} {0.0:<10.3f} {'[reference]':<12}")
    else:
        initial_lag = phase_delays_2d[i]
        opt_lag, min_rms = find_optimal_phase_lag_constrained(
            waveforms_masked_2d[0],
            waveforms_masked_2d[i],
            time_in_periods_2d,
            initial_lag=initial_lag,
            search_width=0.2,
            n_points=100,
        )
        rms_optimal_lags_2d[i] = opt_lag
        rms_optimal_errors_2d[i] = min_rms
        delta_lag = opt_lag - initial_lag
        correction = f"{delta_lag:+.3f} T" if abs(delta_lag) > 0.001 else "negligible"
        print(f"{alt:<10.0f} {initial_lag:<12.3f} {opt_lag:<15.3f} {min_rms:<10.3f} {correction:<12}")

print()
print(f"Average correction: {np.mean(np.abs(rms_optimal_lags_2d[1:] - phase_delays_2d[1:])):.4f} wave periods")
print(f"Maximum correction: {np.max(np.abs(rms_optimal_lags_2d[1:] - phase_delays_2d[1:])):.4f} wave periods")
print(f"RMS errors range: {rms_optimal_errors_2d[1:].min():.4f} to {rms_optimal_errors_2d[1:].max():.4f}")

# %% [markdown]
# ## 7. Compare 1D vs 2D Phase Lag Profiles

# %%
# ============================================================================
# Compute uncertainties (temporal resolution as lower limit)
# ============================================================================
dt_1d = time_1d[1] - time_1d[0]
dt_periods_1d = dt_1d / period_1d

dt_2d = time_2d[1] - time_2d[0]
dt_periods_2d = dt_2d / period_2d

rms_uncertainties_1d = np.full(len(actual_alts_1d), dt_periods_1d)
rms_uncertainties_1d[0] = 0.0

rms_uncertainties_2d = np.full(len(actual_alts_2d), dt_periods_2d)
rms_uncertainties_2d[0] = 0.0

print("\n" + "=" * 80)
print("UNCERTAINTIES")
print("=" * 80)
print(f"1D: ±{dt_periods_1d:.4f} wave periods (temporal resolution)")
print(f"2D: ±{dt_periods_2d:.4f} wave periods (temporal resolution)")
print("\nNote: These represent lower limits on the true uncertainties.")

# %%
# ============================================================================
# Phase Lag Comparison: 1D vs 2D (RMS Optimal Only)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot RMS optimal phase lags for both simulations
ax.errorbar(
    rms_optimal_lags_1d,
    actual_alts_1d,
    xerr=rms_uncertainties_1d,
    fmt="o-",
    color="steelblue",
    label="1D Simulation",
    markersize=10,
    linewidth=2.5,
    capsize=4,
    elinewidth=2,
    zorder=10,
)

ax.errorbar(
    rms_optimal_lags_2d,
    actual_alts_2d,
    xerr=rms_uncertainties_2d,
    fmt="s-",
    color="darkorange",
    label="2D Simulation",
    markersize=10,
    linewidth=2.5,
    capsize=4,
    elinewidth=2,
    zorder=10,
)

ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
ax.set_xlabel("Phase Lag (wave periods)", fontsize=22)
ax.set_ylabel("Altitude (km)", fontsize=22)
# ax.set_title('Vertical Phase Lag Profiles: 1D vs 2D Simulations\n(Hybrid Constrained RMS Optimization)',
#             fontsize=14, fontweight='bold')
ax.legend(fontsize=18, loc="lower right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("wavelength_analysis_1D_2D_hybrid_phase_lags.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# ============================================================================
# Compute Radial Wavelengths from Phase Lag Slopes
# ============================================================================
from scipy.stats import linregress


def weighted_linregress(x, y, yerr):
    """Perform weighted linear regression and return slope, intercept, and uncertainties."""
    # Weights are inverse variance
    weights = 1.0 / (yerr**2)

    # Weighted means
    w_sum = np.sum(weights)
    x_mean = np.sum(weights * x) / w_sum
    y_mean = np.sum(weights * y) / w_sum

    # Weighted covariance and variance
    cov_xy = np.sum(weights * (x - x_mean) * (y - y_mean)) / w_sum
    var_x = np.sum(weights * (x - x_mean) ** 2) / w_sum

    # Slope and intercept
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean

    # Residuals and chi-squared
    y_pred = slope * x + intercept
    chi2 = np.sum(weights * (y - y_pred) ** 2)
    dof = len(x) - 2  # degrees of freedom

    # Uncertainty in slope (includes fit uncertainty)
    # Variance of slope from weighted least squares
    slope_var = 1.0 / np.sum(weights * (x - x_mean) ** 2)
    slope_err = np.sqrt(slope_var)

    # Reduced chi-squared (should be ~1 if errors are correct)
    reduced_chi2 = chi2 / dof if dof > 0 else np.nan

    return slope, intercept, slope_err, reduced_chi2


def compute_wavelength_with_uncertainty(alts, lags, uncertainties, region_name="Full"):
    """
    Compute radial wavelength from phase lag vs altitude.

    λ_r = 1 / (dφ/dz) where φ is in wave periods and z is in km

    Returns wavelength and its uncertainty.
    """
    # Filter out reference altitude (lag = 0) and ensure positive lags
    valid = (lags > 0.001) & (uncertainties > 0)

    if np.sum(valid) < 3:
        return np.nan, np.nan, 0

    x = alts[valid]
    y = lags[valid]
    yerr = uncertainties[valid]

    # Weighted linear regression
    slope, intercept, slope_err, reduced_chi2 = weighted_linregress(x, y, yerr)

    # Wavelength from slope: λ = 1 / slope
    wavelength = 1.0 / slope

    # Uncertainty propagation: if λ = 1/slope, then δλ/λ = δ(slope)/slope
    # δλ = λ * (δ(slope) / slope) = (1/slope) * (δ(slope)/slope) = δ(slope) / slope^2
    wavelength_err = slope_err / (slope**2)

    n_points = np.sum(valid)

    return wavelength, wavelength_err, n_points, reduced_chi2


# Define altitude split around 280 km
split_altitude = 280.0

print("=" * 80)
print("RADIAL WAVELENGTH ANALYSIS")
print("=" * 80)
print(f"\nAltitude range split at {split_altitude:.0f} km")
print(
    f"1D: Lower = {actual_alts_1d.min():.0f}-{split_altitude:.0f} km, Upper = {split_altitude:.0f}-{actual_alts_1d.max():.0f} km"
)
print(
    f"2D: Lower = {actual_alts_2d.min():.0f}-{split_altitude:.0f} km, Upper = {split_altitude:.0f}-{actual_alts_2d.max():.0f} km"
)

# ============================================================================
# 1D Simulation Wavelengths
# ============================================================================
print("\n" + "-" * 80)
print("1D SIMULATION WAVELENGTHS")
print("-" * 80)

# Full range
lambda_1d_full, lambda_1d_full_err, n_1d_full, chi2_1d_full = compute_wavelength_with_uncertainty(
    actual_alts_1d, rms_optimal_lags_1d, rms_uncertainties_1d, "Full"
)
print(f"\nFull Range ({actual_alts_1d.min():.0f}-{actual_alts_1d.max():.0f} km, n={n_1d_full}):")
print(f"  λ_r = {lambda_1d_full:.1f} ± {lambda_1d_full_err:.1f} km")
print(f"  Reduced χ² = {chi2_1d_full:.2f}")

# Lower range
mask_lower_1d = actual_alts_1d <= split_altitude
lambda_1d_lower, lambda_1d_lower_err, n_1d_lower, chi2_1d_lower = compute_wavelength_with_uncertainty(
    actual_alts_1d[mask_lower_1d], rms_optimal_lags_1d[mask_lower_1d], rms_uncertainties_1d[mask_lower_1d], "Lower"
)
print(f"\nLower Range ({actual_alts_1d[mask_lower_1d].min():.0f}-{split_altitude:.0f} km, n={n_1d_lower}):")
print(f"  λ_r = {lambda_1d_lower:.1f} ± {lambda_1d_lower_err:.1f} km")
print(f"  Reduced χ² = {chi2_1d_lower:.2f}")

# Upper range
mask_upper_1d = actual_alts_1d > split_altitude
lambda_1d_upper, lambda_1d_upper_err, n_1d_upper, chi2_1d_upper = compute_wavelength_with_uncertainty(
    actual_alts_1d[mask_upper_1d], rms_optimal_lags_1d[mask_upper_1d], rms_uncertainties_1d[mask_upper_1d], "Upper"
)
print(f"\nUpper Range ({split_altitude:.0f}-{actual_alts_1d[mask_upper_1d].max():.0f} km, n={n_1d_upper}):")
print(f"  λ_r = {lambda_1d_upper:.1f} ± {lambda_1d_upper_err:.1f} km")
print(f"  Reduced χ² = {chi2_1d_upper:.2f}")

# ============================================================================
# 2D Simulation Wavelengths
# ============================================================================
print("\n" + "-" * 80)
print("2D SIMULATION WAVELENGTHS")
print("-" * 80)

# Full range
lambda_2d_full, lambda_2d_full_err, n_2d_full, chi2_2d_full = compute_wavelength_with_uncertainty(
    actual_alts_2d, rms_optimal_lags_2d, rms_uncertainties_2d, "Full"
)
print(f"\nFull Range ({actual_alts_2d.min():.0f}-{actual_alts_2d.max():.0f} km, n={n_2d_full}):")
print(f"  λ_r = {lambda_2d_full:.1f} ± {lambda_2d_full_err:.1f} km")
print(f"  Reduced χ² = {chi2_2d_full:.2f}")

# Lower range
mask_lower_2d = actual_alts_2d <= split_altitude
lambda_2d_lower, lambda_2d_lower_err, n_2d_lower, chi2_2d_lower = compute_wavelength_with_uncertainty(
    actual_alts_2d[mask_lower_2d], rms_optimal_lags_2d[mask_lower_2d], rms_uncertainties_2d[mask_lower_2d], "Lower"
)
print(f"\nLower Range ({actual_alts_2d[mask_lower_2d].min():.0f}-{split_altitude:.0f} km, n={n_2d_lower}):")
print(f"  λ_r = {lambda_2d_lower:.1f} ± {lambda_2d_lower_err:.1f} km")
print(f"  Reduced χ² = {chi2_2d_lower:.2f}")

# Upper range
mask_upper_2d = actual_alts_2d > split_altitude
lambda_2d_upper, lambda_2d_upper_err, n_2d_upper, chi2_2d_upper = compute_wavelength_with_uncertainty(
    actual_alts_2d[mask_upper_2d], rms_optimal_lags_2d[mask_upper_2d], rms_uncertainties_2d[mask_upper_2d], "Upper"
)
print(f"\nUpper Range ({split_altitude:.0f}-{actual_alts_2d[mask_upper_2d].max():.0f} km, n={n_2d_upper}):")
print(f"  λ_r = {lambda_2d_upper:.1f} ± {lambda_2d_upper_err:.1f} km")
print(f"  Reduced χ² = {chi2_2d_upper:.2f}")

# ============================================================================
# Summary Comparison
# ============================================================================
print("\n" + "=" * 80)
print("WAVELENGTH COMPARISON SUMMARY")
print("=" * 80)
print(f"\n{'Region':<20} {'1D Wavelength (km)':<25} {'2D Wavelength (km)':<25}")
print("-" * 70)
print(
    f"{'Full Range':<20} {lambda_1d_full:6.1f} ± {lambda_1d_full_err:4.1f}             {lambda_2d_full:6.1f} ± {lambda_2d_full_err:4.1f}"
)
print(
    f"{'Lower (≤280 km)':<20} {lambda_1d_lower:6.1f} ± {lambda_1d_lower_err:4.1f}             {lambda_2d_lower:6.1f} ± {lambda_2d_lower_err:4.1f}"
)
print(
    f"{'Upper (>280 km)':<20} {lambda_1d_upper:6.1f} ± {lambda_1d_upper_err:4.1f}             {lambda_2d_upper:6.1f} ± {lambda_2d_upper_err:4.1f}"
)
print("\nNote: Uncertainties include both phase lag measurement errors and linear fit uncertainties.")
print(f"Theoretical maximum λ_z = {lambda_z_1D:.1f} km (from 1D dispersion relation)")

# %%
# ============================================================================
# Stacked Waveform Visualization with Phase Lag Indicators
# ============================================================================
# Show stacked waveforms for both 1D and 2D simulations with RMS-optimal phase lag markers

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Select subset of altitudes for visualization (every Nth to avoid overcrowding)
n_display = 12  # Show at most 12 altitudes
step_1d = max(1, len(actual_alts_1d) // n_display)
step_2d = max(1, len(actual_alts_2d) // n_display)
display_indices_1d = range(0, len(actual_alts_1d), step_1d)
display_indices_2d = range(0, len(actual_alts_2d), step_2d)

# --------------------------------------------------------------------------
# Panel (a): 1D Simulation Stacked Waveforms
# --------------------------------------------------------------------------
ax1 = axes[0]
offsets_1d = np.linspace(0, len(display_indices_1d) * 2.5, len(display_indices_1d))

# Find reference peak in 1D waveforms
ref_wf_1d = waveforms_masked_1d[0]
ref_valid_1d = ~np.isnan(ref_wf_1d)
ref_wf_valid_1d = ref_wf_1d[ref_valid_1d]
ref_time_valid_1d = time_in_periods_1d[ref_valid_1d]

# Normalize and find first peak
ref_wf_norm_1d = (ref_wf_valid_1d - np.mean(ref_wf_valid_1d)) / np.std(ref_wf_valid_1d)
peaks_1d, _ = find_peaks(
    ref_wf_norm_1d, prominence=0.5, distance=int(0.8 / (ref_time_valid_1d[1] - ref_time_valid_1d[0]))
)

if len(peaks_1d) > 0:
    ref_peak_time_1d = ref_time_valid_1d[peaks_1d[0]]
else:
    ref_peak_time_1d = ref_time_valid_1d[np.argmax(ref_wf_norm_1d)]

for i, (disp_idx, offset) in enumerate(zip(display_indices_1d, offsets_1d)):
    # color = colors[i]
    wf = waveforms_masked_1d[disp_idx]
    valid = ~np.isnan(wf)

    if np.sum(valid) > 10:
        wf_valid = wf[valid]
        wf_normalized = np.full_like(wf, np.nan)
        wf_normalized[valid] = (wf_valid - np.mean(wf_valid)) / np.std(wf_valid)

        # Plot waveform
        ax1.plot(time_in_periods_1d, wf_normalized + offset, color="black", lw=3, alpha=0.8)

        # Calculate expected peak position based on RMS-optimal phase lag
        phase_lag = rms_optimal_lags_1d[disp_idx]
        expected_peak_time = ref_peak_time_1d + phase_lag

        # Mark expected peak with red dot
        first_valid_idx = np.where(valid)[0][0]
        last_valid_idx = np.where(valid)[0][-1]

        if time_in_periods_1d[first_valid_idx] <= expected_peak_time <= time_in_periods_1d[last_valid_idx]:
            # Plot red dot with uncertainty bar
            ax1.errorbar(
                expected_peak_time,
                offset+1.2,
                xerr=rms_uncertainties_1d[disp_idx],
                fmt="o",
                color="red",
                markersize=8,
                capsize=4,
                elinewidth=2,
                capthick=2,
                zorder=10,
            )

        # Label altitude
        ax1.text(
            time_in_periods_1d[last_valid_idx] + 0.1,
            offset,
            f"{actual_alts_1d[disp_idx]:.0f} km",
            fontsize=20,
            va="center",
            # color=color,
            color="black",
        )

ax1.axvline(
    ref_peak_time_1d,
    color="blue",
    linestyle=":",
    linewidth=2,
    alpha=0.6,
    label=f"Reference peak at {ref_peak_time_1d:.2f} T",
)
ax1.set_xlabel("Time since wave start (wave periods)", fontsize=20)
ax1.set_xlabel("Time since wave start (wave periods)", fontsize=20)
ax1.set_ylabel("Vertical Normalized Density Perturbation + Offset", fontsize=20)
# Set x-axis limits relative to the reference cycle window
cycle_start_in_periods_1d = (cycle_start_time_ref_1d - wave_start_1d) / period_1d
cycle_end_in_periods_1d = (cycle_end_time_ref_1d - wave_start_1d) / period_1d
ax1.set_xlim([cycle_start_in_periods_1d - 0.2, cycle_end_in_periods_1d + max_phase_delay_1d + 0.6])
ax1.grid(alpha=0.3)

# --------------------------------------------------------------------------
# Panel (b): 2D Simulation Stacked Waveforms
# --------------------------------------------------------------------------
ax2 = axes[1]
offsets_2d = np.linspace(0, len(display_indices_2d) * 2.5, len(display_indices_2d))

# Find reference peak in 2D waveforms
ref_wf_2d = waveforms_masked_2d[0]
ref_valid_2d = ~np.isnan(ref_wf_2d)
ref_wf_valid_2d = ref_wf_2d[ref_valid_2d]
ref_time_valid_2d = time_in_periods_2d[ref_valid_2d]

# Normalize and find first peak
ref_wf_norm_2d = (ref_wf_valid_2d - np.mean(ref_wf_valid_2d)) / np.std(ref_wf_valid_2d)
peaks_2d, _ = find_peaks(
    ref_wf_norm_2d, prominence=0.5, distance=int(0.8 / (ref_time_valid_2d[1] - ref_time_valid_2d[0]))
)

if len(peaks_2d) > 0:
    ref_peak_time_2d = ref_time_valid_2d[peaks_2d[0]]
else:
    ref_peak_time_2d = ref_time_valid_2d[np.argmax(ref_wf_norm_2d)]

for i, (disp_idx, offset) in enumerate(zip(display_indices_2d, offsets_2d)):
    color = colors[i]
    wf = waveforms_masked_2d[disp_idx]
    valid = ~np.isnan(wf)

    if np.sum(valid) > 10:
        wf_valid = wf[valid]
        wf_normalized = np.full_like(wf, np.nan)
        wf_normalized[valid] = (wf_valid - np.mean(wf_valid)) / np.std(wf_valid)

        # Plot waveform
        ax2.plot(time_in_periods_2d, wf_normalized + offset, color="black", lw=3, alpha=0.8)

        # Calculate expected peak position based on RMS-optimal phase lag
        phase_lag = rms_optimal_lags_2d[disp_idx]
        expected_peak_time = ref_peak_time_2d + phase_lag

        # Mark expected peak with red dot
        first_valid_idx = np.where(valid)[0][0]
        last_valid_idx = np.where(valid)[0][-1]

        if time_in_periods_2d[first_valid_idx] <= expected_peak_time <= time_in_periods_2d[last_valid_idx]:
            # Plot red dot with uncertainty bar
            ax2.errorbar(
                expected_peak_time+0.04,
                offset+1.2,
                xerr=rms_uncertainties_2d[disp_idx],
                fmt="o",
                color="red",
                markersize=8,
                capsize=4,
                elinewidth=2,
                capthick=2,
                zorder=10,
            )

        # Label altitude
        ax2.text(
            time_in_periods_2d[last_valid_idx] + 0.1,
            offset,
            f"{actual_alts_2d[disp_idx]:.0f} km",
            fontsize=20,
            va="center",
            # color=color,
            color="black",
        )

ax1.tick_params(axis="both", labelsize=18)
ax2.tick_params(axis="both", labelsize=18)

# remove y-axis tick marks and labels
ax1.tick_params(axis="y", which="both", left=False, labelleft=False)
ax2.tick_params(axis="y", which="both", left=False, labelleft=False)

ax2.axvline(
    ref_peak_time_2d,
    color="blue",
    linestyle=":",
    linewidth=2,
    alpha=0.6,
    label=f"Reference peak at {ref_peak_time_2d:.2f} T",
)
ax2.set_xlabel("Time since wave start (wave periods)", fontsize=20)
ax2.set_xlabel("Time since wave start (wave periods)", fontsize=20)
ax2.set_ylabel("Vertical Normalized Density Perturbation + Offset", fontsize=20)
# Set x-axis limits relative to the reference cycle window
cycle_start_in_periods_2d = (cycle_start_time_ref_2d - wave_start_2d) / period_2d
cycle_end_in_periods_2d = (cycle_end_time_ref_2d - wave_start_2d) / period_2d
ax2.set_xlim([cycle_start_in_periods_2d - 0.2, cycle_end_in_periods_2d + max_phase_delay_2d + 0.6])
# ax2.legend(fontsize=10, loc='upper right')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("wavelength_analysis_1D_2D_hybrid_stacked_waveforms.png", dpi=300, bbox_inches="tight")
plt.show()