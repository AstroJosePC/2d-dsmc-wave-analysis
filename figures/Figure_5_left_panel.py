#!/usr/bin/env python3
"""
Complete Averaged Horizontal Waveform Analysis (Figure 5, left panel)

This script implements the full 3-pass phase lag analysis for averaged LEFT+RIGHT waveforms
from the 1024 azimuthal resolution DSMC simulation, including:
- Pass 1: Cross-correlation phase lag estimation
- Pass 2: Adaptive windowing based on phase lags
- Pass 3: RMS optimization for refined phase lags
- Final visualization with faded noisy waveforms and smooth sinusoidal fits
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar, curve_fit
from pathlib import Path
import os

from optimized_loader_zenodo import OptimizedDSMCDataProcessor
from helper_funcs_zenodo import detect_wave_period

sns.set_theme(context="paper", font_scale=2.0, style="whitegrid")

# Output directory
FIG_OUTPUT_DIR = Path("horizontal_phase_lag_figures")
FIG_OUTPUT_DIR.mkdir(exist_ok=True)

# Physical constants
OMEGA = 9.5e-3  # rad/s
WAVE_PERIOD = 2 * np.pi / OMEGA
MARS_RADIUS = 3390e3  # meters

# Analysis parameters
MAX_DISTANCE_KM = 300
REFERENCE_ALTITUDE_KM = 230

# Simulation path
SIM_PATH = os.getenv("DSMC_SIM_PATH")
if SIM_PATH is None:
    raise ValueError("Please set the DSMC_SIM_PATH environment variable to the simulation (N_azi=1024) directory path.")

print("=" * 80)
print("COMPLETE AVERAGED HORIZONTAL WAVEFORM ANALYSIS")
print("=" * 80)
print(f"Wave period: {WAVE_PERIOD:.1f} s")
print(f"Analysis altitude: {REFERENCE_ALTITUDE_KM} km")
print(f"Max horizontal distance: {MAX_DISTANCE_KM} km")
print()

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("[1/8] Loading DSMC data...")
processor = OptimizedDSMCDataProcessor(SIM_PATH)
processor.load_parameters()
processor.load_data()

params = processor.params
time = params.time
radial_cells = params.radial_cells
wave_start = params.start_step_wave * params.timestep
wave_end = params.final_step_wave * params.timestep

bg = processor.get_background_profiles()
osc_cols = processor.compute_osc_cols()
jL, jR = int(osc_cols[0]), int(osc_cols[-1])

print(f"  ✓ Density shape: {processor.density_data.shape}")
print(f"  ✓ Wedge edges: Left={jL}, Right={jR}")
print()

# Detect wave period
r_idx = np.argmin(np.abs(radial_cells - REFERENCE_ALTITUDE_KM))
center_col = int((jL + jR) / 2)

wave_mask = (time >= wave_start) & (time <= wave_end)
wave_times = time[wave_mask]
wave_indices = np.where(wave_mask)[0]

density_ts = processor.density_data[wave_indices, r_idx, center_col]
perturbation = (density_ts - bg["density"][r_idx]) / bg["density"][r_idx] * 100
period, freq, _, _ = detect_wave_period(wave_times, perturbation)

print(f"  ✓ Detected period: {period:.1f} s")

# Compute horizontal resolution
azi_edges_rad = np.deg2rad(params.azi_edges)
mean_dphi = np.mean(np.diff(azi_edges_rad))
r_m = MARS_RADIUS + REFERENCE_ALTITUDE_KM * 1000
d_phi_km = mean_dphi * r_m / 1000.0

print(f"  ✓ Horizontal resolution: {d_phi_km:.2f} km/cell")
print()


# ============================================================================
# 2. EXTRACT AND AVERAGE WAVEFORMS
# ============================================================================
def extract_edge_waveforms(ref_col, target_cols):
    """Extract waveforms for given columns."""
    generous_start = wave_start
    generous_end = wave_start + 7 * period
    generous_mask = (time >= generous_start) & (time <= generous_end)
    generous_indices = np.where(generous_mask)[0]

    waveforms = []
    for col_idx in [ref_col] + target_cols:
        density_ts = processor.density_data[generous_indices, r_idx, col_idx]
        pert = (density_ts - bg["density"][r_idx]) / bg["density"][r_idx] * 100
        waveforms.append(pert)

    return np.array(waveforms), generous_indices


print("[2/8] Extracting LEFT edge waveforms...")
max_steps = int(np.ceil(MAX_DISTANCE_KM / d_phi_km))
left_target_cols = [jL - step for step in range(1, max_steps + 1) if jL - step >= 0]
left_waveforms, generous_indices = extract_edge_waveforms(jL, left_target_cols)
print(f"  ✓ Extracted {len(left_waveforms)} waveforms")

print("[3/8] Extracting RIGHT edge waveforms...")
num_azimuth = processor.density_data.shape[2]
right_target_cols = [jR + step for step in range(1, max_steps + 1) if jR + step < num_azimuth]
right_waveforms, _ = extract_edge_waveforms(jR, right_target_cols)
print(f"  ✓ Extracted {len(right_waveforms)} waveforms")

print("[4/8] Averaging LEFT and RIGHT waveforms...")
min_num = min(len(left_waveforms), len(right_waveforms))
averaged_waveforms = (left_waveforms[:min_num] + right_waveforms[:min_num]) / 2.0
distances = np.array([i * d_phi_km for i in range(min_num)])
print(f"  ✓ Averaged {min_num} waveforms")
print(f"  ✓ Distance range: 0 - {distances[-1]:.1f} km")
print()

# Time arrays
generous_times = time[generous_indices]
time_generous_periods = (generous_times - wave_start) / period


# ============================================================================
# 3. PASS 1: CROSS-CORRELATION PHASE LAG ESTIMATION
# ============================================================================
print("[5/8] Pass 1: Cross-correlation phase lag estimation...")

ref_waveform = averaged_waveforms[0]
ref_norm = (ref_waveform - np.mean(ref_waveform)) / np.std(ref_waveform)

phase_delays_xcorr = [0.0]
dt_step = generous_times[1] - generous_times[0]

for i in range(1, len(averaged_waveforms)):
    target = averaged_waveforms[i]
    target_norm = (target - np.mean(target)) / np.std(target)

    xcorr = np.correlate(target_norm, ref_norm, mode="full")
    lags = np.arange(-len(ref_norm) + 1, len(ref_norm))

    max_lag_samples = int(1.5 * period / dt_step)
    center = len(ref_norm) - 1
    search_start = max(0, center - max_lag_samples)
    search_end = min(len(xcorr), center + max_lag_samples)

    peak_idx = search_start + np.argmax(xcorr[search_start:search_end])
    lag_samples = lags[peak_idx]
    lag_periods = abs(lag_samples * dt_step / period)

    phase_delays_xcorr.append(lag_periods)

phase_delays_xcorr = np.array(phase_delays_xcorr)

# Enforce monotonicity
for i in range(1, len(phase_delays_xcorr)):
    if phase_delays_xcorr[i] < phase_delays_xcorr[i - 1]:
        phase_delays_xcorr[i] = phase_delays_xcorr[i - 1]

print(f"  ✓ Phase lag range: 0 - {phase_delays_xcorr[-1]:.3f} periods")
print()


# ============================================================================
# 4. PASS 2: ADAPTIVE WINDOWING
# ============================================================================
print("[6/8] Pass 2: Adaptive windowing based on phase lags...")

# Find zero crossings in reference
ref_perturbation = averaged_waveforms[0]
crossings = []
for i in range(1, len(ref_perturbation)):
    if ref_perturbation[i - 1] < 0 and ref_perturbation[i] >= 0:
        frac = -ref_perturbation[i - 1] / (ref_perturbation[i] - ref_perturbation[i - 1])
        t_cross = generous_times[i - 1] + frac * (generous_times[i] - generous_times[i - 1])
        crossings.append(t_cross)

print(f"  ✓ Found {len(crossings)} zero crossings")

# Target cycles 4-5 (2-period tracking window)
if len(crossings) >= 4:
    cycle_start = crossings[4]
    cycle_end = crossings[6] if len(crossings) > 6 else cycle_start + 2 * period
else:
    cycle_start = wave_start
    cycle_end = wave_start + 7 * period

print(f"  ✓ Tracking window: {cycle_start:.0f} - {cycle_end:.0f} s")

# Create phase-shifted windows
window_starts = cycle_start + phase_delays_xcorr * period
window_ends = cycle_end + phase_delays_xcorr * period

global_start = window_starts.min()
global_end = min(window_ends.max(), time[-1])

global_mask = (time >= global_start) & (time <= global_end)
global_times = time[global_mask]
global_indices = np.where(global_mask)[0]

# Extract averaged waveforms in global window
waveforms_global = []
for col_idx in list(range(min_num)):
    # Need to re-extract for the global window
    # We'll extract from both edges and average
    left_col = jL - col_idx
    right_col = jR + col_idx

    if left_col >= 0 and right_col < num_azimuth:
        left_dens = processor.density_data[global_indices, r_idx, left_col]
        right_dens = processor.density_data[global_indices, r_idx, right_col]
        avg_dens = (left_dens + right_dens) / 2.0
        pert = (avg_dens - bg["density"][r_idx]) / bg["density"][r_idx] * 100
        waveforms_global.append(pert)
    else:
        waveforms_global.append(np.full(len(global_times), np.nan))

waveforms_global = np.array(waveforms_global)

# Apply masks for each waveform's tracking window
waveforms_masked = []
for i in range(len(waveforms_global)):
    local_mask = (global_times >= window_starts[i]) & (global_times <= window_ends[i])
    pert_masked = np.where(local_mask, waveforms_global[i], np.nan)
    waveforms_masked.append(pert_masked)
waveforms_masked = np.array(waveforms_masked)

time_in_periods = (global_times - wave_start) / period
print(f"  ✓ Global window: {global_start:.0f} - {global_end:.0f} s")
print()


# ============================================================================
# 5. PASS 3: RMS OPTIMIZATION
# ============================================================================
print("[7/8] Pass 3: RMS optimization for refined phase lags...")

rms_optimal_lags = np.zeros(len(distances))
dt_periods = (time[1] - time[0]) / period

for i in range(len(distances)):
    if i == 0:
        rms_optimal_lags[i] = 0.0
    else:

        def compute_rms(shift_periods):
            dt_step = time_in_periods[1] - time_in_periods[0]
            lag_samples = int(round(shift_periods / dt_step))

            wf_ref = waveforms_masked[0]
            wf_target = waveforms_masked[i]

            ref_valid = ~np.isnan(wf_ref)
            target_valid = ~np.isnan(wf_target)

            if np.sum(ref_valid) < 10 or np.sum(target_valid) < 10:
                return np.inf

            wf_ref_norm = np.full_like(wf_ref, np.nan)
            wf_target_norm = np.full_like(wf_target, np.nan)

            wf_ref_norm[ref_valid] = (wf_ref[ref_valid] - np.mean(wf_ref[ref_valid])) / np.std(wf_ref[ref_valid])
            wf_target_norm[target_valid] = (wf_target[target_valid] - np.mean(wf_target[target_valid])) / np.std(
                wf_target[target_valid]
            )

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

            both_valid = (~np.isnan(wf_ref_norm)) & (~np.isnan(wf_shifted))
            if np.sum(both_valid) < 10:
                return np.inf

            residual = wf_ref_norm[both_valid] - wf_shifted[both_valid]
            return np.sqrt(np.mean(residual**2))

        search_width = 0.2
        initial_lag = phase_delays_xcorr[i]
        search_min = max(0.0, initial_lag - search_width)
        search_max = initial_lag + search_width

        result = minimize_scalar(compute_rms, bounds=(search_min, search_max), method="bounded")
        rms_optimal_lags[i] = result.x

uncertainties = np.full(len(distances), dt_periods)
uncertainties[0] = 0.0

print(f"  ✓ Optimized phase lags for {len(distances)} waveforms")
print(f"  ✓ Final phase lag range: 0 - {rms_optimal_lags[-1]:.3f} periods")
print()


# ============================================================================
# 6. FINAL VISUALIZATION
# ============================================================================
print("[8/8] Creating publication-quality visualization...")

# Select subset for display
valid_indices = np.where(distances <= MAX_DISTANCE_KM)[0]
# step = max(1, len(valid_indices) // 12)
# display_indices = valid_indices[::step].tolist()
# if display_indices[-1] != valid_indices[-1]:
#     display_indices.append(int(valid_indices[-1]))

# Force the following altitudes: 0, 40, 80, 120, 160, 200, 240, 280 km, use argmin to find closest indices
target_altitudes = np.arange(0, MAX_DISTANCE_KM + 1, 40)
display_indices = []
for alt in target_altitudes:
    idx = np.argmin(np.abs(distances - alt))
    if idx not in display_indices:
        display_indices.append(idx)

print(f"  ✓ Displaying {len(display_indices)} waveforms")

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(5.5, 7))
offsets = np.arange(len(display_indices)) * 3.8

# Plot each waveform
for i, idx in enumerate(display_indices):
    wf = waveforms_masked[idx]
    wf_generous = averaged_waveforms[idx]
    valid_mask = ~np.isnan(wf)

    if np.sum(valid_mask) < 5:
        continue

    # Normalize using tracked window statistics
    wf_valid = wf[valid_mask]
    wf_mean = np.mean(wf_valid)
    wf_std = np.std(wf_valid)
    wf_norm = (wf_valid - wf_mean) / wf_std
    wf_generous_norm = (wf_generous - wf_mean) / wf_std
    t_valid = time_in_periods[valid_mask]

    # Plot faded generous window data (noisy background)
    ax.plot(time_generous_periods, wf_generous_norm + offsets[i], linewidth=1.5, alpha=0.3, color="gray")

    # Baseline
    ax.hlines(offsets[i], time_in_periods[0], time_in_periods[-1], colors="k", linestyles="--", linewidth=2, alpha=0.5)

    # Fit and overlay smooth sinusoid
    if len(wf_norm) >= 5:
        try:
            offset_init = np.mean(wf_norm)
            amplitude_init = (np.max(wf_norm) - np.min(wf_norm)) / 2

            popt, _ = curve_fit(
                lambda t, amp, phase, off: amp * np.sin(2 * np.pi * t + phase) + off,
                t_valid,
                wf_norm,
                p0=[amplitude_init, 0.0, offset_init],
                maxfev=1000,
            )

            amp, phase, off = popt
            t_smooth = np.linspace(t_valid.min(), t_valid.max(), 200)
            wf_fit = amp * np.sin(2 * np.pi * t_smooth + phase) + off

            ax.plot(t_smooth, wf_fit + offsets[i], linewidth=3, alpha=0.95, color="k")
        except:
            pass

    # Distance label
    dist_label = f"{round(distances[idx], -1):.0f} km"
    ax.text(t_valid[-1] + 0.05, offsets[i] + 0.5, dist_label, fontsize=18, va="center", color="k", fontweight="bold")

# Add phase lag markers
boundary_tolerance = 5 * dt_periods

for i, idx in enumerate(display_indices):
    if idx == 0:
        continue

    wf = waveforms_masked[idx]
    valid_mask = ~np.isnan(wf)

    if np.sum(valid_mask) < 5:
        continue

    t_valid = time_in_periods[valid_mask]
    lag = rms_optimal_lags[idx]
    unc = uncertainties[idx]

    # Find reference peak
    ref_wf = waveforms_masked[0]
    ref_valid = ~np.isnan(ref_wf)

    if np.sum(ref_valid) > 5:
        ref_wf_norm = (ref_wf[ref_valid] - np.mean(ref_wf[ref_valid])) / np.std(ref_wf[ref_valid])
        ref_t = time_in_periods[ref_valid]
        ref_positive = ref_wf_norm > 0.5

        if np.any(ref_positive):
            ref_peak_t = ref_t[ref_positive][0]
            expected_peak_t = ref_peak_t + lag

            if (t_valid.min() - boundary_tolerance) <= expected_peak_t <= (t_valid.max() + boundary_tolerance):
                ax.errorbar(
                    expected_peak_t + 18 * dt_periods,
                    offsets[i] + 1.2,
                    xerr=unc,
                    fmt="o",
                    color="red",
                    markersize=8,
                    capsize=4,
                    elinewidth=2,
                    capthick=2,
                    zorder=10,
                )

# Reference time line
ax.axvline(time_in_periods[0], color="blue", linestyle=":", linewidth=4, alpha=0.6)

# Formatting
ax.set_xlim(2.0, 6.0)
ax.set_xlabel("Time since wave start (wave periods)", fontsize=20)
# ax.set_ylabel("Normalized Density Perturbation + Offset", fontsize=20)
ax.set_ylabel("Horizontal Normalized Density Perturbation + Offset", fontsize=18)
ax.tick_params(axis="both", labelsize=18)
# ax.set_title("Averaged Horizontal Waveforms (LEFT + RIGHT)", fontsize=16, fontweight="bold")
ax.grid(alpha=0.6)
ax.set_ylim(-3, offsets[-1] + 2)
ax.set_yticks([])

plt.tight_layout()

output_path = FIG_OUTPUT_DIR / "horizontal_waveforms_stacked_1024_azi_averaged_complete.png"
# plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\n  ✓ Saved: {output_path}")

print()
print("=" * 80)
print("COMPLETE ANALYSIS FINISHED")
print("=" * 80)
print("Phase lag results:")
print(f"  Cross-correlation range: 0 - {phase_delays_xcorr[-1]:.3f} periods")
print(f"  RMS-optimized range: 0 - {rms_optimal_lags[-1]:.3f} periods")
print(f"  Horizontal wavelength estimate: ~{distances[-1] / rms_optimal_lags[-1]:.0f} km/period")
