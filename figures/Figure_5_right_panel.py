#!/usr/bin/env python3
"""
Multi-Simulation Averaged Phase Lag Analysis

Compares horizontal phase lag measurements across multiple azimuthal resolutions
using averaged LEFT+RIGHT waveforms. Implements the 3-pass analysis with 2-stage
RMS optimization for each simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.odr import ODR, Model, RealData
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
SKIP_128_AZI = True

print("=" * 80)
print("MULTI-SIMULATION AVERAGED PHASE LAG ANALYSIS")
print("=" * 80)
print(f"Wave period: {WAVE_PERIOD:.1f} s")
print(f"Analysis altitude: {REFERENCE_ALTITUDE_KM} km")
print(f"Max horizontal distance: {MAX_DISTANCE_KM} km")
print()

# ============================================================================
# 1. SIMULATION LIST AND LOADING
# ============================================================================
print("[1/4] Loading simulations...")

SIMS_2D = [
    {"path": "S2_256_azi", "label": "256 azi", "n_azi": 256},
    {"path": "S2_384_azi", "label": "384 azi", "n_azi": 384},
    {"path": "S2_512_azi", "label": "512 azi", "n_azi": 512},
    {"path": "S2_640_azi", "label": "640 azi", "n_azi": 640},
    {"path": "S2_1024_azi", "label": "1024 azi", "n_azi": 1024},
]

# Load all simulations
loaded_sims = []
for sim in SIMS_2D:
    if SKIP_128_AZI and sim["n_azi"] == 128:
        continue

    sim_path = sim["path"]
    if not os.path.exists(sim_path):
        raise FileNotFoundError(f"Simulation path not found: {sim_path}")


    print(f"  Loading {sim['label']}...")
    processor = OptimizedDSMCDataProcessor(sim_path)
    processor.load_parameters()
    processor.load_data()

    loaded_sims.append({"sim": sim, "processor": processor})

print(f"\n  ✓ Loaded {len(loaded_sims)} simulations")
print()


# ============================================================================
# 2. ANALYSIS LOOP: PROCESS EACH SIMULATION
# ============================================================================
print("[2/4] Running 3-pass phase lag analysis for each simulation...")
print()

results = []


def shift_with_mask(data, mask, lag_samples):
    """Shift data and mask arrays by lag_samples."""
    if abs(lag_samples) >= len(data):
        return None, None
    if lag_samples > 0:
        pad = np.zeros(lag_samples, dtype=data.dtype)
        pad_mask = np.ones(lag_samples, dtype=bool)
        return np.concatenate([data[lag_samples:], pad]), np.concatenate([mask[lag_samples:], pad_mask])
    if lag_samples < 0:
        pad = np.zeros(-lag_samples, dtype=data.dtype)
        pad_mask = np.ones(-lag_samples, dtype=bool)
        return np.concatenate([pad, data[:lag_samples]]), np.concatenate([pad_mask, mask[:lag_samples]])
    return data.copy(), mask.copy()


for entry in loaded_sims:
    sim = entry["sim"]
    processor = entry["processor"]
    n_azi = sim["n_azi"]

    print(f"  [{sim['label']}] Starting analysis...")

    # Get parameters
    time = processor.params.time
    radial_cells = processor.params.radial_cells
    wave_start = processor.params.start_step_wave * processor.params.timestep
    wave_end = processor.params.final_step_wave * processor.params.timestep
    bg = processor.get_background_profiles()
    r_idx = np.argmin(np.abs(radial_cells - REFERENCE_ALTITUDE_KM))

    # Get wedge edges
    osc_cols = processor.compute_osc_cols()
    jL = int(osc_cols[0])
    jR = int(osc_cols[-1])
    center_col = int((jL + jR) / 2)

    # Detect wave period
    wave_mask = (time >= wave_start) & (time <= wave_end)
    wave_indices = np.where(wave_mask)[0]
    density_ts = processor.density_data[wave_indices, r_idx, center_col]
    perturbation = (density_ts - bg["density"][r_idx]) / bg["density"][r_idx] * 100
    period, _, _, _ = detect_wave_period(time[wave_mask], perturbation)

    # Compute horizontal resolution
    azi_edges_rad = np.deg2rad(processor.params.azi_edges)
    d_phi_km = np.mean(np.diff(azi_edges_rad)) * (MARS_RADIUS + REFERENCE_ALTITUDE_KM * 1000) / 1000.0

    # Generous window
    generous_start = wave_start + 1 * period
    generous_end = min(wave_start + 7 * period, wave_end)
    generous_mask = (time >= generous_start) & (time <= generous_end)
    generous_times = time[generous_mask]
    generous_indices = np.where(generous_mask)[0]

    # Extract and average LEFT+RIGHT waveforms
    num_steps = int(np.ceil(MAX_DISTANCE_KM / d_phi_km))
    num_azimuth = processor.density_data.shape[2]

    waveforms = []
    distances = []

    for step in range(num_steps + 1):
        left_col = jL - step
        right_col = jR + step

        if left_col >= 0 and right_col < num_azimuth:
            left_dens = processor.density_data[generous_indices, r_idx, left_col]
            right_dens = processor.density_data[generous_indices, r_idx, right_col]
            avg_dens = (left_dens + right_dens) / 2.0
            pert = (avg_dens - bg["density"][r_idx]) / bg["density"][r_idx] * 100
            waveforms.append(pert)
            distances.append(step * d_phi_km)

    waveforms = np.array(waveforms)
    distances = np.array(distances)

    if np.isnan(waveforms).any():
        raise ValueError(f"NaNs in waveforms for {sim['label']}")

    # PASS 1: Cross-correlation
    dt_step = generous_times[1] - generous_times[0]
    max_lag_samples = int(1.5 * period / dt_step)
    ref_norm = (waveforms[0] - np.mean(waveforms[0])) / np.std(waveforms[0])

    phase_delays = np.zeros(len(distances))
    for i in range(1, len(distances)):
        wf_norm = (waveforms[i] - np.mean(waveforms[i])) / np.std(waveforms[i])
        xcorr = np.correlate(wf_norm, ref_norm, mode="full")
        lags = np.arange(-len(ref_norm) + 1, len(ref_norm))
        center = len(ref_norm) - 1
        search_slice = slice(max(0, center - max_lag_samples), min(len(xcorr), center + max_lag_samples))
        peak_idx = search_slice.start + np.argmax(xcorr[search_slice])
        phase_delays[i] = abs(lags[peak_idx] * dt_step / period)
        if phase_delays[i] < phase_delays[i - 1]:
            phase_delays[i] = phase_delays[i - 1]

    # PASS 2: Adaptive windowing
    extended_end = min(wave_start + 6 * period, time[-1])
    wave_mask_full = (time >= generous_start) & (time <= extended_end)
    wave_times_full = time[wave_mask_full]
    wave_indices_full = np.where(wave_mask_full)[0]

    ref_left = processor.density_data[wave_indices_full, r_idx, jL]
    ref_right = processor.density_data[wave_indices_full, r_idx, jR]
    ref_pert_full = ((ref_left + ref_right) / 2.0 - bg["density"][r_idx]) / bg["density"][r_idx] * 100

    zero_crossings = []
    for i in range(1, len(ref_pert_full)):
        if ref_pert_full[i - 1] < 0 and ref_pert_full[i] >= 0:
            frac = -ref_pert_full[i - 1] / (ref_pert_full[i] - ref_pert_full[i - 1])
            zero_crossings.append(wave_times_full[i - 1] + frac * (wave_times_full[i] - wave_times_full[i - 1]))
    zero_crossings = np.array(zero_crossings)

    target_cycle = 3 if len(zero_crossings) >= 2 else max(2, len(zero_crossings))
    cycle_start = zero_crossings[target_cycle - 2] if len(zero_crossings) >= target_cycle - 1 else generous_start
    cycle_end = zero_crossings[target_cycle] if target_cycle < len(zero_crossings) else cycle_start + 2 * period

    window_starts = cycle_start + phase_delays * period
    window_ends = cycle_end + phase_delays * period
    global_start = window_starts.min()
    global_end = min(window_ends.max(), time[-1])

    global_mask = (time >= global_start) & (time <= global_end)
    global_times = time[global_mask]
    global_indices = np.where(global_mask)[0]

    waveforms_masked = []
    waveforms_mask = []

    for i in range(len(distances)):
        step = i
        left_col = jL - step
        right_col = jR + step

        if left_col >= 0 and right_col < num_azimuth:
            left_dens = processor.density_data[global_indices, r_idx, left_col]
            right_dens = processor.density_data[global_indices, r_idx, right_col]
            avg_dens = (left_dens + right_dens) / 2.0
            pert = (avg_dens - bg["density"][r_idx]) / bg["density"][r_idx] * 100

            if np.isnan(pert).any():
                raise ValueError(f"NaNs in masked window for {sim['label']}")

            local_mask = (global_times >= window_starts[i]) & (global_times <= window_ends[i])
            waveforms_masked.append(pert)
            waveforms_mask.append(~local_mask)

    waveforms_masked = np.array(waveforms_masked)
    waveforms_mask = np.array(waveforms_mask)

    # PASS 3: 2-stage RMS optimization
    dt_periods = (time[1] - time[0]) / period
    rms_optimal_lags = np.zeros(len(distances))

    for i in range(1, len(distances)):
        ref_mask = waveforms_mask[0]
        target_mask = waveforms_mask[i]
        ref_valid = ~ref_mask
        target_valid = ~target_mask

        if np.sum(ref_valid) < 10 or np.sum(target_valid) < 10:
            rms_optimal_lags[i] = phase_delays[i]
            continue

        ref_data = waveforms_masked[0]
        target_data = waveforms_masked[i]

        ref_std = np.std(ref_data[ref_valid])
        target_std = np.std(target_data[target_valid])
        if ref_std == 0 or target_std == 0:
            raise ValueError(f"Zero std for {sim['label']}")

        ref_mean = np.mean(ref_data[ref_valid])
        target_mean = np.mean(target_data[target_valid])
        ref_n = (ref_data - ref_mean) / ref_std
        target_n = (target_data - target_mean) / target_std

        # Stage 1: Coarse
        search_min = max(0, phase_delays[i] - 0.2)
        search_max = phase_delays[i] + 0.2
        test_lags = np.linspace(search_min, search_max, 80)
        rms_vals = np.full_like(test_lags, np.inf, dtype=float)

        for j, shift in enumerate(test_lags):
            lag_samples = int(round(shift / dt_periods))
            shifted, shifted_mask = shift_with_mask(target_n, target_mask, lag_samples)
            if shifted is None:
                continue
            both_valid = (~ref_mask) & (~shifted_mask)
            if np.sum(both_valid) >= 10:
                rms_vals[j] = np.sqrt(np.mean((ref_n[both_valid] - shifted[both_valid]) ** 2))

        coarse_idx = int(np.argmin(rms_vals))
        coarse = test_lags[coarse_idx]

        # Stage 2: Fine
        refine_min = max(search_min, coarse - 0.05)
        refine_max = min(search_max, coarse + 0.05)
        refine_lags = np.linspace(refine_min, refine_max, 80)
        refine_rms = np.full_like(refine_lags, np.inf, dtype=float)

        for j, shift in enumerate(refine_lags):
            lag_samples = int(round(shift / dt_periods))
            shifted, shifted_mask = shift_with_mask(target_n, target_mask, lag_samples)
            if shifted is None:
                continue
            both_valid = (~ref_mask) & (~shifted_mask)
            if np.sum(both_valid) >= 10:
                refine_rms[j] = np.sqrt(np.mean((ref_n[both_valid] - shifted[both_valid]) ** 2))

        rms_optimal_lags[i] = refine_lags[int(np.argmin(refine_rms))]

    if np.isnan(rms_optimal_lags).any():
        raise ValueError(f"NaNs in rms_optimal_lags for {sim['label']}")

    # Store results
    results.append({
        "label": sim["label"],
        "n_azi": n_azi,
        "distances": distances,
        "phase_lags": rms_optimal_lags,
        "dt_periods": dt_periods,
        "d_phi_km": d_phi_km,
    })

    print(f"    ✓ Phase lag range: 0 - {rms_optimal_lags[-1]:.3f} periods")

print(f"\n  ✓ Completed analysis for {len(results)} simulations")
print()


# ============================================================================
# 3. PHASE LAG VS DISTANCE COMPARISON PLOT
# ============================================================================
print("[3/4] Creating phase lag comparison plot...")

fig, ax = plt.subplots(figsize=(7, 7))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))

# label the simulations from S2-S6 (map the results)
label_map = {
    "256 azi": "S2",
    "384 azi": "S3",
    "512 azi": "S4",
    "640 azi": "S5",
    "1024 azi": "S6",
}

for i, res in enumerate(results):
    distances = res["distances"]
    lags = res["phase_lags"]
    dt_periods = res["dt_periods"]
    d_phi_km = res["d_phi_km"]
    mask = distances <= MAX_DISTANCE_KM

    ax.errorbar(
        lags[mask],
        distances[mask],
        xerr=dt_periods,
        yerr=d_phi_km / 2,
        fmt="o-",
        lw=2,
        markersize=5,
        capsize=3,
        capthick=1.2,
        color=colors[i],
        ecolor=colors[i],
        alpha=0.8,
        label=f"{label_map.get(res['label'], res['label'])}",
    )

ax.set_xlabel("Phase Lag (periods)", fontsize=20)
ax.set_ylabel("Distance from wedge edge (km)", fontsize=20)
ax.set_title(
    f"Averaged Horizontal Phase Lag Comparison at {REFERENCE_ALTITUDE_KM} km altitude\n(LEFT + RIGHT)",
    fontsize=14,
    fontweight="bold",
)
ax.grid(alpha=0.5)
ax.set_xlim(0, None)
ax.set_ylim(0, 330)
ax.legend(fontsize=11)
# ax.tick_params(axis="both", labelsize=14)
ax.tick_params(axis="both", labelsize=18)

plt.tight_layout()

output_path = FIG_OUTPUT_DIR / "phase_lag_comparison_all_simulations_averaged.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: {output_path}")
print()


# ============================================================================
# 4. WAVELENGTH ESTIMATION FOR EACH SIMULATION
# ============================================================================
print("[4/4] Computing horizontal wavelengths...")
print()


def linear_func(B, x):
    """Linear model: y = B[0] + B[1]*x"""
    return B[0] + B[1] * x


wavelength_results = []

print("=" * 80)
print("HORIZONTAL WAVELENGTH ANALYSIS (ODR Linear Regression)")
print("=" * 80)

for res in results:
    n_azi = res["n_azi"]
    distances = res["distances"]
    phase_lags = res["phase_lags"]
    dt_periods = res["dt_periods"]
    d_phi_km = res["d_phi_km"]

    # Apply mask
    mask = distances <= MAX_DISTANCE_KM
    x_data = distances[mask]
    y_data = phase_lags[mask]

    # Uncertainties
    x_err = d_phi_km / 2.0
    y_err = dt_periods

    # ODR fit
    linear_model = Model(linear_func)
    data = RealData(x_data, y_data, sx=x_err, sy=y_err)
    odr = ODR(data, linear_model, beta0=[0.0, 0.001])
    output = odr.run()

    intercept = output.beta[0]
    slope = output.beta[1]
    intercept_err = output.sd_beta[0]
    slope_err = output.sd_beta[1]

    wavelength = 1.0 / slope
    wavelength_err = slope_err / (slope**2)

    wavelength_results.append({
        "n_azi": n_azi,
        "label": res["label"],
        "slope": slope,
        "slope_err": slope_err,
        "intercept": intercept,
        "intercept_err": intercept_err,
        "wavelength_km": wavelength,
        "wavelength_err_km": wavelength_err,
    })

    print(f"\n{res['label']} (averaged):")
    print(f"  Slope:       {slope:.6f} ± {slope_err:.6f} periods/km")
    print(f"  Intercept:   {intercept:.6f} ± {intercept_err:.6f} periods")
    print(f"  Wavelength:  {wavelength:.2f} ± {wavelength_err:.2f} km/period")

print("\n" + "=" * 80)


# ============================================================================
# 5. WAVELENGTH CONVERGENCE PLOT (OPTIONAL)
# ============================================================================
print()
print("[5/4] Creating wavelength convergence plot...")

fig, ax = plt.subplots(figsize=(10, 6))

n_azi_vals = [r["n_azi"] for r in wavelength_results]
wavelengths = [r["wavelength_km"] for r in wavelength_results]
wavelength_errs = [r["wavelength_err_km"] for r in wavelength_results]

ax.errorbar(
    n_azi_vals,
    wavelengths,
    yerr=wavelength_errs,
    marker="o",
    markersize=8,
    capsize=5,
    capthick=2,
    linewidth=2,
    label="Horizontal Wavelength (averaged)",
)

ax.set_xlabel("Number of Azimuthal Cells", fontsize=14)
ax.set_ylabel("Horizontal Wavelength (km/period)", fontsize=14)
ax.set_title(
    "Horizontal Wavelength vs Azimuthal Resolution\n(Averaged LEFT+RIGHT, ODR Fit)",
    fontsize=15,
    fontweight="bold",
)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=12)
ax.tick_params(axis="both", labelsize=12)

plt.tight_layout()

output_path_conv = FIG_OUTPUT_DIR / "horizontal_wavelength_convergence_averaged.png"
plt.savefig(output_path_conv, dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: {output_path_conv}")

print()

