#!/usr/bin/env python3
"""
Horizontal decay profiles from edges of perturbation (Figure 3, right panel)

This script loads horizontal waveform results from all 6 simulations, extracts profiles at specific altitudes,
and plots them together to illustrate the decay of wave amplitudes with distance from the forcing edge.
It also calculates and annotates the 1/e decay distance for each altitude based on the reference simulation (S6).
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy.interpolate import interp1d

# Configure paths and parameters
BASE_DIR = Path(__file__).parent
SIM_FOLDERS = ["S1_128_azi", "S2_256_azi", "S3_384_azi", "S4_512_azi", "S5_640_azi", "S6_1024_azi"]
TARGET_ALTS = [150, 200, 230, 280, 320]

class CrossPlatformUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "WindowsPath": return Path
        return super().find_class(module, name)

# Load transport and simulation data
transport = np.load(BASE_DIR / "pre_wave_transport_profiles.npz")
mfp_interp = interp1d(transport["radial_cells"], transport["mean_free_path"] / 1e3, fill_value="extrapolate")

all_results = []
for folder in SIM_FOLDERS:
    with open(BASE_DIR / folder / "horizontal_results.pkl", "rb") as f:
        all_results.append(CrossPlatformUnpickler(f).load())

ref_res = all_results[-1]
ref_alts = ref_res["altitudes"]
alt_indices = []
for t in TARGET_ALTS:
    idx = np.argmin(np.abs(ref_alts - t))
    alt_indices.append(idx)
    print(f"Target altitude {t} km → index {idx} (actual: {ref_alts[idx]:.1f} km)")

# Plotting setup
fig, axes = plt.subplots(5, 1, figsize=(4.5, 10), sharey=True, sharex=True)
colors = [plt.cm.viridis(i/5) for i in range(6)]

for i, alt_idx in enumerate(alt_indices[::-1]):
    ax = axes[i]
    # Plot S2-S6 simulations
    for idx in range(1, 6):
        res = all_results[idx]
        d = res["left_distances_km"][alt_idx]
        a = res["left_amplitudes"][alt_idx]
        u = res["left_amplitude_uncertainties"][alt_idx]
        mask = d < 320
        ax.errorbar(d[mask], a[mask], yerr=2*u[mask], marker="o", markersize=6, alpha=0.6, lw=4, color=colors[idx], label=f"S{idx+1}")

    # 1/e decay line based on reference (S6)
    rd, ra = ref_res["left_distances_km"][alt_idx], ref_res["left_amplitudes"][alt_idx]
    e_decay_val = ra[-1] * np.exp(-1)
    e_idx = np.where(ra < e_decay_val)[0][-1]
    ax.axvline(rd[e_idx], color="gray", ls="--", lw=2, alpha=0.7)
    print(f"Altitude {ref_alts[alt_idx]:.1f} km: 1/e decay distance = {rd[e_idx]:.1f} km")

    # Formatting
    ax.text(0.98, 0.92, f"$r-R_M \\approx$ {round(ref_alts[alt_idx], -1):.0f} km", transform=ax.transAxes,
            fontsize=16, va="top", ha="right", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.15))
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=20, direction="out", top=False, right=False)

# Global labels and limits
axes[2].set_ylabel("Normalized Density Amplitude", fontsize=22)
axes[-1].set_xlabel("Distance from forcing edge (km)", fontsize=22)
axes[-1].set_yticks([0, 1, 2, 3])
axes[-1].set_ylim(0, 3.1)
fig.subplots_adjust(right=1.045)
plt.show()
