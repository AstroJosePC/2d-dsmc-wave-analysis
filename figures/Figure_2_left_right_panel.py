"""
Density growth profiles. Figure 2 left panel and right panel
This script generates the density growth profiles for the high-resolution 2D spherical DSMC simulation
and compares them with reference datasets, including Tian et al. 2023 and a 1D spherical model.
It also measures the exponential slopes of the growth curves to quantify the differences between the models.
+ The right panel compares the temperature profiles and heating rates, including a reference line with a slope of 9.4 K/H (T23)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

# Import the OptimizedDSMCDataProcessor
from optimized_loader_zenodo import OptimizedDSMCDataProcessor
from helper_funcs_zenodo import measure_exponential_slopes

# compute the scale height
M = 16  # g/mol
R = 8.314  # J/mol/K
R_prime = R / M * 1e3  # J/kg/K
T = 270  # K
g = 3.51  # m/s^2
H = R_prime * T / g / 1e3  # km

sns.set_theme(context="paper", font_scale=2.5, style="whitegrid")

high_res_processor = OptimizedDSMCDataProcessor("S6_1024_azi")
high_res_growth_data = high_res_processor.compute_density_growth()

# Select middle oscillating columns
osc_cols = high_res_processor.compute_osc_cols()
mid_idx = len(osc_cols) // 2
selected_cols_middle = osc_cols[max(0, mid_idx - 1) : min(len(osc_cols), mid_idx + 1)]
print(f"Selected middle oscillating columns: {selected_cols_middle}")

# Get growth curves for selected columns
combined_growth_middle = {}
for col in selected_cols_middle:
    altitudes, mean_growth, lower_ci, upper_ci = high_res_processor.get_density_growth_curves(high_res_growth_data, col, n_last_peaks=2)
    combined_growth_middle[col] = mean_growth

# Select outer oscillating columns
selected_cols_outer = [osc_cols[0], osc_cols[-1]]
print(f"Selected outer oscillating columns: {selected_cols_outer}")

# Get growth curves for selected columns
combined_growth_outer = {}
for col in selected_cols_outer:
    altitudes, mean_growth, lower_ci, upper_ci = high_res_processor.get_density_growth_curves(high_res_growth_data, col, n_last_peaks=2)
    combined_growth_outer[col] = mean_growth

avg_growth_middle = np.mean(list(combined_growth_middle.values()), axis=0)
avg_growth_outer = np.mean(list(combined_growth_outer.values()), axis=0)

# --- reference datasets (Tian; 1D spherical) ---
tian_1d_data_path = os.path.join(os.path.dirname(__file__), 'tian1d_dataset.csv')
tian_1d_data = np.loadtxt(tian_1d_data_path, delimiter=',')

spherical_1d_data_path = os.path.join(os.path.dirname(__file__), 'spherical_1d_data.csv')
spherical_1d_data = np.loadtxt(spherical_1d_data_path, delimiter=',')

plt.figure(figsize=(10, 8))

# Plot all four curves with swapped axes
plt.plot(spherical_1d_data[:, 0], spherical_1d_data[:, 1], ls='-', color='#dccd7d', linewidth=3, label='1D Spherical Model')
plt.plot(avg_growth_middle, altitudes, ls='-', color='#337538', linewidth=3, label='2D Spherical Center')
plt.plot(avg_growth_outer, altitudes, ls='-', color='#2e2585', linewidth=3, label='2D Spherical Edges')

# plot alternative tian data: update: keep the new tian data rather than the old one
plt.plot(tian_1d_data[:, 0], tian_1d_data[:, 1], ls='--', color='r', linewidth=3, label='1D Cartesian')

# Measure and print slopes
slope_inner, r2_inner = measure_exponential_slopes(altitudes, avg_growth_middle, "2D Spherical Center")
slope_outer, r2_outer = measure_exponential_slopes(altitudes, avg_growth_outer, "2D Spherical Edges")
slope_tian, r2_tian = measure_exponential_slopes(tian_1d_data[:, 1], tian_1d_data[:, 0], "Tian et al. 2023")
slope_1d, r2_1d = measure_exponential_slopes(spherical_1d_data[:, 1], spherical_1d_data[:, 0], "1D Spherical")

print(f"2D Spherical Center: {slope_inner:.6f} km⁻¹ (R² = {r2_inner:.4f})")
print(f"2D Spherical Edges: {slope_outer:.6f} km⁻¹ (R² = {r2_outer:.4f})")
print(f"Tian et al. 2023: {slope_tian:.6f} km⁻¹ (R² = {r2_tian:.4f})")
print(f"1D Spherical Model: {slope_1d:.6f} km⁻¹ (R² = {r2_1d:.4f})")

plt.ylabel("Altitude (km)")
plt.xlabel(r"$n(z)$ Amplitude Peak / $n(\text{150 km})$ Amplitude Peak")
# plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xlim(1, 6)
plt.ylim(149, 330)
plt.xticks([1, 2, 3, 4, 5, 6])
plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
plt.gca().xaxis.set_minor_locator(plt.NullLocator())
plt.tight_layout()
plt.show()


# --- Figure 2 right panel: compare heat profiles
high_res_heating_results = high_res_processor.compute_temperature_heating_rate(H=H)

# Load heating rate data
osc_cols = high_res_processor.compute_osc_cols()
# Get two middle oscillating columns
mid_idx = len(osc_cols) // 2

left_temp_last_cycle = high_res_heating_results['heating_rate_results'][555]['temp_last_cycle']
right_temp_last_cycle = high_res_heating_results['heating_rate_results'][556]['temp_last_cycle']
avg = (left_temp_last_cycle + right_temp_last_cycle) / 2

left_edge = high_res_heating_results['heating_rate_results'][osc_cols[0]]['temp_last_cycle']
right_edge = high_res_heating_results['heating_rate_results'][osc_cols[-1]]['temp_last_cycle']

temp_last_cycle = avg
radial_distances = high_res_heating_results['heating_rate_results'][555]['radial_distances']

# Compute best fit lines
x = radial_distances[19:36]
y1 = temp_last_cycle
y2 = 0.5 * (left_edge + right_edge)
slope1, intercept1 = np.polyfit(x, y1, 1)
slope2, intercept2 = np.polyfit(x, y2, 1)

plt.figure(figsize=(10, 8))

# Add a line with slope 9.4 K/H over the same x range
line_94_y = 9.4 * (x - x[0]) / H  # Convert slope from K/H to K/km
line_94_intercept = np.mean([y1[0], y2[0]])  # Start from average of first points
plt.plot(line_94_y + line_94_intercept, x, 'r--', linewidth=3,
         label='1D Cartesian')
        #  label='Tian et al., 2023')

aw1_data = np.load("heating_rate_data_1d_radial.npz")

# Use math text for uncertainty in the slopes
aw1_slope_H = aw1_data['slope_H']
aw1_t_crit = aw1_data['t_critical']
aw1_std_err = aw1_data['std_err']
H = aw1_data['H']
aw1_slope_H_ci = aw1_data['slope_H_ci_lower'], aw1_data['slope_H_ci_upper']
aw1_r_value = aw1_data['r_value']

# Plot AW1 data
plt.scatter(aw1_data['full_temps'], aw1_data['full_alts'], color='#dccd7d', alpha=0.3, ec='k')
# plt.plot(aw1_data['y_pred'], aw1_data['x_pred'], color='#dccd7d', label=f'1D Spherical AW1', lw=4)
plt.plot(aw1_data['y_pred'], aw1_data['x_pred'], color='#dccd7d', label=f'1D Spherical', lw=4)

plt.xlabel(r'$\langle T \rangle - T_0$ (K)')
plt.ylabel('Altitude (km)')

plt.scatter(temp_last_cycle, radial_distances[19:36], c="#337538", alpha=0.3, ec='k')
plt.plot(slope1 * x + intercept1, x, c="#337538", linestyle='-', linewidth=4, label=f'2D Spherical Center')
plt.scatter(0.5 * (left_edge + right_edge), radial_distances[19:36], c="#2e2585", alpha=0.3, ec='k')
plt.plot(slope2 * x + intercept2, x, c="#2e2585", linestyle='-', linewidth=4, label=f'2D Spherical Edges')

print(f"Slope for center: {slope1 * H:.4f} K/H")
print(f"Slope for edges: {slope2 * H:.4f} K/H")

# Plot the lines
plt.legend()
plt.ylim(149, 330)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()