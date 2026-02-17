"""
Figure 6 and 7
6 panels: growth profiles for S1-S6, with 1D references and edge avg
1 panel: amplitude vs #azi cells, with fit and 1D references
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import warnings
from os import path
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import scipy as scp
import matplotlib.cm as cm

sns.set_theme(style="ticks", context="talk")
warnings.simplefilter(action='ignore', category=FutureWarning)
mars_radius = 3390

# --- reference datasets (Tian; 1D spherical) ---
tian_1d_data_path = path.join(path.dirname(__file__), 'tian1d_dataset.csv')
tian_1d_data = np.loadtxt(tian_1d_data_path, delimiter=',')

spherical_1d_data_path = path.join(path.dirname(__file__), 'spherical_1d_data.csv')
spherical_1d_data = np.loadtxt(spherical_1d_data_path, delimiter=',')

with open(path.join(path.dirname(__file__), '2d_spherical_growth_profiles.pkl'), 'rb') as f:
    spherical_2d_growth_profiles = pickle.load(f)

parquet_path = path.join(path.dirname(__file__), '2d_growth_peak_data.parquet')
df = pd.read_parquet(parquet_path)

# prepare distance-from-center used for universal colormap
df['Azi Num'] = df['Azi'].str.strip('azi').astype(int)
config_info = {}
for azi_cells in df['Azi Cells'].unique():
    subset = df[df['Azi Cells'] == azi_cells]
    azi_nums = subset['Azi Num'].unique()
    center_azi_num = (min(azi_nums) + max(azi_nums)) / 2
    azi_width_deg = 360 / azi_cells
    azi_width_rad = np.deg2rad(azi_width_deg)
    cell_width_km = (mars_radius + 100) * azi_width_rad
    config_info[azi_cells] = {'center': center_azi_num, 'cell_width': cell_width_km}

df['Distance from Center (km)'] = df.apply(
    lambda row: (row['Azi Num'] - config_info[row['Azi Cells']]['center']) * config_info[row['Azi Cells']]['cell_width'],
    axis=1
)
df['Abs Distance (km)'] = df['Distance from Center (km)'].abs()

# ---- helper: distance -> palette ----
def get_distance_palette(df_subset, cmap_name='YlOrRd', use_absolute=False):
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    if use_absolute:
        azi_distances = df_subset[['Azi', 'Abs Distance (km)']].drop_duplicates()
        dist_col = 'Abs Distance (km)'
        norm = Normalize(vmin=0, vmax=azi_distances[dist_col].max())
    else:
        azi_distances = df_subset[['Azi', 'Distance from Center (km)']].drop_duplicates()
        dist_col = 'Distance from Center (km)'
        max_dist = max(abs(azi_distances[dist_col].min()), abs(azi_distances[dist_col].max()))
        norm = Normalize(vmin=-max_dist, vmax=max_dist)

    cmap = cm.get_cmap(cmap_name)
    palette = {row['Azi']: cmap(norm(row[dist_col])) for _, row in azi_distances.iterrows()}
    return palette, norm, cmap

# ----------------- Figure 6: S1--S6 panels -----------------
from scipy.signal import savgol_filter

edge_growth_avg_1024 = (
    df.query('`Azi Cells` == 1024 & `Azi Num` in (598,511)')
      .groupby(['Radial Distance'])
      .agg({'Density Amplitude Growth': 'mean'})
      .reset_index()
)

harrah_growth_smooth = savgol_filter(spherical_2d_growth_profiles['growth_peaks_std'].data[:, -1], window_length=5, polyorder=2)

# subset and label S1..S6
df_subset = df.query('`Amp` == 0.25 & `Cell Size` > 6').sort_values(by=['Azi Cells', 'Cell Size', 'Azi']).copy()
azi_cells_to_sim = {128: 'S1', 256: 'S2', 384: 'S3', 512: 'S4', 640: 'S5', 1024: 'S6'}
df_subset['Simulation'] = df_subset['Azi Cells'].map(azi_cells_to_sim)

palette, norm, cmap = get_distance_palette(df_subset, cmap_name='YlOrRd', use_absolute=True)
col_order = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning)
    g = sns.FacetGrid(df_subset, col='Simulation', col_order=col_order, hue='Azi', palette=palette, sharey=False, height=4.5, col_wrap=3)
    g.map_dataframe(sns.lineplot, 'Density Amplitude Growth', 'Radial Distance', orient='y', lw=2.5, dashes=True, estimator=None)
    for ax in g.axes.flat:
        ax.plot(tian_1d_data[:, 0], tian_1d_data[:, 1], label='Tian et al., 2023', lw=2.5, color='k', ls='--')
        ax.plot(spherical_1d_data[:, 0], spherical_1d_data[:, 1], label='Spherical 1D', lw=2.5, color='#ff7f0e', ls='--')
        ax.plot(harrah_growth_smooth, spherical_2d_growth_profiles['altitudes'], label='Harrah 1D', lw=2.5, color='#2ca02c', ls='--')
        ax.plot(edge_growth_avg_1024['Density Amplitude Growth'], edge_growth_avg_1024['Radial Distance'], label='Edge Avg 1024', lw=2.5, color='#1f77b4', ls='--', zorder=5)

g.set_ylabels('Altitude (km)')
for ax in g.axes[-3:]:
    ax.set_xlabel(r"$n(z)$ / $n(150 \rm{km})$ Amplitude Peak")

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = g.figure.colorbar(sm, ax=g.axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('|Distance from Center| (km)')

plt.tight_layout()
plt.show()

# ----------------- Figure 7: amplitude vs #azi cells + fit -----------------

def plot_fit_with_confidence(ax, x, popt, pcov, func, confidence=0.95, color='r', label=None):
    y_fit = func(x, *popt)
    perr = np.sqrt(np.diag(pcov))
    z = scp.stats.norm.ppf(1 - (1 - confidence) / 2)
    # analytic propagation for y = A - B/x
    dy_dA = np.ones_like(x)
    dy_dB = -1 / x
    y_err = np.sqrt((dy_dA * perr[0])**2 + (dy_dB * perr[1])**2 + 2 * dy_dA * dy_dB * pcov[0, 1])
    y_upper = y_fit + z * y_err
    y_lower = y_fit - z * y_err
    ax.plot(x, y_fit, color=color, ls='--', label=label)
    ax.fill_between(x, y_lower, y_upper, color=color, alpha=0.2, label=f'{int(confidence*100)}% CI')

# prepare data and fit
_df = df[(df['Cell Size'] >= 6) & (df['Amp'] == 0.25)]
df_filtered_grouped = _df.groupby('Azi Cells').agg({'Density Amplitude Growth': 'max'}).reset_index()

fig, ax = plt.subplots(figsize=(6.5, 4.5))
sns.scatterplot(data=df_filtered_grouped, x='Azi Cells', y='Density Amplitude Growth', s=90, ax=ax)
ax.set(xlabel=r'Number of Azi Cells $N_\phi$', ylabel=r'$A_\rho(r=320\,\mathrm{km}) / A_\rho(r=150\,\mathrm{km})$')

def func(x, A, B):
    return A - B / x

popt, pcov = curve_fit(func, df_filtered_grouped['Azi Cells'], df_filtered_grouped['Density Amplitude Growth'])
x_fit = np.linspace(df_filtered_grouped['Azi Cells'].min(), df_filtered_grouped['Azi Cells'].max(), 200)

ax.axhline(y=6.14065, color='k', ls='--', lw=2.5, label='1D Cartesian (T23)')
ax.axhline(y=spherical_2d_growth_profiles['growth_peaks_std'].data[:, -1].max(), color='green', ls='--', lw=2.5, label='1D Spherical (P26)')

plot_fit_with_confidence(ax, x_fit, popt, pcov, func, confidence=0.95, color='r', label=r'Fit: $A - \frac{B}{x}$')
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

# end of script