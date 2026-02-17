#!/usr/bin/env python3
"""
Outside Wedge Density Analyzer (Figure 3 left and middle panel)

Analyzes horizontal wave propagation away from the forcing wedge by computing
density amplitude profiles as a function of distance from the sector edge.

Features:
- Shows growth, steady-state, and decay phases
- Visualizes horizontal propagation away from the active perturbed region
- Exports analysis data for publication

Usage:
    python outside_wedge_density_analyzer.py /path/to/results/dir
    python outside_wedge_density_analyzer.py /path/to/results/dir --no-plot
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import seaborn as sns

from optimized_loader_zenodo import OptimizedDSMCDataProcessor

sns.set_theme(context="paper", font_scale=2.0, style="ticks")

# CLI arguments
parser = argparse.ArgumentParser(
    description="Analyze horizontal wave propagation outside the forcing wedge",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python outside_wedge_density_analyzer.py /path/to/S2_512_azi
  python outside_wedge_density_analyzer.py /path/to/S2_512_azi --no-plot
    """,
)
parser.add_argument("data_dir", type=str, help="Path to the simulation results directory")
parser.add_argument("--no-plot", action="store_true", help="Skip generating and showing plots")
parser.add_argument("--max-dist", type=float, default=800, help="Maximum distance from wedge edge in km (default: 800)")
parser.add_argument("--altitude", type=float, default=230.0, help="Altitude for horizontal slice in km (default: 230)")
parser.add_argument(
    "--plot-padding",
    type=float,
    default=1000,
    help="Time padding after wave end for plotting in seconds (default: 1000)",
)
parser.add_argument("--debug", action="store_true", help="Save detailed debugging information to file")
parser.add_argument(
    "--snapshot-alt-min", type=float, default=150.0, help="Minimum altitude for snapshot plot in km (default: 150)"
)
parser.add_argument(
    "--snapshot-alt-max", type=float, default=320.0, help="Maximum altitude for snapshot plot in km (default: 320)"
)
parser.add_argument(
    "--snapshot-ticks",
    type=float,
    nargs="+",
    default=[200, 400, 600, 800, 1000],
    help="Tick positions (distances from edge in km) for snapshot x-axis (default: 200 400 600 800 1000)",
)
parser.add_argument(
    "--snapshot-x-max",
    type=float,
    default=None,
    help="Maximum distance from forcing edge to show on x-axis in km (default: show all)",
)
parser.add_argument(
    "--snapshot-smooth",
    type=str,
    choices=["none", "gaussian", "savgol"],
    default="none",
    help="Smoothing method for standard snapshot plot: none, gaussian, or savgol (default: none)",
)
parser.add_argument(
    "--experimental",
    action="store_true",
    help="Enable experimental 3-panel comparison: original, Gaussian smoothed, Savitzky-Golay smoothed",
)
parser.add_argument(
    "--smooth-sigma",
    type=float,
    nargs=2,
    default=[1.0, 3.0],
    metavar=("ALT", "AZI"),
    help="Gaussian smoothing sigmas [altitude, azimuth] in grid cells (default: 1.0 3.0)",
)
parser.add_argument(
    "--savgol-window",
    type=int,
    nargs=2,
    default=[5, 11],
    metavar=("ALT", "AZI"),
    help="Savitzky-Golay window sizes [altitude, azimuth] - must be odd (default: 5 11)",
)
parser.add_argument("--savgol-order", type=int, default=2, help="Savitzky-Golay polynomial order (default: 2)")
parser.add_argument(
    "--snapshot-vmin", type=float, default=-0.15, help="Minimum value for snapshot colormesh colorbar (default: -0.15)"
)
parser.add_argument(
    "--snapshot-vmax", type=float, default=0.15, help="Maximum value for snapshot colormesh colorbar (default: 0.15)"
)
parser.add_argument(
    "--snapshot-split", action="store_true", help="Use split-panel layout focusing on edges (excludes forcing region)"
)
parser.add_argument(
    "--snapshot-averaged",
    action="store_true",
    help="Average left and right sides into single panel (symmetric assumption)",
)
parser.add_argument(
    "--overlay-mfp",
    action="store_true",
    help="Overlay mean free path profile on altitude-distance snapshot",
)

args = parser.parse_args()
NO_PLOT = bool(args.no_plot)

# Constants (for physical calculations if needed)
mars_radius = 3390e3  # m


def savgol_filter_2d(data, window_size, poly_order):
    """
    Apply 2D Savitzky-Golay filter using separable 1D passes.

    Parameters
    ----------
    data : ndarray
        2D array to smooth
    window_size : tuple of int
        Window sizes (rows, cols) - must be odd
    poly_order : int
        Polynomial order for fitting

    Returns
    -------
    ndarray
        Smoothed 2D array
    """
    win_r, win_c = window_size

    # Ensure window sizes are odd
    if win_r % 2 == 0:
        win_r += 1
    if win_c % 2 == 0:
        win_c += 1

    # Ensure polynomial order is less than window size
    order_r = min(poly_order, win_r - 1)
    order_c = min(poly_order, win_c - 1)

    # Apply along rows (axis=1) first, then columns (axis=0)
    smoothed = savgol_filter(data, win_c, order_c, axis=1, mode="nearest")
    smoothed = savgol_filter(smoothed, win_r, order_r, axis=0, mode="nearest")

    return smoothed


def plot_altitude_azimuth_snapshot(
    amp_cube,
    altitudes,
    azi_edges_km,
    jL,
    jR,
    snapshot_time_idx,
    time_value,
    save_path,
    no_plot=False,
    vmin=-0.15,
    vmax=0.15,
    alt_min=None,
    alt_max=None,
    tick_distances=None,
    x_max=None,
    experimental=False,
    smooth_method="none",
    smooth_sigma=None,
    savgol_window=None,
    savgol_order=2,
    split_layout=False,
    averaged_layout=False,
    mean_cell_width=None,
    overlay_mfp=False,
    mfp_km=None,
    mfp_altitudes=None,
):
    """
    Create altitude vs azimuthal distance colormesh at a single time snapshot.

    X-axis shows distance from forcing region edges (positive outward on both sides).
    Tick marks only appear outside the forcing region.

    Parameters
    ----------
    amp_cube : ndarray
        Amplitude data cube (time, n_r, n_phi)
    altitudes : ndarray
        Altitude values in km
    azi_edges_km : ndarray
        Azimuthal cell edges in km (cumulative)
    jL, jR : int
        Left and right indices of the forcing wedge
    snapshot_time_idx : int
        Time index for the snapshot
    time_value : float
        Time value in seconds (for labeling)
    save_path : Path
        Path to save the figure
    no_plot : bool
        If True, close figure without displaying
    vmin, vmax : float
        Color scale limits
    alt_min, alt_max : float, optional
        Altitude range limits in km (default: use full range)
    tick_distances : array-like, optional
        Distances from edge for tick marks in km (default: [200, 400, 600, 800, 1000])
    x_max : float, optional
        Maximum distance from forcing edge to display in km (default: show all)
    experimental : bool
        If True, create 3-panel comparison with smoothed versions
    smooth_method : str
        Smoothing method for standard mode: "none", "gaussian", or "savgol"
    smooth_sigma : tuple of float, optional
        Gaussian smoothing sigmas (altitude, azimuth) in grid cells
    savgol_window : tuple of int, optional
        Savitzky-Golay window sizes (altitude, azimuth)
    savgol_order : int
        Savitzky-Golay polynomial order
    split_layout : bool
        If True, create split-panel layout with left/right edges (excludes forcing region)
    averaged_layout : bool
        If True, average left/right sides into single panel (assumes symmetry).
        Includes one cell inside forcing region that shares the forcing edge.
    mean_cell_width : float, optional
        Mean azimuthal cell width in km (for grid spacing in averaged layout)
    overlay_mfp : bool
        If True, overlay mean free path profile on the plot
    mfp_km : ndarray, optional
        Mean free path values in km
    mfp_altitudes : ndarray, optional
        Altitude values corresponding to mfp_km
    """
    # Default parameters
    if tick_distances is None:
        tick_distances = np.array([200, 400, 600, 800, 1000])
    else:
        tick_distances = np.array(tick_distances)

    # Extract snapshot data (altitude, azimuth)
    snapshot = amp_cube[snapshot_time_idx, :, :]  # (n_r, n_phi)

    # Filter altitude range if specified
    if alt_min is not None or alt_max is not None:
        alt_min = alt_min if alt_min is not None else altitudes[0]
        alt_max = alt_max if alt_max is not None else altitudes[-1]

        alt_mask = (altitudes >= alt_min) & (altitudes <= alt_max)
        altitudes = altitudes[alt_mask]
        snapshot = snapshot[alt_mask, :]

    # Get azimuthal cell centers
    azi_centers_km = 0.5 * (azi_edges_km[:-1] + azi_edges_km[1:])

    # Forcing region edges in km
    left_edge_km = azi_edges_km[jL]
    right_edge_km = azi_edges_km[jR + 1]

    # Filter azimuthal extent if x_max is specified
    if x_max is not None:
        # Keep only data within x_max distance from forcing edges
        azi_mask = (azi_centers_km >= left_edge_km - x_max) & (azi_centers_km <= right_edge_km + x_max)
        azi_centers_km = azi_centers_km[azi_mask]
        snapshot = snapshot[:, azi_mask]

    # --- Helper function to add MFP overlay ---
    def add_mfp_overlay(ax, mfp_km, mfp_altitudes, altitudes_plot):
        """Add mean free path overlay to axis."""
        from scipy.interpolate import interp1d

        # Interpolate MFP to the plot altitudes
        mfp_interp = interp1d(mfp_altitudes, mfp_km, bounds_error=False, fill_value="extrapolate")
        mfp_at_altitudes = mfp_interp(altitudes_plot)

        # Plot MFP line (white line with black dashed overlay for visibility)
        ax.plot(mfp_at_altitudes, altitudes_plot, 'w-', linewidth=4.0, label='MFP', zorder=10)
        ax.plot(mfp_at_altitudes, altitudes_plot, 'k--', linewidth=2.0, zorder=11)

    # --- Helper function to configure axis ---
    def configure_axis(ax, data, title=None, show_xlabel=True, show_ylabel=True, angle_xaxis=None):
        """Configure a single axis with colormesh and formatting."""
        pcm = ax.pcolormesh(azi_centers_km, altitudes, data, cmap="RdBu_r", shading="nearest", vmin=vmin, vmax=vmax)

        # Add vertical lines at forcing region boundaries
        ax.axvline(x=left_edge_km, color="k", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvline(x=right_edge_km, color="k", linestyle="--", linewidth=1.5, alpha=0.7)

        # Custom x-axis ticks
        left_tick_positions = left_edge_km - tick_distances
        left_tick_positions = left_tick_positions[left_tick_positions >= azi_centers_km[0]]
        right_tick_positions = right_edge_km + tick_distances
        right_tick_positions = right_tick_positions[right_tick_positions <= azi_centers_km[-1]]
        all_tick_positions = np.concatenate([left_tick_positions[::-1], right_tick_positions])

        all_tick_labels = []
        for pos in all_tick_positions:
            if pos <= left_edge_km:
                dist = left_edge_km - pos
            else:
                dist = pos - right_edge_km
            all_tick_labels.append(f"{dist:.0f}")

        ax.set_xticks(all_tick_positions)
        ax.set_xticklabels(all_tick_labels)

        # Add "Forcing" label
        wedge_center_km = 0.5 * (left_edge_km + right_edge_km)
        ax.annotate(
            "Forcing",
            xy=(wedge_center_km, altitudes[0]),
            xytext=(wedge_center_km, altitudes[0] - 8),
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            annotation_clip=False,
        )

        # Rotate x-axis labels if specified
        if angle_xaxis is not None:
            plt.setp(ax.get_xticklabels(), rotation=angle_xaxis, ha="right", rotation_mode="anchor")
        if show_xlabel:
            ax.set_xlabel("Distance from forcing edge (km)", fontsize=22)
        if show_ylabel:
            ax.set_ylabel("Altitude (km)", fontsize=22)
        ax.tick_params(axis="both", labelsize=20)

        if title:
            ax.set_title(title, fontsize=16, fontweight="bold")

        # Add MFP overlay if requested
        if overlay_mfp and mfp_km is not None and mfp_altitudes is not None:
            add_mfp_overlay(ax, mfp_km, mfp_altitudes, altitudes)

        return pcm

    # --- Experimental mode: 3-panel comparison ---
    if experimental:
        # Set default smoothing parameters
        if smooth_sigma is None:
            smooth_sigma = [1.0, 3.0]
        if savgol_window is None:
            savgol_window = [5, 11]

        # Apply smoothing
        snapshot_gaussian = gaussian_filter(snapshot, sigma=smooth_sigma)
        snapshot_savgol = savgol_filter_2d(snapshot, savgol_window, savgol_order)

        # Create 3-panel figure (1 row, 3 columns)
        fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)

        # Panel 1: Original
        pcm1 = configure_axis(axes[0], snapshot, title="Original", show_ylabel=True)

        # Panel 2: Gaussian smoothed
        sigma_str = f"σ=[{smooth_sigma[0]:.1f}, {smooth_sigma[1]:.1f}]"
        pcm2 = configure_axis(axes[1], snapshot_gaussian, title=f"Gaussian Filter ({sigma_str})", show_ylabel=False)

        # Panel 3: Savitzky-Golay smoothed
        savgol_str = f"win=[{savgol_window[0]}, {savgol_window[1]}], order={savgol_order}"
        pcm3 = configure_axis(axes[2], snapshot_savgol, title=f"Savitzky-Golay ({savgol_str})", show_ylabel=False)

        # Single colorbar for all panels
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(pcm1, cax=cbar_ax, label=r"$(n - n_0)/n_0$")
        cbar.ax.tick_params(labelsize=14)

        plt.tight_layout(rect=[0, 0, 0.92, 1])

        # Save experimental figure with different name
        exp_save_path = save_path.parent / (save_path.stem + "_experimental.png")
        fig.savefig(exp_save_path, dpi=300, bbox_inches="tight")
        print(f"Saved experimental 3-panel snapshot: {exp_save_path}")

        if not no_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig, axes

    # --- Standard mode: single panel or split panel ---
    else:
        # Set default smoothing parameters
        if smooth_sigma is None:
            smooth_sigma = [1.0, 3.0]
        if savgol_window is None:
            savgol_window = [5, 11]

        # Apply smoothing based on method
        if smooth_method == "gaussian":
            snapshot_display = gaussian_filter(snapshot, sigma=smooth_sigma)
            method_label = f"Gaussian σ=[{smooth_sigma[0]:.1f}, {smooth_sigma[1]:.1f}]"
        elif smooth_method == "savgol":
            snapshot_display = savgol_filter_2d(snapshot, savgol_window, savgol_order)
            method_label = f"Savitzky-Golay win=[{savgol_window[0]}, {savgol_window[1]}], order={savgol_order}"
        else:  # "none"
            snapshot_display = snapshot
            method_label = None

        # --- Split-panel layout ---
        if split_layout:
            # Create masks for left and right regions
            mask_left = azi_centers_km < left_edge_km
            mask_right = azi_centers_km > right_edge_km

            # Extract left and right data
            azi_left = azi_centers_km[mask_left]
            azi_right = azi_centers_km[mask_right]
            data_left = snapshot_display[:, mask_left]
            data_right = snapshot_display[:, mask_right]

            # Convert to distance from edge (positive outward)
            dist_left = left_edge_km - azi_left  # Increases leftward
            dist_right = azi_right - right_edge_km  # Increases rightward

            # Create 2-panel figure
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

            # Plot left panel (inverted x-axis)
            pcm_left = ax_left.pcolormesh(
                dist_left, altitudes, data_left, cmap="RdBu_r", shading="nearest", vmin=vmin, vmax=vmax
            )
            ax_left.invert_xaxis()  # Distance increases leftward

            # Plot right panel (normal x-axis)
            pcm_right = ax_right.pcolormesh(
                dist_right, altitudes, data_right, cmap="RdBu_r", shading="nearest", vmin=vmin, vmax=vmax
            )

            # Add MFP overlay to both panels if requested
            if overlay_mfp and mfp_km is not None and mfp_altitudes is not None:
                add_mfp_overlay(ax_left, mfp_km, mfp_altitudes, altitudes)
                add_mfp_overlay(ax_right, mfp_km, mfp_altitudes, altitudes)

            # Configure left panel
            ax_left.set_xlabel("Distance from forcing edge (km)", fontsize=22)
            ax_left.set_ylabel("Altitude (km)", fontsize=22)
            ax_left.tick_params(axis="both", labelsize=20)
            ax_left.spines["right"].set_visible(False)

            # Configure right panel
            ax_right.set_xlabel("Distance from forcing edge (km)", fontsize=22)
            ax_right.tick_params(axis="both", labelsize=20, labelleft=False)
            ax_right.spines["left"].set_visible(False)

            # Adjust spacing to create narrow gap
            plt.subplots_adjust(wspace=0.05)

            # Shared colorbar
            fig.subplots_adjust(right=0.92)
            cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(pcm_right, cax=cbar_ax, label=r"$(n - n_0)/n_0$")
            cbar.ax.tick_params(labelsize=18)

            plt.tight_layout(rect=[0, 0, 0.92, 1])

            # Save figure
            split_save_path = save_path.parent / (save_path.stem + "_split.png")
            fig.savefig(split_save_path, dpi=300, bbox_inches="tight")
            if smooth_method != "none":
                print(f"Saved split-panel snapshot ({smooth_method} smoothed): {split_save_path}")
            else:
                print(f"Saved split-panel snapshot: {split_save_path}")

            if not no_plot:
                plt.show()
            else:
                plt.close(fig)

            return fig, (ax_left, ax_right)

        # --- Averaged layout: symmetric average of both sides ---
        elif averaged_layout:
            # Use mean_cell_width if provided, otherwise compute from edges
            cell_spacing = mean_cell_width if mean_cell_width else np.mean(np.diff(azi_edges_km))

            # Create masks for left and right regions (include one cell inside forcing region)
            mask_left = azi_centers_km < left_edge_km + cell_spacing
            mask_right = azi_centers_km > right_edge_km - cell_spacing

            # Extract left and right data
            azi_left = azi_centers_km[mask_left]
            azi_right = azi_centers_km[mask_right]
            data_left = snapshot_display[:, mask_left]
            data_right = snapshot_display[:, mask_right]

            # Convert to distance from edge (positive outward, negative inward)
            dist_left = left_edge_km - azi_left  # Includes one negative value (inside cell)
            dist_right = azi_right - right_edge_km  # Includes one negative value (inside cell)

            # Create common distance grid using explicit cell edges
            dist_max = min(dist_left.max(), dist_right.max())
            # Edges start at -cell_spacing (inside cell left edge) to dist_max + cell_spacing
            dist_edges = np.arange(-cell_spacing, dist_max + cell_spacing, cell_spacing)
            # Cell centers are midpoints of edges
            dist_common = 0.5 * (dist_edges[:-1] + dist_edges[1:])

            # Interpolate both sides to common distance grid and average
            # Need to interpolate for each altitude row
            n_alt = len(altitudes)
            data_averaged = np.zeros((n_alt, len(dist_common)))

            for i_alt in range(n_alt):
                # Sort by distance (ascending) for interpolation
                sort_left = np.argsort(dist_left)
                sort_right = np.argsort(dist_right)

                # Interpolate left side
                from scipy.interpolate import interp1d

                interp_left = interp1d(
                    dist_left[sort_left],
                    data_left[i_alt, sort_left],
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                data_left_interp = interp_left(dist_common)

                # Interpolate right side
                interp_right = interp1d(
                    dist_right[sort_right],
                    data_right[i_alt, sort_right],
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                data_right_interp = interp_right(dist_common)

                # Average (handling NaNs)
                data_averaged[i_alt, :] = np.nanmean([data_left_interp, data_right_interp], axis=0)

            # Create single-panel figure
            fig, ax = plt.subplots(figsize=(7, 10))

            # Plot averaged data (use centers with nearest shading, first cell extends to edge)
            pcm = ax.pcolormesh(
                dist_common, altitudes, data_averaged, cmap="RdBu_r", shading="nearest", vmin=vmin, vmax=vmax
            )

            # Add MFP overlay if requested
            if overlay_mfp and mfp_km is not None and mfp_altitudes is not None:
                add_mfp_overlay(ax, mfp_km, mfp_altitudes, altitudes)

            # Configure axis
            ax.set_xlabel("Distance from forcing edge (km)", fontsize=22)
            ax.set_ylabel("Altitude (km)", fontsize=22)
            ax.set_xlim(-cell_spacing, None)  # Start x-axis at forcing edge (distance=0)
            ax.tick_params(axis="both", labelsize=20)

            # Colorbar
            cbar = fig.colorbar(pcm, ax=ax, label=r"$(n - n_0)/n_0$")
            cbar.ax.tick_params(labelsize=18)

            plt.tight_layout()

            # Save figure
            avg_save_path = save_path.parent / (save_path.stem + "_averaged.png")
            fig.savefig(avg_save_path, dpi=300, bbox_inches="tight")
            if smooth_method != "none":
                print(f"Saved averaged snapshot ({smooth_method} smoothed): {avg_save_path}")
            else:
                print(f"Saved averaged snapshot: {avg_save_path}")

            if not no_plot:
                plt.show()
            else:
                plt.close(fig)

            return fig, ax

        # --- Single-panel layout (original) ---
        else:
            fig, ax = plt.subplots(figsize=(14, 8))
            pcm = configure_axis(ax, snapshot_display, angle_xaxis=75)

            # Colorbar
            cbar = fig.colorbar(pcm, ax=ax, label=r"$(n - n_0)/n_0$")
            cbar.ax.tick_params(labelsize=18)

            plt.tight_layout()

            # Save figure
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            if smooth_method != "none":
                print(f"Saved altitude-azimuth snapshot ({smooth_method} smoothed): {save_path}")
            else:
                print(f"Saved altitude-azimuth snapshot: {save_path}")

            if not no_plot:
                plt.show()
            else:
                plt.close(fig)

            return fig, ax


def main():
    """Main analysis function"""

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Processing: {data_dir}")

    # Initialize data processor
    processor = OptimizedDSMCDataProcessor(str(data_dir))

    # Load simulation parameters and data
    print("Loading parameters...")
    processor.load_parameters()

    print("Loading data...")
    processor.load_data()

    # Get wave time indices
    start_idx, end_idx = processor.get_wave_time_indices()
    # tweak the end idx by 5
    end_idx -= 60

    # Get background profiles
    print("Computing background profiles...")
    background = processor.get_background_profiles()
    background_density = background["density"]  # (n_r,)

    # Get oscillating columns
    print("Computing oscillating columns...")
    osc_cols = processor.compute_osc_cols()
    print(f"Found {len(osc_cols)} oscillating azimuthal columns: {osc_cols[0]} to {osc_cols[-1]}")

    # Extract data from processor
    density_cube = processor.density_data  # (time, n_r, n_phi)
    altitudes = processor.params.radial_cells  # km
    time = processor.params.time  # s

    # Get data dimensions
    n_time, n_r, n_phi = density_cube.shape

    # Get grid information for azimuthal spacing
    azi_edges = processor.params.azi_edges  # deg
    azi_edges = np.deg2rad(azi_edges)  # convert to radians
    dphi = np.diff(azi_edges)  # (n_azi,)

    # Compute azimuthal arc lengths at 100 km altitude
    r100 = mars_radius + 100e3  # m
    azi_edges_km = np.cumsum(np.insert(r100 * dphi, 0, 0)) / 1e3  # km, cumulative

    # Get individual cell widths in km
    cell_widths_km = np.diff(azi_edges_km)
    mean_cell_width = np.mean(cell_widths_km)

    print(f"Azimuthal resolution: {mean_cell_width:.3f} km mean cell width")
    print(f"Total azimuthal cells: {n_phi}")

    # --- Step 1. Identify the wedge edges at 100 km altitude ---
    jL, jR = osc_cols[0], osc_cols[-1]  # wedge indices
    r100_idx = np.argmin(np.abs(altitudes - 100.0))  # index for ~100 km

    # Wedge edges in km
    phiL_edge_km = azi_edges_km[jL]
    phiR_edge_km = azi_edges_km[jR + 1]

    print(f"Wedge spans azimuthal indices {jL} to {jR} ({jR - jL + 1} cells)")
    print(f"Wedge arc length: {phiR_edge_km - phiL_edge_km:.2f} km")

    # --- Step 2. Create distance array by sampling azimuthal cells ---
    # Instead of using physical distance spacing, sample every N cells outward
    # This ensures we always have valid data points
    max_dist_km = args.max_dist

    # Calculate how many cells we can go outward on each side
    max_cells_left = jL  # cells available to the left
    max_cells_right = n_phi - 1 - jR  # cells available to the right
    max_cells_outward = min(max_cells_left, max_cells_right)

    # Determine cell spacing for sampling (adaptive based on resolution)
    if mean_cell_width < 0.5:  # High resolution
        cell_stride = max(1, int(5.0 / mean_cell_width))  # Sample every ~5 km
    elif mean_cell_width < 2.0:  # Medium resolution
        cell_stride = max(1, int(2.0 / mean_cell_width))  # Sample every ~2 km
    else:  # Low resolution
        cell_stride = 1  # Sample every cell

    print(f"Sampling every {cell_stride} cell(s) outward from wedge")

    # Create array of cell offsets
    cell_offsets = np.arange(0, max_cells_outward + 1, cell_stride)

    # Compute actual distances for these offsets
    distances_km = cell_offsets * mean_cell_width

    # For low-resolution simulations, ensure we get enough data points
    # even if max_dist_km is smaller than a few cell widths
    min_samples = 10  # Minimum number of distance points to sample

    if mean_cell_width > 100:  # Very low resolution (cells > 100 km)
        # Prioritize getting enough samples over max_dist constraint
        target_cells = min(max_cells_outward, max(min_samples, int(max_dist_km / mean_cell_width)))
        cell_offsets = np.arange(0, target_cells + 1, cell_stride)
        distances_km = cell_offsets * mean_cell_width

        if distances_km[-1] > max_dist_km:
            print(f"Warning: Cell size ({mean_cell_width:.1f} km) is large.")
            print(f"         Sampling {len(cell_offsets)} cells up to {distances_km[-1]:.1f} km")
            print(f"         (requested max_dist={max_dist_km} km is too small for this resolution)")
    else:
        # Normal case: truncate to max_dist_km
        valid_idx = distances_km <= max_dist_km
        if np.sum(valid_idx) < min_samples and len(distances_km) >= min_samples:
            # If truncation gives too few points, keep at least min_samples
            cell_offsets = cell_offsets[:min_samples]
            distances_km = distances_km[:min_samples]
            print(f"Warning: Keeping {min_samples} samples (up to {distances_km[-1]:.1f} km)")
            print(f"         to ensure sufficient data despite max_dist={max_dist_km} km")
        else:
            cell_offsets = cell_offsets[valid_idx]
            distances_km = distances_km[valid_idx]

    n_dist = len(distances_km)
    print(f"Sampling {n_dist} distance points from 0 to {distances_km[-1]:.1f} km")

    # --- Step 3. Compute density amplitudes ---
    rho0 = background_density[:, None]  # (n_r, 1) to broadcast
    amp_cube = (density_cube - rho0) / rho0  # (time, n_r, n_phi)

    # --- Step 4. Extract left and right regions ---
    amp_out = np.zeros((n_time, n_r, n_dist))

    print("Computing amplitude profiles...")
    valid_count = 0
    for d, j_offset in enumerate(cell_offsets):
        j_left = jL - j_offset
        j_right = jR + 1 + j_offset

        if j_left >= 0 and j_right < n_phi:
            # Average the two symmetric sides
            amp_out[:, :, d] = 0.5 * (amp_cube[:, :, j_left] + amp_cube[:, :, j_right])
            valid_count += 1
        else:
            amp_out[:, :, d] = np.nan  # outside domain

    print(f"Valid distance points: {valid_count}/{n_dist}")

    # Save debug info if requested
    if args.debug:
        debug_dir = data_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        debug_file = debug_dir / "outside_wedge_debug.txt"
        with open(debug_file, "w") as f:
            f.write("=== Outside Wedge Density Analyzer Debug Info ===\n\n")
            f.write(f"Data directory: {data_dir}\n")
            f.write(f"Total azimuthal cells: {n_phi}\n")
            f.write(f"Azimuthal resolution: {mean_cell_width:.6f} km mean cell width\n")
            f.write(f"Cell stride: {cell_stride}\n")
            f.write(f"Wedge indices: jL={jL}, jR={jR} ({jR - jL + 1} cells)\n")
            f.write(f"Wedge arc length: {phiR_edge_km - phiL_edge_km:.3f} km\n")
            f.write(f"Max cells outward: {max_cells_outward}\n")
            f.write(f"Distance points created: {n_dist}\n")
            f.write(f"Valid distance points: {valid_count}/{n_dist}\n")
            f.write(f"\nDistance array (km): {distances_km}\n")
            f.write(f"\nCell offsets: {cell_offsets}\n")
            f.write(f"\nTime array shape: {time.shape}\n")
            f.write(f"Density cube shape: {density_cube.shape}\n")
            f.write(f"Amplitude cube shape: {amp_cube.shape}\n")
            f.write(f"amp_out shape: {amp_out.shape}\n")
        print(f"Debug info saved to: {debug_file}")

    # --- Step 5. Visualization at specified altitude ---
    target_altitude = args.altitude
    r_target_idx = np.argmin(np.abs(altitudes - target_altitude))
    actual_altitude = altitudes[r_target_idx]
    print(f"\nCreating visualization at {actual_altitude:.1f} km altitude")

    # Limit time slice to wave start to wave end + padding
    PLOT_PADDING = args.plot_padding  # seconds
    end_idx_plot = end_idx + int(np.ceil(PLOT_PADDING / (time[2] - time[1])))
    end_idx_plot = min(end_idx_plot, len(time))  # Ensure within bounds
    time_plot = time[start_idx:end_idx_plot]
    amp_out_plot = amp_out[start_idx:end_idx_plot, r_target_idx, :]

    # Debug: Check data validity
    n_valid_data = np.sum(~np.isnan(amp_out_plot))
    n_total_data = amp_out_plot.size
    print(f"Data validity: {n_valid_data}/{n_total_data} non-NaN values ({100 * n_valid_data / n_total_data:.1f}%)")
    print(f"Amplitude range: [{np.nanmin(amp_out_plot):.4f}, {np.nanmax(amp_out_plot):.4f}]")
    print(f"Time range: [{time_plot[0]:.1f}, {time_plot[-1]:.1f}] s")
    print(f"Distance range: [{distances_km[0]:.1f}, {distances_km[-1]:.1f}] km")

    amp_out_pcolormesh = amp_out[start_idx - 5 : end_idx_plot, r_target_idx, :]
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(12, 10))
    # pcm = ax.pcolormesh(time_plot, distances_km,
    pcm = ax.pcolormesh(
        time[start_idx - 5 : end_idx_plot],
        distances_km,
        # amp_out_plot.T,  # (dist, time)
        amp_out_pcolormesh.T,  # (dist, time)
        cmap="RdBu_r",
        shading="auto",
        vmin=-0.15,
        vmax=0.15,
    )
    # add vertical lines for wave start and end
    ax.axvline(x=time[start_idx], color="k", linestyle="--", linewidth=2.0)
    ax.axvline(x=time[end_idx], color="k", linestyle="--", linewidth=2.0)
    ax.set_xlabel("Time (s)", fontsize=24)
    # ax.set_ylabel("Distance from sector edge [km]", fontsize=24)
    ax.set_ylabel("Distance from forcing edge (km)", fontsize=24)
    ax.tick_params(axis="both", labelsize=20)
    fig.colorbar(pcm, ax=ax, label=r"$(n - n_0)/n_0$")
    # ax.set_title(f"Density amplitude at {actual_altitude:.0f} km outside forcing sector", fontsize=13)
    plt.tight_layout()

    # Save figure
    save_dir = data_dir / "figures"
    save_dir.mkdir(exist_ok=True)

    plt.show()

    # --- Load transport profiles for MFP overlay ---
    if args.overlay_mfp:
        # transport_file = data_dir / "exports" / "pre_wave_transport_profiles.npz"
        # use instead the script path / exports / pre_wave_transport_profiles.npz
        transport_file = Path(__file__).parent / "exports" / "pre_wave_transport_profiles.npz"
        if transport_file.exists():
            print(f"\nLoading transport profiles from {transport_file}...")
            transport_profiles = np.load(transport_file)
            mfp_km = transport_profiles["mean_free_path"] / 1e3  # convert to km
            mfp_altitudes = transport_profiles["radial_cells"]  # already in km
            print(f"  MFP loaded: {len(mfp_km)} altitude points")
            print(f"  MFP range: {mfp_km.min():.4f} - {mfp_km.max():.2f} km")
        else:
            print(f"\nWarning: Transport profiles not found at {transport_file}")
            print("         MFP overlay will be disabled.")
            args.overlay_mfp = False
            mfp_km, mfp_altitudes = None, None
    else:
        mfp_km, mfp_altitudes = None, None

    # --- Step 6. Altitude vs Azimuth snapshot at end of wave forcing ---
    print(f"\nCreating altitude-azimuth snapshot at wave end (t = {time[end_idx]:.1f} s)...")
    print(f"  Altitude range: {args.snapshot_alt_min:.1f} - {args.snapshot_alt_max:.1f} km")
    if args.snapshot_x_max:
        print(f"  X-axis extent: ±{args.snapshot_x_max:.1f} km from forcing edges")
    print(f"  Tick positions: {args.snapshot_ticks} km")
    if args.experimental:
        print("  Experimental mode: ON")
        print(f"    Gaussian sigma: {args.smooth_sigma}")
        print(f"    Savitzky-Golay window: {args.savgol_window}, order: {args.savgol_order}")
    else:
        if args.snapshot_smooth != "none":
            print(f"  Smoothing: {args.snapshot_smooth}")
            if args.snapshot_smooth == "gaussian":
                print(f"    Sigma: {args.smooth_sigma}")
            elif args.snapshot_smooth == "savgol":
                print(f"    Window: {args.savgol_window}, order: {args.savgol_order}")
        if args.snapshot_split:
            print("  Split-panel layout: ON")
        if args.snapshot_averaged:
            print("  Averaged layout: ON (symmetric sides)")
    if args.overlay_mfp:
        print("  MFP overlay: ON")
    snapshot_path = save_dir / "altitude_azimuth_snapshot_wave_end.png"
    plot_altitude_azimuth_snapshot(
        amp_cube=amp_cube,
        altitudes=altitudes,
        azi_edges_km=azi_edges_km,
        jL=jL,
        jR=jR,
        snapshot_time_idx=end_idx,
        time_value=time[end_idx],
        save_path=snapshot_path,
        no_plot=NO_PLOT,
        vmin=args.snapshot_vmin,
        vmax=args.snapshot_vmax,
        alt_min=args.snapshot_alt_min,
        alt_max=args.snapshot_alt_max,
        tick_distances=args.snapshot_ticks,
        x_max=args.snapshot_x_max,
        experimental=args.experimental,
        smooth_method=args.snapshot_smooth,
        smooth_sigma=args.smooth_sigma,
        savgol_window=args.savgol_window,
        savgol_order=args.savgol_order,
        split_layout=args.snapshot_split,
        averaged_layout=args.snapshot_averaged,
        mean_cell_width=mean_cell_width,
        overlay_mfp=args.overlay_mfp,
        mfp_km=mfp_km,
        mfp_altitudes=mfp_altitudes,
    )

    # --- Identify the fifth peak at specified altitude edge ---
    # Time series at target altitude and edge (distance 0)
    amp_edge = amp_out_plot[:, 0]  # (time,) at target altitude, dist=0

    print("\nSearching for peaks in edge time series...")
    print(f"Edge amplitude range: [{np.nanmin(amp_edge):.4f}, {np.nanmax(amp_edge):.4f}]")

    # Find peaks (prominent ones)
    peaks, properties = find_peaks(amp_edge, prominence=0.01)
    print(f"Found {len(peaks)} peaks with prominence >= 0.01")

    if len(peaks) >= 5:
        # Sort by prominence, take top 5, then the fifth one
        prominences = properties["prominences"]
        top5_indices = np.argsort(prominences)[-5:]  # indices of top 5 peaks
        top5_peaks = peaks[top5_indices]
        top5_peaks = np.sort(top5_peaks)  # sort by time

        fifth_peak_idx = top5_peaks[4]  # 0-based, fifth is index 4

        # Time of fifth peak
        time_fifth_peak = time_plot[fifth_peak_idx]
        omega = processor.params.omega_freq
        period = processor.params.wave_period

        # === TROUGH TRACKING: Detect and track the trough before fifth peak ===
        def find_fifth_peak_and_trough(time_series, base_prominence=0.01, distance_idx=0, total_distances=1):
            """
            Find the fifth peak and the trough immediately before it.
            Uses adaptive prominence that decreases with distance from sector.

            Returns: (trough_time_idx, trough_amplitude, peak_time_idx) or (None, None, None)
            """
            # Adaptive prominence: decrease with distance from sector
            prominence_decay = 0.7 ** (distance_idx / max(1, total_distances))
            prominence = base_prominence * max(prominence_decay, 0.001)  # Floor at 0.001

            # Find all peaks
            peaks, properties = find_peaks(time_series, prominence=prominence)

            if len(peaks) < 5:
                return None, None, None

            # Sort by prominence, take top 5, then sort by time
            prominences = properties["prominences"]
            top5_indices = np.argsort(prominences)[-5:]
            top5_peaks = peaks[top5_indices]
            top5_peaks_sorted = np.sort(top5_peaks)

            fifth_peak_idx = top5_peaks_sorted[4]

            # Find the trough before the fifth peak
            # Search backward from fifth peak to fourth peak (or start of data)
            if len(top5_peaks_sorted) >= 4:
                search_start = top5_peaks_sorted[3]  # Fourth peak
            else:
                search_start = max(0, fifth_peak_idx - int(period / np.mean(np.diff(time_plot))))

            # Find troughs in the region before fifth peak
            region = time_series[search_start:fifth_peak_idx]
            if len(region) < 3:
                return None, None, None

            troughs_in_region, _ = find_peaks(-region, prominence=prominence * 0.5)

            if len(troughs_in_region) > 0:
                # Take the last trough before the peak (closest to fifth peak)
                trough_relative_idx = troughs_in_region[-1]
                trough_idx = search_start + trough_relative_idx
                return trough_idx, time_series[trough_idx], fifth_peak_idx

            return None, None, None

        def compute_1e_attenuation_distance(position_map, distances_km_arr, use_interpolation=True):
            """
            Compute 1/e attenuation distance from tracked peak/trough amplitudes.

            Parameters:
            -----------
            position_map : dict
                {distance_idx: (time_idx, amplitude)} from tracking
            distances_km_arr : array
                Distance values
            use_interpolation : bool
                If True, interpolate to find exact crossing; else fit exponential

            Returns:
            --------
            dict with: e_folding_distance, tracked_distances, tracked_amplitudes, method
            """
            # Extract tracked data
            d_indices = sorted(position_map.keys())
            tracked_dists = np.array([distances_km_arr[d] for d in d_indices])
            tracked_amps = np.array([position_map[d][1] for d in d_indices])

            # Use absolute values (troughs are negative)
            tracked_amps = np.abs(tracked_amps)

            if len(tracked_amps) < 3:
                return {
                    "e_folding_distance": np.nan,
                    "method": "insufficient_data",
                    "tracked_distances": tracked_dists,
                    "tracked_amplitudes": tracked_amps,
                }

            initial_amp = tracked_amps[0]
            target_amp = initial_amp / np.e

            result = {
                "tracked_distances": tracked_dists,
                "tracked_amplitudes": tracked_amps,
                "initial_amplitude": initial_amp,
                "target_amplitude": target_amp,
            }

            # Method 1: Direct interpolation if data crosses threshold
            if use_interpolation and np.any(tracked_amps <= target_amp):
                # Sort by distance (ascending) so interpolation proceeds outward
                sort_idx = np.argsort(tracked_dists)
                dist_sorted = tracked_dists[sort_idx]
                amp_sorted = tracked_amps[sort_idx]

                if len(dist_sorted) >= 2:
                    # Find first index where amplitude has fallen to or below target
                    crossing = np.where(amp_sorted <= target_amp)[0]
                    if crossing.size > 0:
                        i = crossing[0]
                        # If crossing occurs at the first sample, just take that distance
                        if i == 0:
                            e_fold_dist = float(dist_sorted[0])
                        else:
                            d1, a1 = dist_sorted[i - 1], amp_sorted[i - 1]
                            d2, a2 = dist_sorted[i], amp_sorted[i]
                            # Linear interpolation between bracketing points
                            if a2 == a1:
                                e_fold_dist = float(d2)
                            else:
                                t = (target_amp - a1) / (a2 - a1)
                                e_fold_dist = float(d1 + t * (d2 - d1))

                        result["e_folding_distance"] = e_fold_dist
                        result["method"] = "interpolation"
                        return result

            # Method 2: Fit exponential and compute 1/b
            def exp_decay_func(x, a, b):
                return a * np.exp(-b * x)

            try:
                popt, pcov = curve_fit(
                    exp_decay_func, tracked_dists, tracked_amps, p0=(initial_amp, 0.01), bounds=(0, np.inf), maxfev=2000
                )
                a_fit, b_fit = popt
                e_fold_dist = 1.0 / b_fit

                # Uncertainty from covariance
                b_std = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else np.nan
                e_fold_uncertainty = (1.0 / b_fit**2) * b_std

                result["e_folding_distance"] = e_fold_dist
                result["e_folding_uncertainty"] = e_fold_uncertainty
                result["fit_params"] = popt
                result["fit_cov"] = pcov
                result["method"] = "exponential_fit"
            except (RuntimeError, ValueError):
                result["e_folding_distance"] = np.nan
                result["method"] = "fit_failed"

            return result

        # Build trough position map for all distances
        print("\nDetecting fifth peak and preceding trough at each distance...")

        trough_position_map = {}  # {distance_idx: (trough_time_idx, trough_amplitude)}
        peak_position_map = {}  # {distance_idx: peak_time_idx}

        for d_idx in range(n_dist):
            time_series = amp_out_plot[:, d_idx]

            trough_t_idx, trough_amp, peak_t_idx = find_fifth_peak_and_trough(
                time_series, base_prominence=0.01, distance_idx=d_idx, total_distances=n_dist
            )

            if trough_t_idx is not None:
                trough_position_map[d_idx] = (trough_t_idx, trough_amp)
                peak_position_map[d_idx] = peak_t_idx

        print(f"  Successfully detected at {len(trough_position_map)}/{n_dist} distances")

        # Compute 1/e attenuation from tracked troughs
        print("\nComputing 1/e attenuation distance from tracked troughs...")
        attenuation_result = compute_1e_attenuation_distance(trough_position_map, distances_km, use_interpolation=True)

        if not np.isnan(attenuation_result["e_folding_distance"]):
            print(f"  1/e attenuation distance: {attenuation_result['e_folding_distance']:.1f} km")
            print(f"  Method: {attenuation_result['method']}")
            print(f"  Initial amplitude: {attenuation_result['initial_amplitude']:.4f}")
            print(f"  Points tracked: {len(attenuation_result['tracked_distances'])}")
        else:
            print(f"  Could not compute attenuation: {attenuation_result['method']}")

        # Plot tracked amplitude decay
        if len(attenuation_result.get("tracked_distances", [])) > 0:
            fig_att, ax_att = plt.subplots(figsize=(10, 6))

            # Scatter: tracked data
            ax_att.scatter(
                attenuation_result["tracked_distances"],
                attenuation_result["tracked_amplitudes"],
                s=50,
                c="blue",
                label="Tracked trough amplitude",
                zorder=5,
            )

            # Line: exponential fit (if available)
            if "fit_params" in attenuation_result:
                d_fit = np.linspace(0, attenuation_result["tracked_distances"].max(), 100)
                a, b = attenuation_result["fit_params"]
                ax_att.plot(d_fit, a * np.exp(-b * d_fit), "r-", label=f"Exp fit: 1/e = {1 / b:.1f} km", linewidth=2)

            # Horizontal line at 1/e threshold
            ax_att.axhline(
                attenuation_result["target_amplitude"], color="green", linestyle="--", label="1/e threshold", alpha=0.7
            )

            # Vertical line at 1/e distance
            if not np.isnan(attenuation_result["e_folding_distance"]):
                ax_att.axvline(
                    attenuation_result["e_folding_distance"],
                    color="green",
                    linestyle=":",
                    linewidth=2,
                    label=f"$L_{{att}}$ = {attenuation_result['e_folding_distance']:.1f} km",
                )

            ax_att.set_xlabel("Distance from forcing edge (km)")
            ax_att.set_ylabel("|Amplitude|")
            ax_att.set_title(f"Trough Amplitude Decay at {actual_altitude:.0f} km")
            ax_att.legend()
            ax_att.grid(True, alpha=0.3)

            plt.show()

    # --- Export the analysis data ---
    print("Exporting analysis data...")
    export_dir = data_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    arrays_to_save = {
        "amp_out": amp_out,
        "distances_km": distances_km,
        "time": time,
        "altitudes": altitudes,
    }

    def _to_jsonable(meta: dict) -> dict:
        """Convert numpy types to JSON-serializable types"""
        out = {}
        for k, v in meta.items():
            if isinstance(v, (np.generic,)):
                out[k] = v.item()
            elif isinstance(v, Path):
                out[k] = str(v)
            else:
                out[k] = v
        return out

    metadata = {
        "results_path": str(data_dir),
        "jL": int(jL),
        "jR": int(jR),
        "max_dist_km": float(max_dist_km),
        "target_altitude_km": float(actual_altitude),
        "r100": float(r100),
        "r_target_idx": int(r_target_idx),
        "omega": float(processor.params.omega_freq),
        "wave_period": float(processor.params.wave_period),
        "wave_t0": float(processor.params.start_step_wave * processor.params.timestep),
        "wave_tf": float(processor.params.final_step_wave * processor.params.timestep),
        "units": {
            "amp_out": "dimensionless (rho - rho0) / rho0",
            "distances_km": "km",
            "time": "s",
            "altitudes": "km",
        },
        "notes": "Density amplitude cube outside the forcing wedge, averaged symmetrically from left and right edges.",
    }

    # Save arrays
    np.savez_compressed(export_dir / "outside_wedge_amp_export.npz", **arrays_to_save)

    # Save metadata
    with open(export_dir / "outside_wedge_amp_export_meta.json", "w") as f:
        json.dump(_to_jsonable(metadata), f, indent=2)

    print(f"Exported analysis data to: {export_dir}")
    print("  - outside_wedge_amp_export.npz")
    print("  - outside_wedge_amp_export_meta.json")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
