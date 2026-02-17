import pprint
import re
from glob import glob
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, butter, filtfilt
from scipy.fft import fft, fftfreq, ifft
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, ifft
from scipy.interpolate import UnivariateSpline
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy as sp
import os
import pickle
from typing import Tuple
import math


mars_radius = 3389.5e3  # m

num_part_inject_idx = 7
inject_rate_idx = 11

def extract_grid_edges(grid_ascii_path: str):
    """
    Extract the radial and azimuthal grid edges from Grid.ascii.

    Parameters
    ----------
    grid_ascii_path : str
        Path to .../Info/Grid.ascii

    Returns
    -------
    radial_grid : np.ndarray
        1D array of unique radial grid edges (in meters), sorted.
    azimuth_grid : np.ndarray
        1D array of unique azimuthal grid edges (in degrees), sorted.
    """
    cols = ["cell_num","i_rad","i_zen","i_azi","rad_lower","rad_upper",
            "zen_lower","zen_upper","azi_lower","azi_upper",
            "rad_width","zen_width","azi_width","volume"]
    df = pd.read_csv(grid_ascii_path, header=None, names=cols, sep=r"\s+")

    # Collect and sort unique radial edges
    radial_edges = np.unique(np.concatenate([df["rad_lower"].values, df["rad_upper"].values]))

    # Collect and sort unique azimuthal edges
    azimuth_edges = np.unique(np.concatenate([df["azi_lower"].values, df["azi_upper"].values]))

    return radial_edges, azimuth_edges

def make_volume_lookup(grid_ascii_path: str, prefer_file_volume: bool = True, tol=0.05):
    """
    Build a fast lookup for cell volume V[i_rad, i_azi] from Grid.ascii.
    Indices returned are addressed as 0-based by default (set one_based=True when calling to use file indices).

    --- usage ---
    volume_of = make_volume_lookup(f"{results_path}/Info/Grid.ascii")
    V = volume_of(ir, ia)                 # ir, ia are 0-based
    V_file_indexing = volume_of(1, 1, one_based=True)  # file's 1-based indices

    Parameters
    ----------
    grid_ascii_path : str
        Path to .../Info/Grid.ascii
    prefer_file_volume : bool
        If True and a 'volume' column exists and matches geometry within `tol`, use it.
    tol : float
        Relative tolerance for accepting file 'volume' vs geometric volume.

    Returns
    -------
    cell_volume : callable
        cell_volume(i_rad, i_azi, one_based=False) -> float volume [m^3]
    """
    cols = ["cell_num","i_rad","i_zen","i_azi","rad_lower","rad_upper",
            "zen_lower","zen_upper","azi_lower","azi_upper",
            "rad_width","zen_width","azi_width","volume"]
    df = pd.read_csv(grid_ascii_path, header=None, names=cols, sep=r"\s+")

    # Ensure we have exactly one zenith index (your thick disk slice)
    if df["i_zen"].nunique() != 1:
        raise ValueError(f"Expected a single zenith index; found {df['i_zen'].nunique()}.")

    # Geometry-correct volume from columns (degrees -> radians)
    def geom_vol(row):
        rl, ru = row.rad_lower, row.rad_upper
        th_l, th_u = math.radians(row.zen_lower), math.radians(row.zen_upper)
        ph_l, ph_u = math.radians(row.azi_lower), math.radians(row.azi_upper)
        dphi = ph_u - ph_l
        int_sin = math.cos(th_l) - math.cos(th_u)  # ∫ sinθ dθ
        return (ru**3 - rl**3) * int_sin * dphi / 3.0

    V_geom = df.apply(geom_vol, axis=1).to_numpy()

    # Decide which volume to trust
    V_use = V_geom
    if prefer_file_volume and "volume" in df:
        V_file = df["volume"].to_numpy()
        relerr = np.abs(V_file - V_geom) / np.maximum(np.abs(V_geom), 1e-30)
        if np.all(relerr < tol):
            V_use = V_file  # file volumes are consistent; use them

    # Map to a dense 2D array [i_rad=1..N, i_azi=1..M] -> 0-based numpy array shape (N, M)
    df_sorted = df.sort_values(["i_rad", "i_azi"]).reset_index(drop=True)
    n_rad = int(df_sorted["i_rad"].max())
    n_azi = int(df_sorted["i_azi"].max())

    # Align V_use to the sorted order, then reshape
    V_sorted = V_use[df_sorted.index.to_numpy()]
    V2d = V_sorted.reshape(n_rad, n_azi)

    def cell_volume(i_rad: int, i_azi: int, one_based: bool = False) -> float:
        """
        Return volume [m^3] of cell (i_rad, i_azi).
        - By default, indices are 0-based (NumPy-style).
        - Set one_based=True to pass indices as in Grid.ascii (start at 1).
        """
        ir = i_rad - 1 if one_based else i_rad
        ia = i_azi - 1 if one_based else i_azi
        if not (0 <= ir < n_rad and 0 <= ia < n_azi):
            raise IndexError(f"indices out of bounds: (i_rad={i_rad}, i_azi={i_azi}), one_based={one_based}")
        return float(V2d[ir, ia])

    return cell_volume

# extract timestep from simulation parameters file using regex
def extract_from_info(res_path):
    with open(f"{res_path}/Info/Simulation_Parameters.ascii", "r") as f:
        lines = f.readlines()
        for line in lines:
            # remove spaces before and after the string
            line = line.strip()
            if re.match(r"^Time Step", line):
                timestep = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
            if re.match(r"^Number of Total Simulations", line):
                nsteps = int(re.findall(r"\d+", line)[0])
            if re.match(r"Particle Weight", line):
                # extract number in exponential notation
                part_weight = float(re.findall(r"[-+]?\d*\.\d+[eE][-+]?\d+", line)[0])
            if re.match(r"^Number of Radial Cells", line):
                num_rad_cells = int(re.findall(r"\d+", line)[0])
    return timestep, nsteps, part_weight, num_rad_cells


def extract_from_info_updated(res_path):
    info_dict = {}
    # extract timestep from simulation parameters file using regex
    with open(f"{res_path}/Info/Simulation_Parameters.ascii", "r") as f:
        lines = f.readlines()
        for line in lines:
            # remove spaces before and after the string
            line = line.strip()
            if re.match(r"^Time Step", line):
                info_dict["timestep"] = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
            if re.match(r"^Number of Total Simulations", line):
                info_dict["nsteps"] = int(re.findall(r"\d+", line)[0])
            if re.match(r"Particle Weight", line):
                # extract number in exponential notation
                info_dict["part_weight"] = float(re.findall(r"[-+]?\d*\.\d+[eE][-+]?\d+", line)[0])
            if re.match(r"Number of Radial Cells", line):
                info_dict["num_radial_cells"] = int(re.findall(r"\d+", line)[0])
            if re.match(r"Number of Azimuth Cells", line):
                info_dict["num_azi_cells"] = int(re.findall(r"\d+", line)[0])
            if re.match(r"Start Step of Wave Injection", line):
                info_dict["step0_inject"] = int(re.findall(r"\d+", line)[0])
            if re.match(r"End Step of Wave Injection", line):
                info_dict["stepf_inject"] = int(re.findall(r"\d+", line)[0])
            if re.match(r"Wave Injection Frequency", line):
                # extract number in exponential notation
                info_dict["OMEGA"] = float(re.findall(r"[-+]?\d*\.\d+[eE][-+]?\d+", line)[0])

    num_periods = (
        (info_dict["stepf_inject"] - info_dict["step0_inject"] - 1)
        * info_dict["OMEGA"]
        * info_dict["timestep"]
        / (2 * np.pi)
    )
    info_dict["num_periods"] = num_periods

    return info_dict


macro_headers = header = [
    "Time",
    "Avg Time Interval",
    "Radius",
    "Zenith",
    "Azimuth",
    "Cell Number",
    "Number of Particles",
    "Number Density",
    "Radial Velocity",
    "Zenithal Velocity",
    "Azimuthal Velocity",
    "Thermal Velocity",
    "Radial Temperature",
    "Zenithal Temperature",
    "Azimuthal Temperature",
    "Kinetic Temperature",
    "Translational Temperature",
    "Internal Temperature",
    "Overpopulation Temperature",
    "Theoretical Collision Frequency",
    "Mean Free Path",
    "Scale Height",
    "Knudsen Number",
    "Thermal Conductivity Up",
    "Thermal Conductivity Down",
]

macro_properties = macro_headers[6:]

# create dictionary map of macro header names to their index
macro_header_map = {}
for i in range(len(macro_headers)):
    macro_header_map[macro_headers[i]] = i

# create dictionary map of macro property names to their index
macro_property_map = {}
for i in range(len(macro_properties)):
    macro_property_map[macro_properties[i]] = i


def open_macro(path):
    macro_dataframe = pd.read_csv(path, sep=r"\s+", header=None).drop(columns=[19])
    macro_dataframe.columns = macro_headers
    return macro_dataframe


def open_surface(path):
    surf_dataframe = pd.read_csv(path, sep=r"\s+", header=None)
    surf_dataframe.columns = [
        "Time",
        "Average Time Interval",
        "Radius",
        "Zenith",
        "Azimuth",
        "Cell Number",
        "Area",
        "Number of Particles Injected",
        "Number of Particles Returned",
        "Number of Particles Reemitted",
        "Number of Particles on Surface",
        "Injection Rate",
        "Return Rate",
        "Reemission Rate",
    ]
    return surf_dataframe


def open_surface(path) -> pd.DataFrame:
    surf_dataframe = pd.read_csv(path, sep=r"\s+", header=None)
    surf_dataframe.columns = [
        "Time",
        "Avg TimeInterval",
        "Rad",
        "Zen",
        "Azi",
        "Cell Num",
        "Lower Area",
        "Num Part Inject",
        "Num Part Return",
        "Num Part Reemit",
        "Num Part Surf",
        "Inject Rate",
        "Return Rate",
        "Reemit Rate",
        "Return Time",
        "Inject Energy",
        "Return Energy",
    ]
    return surf_dataframe


def extract_macro_data(res_path, n_properties, num_rad_cells, window_size):
    macro_files = glob(f"{res_path}/Macro/Output*_Proc*_Macroscopic.ascii")
    macro_files.sort(key=lambda x: [int(d) for d in re.split(r"(\d+)", x) if d.isdigit()])

    # extract output number from the last file
    number_of_outputs = int(re.split(r"(\d+)", macro_files[-1])[1])

    macro_data = np.zeros((n_properties, number_of_outputs, num_rad_cells))
    timestamps = np.zeros(number_of_outputs)

    for k, i in enumerate(range(0, len(macro_files), window_size)):
        macro_dfs = [open_macro(macro_files[i]) for i in range(i, i + window_size)]
        df = pd.concat(macro_dfs, axis=0, ignore_index=True).sort_values(by="Cell Number")

        df_reshaped = df.values[:, 6:].T
        macro_data[:, k, :] = df_reshaped
        timestamps[k] = df["Time"][0]

    return macro_data, timestamps, number_of_outputs


macro_props = ["Kinetic Temperature", "Number Density", "Radial Velocity", "Azimuthal Velocity"]


def extract_macro_data(res_path, num_radial_cells, num_azi_cells):
    """
    Extract macroscopic data from the macroscopic output files.
    :param res_path: path to the simulation results
    :param number_of_outputs: number of outputs
    :param num_processes: number of processes used in the simulation
    :param num_radial_cells: number of radial cells
    :param num_azi_cells: number of zenithal cells
    :return: macroscopic data, timestamps, number of particles, number of particles per cell
    """
    macro_files = glob(f"{res_path}/Macro/Output*_Proc*_Macroscopic.ascii")
    macro_files.sort(key=lambda x: [int(d) for d in re.split(r"(\d+)", x) if d.isdigit()])

    number_of_outputs = int(re.split(r"(\d+)", macro_files[-1])[-4])
    num_processes = int(re.split(r"(\d+)", macro_files[-1])[-2]) + 1

    time_stamps = np.zeros(number_of_outputs)
    num_particles = np.zeros((number_of_outputs, num_radial_cells))
    num_particle_data = np.zeros((number_of_outputs, num_radial_cells, num_azi_cells))

    macro_grids_data = {prop: np.zeros((number_of_outputs, num_radial_cells, num_azi_cells)) for prop in macro_props}

    for k, i in enumerate(range(0, len(macro_files), num_processes)):
        macro_dfs = [open_macro(macro_files[i]) for i in range(i, i + num_processes)]
        df = pd.concat(macro_dfs, axis=0, ignore_index=True)
        df_sort_rad = df.sort_values(by="Radius")
        df_sort_cell = df.sort_values(by="Cell Number")

        grouped_df = df_sort_rad.groupby("Radius")
        time_stamps[k] = macro_dfs[0]["Time"].values[0]

        num_particles[k, :] = grouped_df.sum()["Number of Particles"].values
        num_particle_data[k, :, :] = df_sort_cell["Number of Particles"].values.T.reshape(
            (num_radial_cells, num_azi_cells)
        )

        data = df_sort_cell[macro_props].values.T.reshape(len(macro_props), num_radial_cells, num_azi_cells)

        for j, prop in enumerate(macro_props):
            macro_grids_data[prop][k, :, :] = data[j, :, :]

    return macro_grids_data, time_stamps, num_particles, num_particle_data


# def extract_surf_data():
#     surf_rates = np.zeros((time_stamps.size, 2))
#     surf_rates_azi = np.zeros((time_stamps.size, 2))
#     surf_rates_surf = np.zeros((time_stamps.size, info_dict["num_azi_cells"]))

#     for k, i in enumerate(range(0, len(surf_files), 12)):
#         surf_dfs = [open_surface(surf_files[i]) for i in range(i, i + 12)]
#         df = pd.concat(surf_dfs, axis=0, ignore_index=True)
#         surf_rates[k, 0] = df["Num Part Inject"].sum()
#         surf_rates[k, 1] = df["Inject Rate"].sum()

#         # sort df by Azi and sum only first 2 rows
#         df = df.sort_values(by=["Azi"])
#         surf_rates_azi[k, 0] = df["Num Part Inject"].iloc[0:2].sum()
#         surf_rates_azi[k, 1] = df["Inject Rate"].iloc[0:2].sum()

#         surf_rates_surf[k, :] = df["Num Part Inject"].values


def process_temperature_deviation_peaks(
    temperature_deviation_azi, is_oscillating_extended, mid_rad_km, cell_150km, top_radius_idx, num_periods, return_indices=False, last_cycle_offset=0,
    **kwargs
):
    """Process temperature deviation peaks, handling cancelled peak detection with masked arrays"""
    temp_deviation_peaks = np.ma.zeros((top_radius_idx - cell_150km + 1, int(num_periods)))
    temp_deviation_last_cycle = np.ma.zeros(top_radius_idx - cell_150km + 1)
    all_peak_indices = []

    for i in range(0, top_radius_idx - cell_150km + 1):
        temperature_deviation_slice = temperature_deviation_azi[is_oscillating_extended, cell_150km + i]
        peaks, peak_indices = process_amplitude(
            -temperature_deviation_slice,
            1,
            num_periods=num_periods,
            altitude=mid_rad_km[cell_150km + i],
            return_peaks=True,
            **kwargs,
        )

        all_peak_indices.append(peak_indices)

        if np.all(np.isnan(peaks)) or len(peak_indices) == 0:
            # If peaks were cancelled or no peaks found, mask this altitude and all higher altitudes
            temp_deviation_peaks[i:] = np.ma.masked
            temp_deviation_last_cycle[i:] = np.ma.masked
            print(f"Temperature peak detection cancelled at altitude {mid_rad_km[cell_150km + i]:.1f} km. "
                  f"Masking this and higher altitudes.")
            # Break out of the loop immediately
            break

        temp_deviation_peaks[i] = -peaks

        # Only compute last cycle if we have enough peaks
        if len(peak_indices) >= 2:
            temperature_deviations_last_cycle = temperature_deviation_slice[peak_indices[-2-last_cycle_offset]:peak_indices[-1-last_cycle_offset]]
            temp_deviation_last_cycle[i] = temperature_deviations_last_cycle.mean()
        else:
            temp_deviation_last_cycle[i] = np.ma.masked

    if return_indices:
        return temp_deviation_peaks, temp_deviation_last_cycle, all_peak_indices

    return temp_deviation_peaks, temp_deviation_last_cycle


def process_growth_num_amp(
    num_density_amplitude_azi, is_oscillating_extended, mid_rad_km, cell_150km, top_radius_idx, num_periods, use_dynamic=False, **kwargs
):
    """Process growth in density amplitude, handling cancelled peak detection with masked arrays

    Args:
        use_dynamic: Whether to use dynamic parameter adjustment based on previous peaks
    """
    # Convert num_periods to integer if it's a numpy array or float
    num_periods = int(num_periods)

    num_den_amp_150km = num_density_amplitude_azi[is_oscillating_extended, cell_150km]
    baseline_peaks_amps, baseline_peak_indices = process_amplitude(
        num_den_amp_150km, 1, altitude=mid_rad_km[cell_150km],
        num_periods=num_periods, return_peaks=True, **kwargs
    )

    # Check if baseline peaks were cancelled or none found
    if np.all(np.isnan(baseline_peaks_amps)) or len(baseline_peak_indices) == 0:
        print("Peak detection cancelled at baseline altitude (150 km). Cannot proceed with growth analysis.")
        return (
            np.ma.masked_all((top_radius_idx - cell_150km + 1, num_periods)),
            np.ma.masked_array(np.full(num_periods, np.nan), mask=True)
        )

    growth_array_size = top_radius_idx - cell_150km + 1
    growth_num_amp_peaks = np.ma.zeros((growth_array_size, int(num_periods)))
    growth_num_amp_peaks[0] = 1.0  # Baseline altitude is always 1.0

    # Store previous peaks for dynamic adjustment
    previous_peaks = baseline_peak_indices

    for i in range(1, growth_array_size):
        current_altitude = mid_rad_km[cell_150km + i]
        num_density_amp_slice = num_density_amplitude_azi[is_oscillating_extended, cell_150km + i]

        peaks, peak_indices = process_amplitude(
            num_density_amp_slice,
            baseline_peaks_amps,
            altitude=current_altitude,
            previous_peaks=previous_peaks if use_dynamic else None,
            use_dynamic=use_dynamic,
            num_periods=num_periods,
            return_peaks=True,
            **kwargs
        )

        if np.all(np.isnan(peaks)) or len(peak_indices) == 0:
            # If peaks were cancelled or none found, mask this altitude and all higher altitudes
            growth_num_amp_peaks[i:] = np.ma.masked
            print(f"Peak detection cancelled at altitude {current_altitude:.1f} km. "
                  f"Masking this and higher altitudes.")
            # Break out of the loop immediately
            break

        growth_num_amp_peaks[i] = peaks

        # Update previous peaks only if they were successfully found
        if len(peak_indices) > 0:
            previous_peaks = peak_indices

    return growth_num_amp_peaks, baseline_peaks_amps


def read_simulation_parameters(file_path):
    parameters = {}
    current_section = None
    num_pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?")  # compile regex pattern

    with open(file_path, "r") as file:
        for line in file:  # iterate over file object directly
            line = line.strip()
            if line.endswith("Information") or line == "Simulation Parameters":
                current_section = line
                if current_section not in parameters:
                    parameters[current_section] = []
                parameters[current_section].append({})
            elif line and ":" in line and current_section:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                # Try to extract a numerical value
                if key not in ["Name", "Species 1", "Species 2"]:
                    match = num_pattern.search(value)  # use compiled regex pattern
                    if match:
                        num_value = float(match.group())
                        if (
                            num_value.is_integer() and num_value < 1e15
                        ):  # check if num_value can be converted to int without exception
                            num_value = int(num_value)
                        value = num_value
                if key == "Time Step":
                    if "Simulation Parameters" not in parameters:
                        parameters["Simulation Parameters"] = [{}]
                    parameters["Simulation Parameters"][-1][key] = value
                elif key in ["Name", "Collision Type"] and parameters[current_section][-1]:
                    parameters[current_section].append({key: value})
                else:
                    parameters[current_section][-1][key] = value
    return parameters


def identify_peaks(
    amplitude_data,
    num_periods,
    altitude=None,
    previous_peaks=None,
    use_dynamic=False,
    peak_params=None,
):
    """
    Identify and filter peaks in a 1D amplitude series.

    - Applies optional dynamic parameter adjustment using previous_peaks.
    - Calls scipy.signal.find_peaks with provided/derived parameters.
    - When enough peaks are found, returns the most prominent num_periods peaks.

    Args:
        amplitude_data: 1D array of amplitudes over time.
        num_periods: Number of peaks to select.
        altitude: Optional, only for debug-print context.
        previous_peaks: Optional array of previous altitude peak indices.
        use_dynamic: Whether to adjust detection parameters based on previous_peaks.
        peak_params: Dict for find_peaks parameters; keys can include
                     'prominence', 'distance', 'width', 'height'.

    Returns:
        peaks: 1D array of selected peak indices (filtered to num_periods when available).
        props: Dict of peak properties corresponding to returned peaks.
    """
    # Default peak parameters
    if peak_params is None:
        peak_params = {
            "prominence": 0.01,
            "distance": 12,
            "width": None,
            "height": None,
        }
    else:
        # Ensure all required keys exist (fall back to defaults)
        peak_params = {
            "prominence": peak_params.get("prominence", 0.01),
            "distance": peak_params.get("distance", 12),
            "width": peak_params.get("width", None),
            "height": peak_params.get("height", None),
        }

    # Optional dynamic parameter tuning
    if previous_peaks is not None and use_dynamic and not np.any(np.isnan(previous_peaks)):
        peak_spacing = np.diff(previous_peaks)
        if peak_spacing.size > 0:
            peak_params["distance"] = int(np.mean(peak_spacing) * 0.8)  # 80% of average spacing
        prev_peak_heights = amplitude_data[previous_peaks]
        if prev_peak_heights.size > 0:
            peak_params["prominence"] = np.mean(np.abs(prev_peak_heights)) * 0.3

        # Debug print mirrors previous behavior
        # if altitude is not None:
            # print(f"\nDynamically adjusted parameters at altitude {altitude:.1f} km:")
            # print(f"Distance: {peak_params['distance']}")
            # print(f"Prominence: {peak_params['prominence']:.2e}")

    # Find peaks
    peaks, props = find_peaks(amplitude_data, **peak_params)

    # If enough peaks, select top 'num_periods' by prominence
    if len(peaks) >= int(num_periods):
        order = np.argsort(props["prominences"])[::-1]
        keep = order[: int(num_periods)]
        selected_peaks = np.sort(peaks[keep])
        # Filter props to only selected peaks
        selected_props = {k: v[keep] for k, v in props.items()}
        return selected_peaks, selected_props

    # Otherwise, return raw (insufficient) peaks/props
    return peaks, props


def process_amplitude(amplitude_data, baseline_peak_amps, altitude=None, previous_peaks=None, **kwargs):
    """Process amplitudes in time series data with optional dynamic parameter adjustment.

    Args:
        amplitude_data: Array of amplitude values
        baseline_peak_amps: Reference amplitudes for normalization
        altitude: Current altitude for display
        previous_peaks: Information about peaks from previous altitude for dynamic adjustment
        **kwargs: Additional parameters for peak detection
    """
    # Get required parameters with defaults
    num_periods = int(kwargs.pop("num_periods", 5))
    return_peaks = kwargs.pop("return_peaks", False)
    use_dynamic = kwargs.pop("use_dynamic", False)

    # Extract peak finding parameters
    peak_params = {
        'prominence': kwargs.pop("prominence", 0.01),
        'distance': kwargs.pop("distance", 12),
        'width': kwargs.pop("width", None),
        'height': kwargs.pop("height", None)
    }

    # Delegate to the reusable identifier (no change to external behavior)
    peaks, props = identify_peaks(
        amplitude_data=amplitude_data,
        num_periods=num_periods,
        altitude=altitude,
        previous_peaks=previous_peaks,
        use_dynamic=use_dynamic,
        peak_params=peak_params,
    )

    if len(peaks) >= num_periods:
        peak_amps = amplitude_data[peaks]
        peak_amps = peak_amps / baseline_peak_amps
        if return_peaks:
            return peak_amps, peaks
        return peak_amps

    # Make sure any existing plots are cleared
    plt.close('all')

    # Interactive plotting for manual peak selection
    fig, ax = plt.subplots()
    line, = ax.plot(amplitude_data)
    scatter = ax.scatter(peaks, amplitude_data[peaks], color="r")

    if np.iterable(baseline_peak_amps):
        ax.axhline(baseline_peak_amps[-1], color="k", linestyle="--")

    if altitude is not None:
        ax.text(0.01, 0.99, f"Altitude: {altitude:.2f} km", transform=ax.transAxes, va="top", fontsize=12)

    # Show previous peaks if available
    has_legend_elements = False  # Track if we have any labeled elements
    if previous_peaks is not None and not np.any(np.isnan(previous_peaks)):
        ax.axvspan(previous_peaks[0], previous_peaks[-1], color='y', alpha=0.1,
                  label='Previous peak range')
        ax.plot(previous_peaks, amplitude_data[previous_peaks], 'y*',
                label='Previous altitude peaks')
        has_legend_elements = True

    cancel_box = ax.text(
        0.99, 0.01, "Cancel peak detection",
        transform=ax.transAxes, va="bottom", ha="right",
        bbox=dict(boxstyle="round", facecolor="red", alpha=0.3),
        picker=True,
        fontsize=8,
    )

    ax.text(
        0.99, 0.99,
        "Left-click to add, right-click to remove\nClick 'Cancel' for noisy data",
        transform=ax.transAxes, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=8,
    )

    cancelled = False

    def on_pick(event):
        nonlocal cancelled
        if event.artist == cancel_box:
            cancelled = True
            # Close the current figure and clear all plots
            plt.close('all')

    def on_click(event):
        nonlocal peaks
        if event.button is MouseButton.LEFT:
            x, y = event.xdata, event.ydata
            new_peak = np.argmin(np.abs(line.get_xdata() - x))
            peaks = np.append(peaks, new_peak)
            peaks = np.sort(peaks)
            scatter.set_offsets(np.c_[line.get_xdata()[peaks], line.get_ydata()[peaks]])
        elif event.button is MouseButton.RIGHT:
            x, y = event.xdata, event.ydata
            dist = np.hypot(x - line.get_xdata()[peaks], y - line.get_ydata()[peaks])
            closest_peak = np.argmin(dist)
            if dist[closest_peak] < 5:  # Tolerance for click proximity
                peaks = np.delete(peaks, closest_peak)
                scatter.set_offsets(np.c_[line.get_xdata()[peaks], line.get_ydata()[peaks]])

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect('pick_event', on_pick)
    if has_legend_elements:
        plt.legend()

    try:
        plt.show()
    finally:
        # Ensure all plots are closed even if an exception occurs
        plt.close('all')

    if cancelled:
        # Return immediately when cancelled
        if return_peaks:
            return np.full(num_periods, np.nan), np.array([])
        return np.full(num_periods, np.nan)

    # Process peaks after manual selection
    if len(peaks) == num_periods:
        peak_amps = amplitude_data[peaks]
        peak_amps = peak_amps / baseline_peak_amps
        if return_peaks:
            return peak_amps, peaks
        return peak_amps
    else:
        if return_peaks:
            return np.full(num_periods, np.nan), np.array([])
        return np.full(num_periods, np.nan)


def plot_density_amplitude(
    num_density_amplitude_azi,
    is_oscillating_extended,
    rad_cells_df_r1,
    time_stamps,
    altitude,
    color,
    label,
    alpha,
    **kwargs,
) -> None:
    prominence = kwargs.pop("prominence", 0.01)
    filter_window = kwargs.pop("filter_window", 9)
    width = kwargs.pop("width", 3)

    cell = np.argmin(np.abs(rad_cells_df_r1 - (altitude + mars_radius)))
    convolved = savgol_filter(num_density_amplitude_azi[is_oscillating_extended, cell], filter_window, 3)
    peaks, _ = find_peaks(convolved, prominence=prominence, width=width)

    plt.scatter(time_stamps[is_oscillating_extended][peaks], convolved[peaks], color="k", alpha=0.5)
    plt.plot(time_stamps[is_oscillating_extended], convolved, color, alpha=alpha)

    # Check if the label already exists in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if label not in labels:
        plt.plot([], [], color=color, label=label)


def plot_density_amp_examples(num_density_amplitude_azi, is_oscillating_extended, rad_cells_df_r1, time_stamps):
    plot_density_amplitude(
        num_density_amplitude_azi, is_oscillating_extended, rad_cells_df_r1, time_stamps, 150e3, "g--", "150km", 0.5
    )
    plot_density_amplitude(
        num_density_amplitude_azi, is_oscillating_extended, rad_cells_df_r1, time_stamps, 230e3, "b--", "230km", 0.99
    )
    plot_density_amplitude(
        num_density_amplitude_azi, is_oscillating_extended, rad_cells_df_r1, time_stamps, 310e3, "r--", "310km", 0.5
    )

    plt.legend(fontsize=14)
    plt.xlabel("Time [secs]")
    plt.ylabel("$n$ Amplitude")
    plt.tight_layout()
    plt.show()


def plot_heating_rate(valid_temps, valid_alts, full_temps, full_alts, slope, slope_H, stats_data, title=None, save_path=None):
    """
    Plot heating rate data with best fit line, confidence bands, and slope uncertainty.

    Args:
        valid_temps: Temperature values used for fitting
        valid_alts: Altitude values used for fitting
        full_temps: All temperature values including masked/invalid points
        full_alts: All altitude values including masked/invalid points
        slope: Computed slope in K/km
        slope_H: Computed slope in K/H
        stats_data: Dictionary containing statistical data:
            - r_value: R-value from linear regression
            - p_value: p-value from linear regression
            - std_err: Standard error of the slope
            - intercept: Y-intercept of the fit line
        title: Optional title for the plot
        save_path: Optional path to save the plot
    """
    from scipy import stats
    import os
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))
    H = slope_H / slope

    # Generate points for confidence bands
    x_pred = np.linspace(valid_alts.min(), valid_alts.max(), 100)
    y_pred = slope * x_pred + stats_data['intercept']

    # Number of observations and statistics
    n = len(valid_alts)
    x_mean = np.mean(valid_alts)
    x_dev_sq = np.sum((valid_alts - x_mean)**2)

    # Compute residual standard error
    y_pred_data = slope * valid_alts + stats_data['intercept']
    residuals = valid_temps - y_pred_data
    residual_std = np.sqrt(np.sum(residuals**2) / (n - 2))

    # Compute standard error of regression line
    x_new_dev_sq = (x_pred - x_mean)**2
    SE = residual_std * np.sqrt(1/n + x_new_dev_sq/x_dev_sq)

    # Compute confidence intervals
    alpha = 0.05  # 95% confidence level
    df = n - 2
    t_crit = stats.t.ppf(1 - alpha/2, df)
    slope_ci = (slope - t_crit * stats_data['std_err'],
                slope + t_crit * stats_data['std_err'])
    slope_H_ci = (slope_ci[0] * H, slope_ci[1] * H)

    # Compute pointwise confidence bands (95%)
    ci_pointwise = t_crit * SE

    # Compute simultaneous confidence bands (Working-Hotelling)
    f_value = stats.f.ppf(1 - alpha, 2, n - 2)
    ci_simultaneous = np.sqrt(2 * f_value) * SE

    # Plot all points, showing masked/invalid points differently
    # ax.plot(full_temps, full_alts, 'o', alpha=0.5)
    ax.scatter(full_temps, full_alts, color='C0', alpha=0.8)

    # Plot valid points and best fit line with both confidence bands
    # ax.plot(valid_temps, valid_alts, 'go', label='Valid Points', alpha=0.7)
    ax.plot(y_pred, x_pred, 'r-', label='Best Fit Line')

    # Plot confidence bands
    # ax.fill_betweenx(x_pred, y_pred - ci_pointwise, y_pred + ci_pointwise,
    #                     color='red', alpha=0.1, label='95% Pointwise CI')
    ax.fill_betweenx(x_pred, y_pred - ci_simultaneous, y_pred + ci_simultaneous,
                        color='blue', alpha=0.1, label='Confidence Band')

    # Use math text for uncertainty in the slopes
    stats_text = (
        f"$s = {slope:.2f} \\pm {t_crit * stats_data['std_err']:.2f}\\,\\mathrm{{K/km}}$" "\n"
        f"$\\;\\;\\;\\;\\;\\; [{slope_ci[0]:.2f}, {slope_ci[1]:.2f}]$" "\n\n"
        f"$\\bf{{s_H = {slope_H:.2f} \\pm {t_crit * stats_data['std_err'] * H:.2f}\\,\\mathrm{{K/H_0}}}}$" "\n"
        f"$\\bf{{[ {slope_H_ci[0]:.2f}, {slope_H_ci[1]:.2f} ]}}$" "\n\n"
        f"$R^2 = {stats_data['r_value']**2:.3f}$" "\n"
        f"$p = {stats_data['p_value']:.3e}$" "\n"
        f"$\\sigma = {stats_data['std_err']:.3f}$"
    )

    ax.text(0.98, 0.02, stats_text,
            ha='right', va='bottom', transform=ax.transAxes,
            bbox=dict(edgecolor='black', facecolor='white', alpha=0.8), fontsize=12)

    ax.set_xlabel(r'$T(r) - T_0$ (K)')
    ax.set_ylabel('Altitude (km)')
    # ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    # ax.margins(y=0)

    if title:
        ax.set_title(title)

    # ax.legend(loc='upper left', fontsize=16)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    return fig, ax

def compute_heating_rate_slope(
    temp_deviation_last_cycle, mid_rad_km, cell_150km, top_radius_idx, H, visualize=True, save_path=None, export_data_path=None
) -> Tuple[float, float]:
    """
    Compute heating rate slope with both pointwise and simultaneous confidence bands.

    Args:
        temp_deviation_last_cycle: Temperature deviation data
        mid_rad_km: Altitude array in km
        cell_150km: Starting cell index
        top_radius_idx: Ending cell index
        H: Scale height
        visualize: Whether to create visualization
        save_path: Path to save the plot
        export_data_path: Optional path to export data as NPZ file for external plotting

    Returns:
        Tuple of (slope in K/km, slope in K/H)
    """
    from scipy import stats

    # Get valid (unmasked) data points
    if isinstance(temp_deviation_last_cycle, np.ma.MaskedArray):
        valid_mask = ~temp_deviation_last_cycle.mask
    else:
        valid_mask = ~np.isnan(temp_deviation_last_cycle)

    if not np.any(valid_mask):
        print("No valid temperature deviation data points found for slope calculation")
        return np.nan, np.nan

    altitudes = mid_rad_km[cell_150km:top_radius_idx + 1]
    valid_temps = temp_deviation_last_cycle[valid_mask].ravel()
    valid_alts = altitudes[valid_mask].ravel()

    # Ensure we have enough points for fitting
    if len(valid_alts) < 2:
        print("Not enough valid points for slope calculation")
        return np.nan, np.nan

    # Print debug info
    print(f"Computing slope with {len(valid_alts)} valid points")
    print(f"Altitude range: {valid_alts.min():.1f} - {valid_alts.max():.1f} km")

    # Compute linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_alts, valid_temps)
    slope_H = slope * H

    # Prepare statistical data
    stats_data = {
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'intercept': intercept
    }

    # Export data if requested
    if export_data_path is not None:
        export_heating_rate_data(
            valid_temps, valid_alts, temp_deviation_last_cycle, altitudes,
            slope, slope_H, H, stats_data, export_data_path
        )
        print(f"Heating rate data exported to: {export_data_path}")

    if visualize:
        plot_heating_rate(
            valid_temps, valid_alts,
            temp_deviation_last_cycle, altitudes,
            slope, slope_H, stats_data,
            # title='Temperature Deviation Profile',
            save_path=save_path
        )

    return slope, slope_H


def export_heating_rate_data(valid_temps, valid_alts, full_temps, full_alts, slope, slope_H, H, stats_data, export_path):
    """
    Export all necessary data for heating rate visualization to an NPZ file.

    Args:
        valid_temps: Temperature values used for fitting
        valid_alts: Altitude values used for fitting
        full_temps: All temperature values including masked/invalid points
        full_alts: All altitude values including masked/invalid points
        slope: Computed slope in K/km
        slope_H: Computed slope in K/H
        H: Scale height
        stats_data: Dictionary containing statistical data
        export_path: Path to save the NPZ file
    """
    from scipy import stats

    # Compute confidence interval data
    n = len(valid_alts)
    x_mean = np.mean(valid_alts)
    x_dev_sq = np.sum((valid_alts - x_mean)**2)

    # Compute residual standard error
    y_pred_data = slope * valid_alts + stats_data['intercept']
    residuals = valid_temps - y_pred_data
    residual_std = np.sqrt(np.sum(residuals**2) / (n - 2))

    # Generate prediction points for confidence bands
    x_pred = np.linspace(valid_alts.min(), valid_alts.max(), 100)
    y_pred = slope * x_pred + stats_data['intercept']

    # Compute standard error of regression line
    x_new_dev_sq = (x_pred - x_mean)**2
    SE = residual_std * np.sqrt(1/n + x_new_dev_sq/x_dev_sq)

    # Compute confidence intervals
    alpha = 0.05  # 95% confidence level
    df = n - 2
    t_crit = stats.t.ppf(1 - alpha/2, df)
    slope_ci = (slope - t_crit * stats_data['std_err'],
                slope + t_crit * stats_data['std_err'])
    slope_H_ci = (slope_ci[0] * H, slope_ci[1] * H)

    # Compute pointwise confidence bands (95%)
    ci_pointwise = t_crit * SE

    # Compute simultaneous confidence bands (Working-Hotelling)
    f_value = stats.f.ppf(1 - alpha, 2, n - 2)
    ci_simultaneous = np.sqrt(2 * f_value) * SE

    # Package all data for export
    export_data = {
        # Raw data points
        'valid_temps': valid_temps,
        'valid_alts': valid_alts,
        'full_temps': full_temps,
        'full_alts': full_alts,

        # Regression results
        'slope': slope,
        'slope_H': slope_H,
        'intercept': stats_data['intercept'],
        'H': H,

        # Statistical measures
        'r_value': stats_data['r_value'],
        'r_squared': stats_data['r_value']**2,
        'p_value': stats_data['p_value'],
        'std_err': stats_data['std_err'],

        # Confidence intervals for slopes
        'slope_ci_lower': slope_ci[0],
        'slope_ci_upper': slope_ci[1],
        'slope_H_ci_lower': slope_H_ci[0],
        'slope_H_ci_upper': slope_H_ci[1],
        't_critical': t_crit,

        # Prediction data for plotting
        'x_pred': x_pred,
        'y_pred': y_pred,
        'ci_pointwise': ci_pointwise,
        'ci_simultaneous': ci_simultaneous,

        # Additional statistics
        'n_points': n,
        'residual_std': residual_std,
        'degrees_freedom': df,
        'alpha': alpha,

        # Metadata
        'altitude_range_km': [valid_alts.min(), valid_alts.max()],
        'description': 'Heating rate analysis data for temperature deviation vs altitude'
    }

    np.savez_compressed(export_path, **export_data)


def smoothed_last_peak_growth_num(growth_num_amp_peaks, mid_rad_km, cell_150km, top_radius_idx, visualize=True) -> None:
    growth_num_amp_last_peak = growth_num_amp_peaks[:, -1]

    # growth_num_amp_last_peak_smooth = savgol_filter(growth_num_amp_last_peak, 5, 3)
    # try a butter filter
    b, a = sp.signal.butter(1, 0.25)
    growth_num_amp_last_peak_smooth = sp.signal.filtfilt(b, a, growth_num_amp_last_peak)

    if visualize:
        plt.plot(
            growth_num_amp_last_peak_smooth,
            mid_rad_km[cell_150km : top_radius_idx + 1],
            label="mode " + str(i),
            color="#a367c1",
        )

        plt.plot(growth_num_amp_last_peak, mid_rad_km[cell_150km : top_radius_idx + 1], color="k", alpha=0.25)

        plt.show()

    return growth_num_amp_last_peak_smooth


def plot_overview_of_simulation(time_stamps, num_particle_data, mid_rad_km, surf_rates, surface_flow_rate, azi_center):
    total_particles = num_particle_data.sum(axis=(1, 2))
    average_particles = num_particle_data.mean(axis=(0, 2))
    altitude_at_10_particles = mid_rad_km[np.argmin(np.abs(average_particles - 10.0))]

    # Create the GridSpec layout
    fig = plt.figure(figsize=(12, 8))  # Adjust figure size as needed
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # First subplot: Total Particles over Time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_stamps[:], total_particles, label="CO2", lw=5, marker="o", markersize=5)
    # ax1.set_ylim(top=915000)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Total # of Particles")
    ax1.set_title("Total Number of Particles in Domain")

    # Second subplot: Average Particles per Altitude
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(average_particles, mid_rad_km)
    ax2.set_xlabel("Avg # of Particles")
    ax2.set_ylabel("Altitude (km)")
    ax2.set_title("Average # of Particles per Altitude")
    ax2.axvline(x=10, color="red", linestyle="--")
    ax2.axhline(y=altitude_at_10_particles, color="green", linestyle="--", label="Altitude at 10 particles")

    ax2.annotate(
        f"Altitude at 10 particles: {altitude_at_10_particles:.2f} km",
        xy=(10, altitude_at_10_particles),
        xytext=(10, -10),  # Offset from the xy point by 20 points right and 20 points up
        textcoords="offset points",  # Use offset in points for xytext
        arrowprops=dict(facecolor="black", arrowstyle="->"),
        fontsize=12,
        ha="left",
        va="top",
    )

    # Third subplot: Injection Rate over Time
    ax3 = fig.add_subplot(gs[1, :])  # Span both columns in the second row
    # ax3.plot(time_stamps, surf_rates[:, :, inject_rate_idx].sum(axis=-1), label='Live Injection Rate')
    ax3.plot(time_stamps, surf_rates[:, azi_center, inject_rate_idx].mean(axis=1), label="Live Injection Rate")
    # plot surf_rates for every azi index
    # for i in range(num_azi_cells):
    #     ax3.plot(time_stamps, surf_rates[:, i, inject_rate_idx], label=f'Azi: {i}', alpha=0.5)
    ax3.plot(time_stamps, surface_flow_rate, "k--", label="Surface Flow Rate", lw=3)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Injection Rate")
    ax3.set_title("Injection Rate as a function of time")
    ax3.legend(fontsize=12)

    # Adjust overall layout for better spacing
    plt.tight_layout()
    plt.show()


def extract_pickle_data(
    results_path="/mnt/i/ArchiveSims/Results_20240709_173728", pattern="growth_peak_data_3[0-5]azi_*.pkl"
):
    pickle_files = glob(os.path.join(results_path, pattern))
    basenames = [os.path.basename(f) for f in pickle_files]
    pickle_data = {}
    for bname, pfile in zip(basenames, pickle_files):
        with open(pfile, "rb") as f:
            pickle_data[bname] = pickle.load(f)

    return pickle_data, basenames


def filter_temperature_profiles(temp_profiles, filter_type="savgol", **kwargs):
    """
    Apply a Savitzky–Golay or Gaussian filter to each temperature profile to reduce high-frequency noise.

    Args:
        temp_profiles (np.ndarray): A 1D or 2D array of temperature profiles.
        filter_type (str): Filter type: "savgol" or "gaussian".
        **kwargs: For "savgol": window_length (int, default=11), polyorder (int, default=3).
                  For "gaussian": sigma (float, default=2), mode (str, default="reflect").

    Returns:
        np.ndarray: Filtered temperature profiles.
    """
    import numpy as np
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d

    single_profile = False
    if temp_profiles.ndim == 1:
        temp_profiles = temp_profiles[np.newaxis, :]
        single_profile = True

    filtered_profiles = []
    if filter_type == "savgol":
        window_length = kwargs.get("window_length", 11)
        polyorder = kwargs.get("polyorder", 3)
        for profile in temp_profiles:
            filtered_profiles.append(savgol_filter(profile, window_length, polyorder))
    elif filter_type == "gaussian":
        sigma = kwargs.get("sigma", 2)
        mode = kwargs.get("mode", "reflect")
        for profile in temp_profiles:
            filtered_profiles.append(gaussian_filter1d(profile, sigma, mode=mode))
    else:
        raise ValueError("filter_type must be either 'savgol' or 'gaussian'")

    filtered_profiles = np.array(filtered_profiles)
    if single_profile:
        return filtered_profiles[0]
    return filtered_profiles


def interpolate_reference_data(altitudes, values, target_altitude, label="Reference", smoothing=0.):
    """
    Interpolate reference data to get value at specific altitude using splines.

    Parameters:
    - altitudes: altitude array
    - values: corresponding values array
    - target_altitude: target altitude for interpolation
    - label: descriptive label for the data
    - smoothing: spline smoothing parameter (0 = interpolating spline)

    Returns:
    - dict with interpolated value and metadata
    """
    # Clean and sort data to ensure strictly increasing altitudes
    # Remove duplicates and sort by altitude
    combined = list(zip(altitudes, values))
    # Sort by altitude
    combined_sorted = sorted(combined, key=lambda x: x[0])

    # Remove duplicates by keeping only unique altitudes (keep first occurrence)
    clean_altitudes = []
    clean_values = []
    prev_alt = None

    for alt, val in combined_sorted:
        if prev_alt is None or alt != prev_alt:
            clean_altitudes.append(alt)
            clean_values.append(val)
            prev_alt = alt

    clean_altitudes = np.array(clean_altitudes)
    clean_values = np.array(clean_values)

    # Create spline interpolator with cleaned data
    spline = UnivariateSpline(clean_altitudes, clean_values, s=smoothing)

    # Get interpolated value
    interpolated_value = spline(target_altitude)

    # Find nearest data points for context (using original data)
    altitude_diffs = np.abs(altitudes - target_altitude)
    nearest_idx = np.argmin(altitude_diffs)

    return {
        'interpolated_value': float(interpolated_value),
        'target_altitude': target_altitude,
        'nearest_altitude': altitudes[nearest_idx],
        'nearest_value': values[nearest_idx],
        'altitude_difference': abs(altitudes[nearest_idx] - target_altitude),
        'log_interpolated': np.log(interpolated_value),
        'label': label,
        'n_clean_points': len(clean_altitudes),
        'n_original_points': len(altitudes)
    }


def measure_exponential_slopes(x_data, y_data, label):
    """
    Measure the slope of an exponential curve by fitting in log space.

    Parameters:
    - x_data: x values (altitude in km)
    - y_data: y values (growth ratios)
    - label: descriptive label for the curve

    Returns:
    - slope: slope in log space (1/km)
    - r_squared: goodness of fit
    """
    # Remove any invalid values
    valid_mask = (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
    x_clean = x_data[valid_mask]
    y_clean = y_data[valid_mask]

    if len(x_clean) < 2:
        return np.nan, np.nan

    # Fit in log space: log(y) = slope * x + intercept
    log_y = np.log(y_clean)
    coeffs = np.polyfit(x_clean, log_y, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    # Calculate R²
    log_y_pred = slope * x_clean + intercept
    ss_res = np.sum((log_y - log_y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return slope, r_squared

# Detect oscillation period using autocorrelation and FFT
def detect_wave_period(time, signal):
    """Detect dominant oscillation period using FFT and autocorrelation"""
    dt = np.mean(np.diff(time))
    n = len(signal)

    # Remove DC component and linear trend
    signal_detrended = signal - np.mean(signal)
    coeffs = np.polyfit(time, signal_detrended, 1)
    signal_detrended = signal_detrended - np.polyval(coeffs, time)

    # FFT analysis
    fft_signal = fft(signal_detrended)
    freqs = fftfreq(n, dt)
    power_spectrum = np.abs(fft_signal)**2

    # Find dominant frequency (exclude DC)
    pos_freqs = freqs[freqs > 0]
    pos_power = power_spectrum[freqs > 0]

    # Find peak frequency
    peak_idx = np.argmax(pos_power)
    dominant_freq = pos_freqs[peak_idx]
    period = 1.0 / dominant_freq

    return period, dominant_freq, pos_freqs, pos_power

def exponential_decay_model(distance, a, b, c):
    """Standard three-parameter exponential decay model."""
    distance = np.asarray(distance, dtype=float)
    return a * np.exp(-b * distance) + c


def robust_linear_phase_fit(x_values, phase_values, max_iterations=3, sigma_clip=2.5):
    """Iteratively fit phase = slope * x + intercept with sigma clipping."""
    x_values = np.asarray(x_values, dtype=float)
    phase_values = np.asarray(phase_values, dtype=float)

    mask = np.isfinite(x_values) & np.isfinite(phase_values)
    current_x = x_values[mask]
    current_y = phase_values[mask]
    cov = None

    if current_x.size < 3:
        return np.nan, np.nan, None

    for _ in range(max_iterations):
        if current_x.size < 3:
            return np.nan, np.nan, None
        try:
            params, cov = np.polyfit(current_x, current_y, 1, cov=True)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, None

        slope, intercept = params
        fitted = slope * current_x + intercept
        residuals = current_y - fitted
        sigma = np.std(residuals)

        if sigma == 0 or current_x.size <= 3:
            return slope, intercept, cov

        outliers = np.abs(residuals) > sigma_clip * sigma
        if not np.any(outliers):
            return slope, intercept, cov

        keep_mask = ~outliers
        if np.sum(keep_mask) < 3:
            return np.nan, np.nan, None

        current_x = current_x[keep_mask]
        current_y = current_y[keep_mask]

    return slope, intercept, cov


def robust_exponential_decay_fit(distance, amplitude, initial_guess=None, bounds=None,
                                 sigma_clip=2.5, max_iterations=3):
    """
    Robustly fit amplitude ≈ a * exp(-b * distance) + c with iterative sigma clipping.
    Returns (popt, pcov); popt is NaN-filled on failure.
    """
    distance = np.asarray(distance, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)

    mask = np.isfinite(distance) & np.isfinite(amplitude)
    current_x = distance[mask]
    current_y = amplitude[mask]

    if current_x.size < 3:
        return np.full(3, np.nan), None

    if initial_guess is None:
        a0 = np.nanmax(current_y) if current_y.size else 1.0
        b0 = 1.0 / max(np.nanmax(current_x) - np.nanmin(current_x), 1.0)
        c0 = np.nanmin(current_y) if current_y.size else 0.0
        initial_guess = (a0, b0, c0)

    if bounds is None:
        bounds = ([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf])

    for _ in range(max_iterations):
        if current_x.size < 3:
            break
        try:
            popt, pcov = curve_fit(
                exponential_decay_model,
                current_x,
                current_y,
                p0=initial_guess,
                bounds=bounds,
                maxfev=5000,
            )
        except (RuntimeError, ValueError):
            return np.full(3, np.nan), None

        fitted = exponential_decay_model(current_x, *popt)
        residuals = current_y - fitted
        sigma = np.std(residuals)

        if sigma == 0 or current_x.size <= 3:
            return popt, pcov

        outliers = np.abs(residuals) > sigma_clip * sigma
        if not np.any(outliers):
            return popt, pcov

        keep_mask = ~outliers
        if np.sum(keep_mask) < 3:
            return np.full(3, np.nan), None

        current_x = current_x[keep_mask]
        current_y = current_y[keep_mask]

    return popt, pcov