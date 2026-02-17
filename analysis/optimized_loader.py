"""
Optimized DSMC Data Processor

This module provides an optimized version of the DSMCDataProcessor class,
focusing on improved modularity, readability, and Pythonic principles.
It follows the DRY principle, uses meaningful names, and implements
error handling where appropriate.

Key improvements:
- Modular structure with smaller, focused methods
- Better separation of concerns
- Enhanced error handling and validation
- Improved documentation and type hints
- Optimized data loading and processing
"""

from glob import glob
import pickle
import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from numpy.lib.stride_tricks import sliding_window_view

from helper_funcs_zenodo import extract_grid_edges
import seaborn as sns

sns.set_theme(context="paper", font_scale=2.3, style="whitegrid")

# Import helper functions (assuming they exist)
try:
    from helper_funcs_zenodo import (
        compute_heating_rate_slope,
        process_temperature_deviation_peaks,
        process_growth_num_amp,
        TIAN_GROW,
        TIAN_ALTITUDES,
        SPHERICAL_1D_GROW,
        SPHERICAL_1D_ALTITUDES,
        mars_radius as MARS_RADIUS,
    )
except ImportError:
    # Placeholder for helper functions
    def compute_heating_rate_slope(*args, **kwargs):
        raise NotImplementedError("Helper functions not available")

    def process_temperature_deviation_peaks(*args, **kwargs):
        raise NotImplementedError("Helper functions not available")

    def process_growth_num_amp(*args, **kwargs):
        raise NotImplementedError("Helper functions not available")

    TIAN_GROW = []
    TIAN_ALTITUDES = []
    SPHERICAL_1D_GROW = []
    SPHERICAL_1D_ALTITUDES = []


def find_min_mad_window(data, window_sizes):
    """
    Compute the rolling MAD for a range of window sizes using vectorized operations,
    then return the window (its edges) that yields the lowest MAD.
    More robust to handle edge cases and low-density data.
    """
    best_mad = float("inf")
    best_info = None

    # Try to clean data - replace any potential NaN values
    clean_data = np.nan_to_num(data, nan=0.0)

    # As a fallback, use the first 1/4 of data available
    fallback_window_size = min(window_sizes[0], len(clean_data) // 4)
    fallback_start = 0
    fallback_end = fallback_start + fallback_window_size - 1

    for window_size in window_sizes:
        # Skip if window size is larger than the data array
        if window_size >= len(clean_data):
            continue

        windows = sliding_window_view(clean_data, window_size)  # shape: (n-window_size+1, window_size)

        # Handle potential empty windows
        if len(windows) == 0:
            continue

        # Calculate medians for each window
        medians = np.median(windows, axis=1)

        # More lenient condition - accept windows with very low but non-zero median
        # Use a small epsilon value instead of exactly zero
        valid_windows = medians > 1e-10
        if not np.any(valid_windows):
            continue

        # Only compute MAD for valid windows
        valid_mads = np.median(np.abs(windows[valid_windows] - medians[valid_windows, None]), axis=1)

        # Find minimum MAD among valid windows
        if len(valid_mads) > 0:
            idx_min = valid_mads.argmin()
            valid_indices = np.where(valid_windows)[0]  # Get original indices of valid windows
            if valid_mads[idx_min] < best_mad:
                best_mad = valid_mads[idx_min]
                best_info = (window_size, (valid_indices[idx_min], valid_indices[idx_min] + window_size - 1))

    # If no valid windows found, return the fallback window
    if best_info is None:
        return (fallback_window_size, (fallback_start, fallback_end), float("inf"))

    return (best_info[0], best_info[1], best_mad)


@dataclass
class SimulationParameters:
    """Storage class for simulation parameters with validation"""
    mid_azi: np.ndarray
    radial_cells: np.ndarray
    r1_distances: np.ndarray
    time: np.ndarray
    mars_radius: float
    start_step_wave: int
    final_step_wave: int
    timestep: float
    omega_freq: float # Angular frequency in rad/s
    num_periods: float
    radial_edges: np.ndarray
    azi_edges: np.ndarray
    n_rad_cell: int = 1  # Default to 1 if not specified
    n_azi_cell: int = 1  # Default to 1 if not specified

    @property
    def wave_period(self) -> float:
        """Calculate wave period from frequency"""
        return 2 * np.pi / self.omega_freq


class OptimizedDSMCDataProcessor:
    """
    Optimized processor for DSMC simulation data with improved modularity.

    This class provides methods for loading, processing, and analyzing DSMC
    simulation data, with a focus on clean architecture and efficient computation.
    """

    def __init__(self, data_dir: str, background_data_dir: Optional[str] = None):
        """
        Initialize the optimized DSMC data processor.

        Args:
            data_dir: Path to directory containing simulation output files
            background_data_dir: Optional path to directory containing background simulation data
        """
        self.data_dir = Path(data_dir)
        self.params: Optional[SimulationParameters] = None
        self.density_data: Optional[np.ndarray] = None
        self.number_particles_data: Optional[np.ndarray] = None
        self.temperature_data: Optional[np.ndarray] = None
        self.radial_vel_data: Optional[np.ndarray] = None
        self.bg_window: Optional[Dict[str, int]] = None
        self.background_profiles: Optional[Dict[str, np.ndarray]] = None
        self.surface_data: Optional[Dict[str, Any]] = None
        self.escape_data: Optional[Dict[str, Any]] = None
        self.knudsen_data: Optional[np.ndarray] = None
        self.scale_height_data: Optional[np.ndarray] = None
        self._macro_cache: Optional[Dict[str, Any]] = None
        self._macro_data_path: Optional[Path] = None
        self._parameters_loaded = False
        self._data_loaded = False

        # Identify macro data file if exists as macro_grids_data.pkl OR macro_grids.pkl OR "combined_*runs_macro_grids_data.pkl"
        possible_macro_files = list(self.data_dir.glob("combined_*runs_macro_grids_data.pkl")) + \
            list(self.data_dir.glob("macro_grids.pkl")) + \
            list(self.data_dir.glob("macro_grids_data.pkl"))

        if possible_macro_files:
            self._macro_data_path = possible_macro_files[0]

        # Initialize background processor if provided
        self.background_processor = None
        if background_data_dir is not None:
            self.background_processor = OptimizedDSMCDataProcessor(background_data_dir)
            self.background_processor.load_parameters()
            self.background_processor.load_data()

    def load_parameters(self, params_file: str = "params_data.npz", macro_data_filename=None) -> None:
        if self._parameters_loaded:
            return
        params_path = self.data_dir / params_file
        if not params_path.exists():
            # Fallback to ASCII extraction
            self.__extract_from_ascii(macro_data_filename=macro_data_filename)
            self._parameters_loaded = True
            return

        # Load NPZ parameters
        params = np.load(params_path, allow_pickle=True)

        # Extract key parameters with fallbacks
        mars_radius = params.get("mars_radius", params.get("radius", params.get("R", MARS_RADIUS/1e3)))
        omega_freq = params.get("omega_freq", params.get("omega", params.get("frequency", 0.0)))
        timestep = params.get("timestep", params.get("dt", params.get("time_step", 0.0)))
        start_step_wave = params.get("start_step_wave", params.get("wave_start", params.get("step0_inject", 0)))
        final_step_wave = params.get("final_step_wave", params.get("wave_end", params.get("stepf_inject", 0)))
        num_periods = params.get("num_periods", 0.0)

        # Extract arrays
        radial_cells = params.get("radial_cells", params.get("radial_distances", np.array([])))
        # If radial cells[0] is in the order of > 1e5, it's likely in m and has the mars radius added, so convert to km and subtract mars radius
        if len(radial_cells) > 0 and radial_cells[0] > 1e5:
            radial_cells = radial_cells / 1000.0 - mars_radius
        n_rad_cell = len(radial_cells) if len(radial_cells) > 0 else 1
        mid_azi = params.get("mid_azi", np.array([]))
        r1_distances = params.get("r1_distances", np.array([]))
        time = params.get("time", params.get("t", np.array([])))

        # Extract grid edges from Grid.ascii
        radial_edges, azi_edges = extract_grid_edges(f"{self.data_dir}/Info/Grid.ascii")

        try:
            n_azi_cell = len(mid_azi) if len(mid_azi) > 0 else 1
        except TypeError:
            n_azi_cell = 1

        # Validate essential parameters
        if mars_radius <= 0:
            raise ValueError("Invalid Mars radius")
        if omega_freq <= 0:
            raise ValueError("Invalid wave frequency")
        if timestep <= 0:
            raise ValueError("Invalid timestep")

        # Create SimulationParameters instance
        self.params = SimulationParameters(
            radial_cells=radial_cells,
            mid_azi=mid_azi,
            r1_distances=r1_distances,
            time=time,
            mars_radius=mars_radius,
            start_step_wave=start_step_wave,
            final_step_wave=final_step_wave,
            timestep=timestep,
            omega_freq=omega_freq,
            num_periods=num_periods,
            n_azi_cell=n_azi_cell,
            n_rad_cell=n_rad_cell,
            radial_edges=radial_edges,
            azi_edges=azi_edges,
        )
        self._parameters_loaded = True

    def __extract_from_ascii(self, macro_data_filename=None) -> None:
        """
        Fallback method to extract parameters from ASCII files when NPZ is not available.
        """
        import re
        from pathlib import Path

        # Read Simulation_Parameters.ascii
        ascii_path = self.data_dir / "Info" / "Simulation_Parameters.ascii"
        if not ascii_path.exists():
            raise FileNotFoundError(f"ASCII parameter file not found: {ascii_path}")

        # Check if macro_grids.pkl exists to extract time stamps
        if macro_data_filename is None and self._macro_data_path is not None:
            macro_data_filename = self._macro_data_path.name
        elif macro_data_filename is None:
            macro_data_filename = "macro_grids_data.pkl"

        params_dict = self.__read_simulation_parameters(ascii_path)
        sim_params = params_dict.get("Simulation Parameters", {})
        planetary = params_dict.get("Planetary Body Information", {})

        # Extract basic parameters
        timestep = sim_params.get("Time Step", 0.0)
        nsteps = sim_params.get("Number of Total Simulations Steps", 0)
        avg_taken_every = sim_params.get("Averages Taken Every", 1)
        mars_radius = planetary.get("Radius", 0.0)
        omega_freq = sim_params.get("Wave Injection Frequency", 0.0)
        start_step_wave = sim_params.get("Start Step of Wave Injection", 0)
        final_step_wave = sim_params.get("End Step of Wave Injection", 0)
        n_rad_cell = sim_params.get("Number of Radial Cells", 0)
        n_azi_cell = sim_params.get("Number of Azimuth Cells", 1)

        # Compute derived values
        num_periods = round((final_step_wave - start_step_wave - 1) * omega_freq * timestep / (2 * np.pi), 2) if omega_freq > 0 else 0.0

        # Generate time array
        avg_time_interval = avg_taken_every * timestep

        macro_grids_path = self.data_dir / macro_data_filename
        if macro_grids_path.exists():
            try:
                with open(macro_grids_path, "rb") as f:
                    macro_data = pickle.load(f)
                    time = macro_data["Time Stamps"]
                print(f"Time array extracted from {macro_grids_path}")
            except (KeyError, pickle.UnpicklingError, Exception) as e:
                print(f"Warning: Failed to extract time from {macro_grids_path}: {e}. Falling back to generation.")
                time = np.linspace(avg_time_interval, nsteps * timestep, nsteps // avg_taken_every)
        else:
            print(f"Warning: {macro_grids_path} not found. Generating time array.")
            time = np.linspace(avg_time_interval, nsteps * timestep, nsteps // avg_taken_every)

        # Extract grid edges
        radial_edges, azi_edges = extract_grid_edges(f"{self.data_dir}/Info/Grid.ascii")

        # Extract radial cells from Radial_Cells.ascii if available
        radial_cells = (self.__extract_radial_cells() - MARS_RADIUS) / 1e3
        mid_azi = 0.5 * (azi_edges[:-1] + azi_edges[1:]) if len(azi_edges) > 1 else np.array([])
        r1_distances = np.array([])  # Placeholder

        # Validate
        if mars_radius <= 0:
            raise ValueError("Invalid Mars radius from ASCII")
        if omega_freq <= 0:
            raise ValueError("Invalid wave frequency from ASCII")
        if timestep <= 0:
            raise ValueError("Invalid timestep from ASCII")

        # Create SimulationParameters
        self.params = SimulationParameters(
            radial_cells=radial_cells,
            mid_azi=mid_azi,
            r1_distances=r1_distances,
            time=time,
            mars_radius=mars_radius,
            start_step_wave=start_step_wave,
            final_step_wave=final_step_wave,
            timestep=timestep,
            omega_freq=omega_freq,
            num_periods=num_periods,
            n_azi_cell=n_azi_cell,
            n_rad_cell=n_rad_cell,
            radial_edges=radial_edges,
            azi_edges=azi_edges,
        )
        self._parameters_loaded = True

    def __read_simulation_parameters(self, ascii_path: Path) -> dict:
        """
        Read simulation parameters from ASCII file, similar to create_db.py.
        """
        parameters = {}
        current_section = None
        num_pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
        special_keys_no_colon = ["Samples Taken Every", "Averages Taken Every"]

        with open(ascii_path, "r") as file:
            for line in file:
                line = line.strip()
                if line.endswith("Information") or line == "Simulation Parameters":
                    current_section = line
                    if current_section not in parameters:
                        parameters[current_section] = {}
                elif "Time Step:" in line:
                    if "Simulation Parameters" not in parameters:
                        parameters["Simulation Parameters"] = {}
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    match = num_pattern.search(value)
                    if match:
                        num_val = float(match.group())
                        if num_val.is_integer() and num_val < 1e15:
                            num_val = int(num_val)
                        value = num_val
                    parameters["Simulation Parameters"][key] = value
                elif line and ":" in line and current_section:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    if key not in ["Name", "Species 1", "Species 2"]:
                        match = num_pattern.search(value)
                        if match:
                            num_val = float(match.group())
                            if num_val.is_integer() and num_val < 1e15:
                                num_val = int(num_val)
                            value = num_val
                    if current_section in ["Species Information", "Collision Information"]:
                        if not parameters[current_section]:
                            parameters[current_section] = []
                        parameters[current_section].append({key: value})
                    else:
                        parameters[current_section][key] = value
                elif current_section == "Simulation Parameters":
                    for special_key in special_keys_no_colon:
                        if line.startswith(special_key):
                            key = special_key
                            value_str = line[len(special_key):].strip()
                            match = num_pattern.search(value_str)
                            if match:
                                value = float(match.group())
                                if value.is_integer() and value < 1e15:
                                    value = int(value)
                            else:
                                value = value_str
                            parameters["Simulation Parameters"][key] = value
                            break
        return parameters

    def __extract_radial_cells(self) -> np.ndarray:
        """
        Extract radial cell midpoints from Radial_Cells.ascii if available.
        """
        rad_cells_path = self.data_dir / "Info" / "Radial_Cells.ascii"
        if not rad_cells_path.exists():
            return np.array([])

        try:
            # Simple extraction, assuming format from compute_macro_grids.py
            df = pd.read_csv(rad_cells_path, sep=r"\s+", header=None, usecols=[1, 2], names=["R1", "R2"])
            mid_bound = (df["R1"] + df["R2"]) / 2.0
            return mid_bound.values
        except Exception:
            return np.array([])

    def _resolve_macro_filename(self, macro_data_filename: Optional[str]) -> str:
        if macro_data_filename is not None:
            return macro_data_filename
        if self._macro_data_path is not None:
            return self._macro_data_path.name
        return "macro_grids_data.pkl"

    def _load_macro_data(self, macro_filename: str) -> Dict[str, Any]:
        if self._macro_cache is not None:
            return self._macro_cache
        macro_path = self.data_dir / macro_filename
        if not macro_path.exists():
            raise FileNotFoundError(macro_path)
        with open(macro_path, "rb") as f:
            self._macro_cache = pickle.load(f)
        return self._macro_cache

    @staticmethod
    def _squeeze_species(array: np.ndarray) -> np.ndarray:
        if array.ndim == 4 and array.shape[0] == 1:
            return array.squeeze(0)
        return array

    def _load_array_with_fallback(
        self,
        primary_filename: Optional[str],
        alt_filename: Optional[str],
        macro_key: str,
        macro_filename: str,
        required: bool = True,
    ) -> Optional[np.ndarray]:
        candidate_paths: List[str] = []
        for candidate in filter(None, [primary_filename, alt_filename]):
            path = self.data_dir / candidate
            candidate_paths.append(str(path))
            if path.exists():
                return self._squeeze_species(np.load(path))

        try:
            macro_data = self._load_macro_data(macro_filename)
        except FileNotFoundError:
            if required:
                all_paths = candidate_paths + [str(self.data_dir / macro_filename)]
                raise FileNotFoundError(
                    f"{macro_key} data not found in {', '.join(all_paths)}"
                )
            return None

        if macro_key in macro_data:
            return self._squeeze_species(macro_data[macro_key])

        if required:
            raise FileNotFoundError(
                f"{macro_key} data not found in {', '.join(candidate_paths + [str(self.data_dir / macro_filename)])}"
            )
        return None

    def _load_surface_and_escape(self, macro_filename: str) -> None:
        self.surface_data = None
        self.escape_data = None

        surface_escape_path = self.data_dir / "surface_escape_data.npz"
        if surface_escape_path.exists():
            surface_escape_data = np.load(surface_escape_path, allow_pickle=True)
            self.surface_data = {
                "base_data": {
                    name: surface_escape_data["surf_base_data"][:, :, col_idx]
                    for name, col_idx in {
                        "time": 0,
                        "avg_time_interval": 1,
                        "rad": 2,
                        "zen": 3,
                        "azi": 4,
                        "cell_num": 5,
                        "lower_area": 6,
                    }.items()
                },
                "species_data": {
                    name: surface_escape_data["surf_species_data"][:, :, :, col_idx]
                    for name, col_idx in {
                        "num_part_inject": 0,
                        "num_part_return": 1,
                        "num_part_reemit": 2,
                        "num_part_surf": 3,
                        "inject_rate": 4,
                        "return_rate": 5,
                        "reemit_rate": 6,
                        "return_time": 7,
                        "inject_energy": 8,
                        "return_energy": 9,
                    }.items()
                },
                "species_names": surface_escape_data["species_names"],
                "flow_rates": surface_escape_data["flow_rates"],
                "surface_base_data": surface_escape_data["surface_base_data"],
            }
            self.escape_data = {
                "base_data": {
                    name: surface_escape_data["escape_base_data"][:, :, col_idx]
                    for name, col_idx in {
                        "time": 0,
                        "avg_time_interval": 1,
                        "rad": 2,
                        "zen": 3,
                        "azi": 4,
                        "cell_num": 5,
                        "upper_area": 6,
                    }.items()
                },
                "species_data": {
                    name: surface_escape_data["escape_species_data"][:, :, :, col_idx]
                    for name, col_idx in {
                        "num_part_escape": 0,
                        "num_part_ball": 1,
                        "escape_rate": 2,
                        "escape_time": 3,
                        "escape_speed": 4,
                        "escape_energy": 5,
                    }.items()
                },
                "species_names": surface_escape_data["species_names"],
            }
            return

        try:
            macro_data = self._load_macro_data(macro_filename)
        except FileNotFoundError:
            print(
                f"Warning: Neither surface_escape_data.npz nor {macro_filename} found. Skipping surface data loading."
            )
            return

        if "Surface Rates" not in macro_data:
            print(f"Warning: 'Surface Rates' not found in {macro_filename}. Skipping surface data loading.")
            return

        surface_rates = macro_data["Surface Rates"]
        self.load_parameters(macro_data_filename=macro_filename)
        if surface_rates.ndim == 2 and self.params.n_azi_cell == 1 and surface_rates.shape[1] == 17:
            surface_rates = surface_rates[:, None, :]
        elif surface_rates.ndim != 3 or surface_rates.shape[2] != 17:
            print(
                f"Warning: 'Surface Rates' in {macro_filename} does not have expected shape (time, azimuthal, 17). Skipping surface data loading."
            )
            return

        base_data = surface_rates[:, :, 0:7]
        species_data = surface_rates[:, :, 7:17][None, :, :, :]
        self.surface_data = {
            "base_data": {
                "time": base_data[:, :, 0],
                "avg_time_interval": base_data[:, :, 1],
                "rad": base_data[:, :, 2],
                "zen": base_data[:, :, 3],
                "azi": base_data[:, :, 4],
                "cell_num": base_data[:, :, 5],
                "lower_area": base_data[:, :, 6],
            },
            "species_data": {
                "num_part_inject": species_data[:, :, :, 0],
                "num_part_return": species_data[:, :, :, 1],
                "num_part_reemit": species_data[:, :, :, 2],
                "num_part_surf": species_data[:, :, :, 3],
                "inject_rate": species_data[:, :, :, 4],
                "return_rate": species_data[:, :, :, 5],
                "reemit_rate": species_data[:, :, :, 6],
                "return_time": species_data[:, :, :, 7],
                "inject_energy": species_data[:, :, :, 8],
                "return_energy": species_data[:, :, :, 9],
            },
            "species_names": [],
            "flow_rates": None,
            "surface_base_data": base_data,
        }

    def load_data(self, macro_data_filename=None) -> None:
        """
        Load all simulation data files with optimized loading, based on wave_flux_analyzer.py patterns.
        Falls back to extracting from macro_grids_data.pkl if individual .npy files are not available.

        Raises:
            FileNotFoundError: If required data files are missing and cannot be extracted from pickle
        """
        if self._data_loaded:
            return
        self._macro_cache = None
        macro_data_filename = self._resolve_macro_filename(macro_data_filename)

        self.density_data = self._load_array_with_fallback(
            "density_data.npy",
            "macro_Number Density.npy",
            "Number Density",
            macro_data_filename,
        )
        self.temperature_data = self._load_array_with_fallback(
            "temperature_data.npy",
            "macro_Kinetic Temperature.npy",
            "Kinetic Temperature",
            macro_data_filename,
        )
        self.radial_vel_data = self._load_array_with_fallback(
            "rad_vel_data.npy",
            "macro_Radial Velocity.npy",
            "Radial Velocity",
            macro_data_filename,
        )

        self.knudsen_data = None
        knudsen = self._load_array_with_fallback(
            "macro_Knudsen Number.npy",
            None,
            "Knudsen Number",
            macro_data_filename,
            required=False,
        )
        if knudsen is not None:
            self.knudsen_data = knudsen

        self.scale_height_data = None
        scale_height = self._load_array_with_fallback(
            "macro_Scale Height.npy",
            None,
            "Scale Height",
            macro_data_filename,
            required=False,
        )
        if scale_height is not None:
            self.scale_height_data = scale_height

        self.number_particles_data = self._load_array_with_fallback(
            "macro_Number Particles.npy",
            None,
            "Number Particles",
            macro_data_filename,
            required=False,
        )

        self._load_surface_and_escape(macro_data_filename)
        self._data_loaded = True

    def get_wave_time_indices(self) -> Tuple[int, int]:
        """
        Calculate start and end time indices for wave analysis.

        Returns:
            Tuple of (start_idx, end_idx)

        Raises:
            ValueError: If parameters are not loaded
        """
        if self.params is None:
            raise ValueError("Parameters not loaded. Call load_parameters() first.")

        start_idx = np.where(self.params.time > (self.params.start_step_wave * self.params.timestep))[0]
        if len(start_idx) == 0:
            start_idx = 0
        else:
            start_idx = start_idx[0]

        end_idx = np.where(self.params.time > self.params.final_step_wave * self.params.timestep)[0]
        if len(end_idx) == 0:
            end_idx = len(self.params.time) - 1
        else:
            end_idx = end_idx[0] - 5  # Padding as in original

        return start_idx, end_idx

    def print_background_window_info(self) -> None:
        """Print information about the background window."""
        # Implementation placeholder
        raise NotImplementedError("print_background_window_info method not yet implemented")

    def _compute_window_sizes(self, end_idx: int, custom_time_windows: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the window sizes for background analysis.

        Args:
            end_idx: End index for the analysis period
            custom_time_windows: Optional array of custom window sizes in seconds

        Returns:
            Tuple of (time_windows, window_sizes)
        """
        dt = np.mean(np.diff(self.params.time))
        available_time = self.params.time[end_idx]
        max_window_time = min(available_time * 0.5, 2500)

        if custom_time_windows is not None:
            # Convert to numpy array if it's a list
            if isinstance(custom_time_windows, list):
                custom_time_windows = np.array(custom_time_windows)

            # Validate custom windows
            if np.any(custom_time_windows > available_time * 0.5):
                raise ValueError(
                    f"Window sizes must be less than half the available time ({available_time * 0.5:.1f} s)"
                )
            if np.any(custom_time_windows < 100):
                raise ValueError("Window sizes must be at least 100 seconds")
            time_windows = np.array(custom_time_windows)
        else:
            # Use default window sizes
            time_windows = np.linspace(1000, max_window_time, 4)

        window_sizes = np.maximum((time_windows / dt).astype(int), 3)
        return time_windows, window_sizes

    def _find_optimal_window(self, data: np.ndarray, window_sizes: np.ndarray) -> Tuple[int, Tuple[int, int], float]:
        """
        Find optimal window for a time series using MAD.

        Args:
            data: Time series data
            window_sizes: Array of window sizes to try

        Returns:
            Tuple of (window_size, (start_idx, end_idx), mad_value)
        """
        return find_min_mad_window(data, window_sizes)

    def _compute_background_value(self, data: np.ndarray, window_start: int, window_end: int) -> float:
        """
        Compute background value for a given window.

        Args:
            data: Time series data
            window_start: Start index of window
            window_end: End index of window

        Returns:
            Background value (mean within window)
        """
        return np.mean(data[window_start : window_end + 1])

    def get_background_profiles(self, custom_time_windows: Optional[np.ndarray] = None, ref_altitude: float = 230.0) -> Dict[str, np.ndarray]:
        """
        Calculate background profiles using standard method.

        If a background processor is provided, the standard method will be applied
        on the background processor's data instead of the primary processor's data.

        Args:
            custom_time_windows: Optional array of custom window sizes in seconds
            ref_altitude: Reference altitude for window selection

        Returns:
            Dictionary containing background profiles
        """
        # If background_profiles already computed, return them
        if self.background_profiles is not None:
            return self.background_profiles

        # Decide which processor to use for background computation
        processor_to_use = self.background_processor if self.background_processor is not None else self

        # If using background processor, verify compatible altitude grid
        if self.background_processor is not None:
            if self.params is None or self.background_processor.params is None:
                raise ValueError("Parameters not loaded for primary or background processor.")
            if not np.allclose(self.params.radial_cells, self.background_processor.params.radial_cells):
                raise ValueError("Background dataset has different altitude grid")

        # Check if data is loaded in the processor we're using
        if any(x is None for x in [processor_to_use.density_data, processor_to_use.temperature_data]):
            raise ValueError("Data not loaded. Call load_data() first.")

        # Get wave timing from the processor we're using
        start_idx, _ = processor_to_use.get_wave_time_indices()
        end_idx = start_idx - 5

        # Get reference altitude data
        ref_idx = np.abs(processor_to_use.params.radial_cells - ref_altitude).argmin()
        density_ref_series = processor_to_use.density_data[:end_idx, ref_idx, 0]
        temperature_ref_series = processor_to_use.temperature_data[:end_idx, ref_idx, 0]

        # Compute window sizes with custom windows if provided
        time_windows, window_sizes = processor_to_use._compute_window_sizes(end_idx, custom_time_windows)

        # Find optimal windows - check for None results
        density_result = processor_to_use._find_optimal_window(density_ref_series, window_sizes)
        temp_result = processor_to_use._find_optimal_window(temperature_ref_series, window_sizes)

        # Check if valid windows were found - use fallback if not
        if density_result[0] is None or temp_result[0] is None:
            # Calculate a fallback window - use first 30% of data before wave start
            fallback_window_end = max(10, int(end_idx * 0.3))
            fallback_window_start = max(0, fallback_window_end - window_sizes[0])

            # Create fallback results
            d_win_size = temp_win_size = window_sizes[0]
            d_win_start = t_win_start = fallback_window_start
            d_win_end = t_win_end = fallback_window_end
            d_mad = t_mad = np.nan  # We don't have a MAD value for the fallback
        else:
            # Unpack results from optimal window finding
            d_win_size, (d_win_start, d_win_end), d_mad = density_result
            t_win_size, (t_win_start, t_win_end), t_mad = temp_result

        # Use average window
        window_info = {
            "start": (d_win_start + t_win_start) // 2,
            "end": (d_win_end + t_win_end) // 2,
        }

        # Set bg_window in both processors
        processor_to_use.bg_window = window_info
        # print out window info
        print(f"Background window selected: Start index = {window_info['start']}, End index = {window_info['end']}")
        # override, by hardcode
        # window_info = {"start": 280, "end": 287}

        # If we're using background processor, also set the window in the primary processor
        if self.background_processor is not None:
            self.bg_window = {
                "start": window_info["start"],
                "end": window_info["end"],
                "from_external": True,  # Mark that this window is from external processor
            }

        # Compute background profiles
        num_altitudes = processor_to_use.density_data.shape[1]
        density_bg = np.empty(num_altitudes)
        temperature_bg = np.empty(num_altitudes)

        for alt in range(num_altitudes):
            density_bg[alt] = processor_to_use._compute_background_value(
                processor_to_use.density_data[:end_idx, alt, 0],
                window_info["start"],
                window_info["end"],
            )
            temperature_bg[alt] = processor_to_use._compute_background_value(
                processor_to_use.temperature_data[:end_idx, alt, 0],
                window_info["start"],
                window_info["end"],
            )

        # Store background profiles in the current processor
        self.background_profiles = {
            "density": density_bg,
            "temperature": temperature_bg,
            "source": "external" if self.background_processor is not None else "primary",
        }

        return self.background_profiles

    def create_stitched_visualization(self, data_field: str = "density", altitudes: Optional[List[float]] = None, show_bg_window: bool = True) -> None:
        """
        Create a visualization that shows stitched data from background and primary processors.

        Args:
            data_field: The field to visualize (density, temperature, rad_vel)
            altitudes: List of altitudes to plot [km]. If None, uses 150km and 320km.
            show_bg_window: Whether to show the background window used for computation
        """
        # Implementation placeholder
        raise NotImplementedError("create_stitched_visualization method not yet implemented")

    def _print_window_info(self, name: str, win_size: int, start: int, end: int, mad: float) -> None:
        """Print information about an optimal window."""
        # Implementation placeholder
        raise NotImplementedError("_print_window_info method not yet implemented")

    def _plot_optimal_windows(self, optimal_windows: np.ndarray) -> None:
        """
        Visualize the optimal windows chosen for each altitude.

        Args:
            optimal_windows: Array of (start, end) indices for each altitude
        """
        # Implementation placeholder
        raise NotImplementedError("_plot_optimal_windows method not yet implemented")

    def stitch_background_data(self, data_field: str = "density") -> Dict[str, Any]:
        """
        Stitch data from background processor and primary processor.

        Args:
            data_field: The field to stitch (e.g., "density", "temperature")

        Returns:
            Dictionary with stitched data information
        """
        # Implementation placeholder
        raise NotImplementedError("stitch_background_data method not yet implemented")

    def calculate_perturbations(self) -> Dict[str, np.ndarray]:
        """
        Calculate perturbations from background state.

        Returns:
            Dictionary containing perturbation fields
        """
        # Implementation placeholder
        raise NotImplementedError("calculate_perturbations method not yet implemented")

    def get_surface_property(self, property_name: str, species_idx: Optional[int] = None) -> np.ndarray:
        """
        Get surface property data by name.

        Args:
            property_name: Name of the property to retrieve
            species_idx: Species index for species-specific properties

        Returns:
            Property data as numpy array
        """
        # Implementation placeholder
        raise NotImplementedError("get_surface_property method not yet implemented")

    def get_escape_property(self, property_name: str, species_idx: Optional[int] = None) -> np.ndarray:
        """
        Get escape property data by name.

        Args:
            property_name: Name of the property to retrieve
            species_idx: Species index for species-specific properties

        Returns:
            Property data as numpy array
        """
        # Implementation placeholder
        raise NotImplementedError("get_escape_property method not yet implemented")

    def compute_osc_cols(self) -> np.ndarray:
        """
        Compute oscillating azimuthal indices based on surface rates or escape data.

        Returns:
            Array of oscillating azimuthal column indices.

        Raises:
            ValueError: If no oscillating columns found or data not loaded.
        """
        if self.surface_data is None and self.escape_data is None:
            growth_peak_files = sorted(Path(self.data_dir).glob("growth_peak_data_*azi_macro_grids_data.pkl"))
            if growth_peak_files:
                # Load the first available growth peak data file, and extract osc_cols
                with open(growth_peak_files[0], "rb") as f:
                    growth_data = pickle.load(f)
                    if "azi_osci" in growth_data:
                        return np.array(growth_data["azi_osci"])
            raise ValueError("Surface or escape data not loaded. Call load_data() first.")

        if self.surface_data is not None:
            # Use surface rates
            surf_rates = self.surface_data["species_data"]["inject_rate"]
            if surf_rates.ndim == 3:
                baseline = surf_rates[0, 0, :]
                std_over_time = np.std(surf_rates[0] - baseline[None, :], axis=0)
            else:
                baseline = surf_rates[0, :]
                std_over_time = np.std(surf_rates - baseline[None, :], axis=0)
        elif self.escape_data is not None:
            # Use escape rates
            escape_rates = self.escape_data["species_data"]["escape_rate"]
            if escape_rates.ndim == 3:
                baseline = escape_rates[0, :, :]
                std_over_time = np.std(escape_rates - baseline[:, None, :], axis=0)
            else:
                baseline = escape_rates[0, :]
                std_over_time = np.std(escape_rates - baseline[None, :], axis=0)
        else:
            raise ValueError("No suitable data for computing oscillating columns.")

        # Non-zero std indicates oscillation; use tolerance for numerical noise
        osc_mask = ~np.isclose(std_over_time, 0.0, atol=1e-12)
        osc_cols = np.where(osc_mask)[0]
        if len(osc_cols) == 0:
            raise ValueError("No oscillating azimuthal columns found.")

        return osc_cols

    def compute_density_growth(
        self,
        alt_range: Tuple[float, float] = (150, 320),
        azimuthal_indices: Optional[List[int]] = None,
        window_sizes: Optional[np.ndarray] = None,
        use_dynamic_peak_detection: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute density growth curves for specified azimuthal indices.

        Args:
            alt_range: Tuple of (min_altitude, max_altitude) in km
            azimuthal_indices: List of azimuthal indices to compute growth for.
                               If None, uses oscillating indices.
            window_sizes: Optional array of window sizes in seconds
            n_last_peaks: Number of last peaks to average
            use_dynamic_peak_detection: Whether to use dynamic peak parameter adjustment

        Returns:
            Dictionary containing growth results and metadata

        Raises:
            ValueError: If parameters or data not loaded
        """
        if self.params is None:
            raise ValueError("Parameters not loaded. Call load_parameters() first.")
        if self.density_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Determine azimuthal indices to use
        if azimuthal_indices is None:
            try:
                azimuthal_indices = self.compute_osc_cols()
            except ValueError:
                # If no oscillating columns, use all if n_azi_cell == 1, else raise
                if self.params.n_azi_cell == 1:
                    azimuthal_indices = [0]
                else:
                    raise ValueError("No oscillating azimuthal indices found. Please specify azimuthal_indices.")
        else:
            azimuthal_indices = np.array(azimuthal_indices)

        # Get background profiles using standard method
        background = self.get_background_profiles(window_sizes)

        # Compute perturbations
        density_perturbation = (
            self.density_data - background["density"][None, :, None]
        ) / background["density"][None, :, None]

        # Find cell indices for analysis regions
        cell_min_idx = np.where(self.params.radial_cells > alt_range[0])[0][0]
        cell_max_idx = np.where(self.params.radial_cells > alt_range[1])[0][0]

        # Process peaks for each azimuthal index
        growth_results = {}
        for azi_idx in azimuthal_indices:
            if azi_idx >= self.density_data.shape[-1]:
                continue  # Skip invalid indices

            # Process peaks
            base_kwargs = {
                "prominence": 1e-1,
                "distance": 12,
                "sampling_rate": 1 / np.mean(np.diff(self.params.time)),
                "cutoff_frequency": self.params.omega_freq * 1.2,
                "filter_order": 2,
                "use_dynamic": use_dynamic_peak_detection,
            }

            try:
                growth_peaks, baseline_peaks = process_growth_num_amp(
                    density_perturbation[..., azi_idx],
                    (self.params.start_step_wave * self.params.timestep <= self.params.time)
                    & (
                        self.params.time
                        <= self.params.final_step_wave * self.params.timestep
                        + 700  # PADDING
                    ),
                    self.params.radial_cells,
                    cell_min_idx,
                    cell_max_idx,
                    self.params.num_periods,
                    **base_kwargs,
                )

                # Store results for this azimuthal index
                growth_results[azi_idx] = {
                    "growth_peaks": growth_peaks,
                    "baseline_peaks": baseline_peaks,
                    "radial_distances": self.params.radial_cells,
                    "cell_min_idx": cell_min_idx,
                    "cell_max_idx": cell_max_idx,
                }
            except Exception as e:
                print(f"Warning: Failed to process peaks for azimuthal index {azi_idx}: {e}")
                growth_results[azi_idx] = None

        return {
            "growth_results": growth_results,
            "azimuthal_indices": azimuthal_indices,
            "alt_range": alt_range,
        }

    def get_density_growth_curves(
        self,
        growth_data: Dict[str, Any],
        azimuthal_idx: int,
        n_last_peaks: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract density growth curves for a specific azimuthal index.

        Args:
            growth_data: Output from compute_density_growth
            azimuthal_idx: Azimuthal index to extract
            n_last_peaks: Number of last peaks to average

        Returns:
            Tuple of (altitudes, mean_growth, lower_ci, upper_ci)
            If bootstrap not available, lower_ci and upper_ci are None
        """
        if azimuthal_idx not in growth_data["growth_results"]:
            raise ValueError(f"No growth data for azimuthal index {azimuthal_idx}")

        results = growth_data["growth_results"][azimuthal_idx]
        if results is None:
            raise ValueError(f"Growth computation failed for azimuthal index {azimuthal_idx}")

        growth_peaks = results["growth_peaks"]
        radial_distances = results["radial_distances"]
        cell_min_idx = results["cell_min_idx"]
        cell_max_idx = results["cell_max_idx"]
        altitudes = radial_distances[cell_min_idx : cell_max_idx + 1]

        if n_last_peaks < 0:
            raise ValueError("n_last_peaks must be >= 0")

        # Select last N peaks (0 means use all peaks)
        if n_last_peaks == 0:
            last_peaks_data = growth_peaks
        else:
            num_peaks = growth_peaks.shape[1]
            if num_peaks < n_last_peaks:
                last_peaks_data = growth_peaks
            else:
                last_peaks_data = growth_peaks[:, -n_last_peaks:]

        # Compute mean
        mean_growth = np.median(last_peaks_data, axis=1)

        # For now, no bootstrap CI (can be added later)
        lower_ci = None
        upper_ci = None

        return altitudes, mean_growth, lower_ci, upper_ci

    def plot_amplitude_growths(
        self,
        growth_data: Dict[str, Any],
        save_basename: Optional[str] = "amplitude_growths.png",
        show_plot: bool = True,
        n_last_peaks: int = 5,
    ) -> None:
        """
        Plot amplitude growth curves for each oscillating azimuthal index.

        Args:
            growth_data: Output from compute_density_growth method
            save_path: Optional path to save the figure
            show_plot: Whether to display the plot
            n_last_peaks: Number of last peaks to average for growth curve
        """
        if self.params is None:
            raise ValueError("Parameters not loaded. Call load_parameters() first.")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Get azimuthal indices from growth data
        azimuthal_indices = growth_data["azimuthal_indices"]
        colors = plt.cm.viridis(np.linspace(0, 1, len(azimuthal_indices)))

        # Plot each azimuthal index
        valid_growths = []
        for i, azi_idx in enumerate(azimuthal_indices):
            try:
                altitudes, mean_growth, lower_ci, upper_ci = self.get_density_growth_curves(
                    growth_data, azi_idx, n_last_peaks=n_last_peaks,
                )

                # Plot the growth curve
                ax.plot(
                    altitudes,
                    mean_growth,
                    color=colors[i],
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    label=f'Azimuthal Index {azi_idx}'
                )

                # Add confidence intervals if available
                if lower_ci is not None and upper_ci is not None:
                    ax.fill_between(
                        altitudes,
                        lower_ci,
                        upper_ci,
                        color=colors[i],
                        alpha=0.2
                    )

                valid_growths.append(mean_growth)

            except ValueError as e:
                print(f"Warning: Could not plot for azimuthal index {azi_idx}: {e}")
                continue

        # Add reference data if available
        try:
            if TIAN_GROW is not None and TIAN_ALTITUDES is not None and len(TIAN_GROW) > 0 and len(TIAN_ALTITUDES) > 0:
                ax.plot(
                    TIAN_ALTITUDES,
                    TIAN_GROW,
                    'k--',
                    linewidth=2,
                    label='Tian et al. (2019)'
                )
            if SPHERICAL_1D_GROW is not None and SPHERICAL_1D_ALTITUDES is not None and len(SPHERICAL_1D_GROW) > 0 and len(SPHERICAL_1D_ALTITUDES) > 0:
                ax.plot(
                    SPHERICAL_1D_ALTITUDES,
                    SPHERICAL_1D_GROW,
                    'k:',
                    linewidth=2,
                    label='Spherical 1D'
                )
        except NameError:
            pass  # Reference data not available

        # Set labels and title
        ax.set_xlabel('Altitude (km)')
        ax.set_ylabel('Density Amplitude Growth')
        ax.set_title('Density Amplitude Growth vs Altitude\n' +
                    f'Wave Frequency: {self.params.omega_freq:.4f} rad/s, ' +
                    f'Period: {self.params.wave_period:.1f} s')

        # Add grid and legend
        ax.grid(True, alpha=0.3)
        # ax.legend(loc='best', fontsize=10)

        # Set y-axis to log scale if appropriate
        if valid_growths and np.all(np.array(valid_growths) > 0):
            ax.set_yscale('log')

        ax.set_yticks([1, 2, 3, 4, 5, 6])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # remove any minor ticks
        ax.yaxis.set_minor_locator(plt.NullLocator())

        plt.tight_layout()

        # Save if requested (save to root data dir + basename)
        save_path = os.path.join(self.data_dir, "export", save_basename)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        # Show plot if requested
        if show_plot:
            return ax
        else:
            plt.close()

    def plot_amplitude_growth_heatmap(
        self,
        growth_data: Dict[str, Any],
        save_basename: Optional[str] = "amplitude_growth_heatmap.png",
        show_plot: bool = True,
    ) -> None:
        """
        Plot a heatmap of density amplitude growths using pcolormesh.

        Args:
            growth_data: Output from compute_density_growth method
            save_path: Optional path to save the figure
            show_plot: Whether to display the plot
        """
        if self.params is None:
            raise ValueError("Parameters not loaded. Call load_parameters() first.")

        azimuthal_indices = growth_data["azimuthal_indices"]

        # Get altitudes from the first valid azimuthal index
        altitudes = None
        for azi_idx in azimuthal_indices:
            try:
                alt, _, _, _ = self.get_density_growth_curves(growth_data, azi_idx)
                altitudes = alt
                break
            except ValueError:
                continue
        if altitudes is None:
            raise ValueError("No valid growth data found for any azimuthal index.")

        # Build growth matrix with shape (n_altitudes, n_azimuths)
        n_alt = len(altitudes)
        n_azi = len(azimuthal_indices)
        growth_matrix = np.full((n_alt, n_azi), np.nan)
        for j, azi_idx in enumerate(azimuthal_indices):
            try:
                _, mean_growth, _, _ = self.get_density_growth_curves(growth_data, azi_idx)
                growth_matrix[:, j] = mean_growth
            except ValueError:
                continue

        # Mask invalid values
        C = np.ma.masked_invalid(growth_matrix)
        vmin = np.nanmin(growth_matrix)
        vmax = np.nanmax(growth_matrix)

        # Find cell indices for analysis regions
        cell_min_idx = np.where(self.params.radial_cells > growth_data["alt_range"][0])[0][0]
        cell_max_idx = np.where(self.params.radial_cells > growth_data["alt_range"][1])[0][0]

        # Add contour lines at specified levels
        azi_edges = self.params.azi_edges[azimuthal_indices[0]:azimuthal_indices[-1]+2]
        radial_edges = self.params.radial_cells[cell_min_idx:cell_max_idx+2]

        azi_mids = self.params.mid_azi[azimuthal_indices]
        radial_mids = self.params.radial_cells[cell_min_idx:cell_max_idx+1]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        pcm = ax.pcolormesh(
            azi_edges,  # X edges
            radial_edges,  # Y edges
            C,  # C shape (n_alt, n_azi)
            cmap='viridis',
            shading='auto',
            vmin=vmin,
            vmax=vmax,
        )
        contour_levels = [2, 3, 4, 5]
        cs = ax.contour(azi_mids, radial_mids, C, levels=contour_levels, colors='white', linewidths=1.5, alpha=0.8)
        ax.clabel(cs, inline=True, fontsize=10, fmt='%d')  # Label the contours

        ax.set_xlabel('Azimuthal Position (degrees)')
        ax.set_ylabel('Altitude (km)')
        ax.set_title('Density Amplitude Growth Heatmap\n' +
                     f'Wave Frequency: {self.params.omega_freq:.4f} rad/s, ' +
                     f'Period: {self.params.wave_period:.1f} s')

        fig.colorbar(pcm, ax=ax, label='Density Amplitude Growth')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save if requested (save to root data dir + basename)
        save_path = os.path.join(self.data_dir, "export", save_basename)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close()

    # Additional optimized methods can be added here

    def validate_data_consistency(self) -> bool:
        """
        Validate consistency of loaded data.

        Returns:
            True if data is consistent, False otherwise
        """
        # Implementation placeholder
        raise NotImplementedError("validate_data_consistency method not yet implemented")

    def optimize_memory_usage(self) -> None:
        """Optimize memory usage by cleaning up unnecessary data."""
        # Implementation placeholder
        raise NotImplementedError("optimize_memory_usage method not yet implemented")

    def compute_temperature_heating_rate(
        self,
        alt_range: Tuple[float, float] = (150, 320),
        azimuthal_indices: Optional[List[int]] = None,
        window_sizes: Optional[np.ndarray] = None,
        use_dynamic_peak_detection: bool = True,
        H: float = 10.0,
        constant_background_temp=270, # Kelvin
    ) -> Dict[str, Any]:
        """
        Compute temperature heating rate slopes for specified azimuthal indices.

        Args:
            alt_range: Tuple of (min_altitude, max_altitude) in km
            azimuthal_indices: List of azimuthal indices to compute heating rate for.
                               If None, uses oscillating indices.
            window_sizes: Optional array of window sizes in seconds
            n_last_peaks: Number of last peaks to average
            use_dynamic_peak_detection: Whether to use dynamic peak parameter adjustment
            H: Scale height in km for normalizing the slope

        Returns:
            Dictionary containing heating rate results and metadata

        Raises:
            ValueError: If parameters or data not loaded
        """
        if self.params is None:
            raise ValueError("Parameters not loaded. Call load_parameters() first.")
        if self.temperature_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Determine azimuthal indices to use
        if azimuthal_indices is None:
            try:
                azimuthal_indices = self.compute_osc_cols()
            except ValueError:
                # If no oscillating columns, use all if n_azi_cell == 1, else raise
                if self.params.n_azi_cell == 1:
                    azimuthal_indices = [0]
                else:
                    raise ValueError("No oscillating azimuthal indices found. Please specify azimuthal_indices.")
        else:
            azimuthal_indices = np.array(azimuthal_indices)

        if constant_background_temp:
            bg_temperature_profile = np.full(self.temperature_data.shape[-2], constant_background_temp)
        else:
            # Get background profiles using standard method
            background = self.get_background_profiles(window_sizes)
            bg_temperature_profile = background["temperature"]

        # Compute temperature deviations
        temperature_deviation = (
            self.temperature_data - bg_temperature_profile[None, :, None]
        )

        # Find cell indices for analysis regions
        cell_min_idx = np.where(self.params.radial_cells > alt_range[0])[0][0]
        cell_max_idx = np.where(self.params.radial_cells > alt_range[1])[0][0]

        # Process peaks for each azimuthal index
        heating_rate_results = {}
        for azi_idx in azimuthal_indices:
            if azi_idx >= self.temperature_data.shape[-1]:
                continue  # Skip invalid indices

            # Process peaks
            base_kwargs = {
                "prominence": 1e-1,
                "distance": 12,
                "sampling_rate": 1 / np.mean(np.diff(self.params.time)),
                "cutoff_frequency": self.params.omega_freq * 1.2,
                "filter_order": 2,
                "use_dynamic": use_dynamic_peak_detection,
            }

            try:
                temp_peaks, temp_last_cycle = process_temperature_deviation_peaks(
                    temperature_deviation[..., azi_idx],
                    (self.params.start_step_wave * self.params.timestep <= self.params.time)
                    & (
                        self.params.time
                        <= self.params.final_step_wave * self.params.timestep
                        + 700  # PADDING
                    ),
                    self.params.radial_cells,
                    cell_min_idx,
                    cell_max_idx,
                    self.params.num_periods,
                    **base_kwargs,
                )

                # Compute heating rate slope for this azimuthal index
                slope, slope_H = compute_heating_rate_slope(
                    temp_last_cycle,
                    self.params.radial_cells,
                    cell_min_idx,
                    cell_max_idx,
                    H,
                    visualize=False,  # We'll handle visualization separately
                    save_path=None,
                    export_data_path=None,
                )

                # Store results for this azimuthal index
                heating_rate_results[azi_idx] = {
                    "temp_peaks": temp_peaks,
                    "temp_last_cycle": temp_last_cycle,
                    "slope": slope,
                    "slope_H": slope_H,
                    "radial_distances": self.params.radial_cells,
                    "cell_min_idx": cell_min_idx,
                    "cell_max_idx": cell_max_idx,
                }
            except Exception as e:
                print(f"Warning: Failed to process heating rate for azimuthal index {azi_idx}: {e}")
                heating_rate_results[azi_idx] = None

        return {
            "heating_rate_results": heating_rate_results,
            "azimuthal_indices": azimuthal_indices,
            "alt_range": alt_range,
            "H": H,
        }

if __name__ == "__main__":
    # Example usage
    processor = OptimizedDSMCDataProcessor("I:\\ArchiveSims\\Results_20250904_113551.097")
    processor.load_parameters()
    processor.load_data()
    growth_data = processor.compute_density_growth()
    altitudes, mean_growth, lower_ci, upper_ci = processor.get_density_growth_curves(growth_data, azimuthal_idx=growth_data["azimuthal_indices"][0])
    processor.plot_amplitude_growths(growth_data, save_basename="amplitude_growths.png")
    processor.plot_amplitude_growth_heatmap(growth_data, save_basename="amplitude_growth_heatmap.png")

    high_res_processor = OptimizedDSMCDataProcessor("I:\\ArchiveSims\\Results_20240722_041906.318")
    high_res_processor.load_parameters()
    high_res_processor.load_data()
    high_res_growth_data = high_res_processor.compute_density_growth()
    high_res_processor.plot_amplitude_growths(high_res_growth_data, save_basename="amplitude_growths.png")
    high_res_processor.plot_amplitude_growth_heatmap(high_res_growth_data, save_basename="amplitude_growth_heatmap.png")

    osc_cols = high_res_processor.compute_osc_cols()
    # Get two middle oscillating columns
    mid_idx = len(osc_cols) // 2
    selected_cols = osc_cols[max(0, mid_idx - 1) : min(len(osc_cols), mid_idx + 1)]
    print(f"Selected oscillating columns: {selected_cols}")

    # Select the growth curves of those two columns, and average them
    combined_growth = {}
    for col in selected_cols:
        try:
            altitudes, mean_growth, lower_ci, upper_ci = high_res_processor.get_density_growth_curves(high_res_growth_data, col)
            combined_growth[col] = mean_growth
        except ValueError as e:
            print(f"Warning: Could not get growth curve for azimuthal index {col}: {e}")
            continue

    # Average the growth curves
    if combined_growth:
        avg_growth = np.mean(list(combined_growth.values()), axis=0)

    # Plot the averaged growth curve
    plt.figure(figsize=(10, 6))
    plt.plot(altitudes, avg_growth, 'b-', label='Averaged Growth Curve')
    if TIAN_GROW is not None and TIAN_ALTITUDES is not None and len(TIAN_GROW) > 0 and len(TIAN_ALTITUDES) > 0:
        plt.plot(TIAN_ALTITUDES, TIAN_GROW, 'r--', label='Tian et al. (2019)')
    if SPHERICAL_1D_GROW is not None and SPHERICAL_1D_ALTITUDES is not None and len(SPHERICAL_1D_GROW) > 0 and len(SPHERICAL_1D_ALTITUDES) > 0:
        plt.plot(SPHERICAL_1D_ALTITUDES, SPHERICAL_1D_GROW, 'g:', label='Spherical 1D')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Density Amplitude Growth')
    plt.title('Averaged Density Amplitude Growth vs Altitude')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.ylim(1, 6)
    plt.yticks([1, 2, 3, 4, 5, 6])
    plt.gca().get_yaxis().set_major_formatter(plt.ScalarFormatter())
    plt.gca().yaxis.set_minor_locator(plt.NullLocator())
    plt.tight_layout()
    # print(f"Averaged growth figure saved to: {save_path}")
    plt.show()

    # NOW SELECT THE EDGES (OUTERMOST) OSCILLATING COLUMNS, and average them, and plot them
    selected_cols = [osc_cols[0], osc_cols[-1]]
    print(f"Selected outer oscillating columns: {selected_cols}")

    # Select the growth curves of those two columns, and average them
    combined_growth = {}
    for col in selected_cols:
        try:
            altitudes, mean_growth, lower_ci, upper_ci = high_res_processor.get_density_growth_curves(high_res_growth_data, col)
            combined_growth[col] = mean_growth
        except ValueError as e:
            print(f"Warning: Could not get growth curve for azimuthal index {col}: {e}")
            continue

    # Average the growth curves
    if combined_growth:
        avg_growth = np.mean(list(combined_growth.values()), axis=0)

    # Plot the averaged growth curve
    plt.figure(figsize=(10, 6))
    plt.plot(altitudes, avg_growth, 'b-', label='Averaged Growth Curve (Outer Columns)')
    if TIAN_GROW is not None and TIAN_ALTITUDES is not None and len(TIAN_GROW) > 0 and len(TIAN_ALTITUDES) > 0:
        plt.plot(TIAN_ALTITUDES, TIAN_GROW, 'r--', label='Tian et al. (2019)')
    if SPHERICAL_1D_GROW is not None and SPHERICAL_1D_ALTITUDES is not None and len(SPHERICAL_1D_GROW) > 0 and len(SPHERICAL_1D_ALTITUDES) > 0:
        plt.plot(SPHERICAL_1D_ALTITUDES, SPHERICAL_1D_GROW, 'g:', label='Spherical 1D')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Density Amplitude Growth')
    plt.title('Averaged Density Amplitude Growth vs Altitude (Outer Columns)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.ylim(1, 6)
    plt.yticks([1, 2, 3, 4, 5, 6])
    plt.gca().get_yaxis().set_major_formatter(plt.ScalarFormatter())
    plt.gca().yaxis.set_minor_locator(plt.NullLocator())
    plt.tight_layout()