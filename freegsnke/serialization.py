"""
Serialization utilities for FreeGSNKE.

Provides secure and portable JSON serialization and deserialization for machine
configurations, diagnostic configurations, and general simulation parameters,
along with backward-compatible loading of legacy pickle files.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.

This file is part of FreeGSNKE.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE. If not, see <http://www.gnu.org/licenses/>.
"""

import json
import os
import pathlib
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

LEGACY_PICKLE_EXTENSIONS = (".pickle", ".pk", ".pkl")


class FreeGSNKEJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder supporting NumPy types and arrays.

    Converts NumPy arrays to nested Python lists and NumPy scalar numbers to
    native Python ints, floats, or bools.
    """

    def default(self, obj: Any) -> Any:
        """
        Convert numpy arrays and scalar types to standard Python equivalents.

        Parameters
        ----------
        obj : Any
            Object to serialize.

        Returns
        -------
        Any
            JSON-serializable representation of `obj`.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.complexfloating, complex)):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        return super().default(obj)


def save_json(
    obj: Any,
    path_or_buf: Union[str, pathlib.Path, Any],
    indent: int = 2,
    **kwargs: Any,
) -> None:
    """
    Save an object to JSON format using :class:`FreeGSNKEJSONEncoder`.

    Parameters
    ----------
    obj : Any
        Data object to serialize.
    path_or_buf : str, pathlib.Path, or file-like object
        Target file path or open text file buffer.
    indent : int, optional
        Indentation spaces for formatted output. Defaults to 2.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`json.dump`.
    """
    if isinstance(path_or_buf, (str, pathlib.Path)):
        filepath = pathlib.Path(path_or_buf)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(obj, f, cls=FreeGSNKEJSONEncoder, indent=indent, **kwargs)
            f.write("\n")
    else:
        json.dump(obj, path_or_buf, cls=FreeGSNKEJSONEncoder, indent=indent, **kwargs)


def load_json(path_or_buf: Union[str, pathlib.Path, Any], **kwargs: Any) -> Any:
    """
    Load an object from a JSON file or buffer.

    Parameters
    ----------
    path_or_buf : str, pathlib.Path, or file-like object
        File path or open text buffer from which to read JSON.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`json.load`.

    Returns
    -------
    Any
        Deserialized Python object.
    """
    if isinstance(path_or_buf, (str, pathlib.Path)):
        with open(path_or_buf, "r", encoding="utf-8") as f:
            return json.load(f, **kwargs)
    return json.load(path_or_buf, **kwargs)


def _convert_probes_arrays(probe_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert probe positions and orientation vectors from lists to NumPy arrays.

    Parameters
    ----------
    probe_data : dict
        Magnetic probe dictionary containing 'flux_loops' and 'pickups'.

    Returns
    -------
    dict
        Magnetic probe dictionary with NumPy arrays for coordinates.
    """
    if not isinstance(probe_data, dict):
        return probe_data

    if "flux_loops" in probe_data and isinstance(probe_data["flux_loops"], list):
        for fl in probe_data["flux_loops"]:
            if isinstance(fl, dict) and "position" in fl and isinstance(fl["position"], list):
                fl["position"] = np.array(fl["position"], dtype=float)

    if "pickups" in probe_data and isinstance(probe_data["pickups"], list):
        for pu in probe_data["pickups"]:
            if isinstance(pu, dict):
                if "position" in pu and isinstance(pu["position"], list):
                    pu["position"] = np.array(pu["position"], dtype=float)
                if "orientation_vector" in pu and isinstance(pu["orientation_vector"], list):
                    pu["orientation_vector"] = np.array(pu["orientation_vector"], dtype=float)

    return probe_data


def load_machine_file(
    path_or_data: Union[str, pathlib.Path, Dict[str, Any], List[Any]],
    component_name: Optional[str] = None,
) -> Tuple[Any, str]:
    """
    Load machine description data from a file path or return in-memory data.

    Supports JSON format (.json) by default. If a legacy pickle file (.pickle,
    .pk, .pkl) is supplied, it is loaded with a :class:`DeprecationWarning`
    advising migration to JSON.

    Parameters
    ----------
    path_or_data : str, pathlib.Path, dict, or list
        File path to load or direct Python data structure.
    component_name : str, optional
        Component name (e.g. 'active_coils', 'passive_coils', 'limiter',
        'wall', 'magnetic_probes') to extract if loading a unified machine
        dictionary containing top-level component keys.

    Returns
    -------
    data : Any
        Loaded machine description component.
    source_type : str
        Source descriptor: 'json', 'pickle', or 'data'.
    """
    if not isinstance(path_or_data, (str, pathlib.Path)):
        data = path_or_data
        if component_name == "magnetic_probes" and isinstance(data, dict):
            _convert_probes_arrays(data)
        return data, "data"

    path_str = str(path_or_data)
    is_pickle = any(path_str.endswith(ext) for ext in LEGACY_PICKLE_EXTENSIONS)

    if is_pickle:
        warnings.warn(
            f"Loading machine configuration '{path_str}' via pickle is deprecated "
            "and will be removed in a future release. Please migrate to JSON format.",
            DeprecationWarning,
            stacklevel=3,
        )
        with open(path_str, "rb") as f:
            data = pickle.load(f)
        source_type = "pickle"
    else:
        data = load_json(path_str)
        source_type = "json"

    # If loading from a unified machine bundle, extract the component if needed
    if component_name is not None and isinstance(data, dict):
        if component_name in data:
            data = data[component_name]

    if component_name == "magnetic_probes" and isinstance(data, dict):
        _convert_probes_arrays(data)

    return data, source_type


def load_machine_bundle(
    machine_path: Union[str, pathlib.Path],
) -> Dict[str, Any]:
    """
    Load a machine configuration from a unified JSON file or directory.

    If `machine_path` points to a JSON file (e.g. ``machine.json`` or
    ``MAST-U.json``), the file is loaded directly. If `machine_path` points to a
    directory, this function looks for ``machine.json`` first, or individual
    component files (e.g. ``active_coils.json``, ``limiter.json``, etc.).

    Parameters
    ----------
    machine_path : str or pathlib.Path
        Path to a unified JSON file or machine directory.

    Returns
    -------
    dict
        Dictionary containing machine components:
        'active_coils', 'passive_coils', 'limiter', 'wall', 'magnetic_probes'.

    Raises
    ------
    FileNotFoundError
        If `machine_path` does not exist or required machine files are missing.
    ValueError
        If the file content is missing required components.
    """
    path = pathlib.Path(machine_path)
    if not path.exists():
        raise FileNotFoundError(f"Machine configuration path not found: {machine_path}")

    if path.is_file():
        data, _ = load_machine_file(path)
        if isinstance(data, dict) and ("active_coils" in data or "limiter" in data):
            if "magnetic_probes" in data and isinstance(data["magnetic_probes"], dict):
                _convert_probes_arrays(data["magnetic_probes"])
            return data
        raise ValueError(
            f"File '{machine_path}' does not contain recognized machine component keys."
        )

    # If it is a directory:
    # 1. Look for machine.json or <dirname>.json
    candidates = [
        path / "machine.json",
        path / f"{path.name}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return load_machine_bundle(candidate)

    # 2. Look for component files in directory
    def find_file(names: List[str]) -> Optional[pathlib.Path]:
        """Find the first matching file name from a list of candidates."""
        for name in names:
            p = path / name
            if p.is_file():
                return p
        return None

    active_path = find_file(
        [
            f"{path.name}_like_active_coils.json",
            f"{path.name}_active_coils.json",
            "active_coils.json",
        ]
    )
    passive_path = find_file(
        [
            f"{path.name}_like_passive_coils.json",
            f"{path.name}_passive_coils.json",
            "passive_coils.json",
        ]
    )
    limiter_path = find_file(
        [
            f"{path.name}_like_limiter.json",
            f"{path.name}_limiter.json",
            "limiter.json",
        ]
    )
    wall_path = find_file(
        [
            f"{path.name}_like_wall.json",
            f"{path.name}_wall.json",
            "wall.json",
        ]
    )
    probe_path = find_file(
        [
            f"{path.name}_like_magnetic_probes.json",
            f"{path.name}_magnetic_probes.json",
            "magnetic_probes.json",
        ]
    )

    # Fallback to pickle if json not found (with deprecation)
    if active_path is None:
        active_path = find_file(
            [f"{path.name}_active_coils.pickle", f"{path.name}_like_active_coils.pickle", "active_coils.pickle"]
        )
    if passive_path is None:
        passive_path = find_file(
            [f"{path.name}_passive_coils.pickle", f"{path.name}_like_passive_coils.pickle", "passive_coils.pickle"]
        )
    if limiter_path is None:
        limiter_path = find_file(
            [f"{path.name}_limiter.pickle", f"{path.name}_like_limiter.pickle", "limiter.pickle"]
        )
    if wall_path is None:
        wall_path = find_file(
            [f"{path.name}_wall.pickle", f"{path.name}_like_wall.pickle", "wall.pickle"]
        )
    if probe_path is None:
        probe_path = find_file(
            [f"{path.name}_magnetic_probes.pickle", f"{path.name}_like_magnetic_probes.pickle", "magnetic_probes.pickle"]
        )

    bundle: Dict[str, Any] = {}
    if active_path is not None:
        bundle["active_coils"], _ = load_machine_file(active_path, "active_coils")
    if passive_path is not None:
        bundle["passive_coils"], _ = load_machine_file(passive_path, "passive_coils")
    else:
        bundle["passive_coils"] = []
    if limiter_path is not None:
        bundle["limiter"], _ = load_machine_file(limiter_path, "limiter")
    if wall_path is not None:
        bundle["wall"], _ = load_machine_file(wall_path, "wall")
    elif "limiter" in bundle:
        bundle["wall"] = bundle["limiter"]
    if probe_path is not None:
        bundle["magnetic_probes"], _ = load_machine_file(probe_path, "magnetic_probes")
    else:
        bundle["magnetic_probes"] = None

    return bundle
