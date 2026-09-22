"""Unit tests for FreeGSNKE serialization and backward-compatibility."""

import io
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
from freeqdsk import geqdsk

from freegsnke import build_machine, equilibrium_update, mastu_tools
from freegsnke.serialization import (
    FreeGSNKEJSONEncoder,
    load_json,
    load_machine_bundle,
    load_machine_file,
    save_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MACHINE_EXAMPLE_DIR = REPO_ROOT / "machine_configs" / "example"


def test_freegsnke_json_encoder_types(tmp_path):
    """Test that FreeGSNKEJSONEncoder handles NumPy scalars and arrays."""
    data = {
        "int_scalar": np.int64(42),
        "float_scalar": np.float64(3.14159),
        "bool_scalar": np.bool_(True),
        "complex_scalar": complex(1.0, 2.0),
        "array_1d": np.array([1.0, 2.0, 3.0]),
        "array_2d": np.array([[1, 2], [3, 4]]),
        "nested": {"val": np.float32(2.5)},
    }
    file_path = tmp_path / "test_encoder.json"
    save_json(data, file_path)

    loaded = load_json(file_path)
    assert loaded["int_scalar"] == 42
    assert abs(loaded["float_scalar"] - 3.14159) < 1e-5
    assert loaded["bool_scalar"] is True
    assert loaded["complex_scalar"] == {"real": 1.0, "imag": 2.0}
    assert loaded["array_1d"] == [1.0, 2.0, 3.0]
    assert loaded["array_2d"] == [[1, 2], [3, 4]]
    assert abs(loaded["nested"]["val"] - 2.5) < 1e-5


def test_load_machine_file_json(tmp_path):
    """Test loading machine component directly from a JSON file."""
    content = {"coilA": {"R": 1.5, "Z": 0.0}}
    json_path = tmp_path / "active_coils.json"
    save_json(content, json_path)

    loaded, source_type = load_machine_file(json_path, component_name="active_coils")
    assert loaded == content
    assert source_type == "json"


def test_load_machine_file_bundle(tmp_path):
    """Test loading a component from a unified machine bundle."""
    bundle = {
        "active_coils": {"coil1": {"R": 1.0}},
        "passive_coils": [{"name": "pass1"}],
        "limiter": [{"R": 1.0, "Z": 1.0}],
    }
    bundle_path = tmp_path / "machine.json"
    save_json(bundle, bundle_path)

    # With component_name specified
    loaded, source_type = load_machine_file(bundle_path, component_name="active_coils")
    assert loaded == {"coil1": {"R": 1.0}}
    assert source_type == "json"

    # Without component_name specified
    loaded_all, source_type = load_machine_file(bundle_path)
    assert loaded_all == bundle
    assert source_type == "json"


def test_load_machine_file_pickle_warning(tmp_path):
    """Test that loading a legacy pickle file issues a DeprecationWarning."""
    content = {"coil1": {"R": 2.0}}
    pk_path = tmp_path / "active_coils.pickle"
    with open(pk_path, "wb") as f:
        pickle.dump(content, f)

    with pytest.deprecated_call(match="Loading machine configuration"):
        loaded, source_type = load_machine_file(pk_path)

    assert loaded == content
    assert source_type == "pickle"


def test_load_machine_file_errors(tmp_path):
    """Test error handling when file is missing or invalid."""
    with pytest.raises(FileNotFoundError):
        load_machine_file(tmp_path / "nonexistent.json")

    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("hello")
    with pytest.raises(ValueError, match="Unrecognized machine file format"):
        load_machine_file(bad_file)


def test_load_machine_bundle_directory(tmp_path):
    """Test loading machine bundle from a directory with individual JSON files."""
    for comp in ["active_coils", "passive_coils", "limiter", "wall", "magnetic_probes"]:
        save_json({f"key_{comp}": 123}, tmp_path / f"{comp}.json")

    bundle = load_machine_bundle(tmp_path)
    assert "active_coils" in bundle
    assert "passive_coils" in bundle
    assert bundle["active_coils"] == {"key_active_coils": 123}


def test_tokamak_from_machine_json():
    """Test building a tokamak instance from unified machine.json bundle."""
    bundle_path = MACHINE_EXAMPLE_DIR / "machine.json"
    tokamak = build_machine.tokamak(machine_path=str(bundle_path))
    assert tokamak.n_active_coils > 0
    assert tokamak.n_passive_coils > 0
    assert tokamak.limiter is not None


def test_tokamak_from_machine_directory():
    """Test building a tokamak instance from directory path."""
    tokamak = build_machine.tokamak(machine_path=str(MACHINE_EXAMPLE_DIR))
    assert tokamak.n_active_coils > 0
    assert tokamak.n_passive_coils > 0
    assert tokamak.limiter is not None


def test_initialize_from_equilibrium_json(tmp_path):
    """Test initialize_from_equilibrium using JSON input."""
    bundle_path = MACHINE_EXAMPLE_DIR / "machine.json"
    tok = build_machine.tokamak(machine_path=str(bundle_path))
    eq = equilibrium_update.Equilibrium(
        tokamak=tok, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=17, ny=33
    )

    eq_data = {
        "Rmin": 0.1,
        "Rmax": 2.0,
        "Zmin": -1.0,
        "Zmax": 1.0,
        "psi_plasma": np.ones((17, 33)) * 0.42,
    }
    eq_json = tmp_path / "initial_eq.json"
    save_json(eq_data, eq_json)

    eq.equilibrium_path = str(eq_json)
    eq.initialize_from_equilibrium()

    np.testing.assert_allclose(eq.plasma_psi, 0.42, atol=1e-5)


def test_initialize_from_equilibrium_pickle_deprecation(tmp_path):
    """Test initialize_from_equilibrium using legacy pickle input raises DeprecationWarning."""
    bundle_path = MACHINE_EXAMPLE_DIR / "machine.json"
    tok = build_machine.tokamak(machine_path=str(bundle_path))
    eq = equilibrium_update.Equilibrium(
        tokamak=tok, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=17, ny=33
    )

    eq_data = {
        "Rmin": 0.1,
        "Rmax": 2.0,
        "Zmin": -1.0,
        "Zmax": 1.0,
        "psi_plasma": np.ones((17, 33)) * 0.15,
    }
    eq_pickle = tmp_path / "initial_eq.pickle"
    with open(eq_pickle, "wb") as f:
        pickle.dump(eq_data, f)

    eq.equilibrium_path = str(eq_pickle)
    with pytest.deprecated_call(match="Loading initial equilibrium from pickle"):
        eq.initialize_from_equilibrium()

    np.testing.assert_allclose(eq.plasma_psi, 0.15, atol=1e-5)


def test_initialize_from_equilibrium_geqdsk(tmp_path):
    """Test initialize_from_equilibrium using GEQDSK format input."""
    bundle_path = MACHINE_EXAMPLE_DIR / "machine.json"
    tok = build_machine.tokamak(machine_path=str(bundle_path))
    eq = equilibrium_update.Equilibrium(
        tokamak=tok, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=17, ny=33
    )

    # Create dummy GEQDSK structure
    nx, ny = 17, 33
    rleft = 0.1
    rdim = 1.9
    zmid = 0.0
    zdim = 2.0
    zmin = -1.0

    total_psi = np.full((nx, ny), 0.5)

    gdata = {
        "nx": nx,
        "ny": ny,
        "rdim": rdim,
        "zdim": zdim,
        "rcentr": 1.0,
        "rleft": rleft,
        "zmid": zmid,
        "rmagx": 1.0,
        "zmagx": 0.0,
        "simagx": 0.0,
        "sibdry": 0.5,
        "bcentr": 0.5,
        "cpasma": 1e5,
        "fpol": np.ones(nx),
        "pres": np.zeros(nx),
        "ffprime": np.zeros(nx),
        "pprime": np.zeros(nx),
        "psi": total_psi,
        "qpsi": np.ones(nx),
        "nbbbs": 0,
        "limitr": 0,
        "zmin": zmin,
    }

    gpath = tmp_path / "test.geqdsk"
    with open(gpath, "w", encoding="utf-8") as f:
        geqdsk.write(gdata, f)

    eq.equilibrium_path = str(gpath)
    eq.initialize_from_equilibrium()

    # Total flux was 0.5, so plasma_psi = 0.5 - coil_psi
    coil_psi = eq.tokamak.getPsitokamak(eq._vgreen)
    np.testing.assert_allclose(eq.plasma_psi, 0.5 - coil_psi, atol=1e-4)


def test_build_mastu_geometry_pickle_files_warning():
    """Test that build_MASTU_geometry_pickle_files issues DeprecationWarning."""
    with pytest.deprecated_call(match="build_MASTU_geometry_pickle_files is deprecated"):
        # We catch the exception because pyuda client won't connect without credentials/data
        try:
            mastu_tools.build_MASTU_geometry_pickle_files(save_path="/invalid/path")
        except Exception:
            pass
