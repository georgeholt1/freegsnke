"""
Unit tests for SurrogateInitialGuess and its integration with GSstaticsolver.
"""

from pathlib import Path

import numpy as np
import pytest

from freegsnke import GSstaticsolver, SurrogateInitialGuess, build_machine, equilibrium_update
from freegsnke.jtor_update import ConstrainPaxisIp


@pytest.fixture(scope="module")
def mastu_setup():
    """Create a MAST-U equilibrium and profiles for testing."""
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "machine_configs" / "MAST-U"
    tokamak = build_machine.tokamak(
        active_coils_path=str(config_dir / "MAST-U_like_active_coils.pickle"),
        passive_coils_path=str(config_dir / "MAST-U_like_passive_coils.pickle"),
        limiter_path=str(config_dir / "MAST-U_like_limiter.pickle"),
        wall_path=str(config_dir / "MAST-U_like_wall.pickle"),
    )

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-2.2,
        Zmax=2.2,
        nx=65,
        ny=129,
    )
    # Set nominal active coil currents
    active_currents = {
        "Solenoid": 5000.0,
        "PX": 3700.0,
        "D1": 6900.0,
        "D2": 4400.0,
        "D3": 2900.0,
        "Dp": -2500.0,
        "D5": 250.0,
        "D6": -315.0,
        "D7": 510.0,
        "P4": -3500.0,
        "P5": -3900.0,
        "P6": 0.0,
    }
    for k, v in active_currents.items():
        eq.tokamak.set_coil_current(k, v)

    profiles = ConstrainPaxisIp(
        eq=eq,
        paxis=8.1e3,
        Ip=6.2e5,
        fvac=0.5,
        alpha_m=1.8,
        alpha_n=1.2,
    )

    eq.tokamak_psi = eq.tokamak.getPsitokamak(vgreen=eq._vgreen)
    solver = GSstaticsolver.NKGSsolver(eq, gs_operator_order=4)
    solver.forward_solve(
        eq,
        profiles,
        target_relative_tolerance=1e-5,
        max_solving_iterations=30,
        suppress=True,
    )
    return eq, profiles, solver


def test_surrogate_model_loading():
    """Test loading the bundled surrogate model and inspecting weight shapes."""
    surrogate = SurrogateInitialGuess()
    assert surrogate.grid_shape == (65, 129)
    assert surrogate.n_modes > 0
    assert len(surrogate.feature_names) == 17
    assert surrogate.pca_components.shape == (surrogate.n_modes, 65 * 129)
    assert surrogate.pca_mean.shape == (65 * 129,)


def test_surrogate_feature_extraction(mastu_setup):
    """Test feature extraction from equilibrium and profile objects."""
    eq, profiles, _ = mastu_setup
    surrogate = SurrogateInitialGuess()
    features = surrogate.extract_features(eq, profiles)

    assert isinstance(features, np.ndarray)
    assert features.shape == (17,)
    assert features[0] == pytest.approx(5000.0)  # Solenoid
    assert features[12] == pytest.approx(6.2e5)  # Ip
    assert features[13] == pytest.approx(8.1e3)  # paxis


def test_surrogate_prediction_shape_and_finite(mastu_setup):
    """Test surrogate 2D flux prediction outputs valid numeric values."""
    eq, profiles, _ = mastu_setup
    surrogate = SurrogateInitialGuess()
    psi_guess = surrogate.predict(eq, profiles)

    assert psi_guess.shape == (65, 129)
    assert np.all(np.isfinite(psi_guess))
    assert np.ptp(psi_guess) > 0.01  # Non-trivial flux variation


def test_surrogate_apply(mastu_setup):
    """Test that surrogate.apply correctly modifies eq.plasma_psi."""
    eq, profiles, _ = mastu_setup
    aux_eq = eq.create_auxiliary_equilibrium()
    surrogate = SurrogateInitialGuess()

    surrogate.apply(aux_eq, profiles)
    assert np.all(np.isfinite(aux_eq.plasma_psi))
    assert not aux_eq.solved


def test_solver_forward_solve_with_surrogate(mastu_setup):
    """Test that NKGSsolver.forward_solve runs and converges when surrogate=True."""
    eq, profiles, solver = mastu_setup
    test_eq = eq.create_auxiliary_equilibrium()

    solver.forward_solve(
        test_eq,
        profiles,
        target_relative_tolerance=1e-8,
        max_solving_iterations=60,
        suppress=True,
        surrogate=True,
    )
    assert test_eq.solved
    assert solver.relative_change <= 1e-8


def test_solver_with_custom_surrogate_instance(mastu_setup):
    """Test providing an instantiated SurrogateInitialGuess directly to forward_solve."""
    eq, profiles, solver = mastu_setup
    test_eq = eq.create_auxiliary_equilibrium()
    surr = SurrogateInitialGuess()

    solver.forward_solve(
        test_eq,
        profiles,
        target_relative_tolerance=1e-8,
        max_solving_iterations=60,
        suppress=True,
        surrogate=surr,
    )
    assert test_eq.solved
    assert solver.relative_change <= 1e-8
