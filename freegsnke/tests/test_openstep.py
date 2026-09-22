import pickle
from pathlib import Path

import numpy as np
import pytest
import scipy.interpolate

from freegsnke import GSstaticsolver, build_machine, equilibrium_update
from freegsnke.jtor_update import GeneralPprimeFFprime

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENSTEP_CONFIG_DIR = REPO_ROOT / "machine_configs" / "OpenSTEP"

EXPECTED_CIRCUITS = {"p3", "p4", "p5", "p6", "p9", "s1", "s2"}


def test_openstep_machine_build():
    """Verify that the OpenSTEP tokamak builds cleanly from pickled configuration."""
    tokamak = build_machine.tokamak(
        active_coils_path=str(OPENSTEP_CONFIG_DIR / "OpenSTEP_active_coils.pickle"),
        passive_coils_path=str(OPENSTEP_CONFIG_DIR / "OpenSTEP_passive_coils.pickle"),
        limiter_path=str(OPENSTEP_CONFIG_DIR / "OpenSTEP_limiter.pickle"),
        wall_path=str(OPENSTEP_CONFIG_DIR / "OpenSTEP_wall.pickle"),
    )

    circuits = set(tokamak.getCurrents().keys())
    assert (
        circuits == EXPECTED_CIRCUITS
    ), f"Expected circuits {EXPECTED_CIRCUITS}, got {circuits}"
    assert (
        len(tokamak.wall.R) == 514
    ), f"Expected 514 wall points, got {len(tokamak.wall.R)}"
    assert (
        len(tokamak.limiter.R) == 514
    ), f"Expected 514 limiter points, got {len(tokamak.limiter.R)}"


def test_openstep_shipped_currents():
    """Verify that shipped OpenSTEP coil currents match expected circuits."""
    with (OPENSTEP_CONFIG_DIR / "OpenSTEP_coil_currents.pickle").open("rb") as f:
        currents = pickle.load(f)

    assert set(currents.keys()) == EXPECTED_CIRCUITS
    assert np.isclose(currents["s1"], 5.1e6, atol=1.0)
    assert np.isclose(currents["p3"], -4790870.39, atol=1.0)


def test_openstep_forward_static_solve():
    """Verify forward static solve converges to the OpenSTEP reference equilibrium."""
    tokamak = build_machine.tokamak(
        active_coils_path=str(OPENSTEP_CONFIG_DIR / "OpenSTEP_active_coils.pickle"),
        passive_coils_path=str(OPENSTEP_CONFIG_DIR / "OpenSTEP_passive_coils.pickle"),
        limiter_path=str(OPENSTEP_CONFIG_DIR / "OpenSTEP_limiter.pickle"),
        wall_path=str(OPENSTEP_CONFIG_DIR / "OpenSTEP_wall.pickle"),
    )

    with (OPENSTEP_CONFIG_DIR / "OpenSTEP_coil_currents.pickle").open("rb") as f:
        currents = pickle.load(f)

    for coil, current in currents.items():
        tokamak.set_coil_current(coil_label=coil, current_value=current)

    with (OPENSTEP_CONFIG_DIR / "OpenSTEP_plasma_psi.pickle").open("rb") as f:
        plasma_ref = pickle.load(f)

    with (OPENSTEP_CONFIG_DIR / "OpenSTEP_profiles.pickle").open("rb") as f:
        prof_data = pickle.load(f)

    nx, ny = 65, 129
    r_grid = np.linspace(0.5, 9.0, nx)
    z_grid = np.linspace(-10.0, 10.0, ny)

    interp_plasma = scipy.interpolate.RectBivariateSpline(
        plasma_ref["R"], plasma_ref["Z"], plasma_ref["plasma_psi"]
    )
    psi_init = interp_plasma(r_grid, z_grid)

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=0.5,
        Rmax=9.0,
        Zmin=-10.0,
        Zmax=10.0,
        nx=nx,
        ny=ny,
        psi=psi_init,
    )

    profiles = GeneralPprimeFFprime(
        eq=eq,
        Ip=prof_data["Ip"],
        fvac=prof_data["fvac"],
        psi_n=prof_data["psi_n"],
        pprime_data=prof_data["pprime"],
        ffprime_data=prof_data["ffprime"],
        Raxis=prof_data["Raxis"],
        Ip_logic=True,
        interpolator="cubic_spline",
    )

    solver = GSstaticsolver.NKGSsolver(eq, gs_operator_order=4)
    solver.solve(
        eq=eq,
        profiles=profiles,
        constrain=None,
        target_relative_tolerance=1e-8,
        verbose=False,
    )

    # Check total psi against OpenSTEP reference
    interp_tot = scipy.interpolate.RectBivariateSpline(
        plasma_ref["R"], plasma_ref["Z"], plasma_ref["total_psi"]
    )
    psi_tot_ref = interp_tot(r_grid, z_grid)
    psi_tot_solved = eq.psi()

    diff = np.abs(psi_tot_solved - psi_tot_ref)
    max_ref = np.max(np.abs(psi_tot_ref))
    rel_error = np.max(diff) / max_ref

    assert np.isclose(eq.plasmaCurrent(), prof_data["Ip"], rtol=1e-3)
    assert np.isclose(eq.psi_axis, plasma_ref["psi_axis"], atol=0.05)
    assert np.isclose(eq.psi_bndry, plasma_ref["psi_bndry"], atol=0.02)
    assert rel_error < 0.005  # Within 0.5% across entire 2D grid
    assert eq.plasmaVolume() > 600.0  # Diverted large tokamak plasma volume (~712 m^3)
