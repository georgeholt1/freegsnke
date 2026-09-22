import pickle
from pathlib import Path

import numpy as np
import pytest

from freegsnke import GSstaticsolver, build_machine, equilibrium_update
from freegsnke.jtor_update import ConstrainBetapIp

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
    """Verify a forward static solve converges using the OpenSTEP machine and shipped currents."""
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

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=0.5,
        Rmax=9.0,
        Zmin=-10.0,
        Zmax=10.0,
        nx=65,
        ny=129,
    )

    profiles = ConstrainBetapIp(
        eq=eq,
        betap=1.042,
        Ip=22.76e6,
        fvac=11.52,
        alpha_m=1.0,
        alpha_n=1.5,
    )

    solver = GSstaticsolver.NKGSsolver(eq, gs_operator_order=4)
    solver.solve(
        eq=eq,
        profiles=profiles,
        constrain=None,
        target_relative_tolerance=1e-6,
        verbose=False,
    )

    assert np.isclose(eq.plasmaCurrent(), 22.76e6, rtol=1e-3)
    assert eq.poloidalBeta() > 0.0
    assert eq.psi_axis > eq.psi_bndry
