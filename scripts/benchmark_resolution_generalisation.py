#!/usr/bin/env python3
"""
Benchmark resolution generalisation of the surrogate initial guess.

Evaluates Default (Gaussian) vs. Surrogate (POD/PCA + MLP with 2D spline interpolation)
across a spectrum of grid resolutions:
- 33x65 (coarse)
- 49x97 (intermediate-coarse)
- 65x129 (native training resolution)
- 97x193 (intermediate-fine)
- 129x257 (fine)
"""

import argparse
import time
from pathlib import Path
import numpy as np

from freegsnke import GSstaticsolver, SurrogateInitialGuess, build_machine, equilibrium_update
from freegsnke.jtor_update import ConstrainPaxisIp

ACTIVE_COILS = [
    "Solenoid",
    "PX",
    "D1",
    "D2",
    "D3",
    "Dp",
    "D5",
    "D6",
    "D7",
    "P4",
    "P5",
    "P6",
]

RESOLUTIONS = [
    (33, 65),
    (49, 97),
    (65, 129),
    (97, 193),
    (129, 257),
]


def run_resolution_benchmark(
    dataset_path: str = "data/surrogate_dataset_mastu.npz",
    model_path: str = "freegsnke/models/mastu_paxis_ip_surrogate.npz",
    n_cases: int = 15,
    target_tolerance: float = 1e-5,
    max_solving_iterations: int = 50,
):
    repo_root = Path(__file__).resolve().parents[1]
    data = np.load(repo_root / dataset_path, allow_pickle=True)
    X = data["X"]
    test_idx = data["test_idx"][:n_cases]

    # Load machine once
    config_dir = repo_root / "machine_configs" / "MAST-U"
    tokamak = build_machine.tokamak(
        active_coils_path=str(config_dir / "MAST-U_like_active_coils.pickle"),
        passive_coils_path=str(config_dir / "MAST-U_like_passive_coils.pickle"),
        limiter_path=str(config_dir / "MAST-U_like_limiter.pickle"),
        wall_path=str(config_dir / "MAST-U_like_wall.pickle"),
    )

    surrogate = SurrogateInitialGuess(model_path=repo_root / model_path)
    print(f"Loaded surrogate model (native grid: {surrogate.grid_shape}, modes: {surrogate.n_modes})")
    print(f"Testing {len(test_idx)} equilibria across resolutions: {RESOLUTIONS}\n")

    benchmark_summary = {}

    for nx, ny in RESOLUTIONS:
        print(f"============================================================")
        print(f"Evaluating Resolution: {nx} x {ny} ({nx*ny} points)")
        print(f"============================================================")

        # Build base equilibrium for this resolution
        base_eq = equilibrium_update.Equilibrium(
            tokamak=tokamak,
            Rmin=0.1,
            Rmax=2.0,
            Zmin=-2.2,
            Zmax=2.2,
            nx=nx,
            ny=ny,
        )
        base_eq.tokamak_psi = base_eq.tokamak.getPsitokamak(vgreen=base_eq._vgreen)
        solver = GSstaticsolver.NKGSsolver(base_eq, gs_operator_order=4)

        res_results = {
            "default_iters": [],
            "surrogate_iters": [],
            "default_time": [],
            "surrogate_solve_time": [],
            "surrogate_inf_time": [],
            "surrogate_net_time": [],
            "speedup": [],
            "default_success": [],
            "surrogate_success": [],
            "default_init_res": [],
            "surrogate_init_res": [],
        }

        for i, idx in enumerate(test_idx):
            x = X[idx]
            currents = {name: x[j] for j, name in enumerate(ACTIVE_COILS)}
            Ip = x[12]
            paxis = x[13]
            fvac = x[14]
            alpha_m = x[15]
            alpha_n = x[16]

            # 1. Default solve
            eq_def = base_eq.create_auxiliary_equilibrium()
            for name, val in currents.items():
                eq_def.tokamak.set_coil_current(name, val)
            prof_def = ConstrainPaxisIp(
                eq=eq_def, paxis=paxis, Ip=Ip, fvac=fvac, alpha_m=alpha_m, alpha_n=alpha_n
            )
            eq_def.plasma_psi = eq_def.create_psi_plasma_default(adaptive_centre=True)
            eq_def.solved = False
            eq_def.adjust_psi_plasma()

            t0_def = time.perf_counter()
            solver.forward_solve(
                eq_def,
                prof_def,
                target_relative_tolerance=target_tolerance,
                max_solving_iterations=max_solving_iterations,
                suppress=True,
            )
            t_def = time.perf_counter() - t0_def
            def_iters = len(solver.norm_rel_change)
            def_init_res = float(solver.norm_rel_change[0]) if len(solver.norm_rel_change) > 0 else 1.0

            # 2. Surrogate solve
            eq_surr = base_eq.create_auxiliary_equilibrium()
            for name, val in currents.items():
                eq_surr.tokamak.set_coil_current(name, val)
            prof_surr = ConstrainPaxisIp(
                eq=eq_surr, paxis=paxis, Ip=Ip, fvac=fvac, alpha_m=alpha_m, alpha_n=alpha_n
            )

            t0_inf = time.perf_counter()
            surrogate.apply(eq_surr, prof_surr)
            t_inf = time.perf_counter() - t0_inf

            t0_solve = time.perf_counter()
            solver.forward_solve(
                eq_surr,
                prof_surr,
                target_relative_tolerance=target_tolerance,
                max_solving_iterations=max_solving_iterations,
                suppress=True,
            )
            t_solve = time.perf_counter() - t0_solve
            t_net = t_inf + t_solve
            surr_iters = len(solver.norm_rel_change)
            surr_init_res = float(solver.norm_rel_change[0]) if len(solver.norm_rel_change) > 0 else 1.0

            speedup = t_def / t_net if t_net > 0 else 1.0

            res_results["default_iters"].append(def_iters)
            res_results["surrogate_iters"].append(surr_iters)
            res_results["default_time"].append(t_def)
            res_results["surrogate_solve_time"].append(t_solve)
            res_results["surrogate_inf_time"].append(t_inf)
            res_results["surrogate_net_time"].append(t_net)
            res_results["speedup"].append(speedup)
            res_results["default_success"].append(eq_def.solved)
            res_results["surrogate_success"].append(eq_surr.solved)
            res_results["default_init_res"].append(def_init_res)
            res_results["surrogate_init_res"].append(surr_init_res)

            print(
                f"  [{i+1}/{len(test_idx)}] Case #{idx}: Def {def_iters} iters ({t_def*1000:.1f}ms) | "
                f"Surr {surr_iters} iters ({t_net*1000:.1f}ms, inf={t_inf*1000:.2f}ms) | "
                f"Speedup: {speedup:.2f}x"
            )

        grid_key = f"{nx}x{ny}"
        benchmark_summary[grid_key] = res_results

    return benchmark_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=15)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    results = run_resolution_benchmark(n_cases=args.cases, target_tolerance=args.tolerance)

    # Print summary table
    print("\n" + "=" * 90)
    print("RESOLUTION GENERALISATION BENCHMARK SUMMARY")
    print("=" * 90)
    print(
        f"{'Grid':<10} | {'Def Iters':<10} | {'Surr Iters':<10} | {'Iter Reduction':<15} | "
        f"{'Def Time':<10} | {'Surr Time':<10} | {'Inf Time':<10} | {'Speedup':<8}"
    )
    print("-" * 90)
    for grid, res in results.items():
        def_iters_mean = np.mean(res["default_iters"])
        surr_iters_mean = np.mean(res["surrogate_iters"])
        iter_red = (def_iters_mean - surr_iters_mean) / def_iters_mean * 100
        def_time_med = np.median(res["default_time"]) * 1000
        surr_time_med = np.median(res["surrogate_net_time"]) * 1000
        inf_time_mean = np.mean(res["surrogate_inf_time"]) * 1000
        speedup_med = np.median(res["speedup"])

        print(
            f"{grid:<10} | {def_iters_mean:<10.1f} | {surr_iters_mean:<10.1f} | {iter_red:<14.1f}% | "
            f"{def_time_med:<8.1f}ms | {surr_time_med:<8.1f}ms | {inf_time_mean:<8.2f}ms | {speedup_med:<7.2f}x"
        )
