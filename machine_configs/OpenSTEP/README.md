# OpenSTEP Machine Configuration

This directory contains the machine configuration for the **OpenSTEP** spherical tokamak, derived from the official UKAEA OpenSTEP repository ([https://github.com/ukaea/OpenSTEP](https://github.com/ukaea/OpenSTEP)).

## Contents

- `OpenSTEP_active_coils.pickle`: Active PF and divertor coil circuits (`p3`, `p4`, `p5`, `p6`, `p9`, `s1`, `s2`) with multi-filament coordinates, dimensions, copper resistivity, and circuit multipliers.
- `OpenSTEP_wall.pickle`: 514-point closed first-wall and divertor baffle contour.
- `OpenSTEP_limiter.pickle`: Limiter boundary definition (identical to wall contour).
- `OpenSTEP_passive_coils.pickle`: Passive structures (empty list, as static solves are current-specified).
- `OpenSTEP_coil_currents.pickle` / `.json`: Reference equilibrium coil currents shipped with OpenSTEP.
- `OpenSTEP_profiles.pickle`: Reference plasma current profile data (1D $p'(\psi_N)$ and $FF'(\psi_N)$, $I_p$, $\beta_p$, $f_{\text{vac}}$, $R_0$, $B_0$, $R_{\text{axis}} = 1.0$).
- `OpenSTEP_plasma_psi.pickle`: 2D reference plasma and total flux data ($\psi_{\text{plasma}}$, $\psi_{\text{tot}}$, $\psi_{\text{axis}} = 6.1109$ Wb/rad, $\psi_{\text{bndry}} = 1.6798$ Wb/rad) for initializing forward solves to the authentic diverted state.

## Reference Coil Currents (from EBCC Free-Boundary Equilibrium)

| Circuit | Total Current [A] | Description / Location |
| :--- | :--- | :--- |
| `p3` | -4,790,870.39 | Outer midplane PF shaping coil ($R \approx 8.30$ m, $Z = \pm 2.10$ m) |
| `p4` | 5,155,466.64 | Upper/lower inner vertical PF coil ($R \approx 2.15$ m, $Z = \pm 8.50$ m) |
| `p5` | 2,254,704.04 | Upper/lower divertor PF coil ($R \approx 3.25$ m, $Z = \pm 9.50$ m) |
| `p6` | 5,458,947.31 | Upper/lower outer vertical PF coil ($R \approx 6.85$ m, $Z = \pm 9.60$ m) |
| `p9` | -7,854,367.89 | Upper/lower intermediate PF coil ($R \approx 6.85$ m, $Z = \pm 6.35$ m) |
| `s1` | 5,100,000.00 | Central solenoid segment 1 ($R \approx 0.95$ m, $Z = \pm 6.42$ m) |
| `s2` | -2,274,646.53 | Central solenoid segment 2 ($R \approx 0.95$ m, $Z = \pm 5.40$ m) |

## Equilibrium Parameters

- Plasma current $I_p$: 22.76 MA ($22,760,461.2$ A)
- Vacuum toroidal field: $B_0 = 2.4338$ T at $R_0 = 4.7334$ m ($f_{\text{vac}} = R_0 B_0 = 11.52\text{ T}\cdot\text{m}$)
- Poloidal beta $\beta_p$: 1.042
- Central pressure $p_{\text{axis}}$: 1.33 MPa
