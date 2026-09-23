use std::f64::consts::PI;
use rayon::prelude::*;
use crate::elliptic_integrals::ellipke;

pub const MU_0: f64 = 4.0 * PI * 1e-7;
const MU_0_OVER_2PI: f64 = 2e-7;

/// Calculate poloidal flux at (r, z) due to a unit current at (rc, zc)
/// using the standard Green's function for the axisymmetric toroidal operator.
#[inline]
pub fn greens(rc: f64, zc: f64, r: f64, z: f64) -> f64 {
    let r_plus_rc = r + rc;
    let dz = z - zc;
    let denom = r_plus_rc * r_plus_rc + dz * dz;
    let mut k2 = 4.0 * r * rc / denom;

    // Clip to between 1e-10 and 1 - 1e-10 to avoid singularity
    if k2 < 1e-10 {
        k2 = 1e-10;
    } else if k2 > 1.0 - 1e-10 {
        k2 = 1.0 - 1e-10;
    }

    let k = k2.sqrt();
    let (ellip_k, ellip_e) = ellipke(k2);

    MU_0_OVER_2PI * (r * rc).sqrt() * ((2.0 - k2) * ellip_k - 2.0 * ellip_e) / k
}

/// Compute boundary Green's matrix Gbnd of size (n_bnd, n_sources)
/// where each entry (i, j) is greens(rc[j], zc[j], rbnd[i], zbnd[i]).
pub fn compute_greens_matrix(
    rc: &[f64],
    zc: &[f64],
    rbnd: &[f64],
    zbnd: &[f64],
) -> Vec<f64> {
    let n_bnd = rbnd.len();
    let n_sources = rc.len();
    let mut mat = vec![0.0; n_bnd * n_sources];

    mat.par_chunks_exact_mut(n_sources)
        .enumerate()
        .for_each(|(i, row)| {
            let rb = rbnd[i];
            let zb = zbnd[i];
            for j in 0..n_sources {
                row[j] = greens(rc[j], zc[j], rb, zb);
            }
        });

    mat
}

/// Fast parallel matrix-vector product for boundary flux:
/// psi_bnd = Gbnd @ jtor_inside * (dR * dZ)
pub fn boundary_flux_from_jtor(
    g_bnd: &[f64],
    _n_bnd: usize,
    n_sources: usize,
    jtor_inside: &[f64],
    dr_dz: f64,
    out_psi_bnd: &mut [f64],
) {
    out_psi_bnd
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, val)| {
            let row = &g_bnd[i * n_sources..(i + 1) * n_sources];
            let mut sum = 0.0;
            for j in 0..n_sources {
                sum += row[j] * jtor_inside[j];
            }
            *val = sum * dr_dz;
        });
}

use std::cell::RefCell;

thread_local! {
    static GATHER_SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Compute boundary flux directly from full jtor and source indices:
/// psi_bnd[i] = sum_j (greenfunc[i, j] * jtor_full[source_indices[j]])
pub fn boundary_flux_gather(
    greenfunc: &[f64],
    _n_bnd: usize,
    n_sources: usize,
    jtor_full: &[f64],
    source_indices: &[usize],
    out_psi_bnd: &mut [f64],
) {
    GATHER_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        if scratch.len() < n_sources {
            scratch.resize(n_sources, 0.0);
        }
        let gathered = &mut scratch[..n_sources];
        for (dst, &src_idx) in gathered.iter_mut().zip(source_indices.iter()) {
            *dst = unsafe { *jtor_full.get_unchecked(src_idx) };
        }

        for (i, val) in out_psi_bnd.iter_mut().enumerate() {
            let row = &greenfunc[i * n_sources..(i + 1) * n_sources];
            let mut sum = 0.0;
            for (&g, &j) in row.iter().zip(gathered.iter()) {
                sum += g * j;
            }
            *val = sum;
        }
    });
}

/// Compute vacuum poloidal flux on (nx, ny) grid from coils:
/// psi_coils[i, j] = sum_k (coil_currents[k] * greens(coil_r[k], coil_z[k], r[i], z[j]))
pub fn vacuum_flux_grid(
    coil_r: &[f64],
    coil_z: &[f64],
    coil_currents: &[f64],
    r_grid: &[f64],
    z_grid: &[f64],
    _nx: usize,
    ny: usize,
    out_psi: &mut [f64],
) {
    let n_coils = coil_r.len();

    out_psi
        .par_chunks_exact_mut(ny)
        .enumerate()
        .for_each(|(ix, col)| {
            let r = r_grid[ix];
            for iy in 0..ny {
                let z = z_grid[iy];
                let mut sum = 0.0;
                for k in 0..n_coils {
                    let ic = coil_currents[k];
                    if ic != 0.0 {
                        sum += ic * greens(coil_r[k], coil_z[k], r, z);
                    }
                }
                col[iy] = sum;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greens_basic() {
        let g = greens(1.0, 0.0, 1.2, 0.1);
        assert!(g > 0.0);
        // Reciprocity: G(1, 2) == G(2, 1)
        let g_rev = greens(1.2, 0.1, 1.0, 0.0);
        assert!((g - g_rev).abs() < 1e-14);
    }
}
