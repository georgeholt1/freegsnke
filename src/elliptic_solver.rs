use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::sparse::linalg::lu::{
    factorize_symbolic_lu, LuSymbolicParams, NumericLu, SymbolicLu,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{MatMut, Par};

thread_local! {
    static SOLVE_SCRATCH: std::cell::RefCell<Option<MemBuffer>> = const { std::cell::RefCell::new(None) };
}

/// 2D Grad-Shafranov elliptic PDE solver on a regular grid (R, Z):
/// Δ* ψ = d^2ψ/dR^2 - (1/R) dψ/dR + d^2ψ/dZ^2 = RHS
/// with Dirichlet boundary conditions ψ_bnd.
pub struct EllipticSolver {
    pub nx: usize,
    pub ny: usize,
    pub rmin: f64,
    pub rmax: f64,
    pub zmin: f64,
    pub zmax: f64,
    pub dr: f64,
    pub dz: f64,
    symbolic_lu: SymbolicLu<usize>,
    numeric_lu: NumericLu<usize, f64>,
    solve_scratch_req: StackReq,
}

impl EllipticSolver {
    pub fn new(
        nx: usize,
        ny: usize,
        rmin: f64,
        rmax: f64,
        zmin: f64,
        zmax: f64,
    ) -> Result<Self, String> {
        let dr = (rmax - rmin) / (nx - 1) as f64;
        let dz = (zmax - zmin) / (ny - 1) as f64;
        let n = nx * ny;

        let inv_dr2 = 1.0 / (dr * dr);
        let inv_dz2 = 1.0 / (dz * dz);

        // Build CSC representation:
        // Accumulate (row, col, value) triples, then sort by (col, row).
        let mut triples: Vec<(usize, usize, f64)> = Vec::with_capacity(5 * n);

        for x in 0..nx {
            let r = rmin + dr * x as f64;
            for y in 0..ny {
                let row = x * ny + y;
                // Check if boundary
                if x == 0 || x == nx - 1 || y == 0 || y == ny - 1 {
                    triples.push((row, row, 1.0));
                } else {
                    // Interior point
                    // y - 1
                    triples.push((row, row - 1, inv_dz2));
                    // x - 1
                    let coeff_xm1 = inv_dr2 + 1.0 / (2.0 * r * dr);
                    triples.push((row, row - ny, coeff_xm1));
                    // diagonal
                    triples.push((row, row, -2.0 * (inv_dr2 + inv_dz2)));
                    // x + 1
                    let coeff_xp1 = inv_dr2 - 1.0 / (2.0 * r * dr);
                    triples.push((row, row + ny, coeff_xp1));
                    // y + 1
                    triples.push((row, row + 1, inv_dz2));
                }
            }
        }

        // Sort by col, then by row
        triples.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let mut col_ptrs = Vec::with_capacity(n + 1);
        let mut row_indices = Vec::with_capacity(triples.len());
        let mut values = Vec::with_capacity(triples.len());

        let mut current_col = 0;
        col_ptrs.push(0);

        for (row, col, val) in triples {
            while current_col < col {
                col_ptrs.push(row_indices.len());
                current_col += 1;
            }
            row_indices.push(row);
            values.push(val);
        }
        while col_ptrs.len() <= n {
            col_ptrs.push(row_indices.len());
        }

        let a_sym = SymbolicSparseColMatRef::new_checked(
            n,
            n,
            &col_ptrs,
            None,
            &row_indices,
        );
        let a_mat = SparseColMatRef::new(a_sym, &values);

        let symbolic_lu = factorize_symbolic_lu(a_sym, LuSymbolicParams::default())
            .map_err(|e| format!("Symbolic LU failed: {:?}", e))?;

        let mut numeric_lu = NumericLu::<usize, f64>::new();
        let scratch_size = symbolic_lu.factorize_numeric_lu_scratch::<f64>(
            Par::Seq,
            Default::default(),
        );
        let mut mem = MemBuffer::new(scratch_size);
        let mut stack = MemStack::new(&mut mem);

        symbolic_lu
            .factorize_numeric_lu(
                &mut numeric_lu,
                a_mat,
                Par::Seq,
                &mut stack,
                Default::default(),
            )
            .map_err(|e| format!("Numeric LU failed: {:?}", e))?;

        let solve_scratch_req = symbolic_lu.solve_in_place_scratch::<f64>(1, Par::Seq);

        Ok(Self {
            nx,
            ny,
            rmin,
            rmax,
            zmin,
            zmax,
            dr,
            dz,
            symbolic_lu,
            numeric_lu,
            solve_scratch_req,
        })
    }

    /// Construct solver from CSC format arrays.
    pub fn from_csc(
        nrows: usize,
        ncols: usize,
        col_ptrs: &[usize],
        row_indices: &[usize],
        values: &[f64],
        nx: usize,
        ny: usize,
    ) -> Result<Self, String> {
        let a_sym = SymbolicSparseColMatRef::new_checked(
            nrows,
            ncols,
            col_ptrs,
            None,
            row_indices,
        );
        let a_mat = SparseColMatRef::new(a_sym, values);

        let symbolic_lu = factorize_symbolic_lu(a_sym, LuSymbolicParams::default())
            .map_err(|e| format!("Symbolic LU failed: {:?}", e))?;

        let mut numeric_lu = NumericLu::<usize, f64>::new();
        let scratch_size = symbolic_lu.factorize_numeric_lu_scratch::<f64>(
            Par::Seq,
            Default::default(),
        );
        let mut mem = MemBuffer::new(scratch_size);
        let mut stack = MemStack::new(&mut mem);

        symbolic_lu
            .factorize_numeric_lu(
                &mut numeric_lu,
                a_mat,
                Par::Seq,
                &mut stack,
                Default::default(),
            )
            .map_err(|e| format!("Numeric LU failed: {:?}", e))?;

        let solve_scratch_req = symbolic_lu.solve_in_place_scratch::<f64>(1, Par::Seq);

        Ok(Self {
            nx,
            ny,
            rmin: 0.0,
            rmax: 0.0,
            zmin: 0.0,
            zmax: 0.0,
            dr: 0.0,
            dz: 0.0,
            symbolic_lu,
            numeric_lu,
            solve_scratch_req,
        })
    }

    /// Solve A * x = rhs in-place.
    /// `rhs` must be a slice of length nx * ny. On output, contains the solution x.
    pub fn solve(&self, rhs: &mut [f64]) -> Result<(), String> {
        let n = self.nx * self.ny;
        if rhs.len() != n {
            return Err(format!(
                "RHS length {} does not match system dimension {}",
                rhs.len(),
                n
            ));
        }

        SOLVE_SCRATCH.with(|cell| {
            let mut opt = cell.borrow_mut();
            let need_new = match opt.as_ref() {
                Some(buf) => {
                    let req_size = self.solve_scratch_req.layout().map(|l| l.size()).unwrap_or(0);
                    buf.as_ref().len() < req_size
                }
                None => true,
            };
            if need_new {
                *opt = Some(MemBuffer::new(self.solve_scratch_req));
            }
            let mem = opt.as_mut().unwrap();
            let mut stack_solve = MemStack::new(mem);

            let mut mat_rhs = MatMut::from_column_major_slice_mut(rhs, n, 1);

            let lu_ref = faer::sparse::linalg::lu::LuRef::new_unchecked(
                &self.symbolic_lu,
                &self.numeric_lu,
            );

            lu_ref.solve_in_place_with_conj(
                faer::Conj::No,
                mat_rhs.as_mut(),
                Par::Seq,
                &mut stack_solve,
            );
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elliptic_solver_dirichlet() {
        // Grid 9x9, R in [1.0, 2.0], Z in [-1.0, 1.0]
        let solver = EllipticSolver::new(9, 9, 1.0, 2.0, -1.0, 1.0).unwrap();
        let mut rhs = vec![0.0; 81];
        // Set Dirichlet boundary condition = 10.0
        for x in 0..9 {
            for y in 0..9 {
                if x == 0 || x == 8 || y == 0 || y == 8 {
                    rhs[x * 9 + y] = 10.0;
                }
            }
        }
        solver.solve(&mut rhs).unwrap();
        // Laplace equation with constant boundary should have constant solution = 10.0
        for val in rhs.iter() {
            assert!((val - 10.0).abs() < 1e-10);
        }
    }
}
