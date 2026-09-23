pub mod elliptic_integrals;
pub mod elliptic_solver;
pub mod greens;

use numpy::ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use elliptic_solver::EllipticSolver;

#[pyfunction]
fn rust_backend_info() -> String {
    format!("FreeGSNKE Rust backend v{}", env!("CARGO_PKG_VERSION"))
}

#[pyfunction]
fn py_greens(rc: f64, zc: f64, r: f64, z: f64) -> f64 {
    greens::greens(rc, zc, r, z)
}

#[pyfunction]
fn py_compute_greens_matrix<'py>(
    py: Python<'py>,
    rc: PyReadonlyArray1<f64>,
    zc: PyReadonlyArray1<f64>,
    rbnd: PyReadonlyArray1<f64>,
    zbnd: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rc_s = rc.as_slice()?;
    let zc_s = zc.as_slice()?;
    let rbnd_s = rbnd.as_slice()?;
    let zbnd_s = zbnd.as_slice()?;

    let n_bnd = rbnd_s.len();
    let n_sources = rc_s.len();

    let mat_vec = py.allow_threads(|| greens::compute_greens_matrix(rc_s, zc_s, rbnd_s, zbnd_s));
    let arr = Array2::from_shape_vec((n_bnd, n_sources), mat_vec)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(arr.into_pyarray_bound(py))
}

#[pyfunction]
fn py_boundary_flux_from_jtor<'py>(
    py: Python<'py>,
    g_bnd: PyReadonlyArray2<f64>,
    jtor_inside: PyReadonlyArray1<f64>,
    dr_dz: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let g_shape = g_bnd.shape();
    let n_bnd = g_shape[0];
    let n_sources = g_shape[1];

    let g_slice = g_bnd.as_slice()?;
    let jtor_slice = jtor_inside.as_slice()?;

    if jtor_slice.len() != n_sources {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: Gbnd has {} cols, but jtor has {} elements",
            n_sources,
            jtor_slice.len()
        )));
    }

    let mut out = vec![0.0; n_bnd];
    greens::boundary_flux_from_jtor(
        g_slice,
        n_bnd,
        n_sources,
        jtor_slice,
        dr_dz,
        &mut out,
    );

    let arr = Array1::from_vec(out);
    Ok(arr.into_pyarray_bound(py))
}

#[pyfunction]
fn py_boundary_flux_gather<'py>(
    py: Python<'py>,
    greenfunc: PyReadonlyArray2<f64>,
    jtor_full: PyReadonlyArray1<f64>,
    source_indices: PyReadonlyArray1<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let g_shape = greenfunc.shape();
    let n_bnd = g_shape[0];
    let n_sources = g_shape[1];

    let g_slice = greenfunc.as_slice()?;
    let jtor_slice = jtor_full.as_slice()?;
    let idx_slice = source_indices.as_slice()?;

    if idx_slice.len() != n_sources {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: greenfunc has {} cols, but source_indices has {} elements",
            n_sources,
            idx_slice.len()
        )));
    }

    let mut out = vec![0.0; n_bnd];
    py.allow_threads(|| {
        greens::boundary_flux_gather(
            g_slice,
            n_bnd,
            n_sources,
            jtor_slice,
            idx_slice,
            &mut out,
        )
    });

    let arr = Array1::from_vec(out);
    Ok(arr.into_pyarray_bound(py))
}

#[pyfunction]
fn py_vacuum_flux_grid<'py>(
    py: Python<'py>,
    coil_r: PyReadonlyArray1<f64>,
    coil_z: PyReadonlyArray1<f64>,
    coil_currents: PyReadonlyArray1<f64>,
    r_grid: PyReadonlyArray1<f64>,
    z_grid: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cr = coil_r.as_slice()?;
    let cz = coil_z.as_slice()?;
    let ci = coil_currents.as_slice()?;
    let rg = r_grid.as_slice()?;
    let zg = z_grid.as_slice()?;

    let nx = rg.len();
    let ny = zg.len();

    let mut out = vec![0.0; nx * ny];
    py.allow_threads(|| greens::vacuum_flux_grid(cr, cz, ci, rg, zg, nx, ny, &mut out));

    let arr = Array2::from_shape_vec((nx, ny), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(arr.into_pyarray_bound(py))
}

#[pyfunction]
fn py_coil_greens_grid<'py>(
    py: Python<'py>,
    rc: f64,
    zc: f64,
    r_grid: PyReadonlyArray1<f64>,
    z_grid: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rg = r_grid.as_slice()?;
    let zg = z_grid.as_slice()?;
    let nx = rg.len();
    let ny = zg.len();

    let mut out = vec![0.0; nx * ny];
    py.allow_threads(|| greens::vacuum_flux_grid(&[rc], &[zc], &[1.0], rg, zg, nx, ny, &mut out));

    let arr = Array2::from_shape_vec((nx, ny), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

#[pyclass(name = "RustEllipticSolver")]
struct PyRustEllipticSolver {
    solver: EllipticSolver,
}

impl PyRustEllipticSolver {
    fn solve_generic<'py>(
        &self,
        py: Python<'py>,
        rhs: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        let n = self.solver.nx * self.solver.ny;
        if let Ok(arr2) = rhs.extract::<PyReadonlyArray2<f64>>() {
            let shape = arr2.shape();
            let nx = shape[0];
            let ny = shape[1];
            if nx != self.solver.nx || ny != self.solver.ny {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "RHS shape ({}, {}) does not match solver shape ({}, {})",
                    nx, ny, self.solver.nx, self.solver.ny
                )));
            }
            let mut data = arr2.as_slice()?.to_vec();
            py.allow_threads(|| self.solver.solve(&mut data))
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            let res = Array2::from_shape_vec((nx, ny), data)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(res.into_pyarray_bound(py).into_any().unbind())
        } else if let Ok(arr1) = rhs.extract::<PyReadonlyArray1<f64>>() {
            let len = arr1.len();
            if len != n {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "RHS length {} does not match solver dimension {}",
                    len, n
                )));
            }
            let mut data = arr1.as_slice()?.to_vec();
            py.allow_threads(|| self.solver.solve(&mut data))
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            let res = Array1::from_vec(data);
            Ok(res.into_pyarray_bound(py).into_any().unbind())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "RHS must be a 1D or 2D float64 numpy array",
            ))
        }
    }
}

#[pymethods]
impl PyRustEllipticSolver {
    #[new]
    fn new(
        nx: usize,
        ny: usize,
        rmin: f64,
        rmax: f64,
        zmin: f64,
        zmax: f64,
    ) -> PyResult<Self> {
        let solver = EllipticSolver::new(nx, ny, rmin, rmax, zmin, zmax)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(Self { solver })
    }

    #[staticmethod]
    fn from_csc(
        nrows: usize,
        ncols: usize,
        indptr: PyReadonlyArray1<i32>,
        indices: PyReadonlyArray1<i32>,
        data: PyReadonlyArray1<f64>,
        nx: usize,
        ny: usize,
    ) -> PyResult<Self> {
        let col_ptrs: Vec<usize> = indptr.as_slice()?.iter().map(|&x| x as usize).collect();
        let row_indices: Vec<usize> = indices.as_slice()?.iter().map(|&x| x as usize).collect();
        let values: &[f64] = data.as_slice()?;

        let solver = EllipticSolver::from_csc(
            nrows,
            ncols,
            &col_ptrs,
            &row_indices,
            values,
            nx,
            ny,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(Self { solver })
    }

    /// Callable interface drop-in for MGDirect: solver(psi_boundary, rhs)
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        _psi_boundary: &Bound<'py, PyAny>,
        rhs: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        self.solve_generic(py, rhs)
    }

    /// Solve the elliptic system given 1D or 2D RHS.
    fn solve<'py>(
        &self,
        py: Python<'py>,
        rhs: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        self.solve_generic(py, rhs)
    }
}

#[pymodule]
fn _freegsnke_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_backend_info, m)?)?;
    m.add_function(wrap_pyfunction!(py_greens, m)?)?;
    m.add_function(wrap_pyfunction!(py_coil_greens_grid, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_greens_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_boundary_flux_from_jtor, m)?)?;
    m.add_function(wrap_pyfunction!(py_boundary_flux_gather, m)?)?;
    m.add_function(wrap_pyfunction!(py_vacuum_flux_grid, m)?)?;
    m.add_class::<PyRustEllipticSolver>()?;
    Ok(())
}
