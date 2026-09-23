use faer::sparse::*;
use faer::sparse::linalg::lu::*;
use faer::mat;
use faer::Par;
use faer::dyn_stack::{MemBuffer, MemStack};

#[test]
fn test_sparse_lu_solve() {
    let col_ptrs: Vec<usize> = vec![0, 2, 5, 7];
    let row_indices: Vec<usize> = vec![0, 1, 0, 1, 2, 1, 2];
    let values: Vec<f64> = vec![2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];

    let a_sym = SymbolicSparseColMatRef::new_checked(
        3, 3, &col_ptrs, None, &row_indices
    );
    let a_mat = SparseColMatRef::new(a_sym, &values);

    let symbolic_lu = factorize_symbolic_lu(a_sym, LuSymbolicParams::default()).unwrap();
    let mut numeric_lu = NumericLu::<usize, f64>::new();
    
    let scratch_size = symbolic_lu.factorize_numeric_lu_scratch::<f64>(
        Par::Seq, Default::default()
    );
    let mut mem = MemBuffer::new(scratch_size);
    let mut stack = MemStack::new(&mut mem);
    
    let lu_ref = symbolic_lu.factorize_numeric_lu(
        &mut numeric_lu,
        a_mat,
        Par::Seq,
        &mut stack,
        Default::default(),
    ).unwrap();

    let mut b = mat![[3.0f64], [5.0f64], [3.0f64]];
    let solve_scratch = symbolic_lu.solve_in_place_scratch::<f64>(1, Par::Seq);
    let mut mem_solve = MemBuffer::new(solve_scratch);
    let mut stack_solve = MemStack::new(&mut mem_solve);

    lu_ref.solve_in_place_with_conj(
        faer::Conj::No,
        b.as_mut(),
        Par::Seq,
        &mut stack_solve,
    );

    println!("Solution x: {:?}", b);
    assert!((b[(0, 0)] - 1.0).abs() < 1e-10);
    assert!((b[(1, 0)] - 1.0).abs() < 1e-10);
    assert!((b[(2, 0)] - 1.0).abs() < 1e-10);
}
