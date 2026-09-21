#!/usr/bin/env python3
"""
Train a POD/PCA + MLP surrogate model on static forward equilibria dataset
and export model parameters to a pure NumPy-compatible .npz archive.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def train_surrogate(
    dataset_path: str = "data/surrogate_dataset_mastu.npz",
    output_model_path: str = "freegsnke/models/mastu_paxis_ip_surrogate.npz",
    n_modes: int = 12,
    hidden_layers: tuple = (64, 64),
    seed: int = 42,
):
    repo_root = Path(__file__).resolve().parents[1]
    data_file = repo_root / dataset_path
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at {data_file}")

    data = np.load(data_file, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]  # (N, 65, 129)
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    feature_names = [str(f) for f in data["feature_names"]]
    grid_shape = Y.shape[1:]

    N, nx, ny = Y.shape
    Y_flat = Y.reshape(N, -1)

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train_flat, Y_test_flat = Y_flat[train_idx], Y_flat[test_idx]

    print(f"Loaded dataset: {N} total samples (train: {len(train_idx)}, test: {len(test_idx)})")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Grid shape: {grid_shape} ({nx * ny} points)")

    # 1. Dimensionality reduction with PCA (POD)
    pca = PCA(n_components=n_modes, random_state=seed)
    A_train = pca.fit_transform(Y_train_flat)
    A_test = pca.transform(Y_test_flat)

    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA with {n_modes} modes explains {explained_var * 100:.3f}% of flux variance")

    # 2. Input standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Fit Multi-Layer Perceptron (MLP)
    print(f"Training MLP regressor with architecture {hidden_layers} on PCA mode amplitudes...")
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=1500,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(X_train_scaled, A_train)

    # 4. Evaluation on held-out test set
    A_test_pred = mlp.predict(X_test_scaled)
    Y_test_pred_flat = pca.inverse_transform(A_test_pred)

    # Relative Frobenius norm error
    test_rel_error = np.linalg.norm(Y_test_pred_flat - Y_test_flat) / np.linalg.norm(Y_test_flat)
    # Per-sample relative errors
    sample_rel_errors = np.linalg.norm(Y_test_pred_flat - Y_test_flat, axis=1) / np.linalg.norm(
        Y_test_flat, axis=1
    )
    median_error = np.median(sample_rel_errors)
    max_error = np.max(sample_rel_errors)

    print("\n--- Test Set Evaluation ---")
    print(f"Relative Frobenius Error : {test_rel_error * 100:.2f}%")
    print(f"Median Sample Error     : {median_error * 100:.2f}%")
    print(f"Max Sample Error        : {max_error * 100:.2f}%")

    # 5. Export weights to pure NumPy archive
    out_path = repo_root / output_model_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        pca_mean=pca.mean_.astype(np.float64),
        pca_components=pca.components_.astype(np.float64),
        x_mean=scaler.mean_.astype(np.float64),
        x_scale=scaler.scale_.astype(np.float64),
        weights_0=mlp.coefs_[0].astype(np.float64),
        bias_0=mlp.intercepts_[0].astype(np.float64),
        weights_1=mlp.coefs_[1].astype(np.float64),
        bias_1=mlp.intercepts_[1].astype(np.float64),
        weights_2=mlp.coefs_[2].astype(np.float64),
        bias_2=mlp.intercepts_[2].astype(np.float64),
        feature_names=np.array(feature_names, dtype=str),
        grid_shape=np.array(grid_shape, dtype=int),
        n_modes=np.array([n_modes], dtype=int),
    )

    size_kb = out_path.stat().st_size / 1024
    print(f"\nSuccessfully exported trained surrogate model to {out_path} ({size_kb:.1f} KB)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train surrogate model for static forward equilibria")
    parser.add_argument("--dataset", type=str, default="data/surrogate_dataset_mastu.npz")
    parser.add_argument("--output", type=str, default="freegsnke/models/mastu_paxis_ip_surrogate.npz")
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_surrogate(
        dataset_path=args.dataset,
        output_model_path=args.output,
        n_modes=args.modes,
        seed=args.seed,
    )
