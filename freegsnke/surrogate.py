"""
Surrogate model for initialising FreeGSNKE forward static simulations.

Provides lightweight, pure-NumPy inference using pre-trained POD/PCA basis modes
and a Multi-Layer Perceptron (MLP) regressor.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np


class SurrogateInitialGuess:
    """
    Lightweight, pure-NumPy surrogate model for predicting initial plasma flux guesses.

    Predicts the 2D poloidal flux distribution `plasma_psi(R, Z)` for forward static
    Grad-Shafranov solves given active coil currents and profile parameters.

    Inference relies entirely on pre-extracted weights and PCA modes loaded from
    a compact .npz file, requiring zero external machine learning dependencies
    (such as PyTorch or scikit-learn) at runtime.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialise the surrogate guess provider.

        Parameters
        ----------
        model_path : str or Path, optional
            Path to the .npz archive containing surrogate weights and PCA components.
            If None, loads the default bundled model for MAST-U.
        """
        if model_path is None:
            default_path = (
                Path(__file__).resolve().parent / "models" / "mastu_paxis_ip_surrogate.npz"
            )
            model_path = default_path

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Surrogate model file not found at: {model_path}. "
                "Ensure the bundled model is present or provide a valid path."
            )

        data = np.load(model_path, allow_pickle=True)
        self.pca_mean = data["pca_mean"]  # (nx*ny,)
        self.pca_components = data["pca_components"]  # (n_modes, nx*ny)
        self.x_mean = data["x_mean"]  # (n_features,)
        self.x_scale = data["x_scale"]  # (n_features,)
        self.weights_0 = data["weights_0"]  # (n_features, h1)
        self.bias_0 = data["bias_0"]  # (h1,)
        self.weights_1 = data["weights_1"]  # (h1, h2)
        self.bias_1 = data["bias_1"]  # (h2,)
        self.weights_2 = data["weights_2"]  # (h2, n_modes)
        self.bias_2 = data["bias_2"]  # (n_modes,)
        self.feature_names = [str(f) for f in data["feature_names"]]
        self.grid_shape = tuple(data["grid_shape"])
        self.n_modes = int(data["n_modes"][0])
        self.model_path = model_path

    def extract_features(self, eq, profiles) -> np.ndarray:
        """
        Extract the input feature vector from an Equilibrium and Profile object.

        Parameters
        ----------
        eq : Equilibrium
            FreeGSNKE equilibrium object containing tokamak coil currents.
        profiles : Profile
            FreeGSNKE profile object containing core plasma parameters.

        Returns
        -------
        np.ndarray
            1D array of ordered input features matching `self.feature_names`.
        """
        features = []
        for name in self.feature_names:
            if hasattr(eq, "tokamak") and hasattr(eq.tokamak, "coil_order") and name in eq.tokamak.coil_order:
                features.append(float(eq.tokamak[name].current))
            elif hasattr(profiles, name):
                val = getattr(profiles, name)
                features.append(float(val() if callable(val) else val))
            elif hasattr(profiles, f"_{name}"):
                val = getattr(profiles, f"_{name}")
                features.append(float(val() if callable(val) else val))
            else:
                raise AttributeError(
                    f"Required feature '{name}' not found in equilibrium or profile object."
                )
        return np.array(features, dtype=np.float64)

    def predict_modes(self, x: np.ndarray) -> np.ndarray:
        """
        Perform forward pass through the pure-NumPy MLP to predict PCA mode amplitudes.

        Parameters
        ----------
        x : np.ndarray
            1D feature vector of shape `(n_features,)`.

        Returns
        -------
        np.ndarray
            1D array of PCA mode amplitudes of shape `(n_modes,)`.
        """
        # Standardise inputs
        x_norm = (x - self.x_mean) / self.x_scale

        # Hidden layer 1 (Dense + ReLU)
        h1 = np.maximum(0.0, np.dot(x_norm, self.weights_0) + self.bias_0)

        # Hidden layer 2 (Dense + ReLU)
        h2 = np.maximum(0.0, np.dot(h1, self.weights_1) + self.bias_1)

        # Output layer (Linear)
        modes = np.dot(h2, self.weights_2) + self.bias_2
        return modes

    def predict(self, eq, profiles) -> np.ndarray:
        """
        Predict the 2D initial plasma flux distribution.

        Parameters
        ----------
        eq : Equilibrium
            FreeGSNKE equilibrium object.
        profiles : Profile
            FreeGSNKE profile object.

        Returns
        -------
        np.ndarray
            2D predicted plasma flux grid of shape `self.grid_shape`.
        """
        x = self.extract_features(eq, profiles)
        modes = self.predict_modes(x)

        # Linear combination of PCA components + mean flux
        psi_flat = self.pca_mean + np.dot(modes, self.pca_components)
        psi_2d = psi_flat.reshape(self.grid_shape)
        return psi_2d

    def apply(self, eq, profiles) -> np.ndarray:
        """
        Predict and directly assign the surrogate flux guess to the equilibrium.

        Parameters
        ----------
        eq : Equilibrium
            FreeGSNKE equilibrium object to be updated.
        profiles : Profile
            FreeGSNKE profile object.

        Returns
        -------
        np.ndarray
            The predicted 2D plasma flux array assigned to `eq.plasma_psi`.
        """
        psi_guess = self.predict(eq, profiles)
        eq._updatePlasmaPsi(psi_guess)
        eq.solved = False
        return psi_guess

    def __call__(self, eq, profiles=None) -> np.ndarray:
        """Callable interface for initializing equilibria."""
        if profiles is None:
            if hasattr(eq, "profiles") and eq.profiles is not None:
                profiles = eq.profiles
            else:
                raise ValueError("Profiles object must be provided to evaluate surrogate guess.")
        return self.predict(eq, profiles)
