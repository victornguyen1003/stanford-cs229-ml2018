import numpy as np # type: ignore
from typing import Optional

class LinearModel:
    """Base class for linear models."""

    def __init__(self, step_size: float = 0.2, max_iter: int = 100, eps: float = 1e-5, theta_0: Optional[np.ndarray] = None, verbose: bool = False):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.    
        """
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.theta = theta_0
        self.verbose = verbose

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Run solver to fit linear models.
        
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        raise NotImplementedError("Subclass of LinearModel must implement fit method.")
    
    def predict(self, x: np.ndarray):
        """Make a prediction given new inputs x.
        
        Args:
            x: Inputs of shape (m, n)
        
        Returns:
            Outputs of shape (m,)
        """

        raise NotImplementedError("Subclass of LinearModel must implement predict method.")