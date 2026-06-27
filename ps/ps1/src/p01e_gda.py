import numpy as np # type: ignore
import util
from typing import Optional, Tuple

import logging
logger = logging.getLogger(__name__)

from linear_model import LinearModel

def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA).
    
    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    logger.info("Starting GDA.")

    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    clf = GDA()
    clf.fit(x_train, y_train)
    logger.info(f"Theta: {clf.theta}")
    
    y_eval_predicted = clf.predict(x_eval)
    np.savetxt(pred_path, y_eval_predicted, delimiter=',')

    error_rate = np.mean(np.absolute(np.round(y_eval - y_eval_predicted)))
    logger.info(f"Error rate: {error_rate}")

    return clf

class GDA(LinearModel):
    """Gaussian Discriminant Analysis.
    
    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self):
        super().__init__()

    def fit(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Fit a GDA model to training set given by x and y.
        
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.    
        """

        logger.debug("Starting training for GDA.")

        m, n = x.shape[0], x.shape[1]
        Phi = np.mean(y)
        mu_0 = ((1-y).reshape(-1,1) * x).sum(axis=0) / (1-y).sum()
        mu_1 = (y.reshape(-1,1) * x).sum(axis=0) / y.sum()

        mean_matrix = y.reshape(-1,1) * mu_1.reshape(1,-1) + (1-y).reshape(-1,1) * mu_0.reshape(1,-1)
        Sigma = (x - mean_matrix).T @ (x - mean_matrix) / m

        self.theta = (Phi, mu_0, mu_1, Sigma)

        logger.debug(f"Phi, mu_0, mu_1, Sigma: {self.theta}")
        if self.verbose:
            print(f"Phi, mu_0, mu_1, Sigma: {self.theta}")

        return self.theta
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make a prediction given new inputs x.
        
        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        logger.debug("Starting prediction for GDA.")

        m, n = x.shape[0], x.shape[1]
        Phi, mu_0, mu_1, Sigma = self.theta[0], self.theta[1], self.theta[2], self.theta[3]
        p_y_1 = Phi
        p_y_0 = 1 - p_y_1 
        p_x_given_y_1 = 1/((2 * np.pi)**(n/2) * np.linalg.det(Sigma)**(1/2)) * np.exp(np.sum(-1/2 * (x-mu_1.reshape(1,-1)) @ np.linalg.inv(Sigma) * (x-mu_1.reshape(1,-1)), axis=1))
        p_x_given_y_0 = 1/((2 * np.pi)**(n/2) * np.linalg.det(Sigma)**(1/2)) * np.exp(np.sum(-1/2 * (x-mu_0.reshape(1,-1)) @ np.linalg.inv(Sigma) * (x-mu_0.reshape(1,-1)), axis=1))
    
        p_y_1_given_x = p_x_given_y_1 * p_y_1 / (p_x_given_y_1 * p_y_1 + p_x_given_y_0 * p_y_0)
        logger.debug(f"Predictions: {p_y_1_given_x}")

        return p_y_1_given_x