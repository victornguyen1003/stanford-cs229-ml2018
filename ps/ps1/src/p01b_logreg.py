import numpy as np # type: ignore
import util

import logging
logger = logging.getLogger(__name__)

from linear_model import LinearModel

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions
    """
    logger.info("Starting logistic regression.")
    
    X_train, y_train = util.load_dataset(train_path, 'y', True)
    X_eval, y_eval_actual = util.load_dataset(eval_path, 'y', True)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    logger.info(f"Theta: {clf.theta}")

    y_eval_predicted = clf.predict(X_eval)
    np.savetxt(pred_path, y_eval_predicted, delimiter=',')

    error_rate = np.mean(np.absolute(np.round(y_eval_predicted) - y_eval_actual))
    logger.info(f"Error rate: {error_rate}")

    return clf

class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.
    
    Example usage:
        > clf = LogisticRegression()
        > clf.fit(X_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self):
        super().__init__()

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Run Newton's Method to minimize J(theta) for logistic regression
        
        Args:
            x: Training example inputs. Shape(m, n).
            y: Training example labels. Shape (m,).
        """

        logger.debug("Starting training for logistic regression.")

        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        for i in range(self.max_iter):
            h = 1 / (1 + np.exp(-np.matmul(x, self.theta)))
            logger.debug(f"Hypothesis at iteration {i+1}: {h}.")

            grad = np.matmul(x.T, y-h) / x.shape[0]
            logger.debug(f"Gradient at iteration {i+1}: {grad}.")

            H = x.T @ np.diag(h * (1-h)) @ x / x.shape[0]
            logger.debug(f"Hessian at iteration {i+1}: {H}.")

            theta_new = self.theta - np.linalg.pinv(H) @ grad 
            logger.debug(f"Theta at iteration {i+1}: {theta_new}.")

            if self.verbose:
                print(f"Theta value at interation {i+1}: {theta_new}.")

            dist = np.linalg.norm(theta_new - self.theta)
            self.theta = theta_new

            if dist <= self.eps:
                logger.debug(f"Theta value converged after {i} iterations at {self.theta}.")
                if self.verbose:
                    print(f"Theta value converged after {i} iterations at {self.theta}.")
                break

            if dist > self.eps and i == self.max_iter:
                logger.debug(f"Theta value does not converge after {i} iterations at which its value: {self.theta}.")
                if self.verbose:
                    print(f"Theta value does not converge after {i} iterations at which its value: {self.theta}.")
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions given new inputs x.
        
        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        logger.debug("Starting prediction for logistic regression.")

        predictions = 1/(1 + np.exp(-np.matmul(x, self.theta))).reshape(x.shape[0],)

        logger.debug(f"Predictions: {predictions}")
        if self.verbose:
            print(f"Predictions: {predictions}")

        return predictions