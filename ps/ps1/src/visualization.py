from util import load_dataset, add_intercept
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

from p01b_logreg import LogisticRegression
from p01e_gda import GDA

import logging
logger = logging.getLogger(__name__)

def main():
    x1, y1 = load_dataset('../data/ds1_train.csv', add_intercept=False)
    plot(x1, y1, 'ds1_train', '../reports/figures')

    x2, y2 = load_dataset('../data/ds2_train.csv', add_intercept=False)
    plot(x2, y2, 'ds2_train', '../reports/figures')

    x1_trans = x1 / np.abs(x1) * np.log(np.abs(x1)+1)
    plot(x1_trans, y1, 'ds1_train_trans', '../reports/figures')

def plot_logistic_boundary(ax: plt.ax, x: np.ndarray, y: np.ndarray, correction: float = 1.0) -> None:
    """Plot decision boundary for logistic regression.

    Args:
        ax: Axis to plot
        x: Inputs of shape (m, 3)
        y: Labels of shape (m,)
        correction: Correction factor to apply (Problem 2(e) only).
    """

    clf = LogisticRegression()
    clf.fit(x, y)
    theta = clf.theta

    x1 = np.arange(x[:,1].min(), x[:,1].max(), 0.01)
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    ax.plot(x1, x2, c='red', linewidth=2, label='logistic regression db')

def plot_gda_boundary(ax: plt.ax, x: np.ndarray, y: np.ndarray, correction: float = 1.0) -> None:
    """Plot decision boundary for gda.

    Args:
        ax: Axis to plot
        x: Inputs of shape (m, 2)
        y: Labels of shape (m,)
        correction: Correction factor to apply (Problem 2(e) only).
    """
    clf = GDA()
    clf.fit(x, y)
    Phi, mu_0, mu_1, Sigma = clf.theta[0], clf.theta[1], clf.theta[2], clf.theta[3]
    theta = np.linalg.pinv(Sigma) @ (mu_1 - mu_0)
    theta_0 = -1/2 * mu_1.T @ np.linalg.pinv(Sigma) @ mu_1 + 1/2 * mu_0.T @ np.linalg.pinv(Sigma) @ mu_0 - np.log((1-Phi)/Phi)

    x1 = np.arange(x[:,0].min(), x[:,0].max(), 0.01)
    x2 = -(theta_0 + theta[0] * x1) / theta[1]
    ax.plot(x1, x2, c='orange', linewidth=2, label='gda db')

def plot(x: np.ndarray, y: np.ndarray, file_name: str, save_dir: str) -> None:
    """Plot dataset and decision bounderies in training sample.
    
    Args:
        x: Inputs of shape (m,n)
        y: Labels of shape (m,)
        save_dir: Directory to save plot.
    """

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(x=x[y == 0, 0], y=x[y == 0, 1], c='b', marker='x')
    ax.scatter(x=x[y == 1, 0], y=x[y == 1, 1], c='g', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    margin0 = ((x[:,0].max()-x[:,0].min())*0.05)
    margin1 = ((x[:,1].max()-x[:,1].min())*0.05)
    ax.set_xlim(x[:,0].min()-margin0, x[:,0].max()+margin0)
    ax.set_ylim(x[:,1].min()-margin1, x[:,1].max()+margin1)

    plot_logistic_boundary(ax, add_intercept(x), y)
    plot_gda_boundary(ax, x, y)
    ax.legend(loc='upper right')

    plt.savefig("{}/{}.jpg".format(save_dir, file_name))
    plt.show()

if __name__ == '__main__':
    main()