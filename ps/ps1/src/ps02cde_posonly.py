import numpy as np # type: ignore
import util
from p01b_logreg import LogisticRegression

import logging
logger = logging.getLogger(__name__)

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD= 'X'

def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.
    
    Run under the following conditions:
        1. on y-labels
        2. on t-labels
        3. on t-labels with correlation factor alpha

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """

    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    clf = LogisticRegression()

    # Part c
    X_train, y_train = util.load_dataset(train_path, 't', True)
    logger.debug(f"X_train: {X_train}, y_train {y_train}")

    X_test, y_test = util.load_dataset(test_path, 't', True)
    logger.debug(f"X_test: {X_test}, y_test {y_test}")

    logger.info("Starting training process for logistic regression on t-labels only.")
    clf.fit(X_train, y_train)
    y_test_predicted = clf.predict(X_test)
    logger.debug(f"y_test_predicted: {y_test_predicted}")

    error_rate = np.mean(np.round(y_test_predicted) - y_test)
    logger.info(f"error_rate: {error_rate}")

    np.savetxt(pred_path_c, y_test_predicted, delimiter=',')

    # Part d
    X_train, y_train = util.load_dataset(train_path, 'y', True)
    logger.debug(f"X_train: {X_train}, y_train {y_train}")

    X_test, y_test = util.load_dataset(test_path, 'y', True)
    logger.debug(f"X_test: {X_test}, y_test {y_test}")
    
    logger.info("Starting training process for logistic regression on t-labels only.")
    clf.fit(X_train, y_train)
    y_test_predicted = clf.predict(X_test)
    logger.debug(f"y_test_predicted: {y_test_predicted}")

    error_rate = np.mean(np.round(y_test_predicted) - y_test)
    logger.info(f"error_rate: {error_rate}")

    np.savetxt(pred_path_d, y_test_predicted, delimiter=',')

    # Part e
    X_valid, y_valid = util.load_dataset(valid_path, 'y', True)

    y_valid_predicted = clf.predict(X_valid)
    logger.debug(f"y_valid_predicted: {y_valid_predicted}")

    alpha = np.mean(y_valid_predicted)
    logger.info(f"alpha: {alpha}")

    y_test_predicted / alpha
    
    error_rate = np.mean(np.round(y_test_predicted) - y_test)
    logger.info(f"error_rate: {error_rate}")

    np.savetxt(pred_path_d, y_test_predicted, delimiter=',')