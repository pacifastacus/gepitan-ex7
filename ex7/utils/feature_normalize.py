import numpy as np


def feature_normalize(X):
    """
    Normalizes the features in X

    feature_normalize(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.

    :param X: input dataset

    :return: normalized dataset
    """
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm = X_norm / sigma

    return X_norm, mu, sigma
