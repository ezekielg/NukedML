import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class UppercaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        """

        :param X: {array-like, dataframe} of shape (n_samples, n_features)
        :return: np.ndarray of shape (n_samples, n_feature_new)
        """
        try:
            if not isinstance(X, (pd.Series, pd.DataFrame)):
                X_out = pd.DataFrame(np.asarray(X))

            X_out = X.apply(lambda x: x.str.upper())

        except AttributeError as e:
            raise TypeError("Variable 'data' must be array-like or np.ndarray")

        return X_out


class SparkjoyTransformer(BaseEstimator, TransformerMixin):
    """
    Too many things in a column? Let's throw out the things that don't spark joy.
    """
    def __init__(self, cutoff, label):
        """

        :param cutoff: Cutoff value for the value count of any individual category.
        :param label: Label to lump cutoff values under.
        """
        self.cutoff = cutoff
        self.label = label
        self._elims = {}

    def fit(self, X, y=None):
        """
        Fit SparkjoyTransformer to X.

        :param X: array-like of shape [n_samples, n_features]
        :param y: Ignored.
        :return: self
        """

        for c in sorted(X):
            truth = X[c].value_counts < self.cutoff
            cutouts = truth.index[truth].to_list()

            self._elims[c] = cutouts

        return self

    def transform(self, X):
        """
        Transform X by re-labeling categories with too small of a sample under the same name.

        :param X: array-like with shape [n_sample, n_features] of the data to re-label.
        :return X_out: The transformed data.
        """

        try:
            if not isinstance(X, (pd.Series, pd.DataFrame)):
                X_out = pd.DataFrame(np.asarray(X))

            X_out = X.apply(lambda x: np.where(x.map(x.value_counts()) < self.cutoff, self.label, x))

        except AttributeError as e:
            raise TypeError("Variable 'data' must be array-like or np.ndarray")

        return X_out
