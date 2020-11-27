import numpy as np
import pandas as pd
import json

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class JSONTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping_ = None  # Shows the relationship between original keys and new keys.

    def fit(self, X, y=None):
        self.mapping_ = {}

        return self

    def transform(self, X):
        if self.mapping_ is None:
            raise NotFittedError("This JSONTransformer instance is not fitted yet.")

        if not isinstance(X, (pd.Series, pd.DataFrame)):
            X = pd.DataFrame(np.asarray(X))

        try:
            X_out = X.apply(lambda x: x.apply(JSONTransformer.smash_json))
        except AttributeError:
            raise TypeError("Variable 'X' must be array-like.")
        except TypeError:
            raise TypeError("X's elements must be str")

        return X_out

    @staticmethod
    def smash_json(json_in) -> dict:
        json_out = {}

        if isinstance(json_in, dict):
            for k in json_in.keys():
                if isinstance(json_in[k], (bool, int, float, str)):
                    json_out[k] = json_in[k]

                elif isinstance(json_in[k], (list, dict)):
                    f = JSONTransformer.smash_json(json_in[k])

                    for fkey, fval in f.items():
                        json_out['{0}_{1}'.format(k, fkey)] = fval

                else:
                    print("Nada.")

        elif isinstance(json_in, list):
            for i, v in enumerate(json_in):
                if isinstance(v, (bool, int, float, str)):
                    json_out['{0}'.format(i)] = v

                elif isinstance(v, (list, dict)):
                    f = JSONTransformer.smash_json(v)

                    for fkey, fval in f.items():
                        json_out['{0}_{1}'.format(i, fkey)] = fval

                else:
                    print("Nada.")

        return json_out
