import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class JSONTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._mapping = {}  # Shows the relationship between original keys and new keys.

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pass

    @staticmethod
    def smash_json(json_in: dict) -> dict:
        json_out = {}

        for k in json_in.keys():
            if isinstance(json_in[k], (bool, int, float, str)):
                json_out[k] = json_in[k]

            elif isinstance(json_in[k], dict):
                f = JSONTransformer.smash_json(json_in[k])

                for fkey, fval in f.items():
                    json_out['{0}_{1}'.format(k, fkey)] = fval

            elif isinstance(json_in[k], list):
                for i, v in enumerate(json_in[k]):
                    if isinstance(v, dict):
                        f = JSONTransformer.smash_json(v)

                        for fkey, fval in f.items():
                            json_out['{0}_{1}_{2}'.format(k, i, fkey)] = fval

                    elif isinstance(v, (bool, int, float, str)):
                        json_out['{0}_{1}'.format(k, i)] = v

                    elif isinstance(v, list):
                        print("Not sure how to handle lists yet, to be honest.")

            else:
                print("Nada.")

        return json_out
