import numpy as np
import pandas as pd


class Faker:
    def __init__(self, config_json, seed=6):
        self._rng = np.random.RandomState(seed)
        self._config_json = config_json
        self._config = {k: pd.Series(v) for k, v in self._config_json.items()}

    def fabricate(self, size=100):
        faked_df = pd.DataFrame({})

        for k, v in self._config.items():
            if pd.api.types.is_integer_dtype(v):
                faked_df[k] = self._rng.randint(low=v.min(), high=v.max()+1, size=size)
            elif pd.api.types.is_float_dtype(v):
                faked_df[k] = self._rng.uniform(low=v.min(), high=v.max(), size=size)
            elif pd.api.types.is_string_dtype(v):
                faked_df[k] = self._rng.choice(a=v, size=size)
            elif pd.api.types.is_bool_dtype(v):
                faked_df[k] = self._rng.choice(a=[True, False], size=size)
            elif pd.api.types.is_datetime64_any_dtype(v):
                dates = pd.date_range(start=v.min(), end=v.max(), periods=(size*1.25)//1)
                faked_df[k] = self._rng.choice(a=dates, size=size)

        return faked_df[sorted(faked_df)]

    @staticmethod
    def ditto_dataframe(df, size=100, seed=6):
        config_json = {}

        for c in sorted(df.select_dtypes(include=['number', 'datetime', 'datetimetz'])):
            config_json[c] = [df[c].min(), df[c].max()]

        for c in sorted(df.select_dtypes(include=['object'])):
            config_json[c] = df[c].apply(str.upper).unique().tolist()  # Do we really want to uppercase here?

        for c in sorted(df.select_dtypes(include=['bool'])):
            config_json[c] = [True, False]

        dittoer = Faker(config_json=config_json, seed=seed)
        faked_df = dittoer.fabricate(size=size)

        return faked_df[sorted(faked_df)]

    @staticmethod
    def ditto_series(*args, **kwargs):
        print("Don't be silly, just use sampling.")

    @staticmethod
    def ditto_json(*args, **kwargs):
        print("Coming soon.")
