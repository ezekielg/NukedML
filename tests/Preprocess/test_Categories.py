import pytest
import pandas as pd
import datetime

from sklearn.exceptions import NotFittedError
from NukedML.Preprocess.Categories import RegexTransformer, UppercaseTransformer, SparkjoyTransformer

default_json = {
    'column_1': ['Apple', 'ApPlE', 'banana', 'BANANA', 'Banana', 'Orange', 'Apple', 'APPLE', 'Cherry', 'Cherry'],
    'column_2': [1, 2, 3, -1, -2, -3, 4, 6, 1, -1],
    'column_3': [True, True, True, False, True, False, False, False, False, False],
    'column_4': [1.0, 2.0, 3.0, -3.14, -3.14, -42.69, -69.42, -69.42, -100.1, 99],
    'column_5': ['2017-03-25 15:16:45', '2017-03-25 15:16:45', '2017-03-19 15:16:45', '2017-03-22 15:16:45',
                 '2017-03-22 15:16:45', '2017-03-19 15:16:45', '2017-03-30 15:16:45', '2017-03-22 15:16:45',
                 '2017-03-30 15:16:45', '2017-03-25 15:16:45']
}

for i, _ in enumerate(default_json['column_5']):
    default_json['column_5'][i] = datetime.datetime.strptime(default_json['column_5'][i], '%Y-%m-%d %H:%M:%S')


class TestTransformers:
    @pytest.mark.parametrize(
        "X, X_out",
        [({}, pd.DataFrame({}))],
        ids=["Empty DataFrame"]
    )
    @pytest.mark.parametrize(
        "transformer",
        [RegexTransformer(), SparkjoyTransformer(), UppercaseTransformer()],
        ids=["RegexTransformer", "SparkjoyTransformer", "UppercaseTransformer"]
    )
    def test_transform_without_fitting(self, transformer, X, X_out):
        with pytest.raises(NotFittedError):
            assert transformer.transform(X)

    @pytest.mark.parametrize(
        "transformer",
        [RegexTransformer(), SparkjoyTransformer(), UppercaseTransformer()],
        ids=["RegexTransformer", "SparkjoyTransformer", "UppercaseTransformer"]
    )
    def test_fit(self, transformer, X, y=None, **kwargs):
        transformer.fit(X, y, **kwargs)
