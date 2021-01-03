import datetime

import pandas as pd
import pytest

from NukedML.Utilities.DataDitto import Faker

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


class TestDittoDataFrame:
    @pytest.mark.parametrize(
        "df_json, size, expected",
        [
            ({}, 0, (0, 0)),
            ({}, 42, (0, 0)),
            (default_json, 0, (0, 5)),
            (default_json, 42, (42, 5))
        ]
    )
    def test_data_frame_shapes(self, df_json, size, expected):
        df = pd.DataFrame(df_json)
        fakey = Faker.ditto_dataframe(df, size=size)

        assert fakey.shape == expected

    @pytest.mark.parametrize(
        "df_json, size",
        [
            ({}, 0),
            ({}, 42),
            (default_json, 0),
            (default_json, 42)
        ]
    )
    def test_data_frame_columns(self, df_json, size):
        df = pd.DataFrame(df_json)
        fakey = Faker.ditto_dataframe(df, size=size)

        assert sorted(fakey) == sorted(df_json.keys())
