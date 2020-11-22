import pytest
import pandas as pd
import datetime

from NukedML.Utilities.DataDitto import Faker


class TestDittoDataFrameShapes:
    def test_empty_df_size_eq_0(self):
        df_json = {}
        df = pd.DataFrame(df_json)
        fakey = Faker.ditto_dataframe(df, size=0)

        assert fakey.shape == (0, 0)

    def test_empty_df_size_gt_0(self):
        df_json = {}
        df = pd.DataFrame(df_json)
        fakey = Faker.ditto_dataframe(df, size=42)

        assert fakey.shape == (0, 0)

    def test_filled_df_size_eq_0(self):
        df_json = {
            'column_1': ['Apple', 'ApPlE', 'banana', 'BANANA', 'Banana', 'Orange', 'Apple', 'APPLE', 'Cherry',
                         'Cherry'],
            'column_2': [1, 2, 3, -1, -2, -3, 4, 6, 1, -1],
            'column_3': [True, True, True, False, True, False, False, False, False, False],
            'column_4': [1.0, 2.0, 3.0, -3.14, -3.14, -42.69, -69.42, -69.42, -100.1, 99],
            'column_5': ['2017-03-25 15:16:45', '2017-03-25 15:16:45', '2017-03-19 15:16:45', '2017-03-22 15:16:45',
                         '2017-03-22 15:16:45', '2017-03-19 15:16:45', '2017-03-30 15:16:45', '2017-03-22 15:16:45',
                         '2017-03-30 15:16:45', '2017-03-25 15:16:45']
        }

        for i, _ in enumerate(df_json['column_5']):
            df_json['column_5'][i] = datetime.datetime.strptime(df_json['column_5'][i], '%Y-%m-%d %H:%M:%S')

        df = pd.DataFrame(df_json)
        fakey = Faker.ditto_dataframe(df, size=0)

        assert fakey.shape == (0, 5)

    def test_filled_df_size_gt_0(self):
        df_json = {
            'column_1': ['Apple', 'ApPlE', 'banana', 'BANANA', 'Banana', 'Orange', 'Apple', 'APPLE', 'Cherry',
                         'Cherry'],
            'column_2': [1, 2, 3, -1, -2, -3, 4, 6, 1, -1],
            'column_3': [True, True, True, False, True, False, False, False, False, False],
            'column_4': [1.0, 2.0, 3.0, -3.14, -3.14, -42.69, -69.42, -69.42, -100.1, 99],
            'column_5': ['2017-03-25 15:16:45', '2017-03-25 15:16:45', '2017-03-19 15:16:45', '2017-03-22 15:16:45',
                         '2017-03-22 15:16:45', '2017-03-19 15:16:45', '2017-03-30 15:16:45', '2017-03-22 15:16:45',
                         '2017-03-30 15:16:45', '2017-03-25 15:16:45']
        }

        for i, _ in enumerate(df_json['column_5']):
            df_json['column_5'][i] = datetime.datetime.strptime(df_json['column_5'][i], '%Y-%m-%d %H:%M:%S')

        df = pd.DataFrame(df_json)
        fakey = Faker.ditto_dataframe(df, size=42)

        assert fakey.shape == (42, 5)


class TestDittoDataFrameColumnSizes:
    def test_empty_df_size_eq_0(self):
        df_json = {}
        df = pd.DataFrame(df_json)
        fakey = Faker.ditto_dataframe(df, size=0)

        assert fakey.columns.size == 0

    def test_empty_df_size_gt_0(self):
        df_json = {}
        df = pd.DataFrame(df_json)
        fakey = Faker.ditto_dataframe(df, size=42)

        assert fakey.columns.size == 0

    def test_filled_df_size_eq_0(self):
        df_json = {
            'column_1': ['Apple', 'ApPlE', 'banana', 'BANANA', 'Banana', 'Orange', 'Apple', 'APPLE', 'Cherry',
                         'Cherry'],
            'column_2': [1, 2, 3, -1, -2, -3, 4, 6, 1, -1],
            'column_3': [True, True, True, False, True, False, False, False, False, False],
            'column_4': [1.0, 2.0, 3.0, -3.14, -3.14, -42.69, -69.42, -69.42, -100.1, 99],
            'column_5': ['2017-03-25 15:16:45', '2017-03-25 15:16:45', '2017-03-19 15:16:45', '2017-03-22 15:16:45',
                         '2017-03-22 15:16:45', '2017-03-19 15:16:45', '2017-03-30 15:16:45', '2017-03-22 15:16:45',
                         '2017-03-30 15:16:45', '2017-03-25 15:16:45']
        }

        for i, _ in enumerate(df_json['column_5']):
            df_json['column_5'][i] = datetime.datetime.strptime(df_json['column_5'][i], '%Y-%m-%d %H:%M:%S')

        df = pd.DataFrame(df_json)
        fakey = Faker.ditto_dataframe(df, size=0)

        assert fakey.columns.size == 5

    def test_filled_df_size_gt_0(self):
        df_json = {
            'column_1': ['Apple', 'ApPlE', 'banana', 'BANANA', 'Banana', 'Orange', 'Apple', 'APPLE', 'Cherry',
                         'Cherry'],
            'column_2': [1, 2, 3, -1, -2, -3, 4, 6, 1, -1],
            'column_3': [True, True, True, False, True, False, False, False, False, False],
            'column_4': [1.0, 2.0, 3.0, -3.14, -3.14, -42.69, -69.42, -69.42, -100.1, 99],
            'column_5': ['2017-03-25 15:16:45', '2017-03-25 15:16:45', '2017-03-19 15:16:45', '2017-03-22 15:16:45',
                         '2017-03-22 15:16:45', '2017-03-19 15:16:45', '2017-03-30 15:16:45', '2017-03-22 15:16:45',
                         '2017-03-30 15:16:45', '2017-03-25 15:16:45']
        }

        for i, _ in enumerate(df_json['column_5']):
            df_json['column_5'][i] = datetime.datetime.strptime(df_json['column_5'][i], '%Y-%m-%d %H:%M:%S')

        df = pd.DataFrame(df_json)
        fakey = Faker.ditto_dataframe(df, size=42)

        assert fakey.columns.size == 5


class TestDittoDataFrameColumnNames:
    def test_empty_df_size_eq_0(self):
        df_json = {}
        df = pd.DataFrame(df_json)
        fakey = Faker.ditto_dataframe(df, size=0)

        assert sorted(fakey) == sorted(df_json.keys())

    def test_empty_df_size_gt_0(self):
        df_json = {}
        df = pd.DataFrame(df_json)
        fakey = Faker.ditto_dataframe(df, size=42)

        assert sorted(fakey) == sorted(df_json.keys())

    def test_filled_df_size_eq_0(self):
        df_json = {
            'column_9': ['Apple', 'ApPlE', 'banana', 'BANANA', 'Banana', 'Orange', 'Apple', 'APPLE', 'Cherry',
                         'Cherry'],
            'column_2': [1, 2, 3, -1, -2, -3, 4, 6, 1, -1],
            'column_3': [True, True, True, False, True, False, False, False, False, False],
            'column_4': [1.0, 2.0, 3.0, -3.14, -3.14, -42.69, -69.42, -69.42, -100.1, 99],
            'column_5': ['2017-03-25 15:16:45', '2017-03-25 15:16:45', '2017-03-19 15:16:45', '2017-03-22 15:16:45',
                         '2017-03-22 15:16:45', '2017-03-19 15:16:45', '2017-03-30 15:16:45', '2017-03-22 15:16:45',
                         '2017-03-30 15:16:45', '2017-03-25 15:16:45']
        }

        for i, _ in enumerate(df_json['column_5']):
            df_json['column_5'][i] = datetime.datetime.strptime(df_json['column_5'][i], '%Y-%m-%d %H:%M:%S')

        df = pd.DataFrame(df_json)
        fakey = Faker.ditto_dataframe(df, size=0)

        assert sorted(fakey) == sorted(df_json.keys())

    def test_filled_df_size_gt_0(self):
        df_json = {
            'column_9': ['Apple', 'ApPlE', 'banana', 'BANANA', 'Banana', 'Orange', 'Apple', 'APPLE', 'Cherry',
                         'Cherry'],
            'column_2': [1, 2, 3, -1, -2, -3, 4, 6, 1, -1],
            'column_3': [True, True, True, False, True, False, False, False, False, False],
            'column_4': [1.0, 2.0, 3.0, -3.14, -3.14, -42.69, -69.42, -69.42, -100.1, 99],
            'column_5': ['2017-03-25 15:16:45', '2017-03-25 15:16:45', '2017-03-19 15:16:45', '2017-03-22 15:16:45',
                         '2017-03-22 15:16:45', '2017-03-19 15:16:45', '2017-03-30 15:16:45', '2017-03-22 15:16:45',
                         '2017-03-30 15:16:45', '2017-03-25 15:16:45']
        }

        for i, _ in enumerate(df_json['column_5']):
            df_json['column_5'][i] = datetime.datetime.strptime(df_json['column_5'][i], '%Y-%m-%d %H:%M:%S')

        df = pd.DataFrame(df_json)
        fakey = Faker.ditto_dataframe(df, size=42)

        assert sorted(fakey) == sorted(df_json.keys())
