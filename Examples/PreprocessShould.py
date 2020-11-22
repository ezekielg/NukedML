import datetime
import numpy as np
import pandas as pd
import pprint as pp

from NukedML.Utilities.DataDitto import Faker
from NukedML.Preprocess.Categories import UppercaseTransformer


def test_preprocess():
    df_json = {
        'column_1': ['Apple', 'ApPlE', 'banana', 'BANANA', 'Banana', 'Orange', 'Apple', 'APPLE', 'Cherry', 'Cherry'],
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

    fakey = Faker.ditto_dataframe(df, size=1000)

    ut = UppercaseTransformer()

    ut.fit_transform(X=fakey[['column_1']])

    return fakey


def make_ndarray_from_dataframe():
    test = {
        'age': [1, 99],
        'levels': [-0.234, 9.431],
        'dates': [datetime.datetime.strptime('2020-04-20 04:20:00', '%Y-%m-%d %H:%M:%S'),
                  datetime.datetime.strptime('2020-05-05 05:15:42', '%Y-%m-%d %H:%M:%S')],
        'fruit': ['apple', 'banana', 'pineapple', 'cherry'],
        'really': [True, False],
        'colors': ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    }

    f = Faker(config_json=test, seed=420)

    faked_df = f.fabricate(size=1000)

    test = faked_df[['fruit', 'colors']]
    test_array = np.asarray(test)

    return test_array


if __name__ == '__main__':
    fakey = make_ndarray_from_dataframe()
    pp.pprint(fakey.head())

    print('\n\n')

    pp.pprint(fakey[['fruit', 'colors']].head())

    pp.pprint(type(fakey[['fruit', 'colors']]))

    pp.pprint(np.array(fakey[['fruit', 'colors']]))
