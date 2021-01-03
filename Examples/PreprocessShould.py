import datetime
import pprint as pp

import numpy as np
from sklearn.compose import ColumnTransformer

from NukedML.Preprocess.Categories import UppercaseTransformer, SparkjoyTransformer
from NukedML.Preprocess.JSON import JSONTransformer
from NukedML.Utilities.DataDitto import Faker


def test_preprocess():
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

    ct = ColumnTransformer(
        [
            ('uppercase', UppercaseTransformer(), ['fruit', 'colors'])
        ]
    )

    fakey = ct.fit_transform(X=faked_df)

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


def test_json_fuckery():
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

    jt = JSONTransformer()
    fakey = jt.fit_transform(X=faked_df[['column_1']])


def sparkjoy_test():
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

    st = SparkjoyTransformer()

    return st.fit_transform(X=faked_df[['colors']])


if __name__ == '__main__':
    pp.pprint(test_preprocess())

    pp.pprint(sparkjoy_test())
