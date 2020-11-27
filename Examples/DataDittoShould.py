import datetime
import pandas as pd
import pprint as pp

from NukedML.Utilities.DataDitto import Faker


def test_faker_class():
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
    fakey = Faker.ditto_dataframe(df, size=100)

    pp.pprint(fakey)


def test_config_json():
    test = {
        'a': [1, 99],
        'b': [-0.234, 9.431],
        'c': [datetime.datetime.strptime('2020-04-20 04:20:00', '%Y-%m-%d %H:%M:%S'),
              datetime.datetime.strptime('2020-05-05 05:15:42', '%Y-%m-%d %H:%M:%S')],
        'x': ['apple', 'banana', 'OraNgE'],
        'okay': [True, False]
    }

    f = Faker(config_json=test, seed=420)

    pp.pprint(f.fabricate(size=100))


if __name__ == '__main__':
    test_faker_class()
    test_config_json()
