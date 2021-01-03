import datetime
import pprint as pp

import pandas as pd
from sklearn.compose import ColumnTransformer

from NukedML.Preprocess.JSON import JSONTransformer
from NukedML.Utilities.DataDitto import Faker


def smash_json_example():
    test = {
        'age': [1, 99],
        'levels': [-0.234, 9.431],
        'dates': [datetime.datetime.strptime('2020-04-20 04:20:00', '%Y-%m-%d %H:%M:%S'),
                  datetime.datetime.strptime('2020-05-05 05:15:42', '%Y-%m-%d %H:%M:%S')],
        'fruit': ['apple', 'banana', 'pineapple', 'cherry'],
        'really': [True, False],
        'colors': ['red', 'orange', 'yellow', 'green', 'blue', 'purple'],
        'jason_1': ['{"key_1": "value_1", "key_2": "value_2", "key_3": "value_3"}',
                    '{"key_z": "value_a", "key_b": "value_b", "key_c": "value_c", "key_surprise": {"key_x1": [99, 98, '
                    '97], "key_x2": {"a": 1, "b": 2}}}'
                    ],
        'jason_2': ['{"key_1": "value_1", "key_2": "value_2", "key_3": "value_3"}',
                    '{"key_z": "value_a", "key_b": "value_b", "key_c": "value_c", "key_surprise": {"key_x1": [99, 98, '
                    '97], "key_x2": {"a": 1, "b": 2}}}'
                    ]
    }

    f = Faker(config_json=test, seed=420)

    faked_df = f.fabricate(size=1000)

    ct = ColumnTransformer(
        [
            ('jaysawn', JSONTransformer(), ['jason_1', 'jason_2'])
        ]
    )

    fakey = ct.fit_transform(X=faked_df)

    pp.pprint(pd.DataFrame(fakey).head())
    pp.pprint(type(fakey[0][0]))


if __name__ == '__main__':
    smash_json_example()

