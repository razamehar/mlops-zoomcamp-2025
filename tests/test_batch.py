import pandas as pd
from datetime import datetime
from home6 import prepare_data


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),          # Valid (9 min)
        (1, 1, dt(1, 2), dt(1, 10)),                # Valid (8 min)
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),       # Valid (59 sec → < 1 min → filtered out)
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),           # Invalid (over 60 min → filtered out)
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    actual_df = prepare_data(df, categorical)

    # Expected: only first 2 rows pass (between 1 and 60 mins)
    expected_data = [
        ('-1', '-1', 9.0),
        ('1', '1', 8.0),
    ]
    expected_df = pd.DataFrame(expected_data, columns=['PULocationID', 'DOLocationID', 'duration'])

    # Compare as dicts (easier and avoids index mismatch)
    assert actual_df[['PULocationID', 'DOLocationID', 'duration']].to_dict(orient='records') == \
           expected_df.to_dict(orient='records')
