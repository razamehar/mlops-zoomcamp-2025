import prefect
from prefect import flow, task
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pickle


@task
def load_data():
    df = pd.read_parquet('yellow_tripdata_2023-03.parquet')
    print("Data loaded")
    return df


@task
def preprocess_data(df):
    print(f"Number of records before filtering: {len(df)}")
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df['duration'].dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"Number of records after filtering: {len(df)}")
    return df


@task
@task
def train(df):
    features = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    target = df['duration']

    dv = DictVectorizer()
    X_train = dv.fit_transform(features)

    lr = LinearRegression()
    lr.fit(X_train, target)

    print(f"Intercept: {lr.intercept_}")

    # Save DictVectorizer
    with open('dv.pkl', 'wb') as f_out:
        pickle.dump(dv, f_out)

    # Save LinearRegression model
    with open('model.pkl', 'wb') as f_out:
        pickle.dump(lr, f_out)

    print("Model and vectorizer saved to 'model.pkl' and 'dv.pkl'")
    return dv, lr


@flow
def main():
    print(f"Prefect version: {prefect.__version__}")

    raw_df = load_data()
    cleaned_df = preprocess_data(raw_df)
    train(cleaned_df)


if __name__ == "__main__":
    main()