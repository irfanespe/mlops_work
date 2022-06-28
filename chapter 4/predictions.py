import argparse
from fileinput import filename
import pickle
import pandas as pd


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PUlocationID', 'DOlocationID']


def read_data(filename):
    print("get data from cloud")
    df = pd.read_parquet(filename)
    
    print("feature engineering")
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def predict(df, year, month):
    dicts = df[categorical].to_dict(orient='records')
    
    print("get predictions")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    # create id
    # year = '2021'
    # month = '02'
    df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')

    # construnct output dataframe
    df_result = df[['ride_id']].copy()
    df_result['predictions'] = y_pred

    print('error mean from predictions : {}'.format(y_pred.mean()))

    output_file = 'predictions.parquet'

    # save output to parquet             
    print("save output file into parquet")
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

def main(filename, year, month):
    df = read_data(filename)
    predict(df,year, month)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        default="2021",
        help="the location where the processed NYC taxi trip data was saved."
    )
    parser.add_argument(
        "--month",
        default="03",
        help="the location where the processed NYC taxi trip data was saved."
    )
    args = parser.parse_args()

    year = args.year
    month = args.month

    filenames = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-{month}.parquet'
    main(filenames, year, month)