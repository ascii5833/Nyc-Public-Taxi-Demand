import pandas as pd
from pathlib import Path

#configs
ROOT_DIR = Path(__file__).resolve().parent.parent
save_path_raw = ROOT_DIR / "data" / "raw" / "taxi_demand_features_combined_raw.parquet"
save_path_processed = ROOT_DIR / "data" / "processed" / "taxi_demand_features_combined.parquet"


#read the data files
def read_data(data_path : str)-> list[pd.DataFrame]:
    '''
    fetch data from a parquet file.
    
    Inptus:
    data_path: path string to where the data is scored
    
    returns:
    dataframe containing the read document
    '''
    
    df = pd.read_parquet(data_path)
    return df


def process_data(df_list : list[pd.DataFrame]) -> pd.DataFrame:
    '''
    Generate combined dataframe from the timeseries taxi dataset list.
    
    Inputs:
    df_list: list containing pandas dataframes for each month
    
    returns:
    combined dataframe
    '''
    combined_df = pd.concat(df_list, axis = 0)
    
    combined_df['tpep_pickup_datetime'] = pd.to_datetime(combined_df['tpep_pickup_datetime'])
    
    combined_df = combined_df[
        (combined_df["tpep_pickup_datetime"] >= "2026-01-01") &
        (combined_df["tpep_pickup_datetime"] < "2026-06-01")
    ].copy()
    
    #pickup hour
    combined_df['pickup_hour'] = combined_df['tpep_pickup_datetime'].dt.floor('h')
    #aggregate pickup hour and location lolz
    hourly_df = (
                combined_df.groupby(['pickup_hour', 'PULocationID'])
                .agg(
                    number_pickups = ("VendorID", "count")
                )
                .reset_index()
                )

    #index with all hours range like 01-01-2026 to the maximum date in the dataset
    all_hours = pd.date_range(
        start = hourly_df['pickup_hour'].min(),
        end = hourly_df['pickup_hour'].max(),
        freq = 'h'
    )

    all_locations = hourly_df['PULocationID'].unique()

    #index with both combined
    combined_index = pd.MultiIndex.from_product(
        [all_hours, all_locations], names = ['pickup_hour', 'PULocationID']
    )

    hourly_df = (
        hourly_df.set_index(['pickup_hour', 'PULocationID'])
        .reindex(combined_index, fill_value = 0)
        .reset_index()
    )
    
    hourly_df = hourly_df.sort_values(["PULocationID", "pickup_hour"])
    #hour and day of week on agg data
    hourly_df['hour'] = hourly_df['pickup_hour'].dt.hour
    hourly_df['day_of_week'] = hourly_df['pickup_hour'].dt.dayofweek
    #previous hour
    hourly_df['previous_hour'] = hourly_df.groupby(['PULocationID'])['number_pickups'].shift(1)
    #past 24 hours(previous day)
    hourly_df['previous_day'] = (hourly_df.groupby('PULocationID')['number_pickups'].shift(24))
    #previous week demand
    hourly_df['previous_week'] = (hourly_df.groupby('PULocationID')['number_pickups'].shift(24*7))
    #rolling 24 hour
    hourly_df['rolling_24h'] = (hourly_df.groupby('PULocationID')['number_pickups']
        .transform(lambda x : x.shift(1).rolling(24).mean()))
    #rolling 3 hour
    hourly_df['rolling_3h'] = (hourly_df.groupby('PULocationID')['number_pickups'].transform(lambda x: x.shift(1).rolling(3).mean()))
    
        
    return hourly_df   
        

                                        
    
#path_list
data_list = [ROOT_DIR / "data" / "raw" / f"yellow_tripdata_2026-0{i+1}.parquet" for i in range(5)] 



if __name__ == "__main__":
    print('Processing datasets ...')
    df_list = list(map(read_data, data_list))
    combined_df = process_data(df_list)
    print('Datasets processed...')
    print('\n Saving datasets...')
    combined_df.to_parquet(path = save_path_raw, index = False)
    
    #remove na values
    combined_df = combined_df.dropna().reset_index(drop = True)
    combined_df.to_parquet(path = save_path_processed, index = False)
    print("\n Datasets saved")
    