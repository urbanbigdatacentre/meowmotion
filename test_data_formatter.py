import json
import os
from datetime import datetime

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

from meowmotion.data_formatter import (
    featureEngineering,
    generateTrajStats,
    getLoadBalancedBuckets,
    readRawData,
    readTripData,
)

load_dotenv()

if __name__ == "__main__":

    print(f"{datetime.now()}: Starting Process...")
    city = os.getenv("CITY")  # "Glasgow"
    years = json.loads(os.getenv("YEARS"))  # [2022, 2023]
    CORES = int(os.getenv("CORES"))
    bus_stops_shape_file = os.getenv("SHAPE_FILE_BUS")
    train_stops_shape_file = os.getenv("SHAPE_FILE_TRAIN")
    metro_stops_shape_file = os.getenv("SHAPE_FILE_METRO")
    gspace_file = os.getenv("SHAPE_FILE_GREEN_SPACES")
    output_dir = os.getenv("OUTPUT_DIR")

    print(f"{datetime.now()}: Reading Bus Stops Shape File")
    bus_stops = gpd.read_file(bus_stops_shape_file)
    bus_stops.sindex

    print(f"{datetime.now()}: Reading Train Stops Shape File")
    train_stops = gpd.read_file(train_stops_shape_file)
    train_stops.sindex

    print(f"{datetime.now()}: Reading Metro Stops Shape File")
    metro_stops = gpd.read_file(metro_stops_shape_file)
    metro_stops.sindex

    print(f"{datetime.now()}: Reading Green Space Shape File")
    green_space_df = gpd.read_file(gspace_file)
    if green_space_df.crs is not None and green_space_df.crs != "EPSG:4326":
        green_space_df = green_space_df.to_crs(epsg=4326)
    green_space_df.sindex
    print(f"{datetime.now()}: Finished Reading Shape Files")

    shape_files = [bus_stops, train_stops, metro_stops, green_space_df]
    for year in years:
        print(
            f"""
        City: {city}
        Year: {year}
        """
        )
        raw_data_dir = f"{os.getenv('RAW_DATA_DIR')}/{city}/{year}"
        trip_data_dir = f"{os.getenv('TRIP_DATA_DIR')}"
        print(f"{datetime.now()}: Reading raw data")
        raw_df = readRawData(raw_data_dir, CORES)
        print(f"{datetime.now()}: Reading trip & NA flow data")
        trip_df = readTripData(year, city, trip_data_dir)
        print(f"{datetime.now()}: Merging raw data with trip data to get datetime")
        trip_df = trip_df.merge(
            raw_df[["uid", "datetime", "lat", "lng"]],
            on=["uid", "lat", "lng"],
            how="left",
        )
        print(f"{datetime.now()}: Converting datetime to datetime object")
        trip_df["datetime"] = pd.to_datetime(trip_df["datetime"])
        trip_df = trip_df[
            trip_df["datetime"].between(
                trip_df["org_leaving_time"], trip_df["dest_arival_time"]
            )
        ].reset_index(drop=True)
        assert trip_df["datetime"].isna().sum() == 0
        print(f"{datetime.now()}: Validation Done")
        del raw_df
        print(f"{datetime.now()}: Get Load Balanced Buckets")
        df_collection = getLoadBalancedBuckets(trip_df, CORES)
        del trip_df
        print(f"{datetime.now()}: Feature Engineering")
        trip_df = featureEngineering(df_collection, shape_files, CORES)
        del df_collection
        print(f"{datetime.now()}: Saving Processed Data")
        os.makedirs(f"{output_dir}/{city}/{year}", exist_ok=True)
        trip_df.to_csv(
            f"{output_dir}/{city}/{year}/processed_trip_points_data.csv",
            index=False,
        )
        print(f"{datetime.now()}: Generating Huq Stats")
        huq_stats = generateTrajStats(trip_df)
        huq_stats.to_csv(
            f"{output_dir}/{city}/{year}/huq_stats_df_for_ml.csv",
            index=False,
        )

    print(f"{datetime.now()}: Finished.")
