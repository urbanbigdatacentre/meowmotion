# import json
import os
from datetime import datetime
from multiprocessing import Pool

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

from meowmotion.data_formatter import (  # getLoadBalancedBuckets,; readRawData,
    featureEngineering,
    generateTrajStats,
    processTripData,
)
from meowmotion.process_data import readJsonFiles, saveFile

load_dotenv()

city = "Glasgow"
shape_file = "U:/Projects/Huq/Faraz/huq_city_data/Shapefiles/msoa_intzone_boundaries/glasgow/msoa_glasgow.shp"
hldf_file = (
    "U:\\Projects\\Huq\\Faraz\\huq_city_data\\homeResults\\Glasgow_2019_results.csv"
)
adult_population_file = "U:/Projects/Huq/od_project/outputs/imd_population_estimates/Glasgow_IMD_summary.csv"
year = 2019
root = f"U:/Operations/SCO/Faraz/huq_compiled/{city}/{year}"
trip_point_data_file = (
    "U:\\Projects\\Huq\\Faraz\\package_testing\\trip_points\\trip_points.csv"
)
na_flow_data_file = "U:\\Projects\\Huq\\Faraz\\package_testing\\na_flows\\na_flows.csv"
cpu_cores = 12
impr_acc = 100
month = 1
radius = 500
time_th = 5  # in minutes
output_dir = "U:/Projects/Huq/Faraz/package_testing"
org_loc_cols = ("org_lng", "org_lat")
dest_loc_cols = ("dest_lng", "dest_lat")


def readData():
    ##################################################################################
    #                                                                                #
    #                           Fetching Data From DB                                #
    #                                                                                #
    ##################################################################################

    start_time = datetime.now()

    print(f"{start_time}: Fetching data from Database")
    print(f"{datetime.now()}: Fetching data from Json Files")
    month_files = [
        f for f in os.listdir(root) if f.split("_")[-1].split(".")[0] == str(month)
    ]  # Getting the files for the specific month
    args = [(root, mf) for mf in month_files]
    with Pool(cpu_cores) as p:
        results = p.starmap(readJsonFiles, args)

    print(f"{datetime.now()}: Data Concatination")
    df = pd.concat(results)  # Concatinating the data fetched from the database

    del results  # Deleting the results to free up the memory
    print(f"{datetime.now()}: Data Concatination Completed")
    print(f"{datetime.now()}: Data fetching completed\n\n")
    print(f"Number of Records: {df.shape[0]}")

    # df = pd.read_csv('data/geolife_sample_data.csv')

    return df


if __name__ == "__main__":

    bus_stops_shape_file = os.getenv("SHAPE_FILE_BUS")
    train_stops_shape_file = os.getenv("SHAPE_FILE_TRAIN")
    metro_stops_shape_file = os.getenv("SHAPE_FILE_METRO")
    gspace_file = os.getenv("SHAPE_FILE_GREEN_SPACES")

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

    print(f"{datetime.now()}: Reading raw data")
    raw_df = readData()  # Reading the data from the database

    print(f"{datetime.now()}: Reading Trip Point Data")
    tp_df = pd.read_csv(trip_point_data_file)  # Reading the data from the database

    print(f"{datetime.now()}: Reading NA-flow Data")
    naf_df = pd.read_csv(na_flow_data_file)  # Reading the data from the database

    print(f"{datetime.now()}: Processing Trip Point Data")

    trip_df = processTripData(trip_point_df=tp_df, na_flow_df=naf_df, raw_df=raw_df)

    print(f"{datetime.now()}: Feature Engineering")
    trip_df = featureEngineering(
        trip_df=trip_df, shape_files=shape_files, cores=cpu_cores
    )

    print(f"{datetime.now()}: Saving Processed Data")
    saveFile(f"{output_dir}/tmd", "processed_trip_points_data.csv", trip_df)

    print(f"{datetime.now()}: Generating Huq Stats")
    trip_stats_df = generateTrajStats(trip_df)
    saveFile(f"{output_dir}/tmd", "trip_stats_data.csv", trip_stats_df)
    print(f"{datetime.now()}: Finished.")
