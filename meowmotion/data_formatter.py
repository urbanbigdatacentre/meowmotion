import ast
import json
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
from haversine import Unit, haversine
from scipy import stats
from shapely.geometry import Point
from tqdm import tqdm

from meowmotion.process_data import getLoadBalancedBuckets, readJsonFiles


def readTripData(year: int, city: str, data_dir: str) -> pd.DataFrame:
    """
    Description:
    Reads trip data for a given year and city, processes
    trip points, and merges it with origin-destination flow
     data to get the name of Origin and Destination.

    Parameters:
    - year (int): The year of the trip data.
    - city (str): The name of the city for which the trip data
    is being retrieved.

    Returns:
    - pd.DataFrame: A DataFrame containing processed trip data
    with latitude and longitude information for each trip point,
    along with origin and destination coordinates.

    Example:
    >>> df = readTripData(2023, "London")
    >>> print(df.head())
    """

    trip_file_path = f"{data_dir}/{city}/{year}/trip_points"
    trip_df = pd.read_csv(f"{trip_file_path}/trip_points_500m_{year}.csv")
    trip_df["trip_points"] = trip_df["trip_points"].apply(ast.literal_eval)
    trip_df = trip_df.explode("trip_points")
    trip_df.dropna(subset=["trip_points"], inplace=True)
    trip_df[["lat", "lng"]] = pd.DataFrame(
        trip_df["trip_points"].tolist(), index=trip_df.index
    )
    trip_df.drop(columns=["trip_points"], inplace=True)
    na_flows_file_path = f"{data_dir}/{city}/{year}/na_flows"
    tdf = pd.read_csv(na_flows_file_path + f"/na_flows_500m_{year}.csv")
    tdf["org_arival_time"] = pd.to_datetime(tdf["org_arival_time"])
    tdf["org_leaving_time"] = pd.to_datetime(tdf["org_leaving_time"])
    tdf["dest_arival_time"] = pd.to_datetime(tdf["dest_arival_time"])
    trip_df = trip_df.merge(
        tdf[
            [
                "uid",
                "trip_id",
                "org_lat",
                "org_lng",
                "dest_lat",
                "dest_lng",
                "org_arival_time",
                "org_leaving_time",
                "dest_arival_time",
            ]
        ],
        on=["uid", "trip_id"],
        how="left",
    )
    trip_df = trip_df[
        (trip_df["dest_arival_time"] - trip_df["org_leaving_time"]).dt.total_seconds()
        / 3600
        <= 24
    ]
    return trip_df


def readRawData(data_dir: str, cores: int = max(1, cpu_count() // 2)) -> pd.DataFrame:
    """
    Description:
    Reads and compiles raw JSON data files for a given year and city by parallel processing
    multiple monthly files.

    Parameters:
    - cores (int): The number of CPU cores to be used for parallel processing.
    - data_dir (str): The directory where the raw data files are stored.

    Returns:
    - pd.DataFrame: A DataFrame containing compiled raw data from all monthly files.

    Example:
    >>> df = readRawData(2023, "path_to_root/city/year")
    >>> print(df.head())
    """
    root = data_dir
    month_files = os.listdir(root)
    args = [(root, mf) for mf in month_files]
    with Pool(cores) as p:
        df = p.starmap(readJsonFiles, args)
    return pd.concat(df, ignore_index=True)


def processData(df: pd.DataFrame, shape_files: List[gpd.GeoDataFrame]) -> pd.DataFrame:
    """
    Description:
        Processes the trip data by calculating various features such as speed, acceleration,
        jerk, angular deviation, and straightness index. It also checks if the trip starts or ends
        at a bus, train, or metro stop and if it is found at a green space.
        The function also filters out trips with less than 5 impressions and calculates the time taken
        and distance covered for each trip.

    Parameters:
        df (pd.DataFrame): DataFrame containing trip data with latitude and longitude columns.
        shape_files (List[gpd.GeoDataFrame]): List of GeoDataFrames containing bus, train, metro stops and green space polygons.

    Returns:
        pd.DataFrame: Processed DataFrame with additional features and filtered trips.

    Example:
        >>> processed_df = processData(df, shape_files)
        >>> print(processed_df.head())
    """

    temp_df = df.copy()

    # In some trips, for very same timestamp, we observed multiple datapoints. To deal with that, dropping all the duplicate timestamps and keeping the first one
    temp_df = (
        temp_df.groupby(["uid", "trip_id"])
        .apply(lambda x: x.drop_duplicates(subset=["datetime"], keep="first"))
        .reset_index(drop=True)
    )

    # Counting number of impressions in each trip
    temp_df["num_of_impressions"] = temp_df.groupby(["uid", "trip_id"])[
        ["datetime"]
    ].transform(lambda x: len(x))

    # Filtering every trip with less than 5 impressions
    temp_df = temp_df[temp_df.num_of_impressions >= 5]

    # Cacluting the time taken (in seconds) to move from point to the next one in a trip
    temp_df["time_taken"] = temp_df.groupby(["uid", "trip_id"])["datetime"].transform(
        lambda x: x.diff().dt.total_seconds()
    )

    # Calculating Distance covered from previous point to current point
    temp_df["prev_lat"] = temp_df.groupby(["uid", "trip_id"])["lat"].transform(
        lambda x: x.shift(1)
    )
    temp_df["prev_long"] = temp_df.groupby(["uid", "trip_id"])["lng"].transform(
        lambda x: x.shift(1)
    )
    temp_df.dropna(subset=["prev_lat"], inplace=True)
    temp_df["distance_covered"] = temp_df.apply(
        lambda row: haversine(
            (row["lat"], row["lng"]),
            (row["prev_lat"], row["prev_long"]),
            unit=Unit.METERS,
        ),
        axis=1,
    )

    # Calculating Speed with which the distance was covered
    temp_df["speed"] = temp_df.distance_covered / temp_df.time_taken
    temp_df["date"] = temp_df["datetime"].dt.date
    temp_df["hour"] = temp_df["datetime"].dt.hour
    temp_df = temp_df.astype({"date": "datetime64[ns]"})
    assert temp_df[np.isinf(temp_df["speed"])].shape[0] == 0
    temp_df["speed_z_score"] = temp_df.groupby(["uid", "trip_id"])[["speed"]].transform(
        lambda x: abs(stats.zscore(x))
    )

    # Calculate Acceleration
    temp_df["new_speed"] = temp_df.groupby(["uid", "trip_id"], group_keys=False)[
        ["speed", "speed_z_score"]
    ].apply(removeOutlier)["speed"]
    temp_df["accelaration"] = temp_df.groupby(["uid", "trip_id"])[
        "new_speed"
    ].transform(lambda x: x.shift(+1))
    temp_df["accelaration"] = (
        temp_df["new_speed"] - temp_df["accelaration"]
    ) / temp_df["time_taken"]
    temp_df["jerk"] = temp_df.groupby(["uid", "trip_id"])["accelaration"].transform(
        lambda x: x.diff()
    )
    temp_df["jerk"] = temp_df["jerk"] / temp_df["time_taken"]
    temp_df["bearing"] = calculateBearing(
        temp_df["prev_lat"], temp_df["prev_long"], temp_df["lat"], temp_df["lng"]
    )
    temp_df["angular_deviation"] = temp_df.groupby(["uid", "trip_id"])[
        "bearing"
    ].transform(lambda x: np.abs(x.diff()))
    temp_df["month"] = temp_df["datetime"].dt.month
    temp_df["hour"] = temp_df["datetime"].dt.hour
    temp_df["is_weekend"] = temp_df["datetime"].dt.dayofweek
    temp_df["is_weekend"] = temp_df.is_weekend.map({5: 1, 6: 1}).fillna(0)
    temp_df = temp_df.astype({"is_weekend": "int32"})
    conditions = [
        (temp_df.hour >= 0) & (temp_df.hour < 6),
        (temp_df.hour >= 6) & (temp_df.hour < 12),
        (temp_df.hour >= 12) & (temp_df.hour < 18),
        (temp_df.hour >= 18) & (temp_df.hour <= 23),
    ]
    category = [0, 1, 2, 3]  # Night, Morning, Afternoon, Evening
    temp_df["hour_category"] = np.select(conditions, category)

    group = temp_df.groupby(["uid", "trip_id"])
    temp_df["trip_group"] = group.ngroup()
    temp_df.sort_values(by=["trip_group"], ascending=True, inplace=True)

    ###########################################################################################

    new_df = []
    for i in tqdm(
        range(temp_df["trip_group"].max()), desc="Adding Stops and Green Space Features"
    ):
        tdf = temp_df[temp_df["trip_group"] == i]
        tdf = tdf.copy()
        tdf.sort_values(by=["datetime"], ascending=True, inplace=True)
        tdf["start_end_at_bus_stop"] = checkIfNearStop(tdf, shape_files[0])
        tdf["start_end_at_train_stop"] = checkIfNearStop(tdf, shape_files[1])
        tdf["start_end_at_metro_stop"] = checkIfNearStop(tdf, shape_files[2])
        tdf["found_at_green_space"] = checkIfAtGrrenSpace(tdf, shape_files[3])
        tdf["straightness_index"] = calculateStraightnessIndex(tdf)
        new_df.append(tdf)

    temp_df = pd.concat(new_df)
    del new_df
    del tdf
    temp_df = temp_df[
        temp_df["straightness_index"] <= 1
    ]  # Filtering out the trips with straightness index greater than 1
    temp_df.drop(columns=["trip_group"], inplace=True)

    ###########################################################################################
    return temp_df


def removeOutlier(group: pd.DataFrame) -> pd.DataFrame:
    """
    Description:
        Removes outliers from the speed column of the DataFrame based on z-score.
        If the z-score is greater than or equal to 3, the speed is replaced with the median speed.

    parameters:
        group (pd.DataFrame): DataFrame containing speed and speed_z_score columns.

    Returns:
        pd.DataFrame: DataFrame with outliers removed from the speed column.

    example:
        >>> print(removeOutlier(df))
    """

    group_speed = group["speed"]
    group_z_score = group["speed_z_score"]
    group_median_speed = np.median(group_speed)
    group_speed[group_z_score >= 3] = group_median_speed
    group["speed"] = group_speed
    return group


def calculateBearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Description:
        Calculates the initial bearing (or forward azimuth) between two geographical coordinates.
        This is the angle between the north direction and the line connecting the start point to the end point.
        The result is normalized to a value between 0 and 360 degrees.

    Parameters:
        lat1 (float): Latitude of the starting point in decimal degrees.
        lon1 (float): Longitude of the starting point in decimal degrees.
        lat2 (float): Latitude of the ending point in decimal degrees.
        lon2 (float): Longitude of the ending point in decimal degrees.

    Returns:
        float: Initial bearing from the starting point to the ending point in degrees.

    Example:
        >>> calculateBearing(12.9716, 77.5946, 13.0827, 80.2707)
        76.123456789
    """

    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    bearing = np.rad2deg(bearing)
    bearing = (bearing + 360) % 360
    return bearing


def checkIfNearStop(df: pd.DataFrame, sdf: gpd.GeoDataFrame) -> List[int]:
    """
    Description:
        Checks if the first and last points of a trip are within a bus stop area.

    Parameters:
        df (pd.DataFrame): DataFrame containing trip data with latitude and longitude columns.
        sdf (gpd.GeoDataFrame): GeoDataFrame containing bus stop polygons.

    Returns:
        List[int]: A list of integers indicating whether the first and/or last points are within a bus stop area.

    Example:
        >>> checkIfNearStop(df, sdf)
        [1]
    """
    total_points = df.shape[0] - 1
    first_lat = df.iloc[0]["lat"]
    first_lon = df.iloc[0]["lng"]
    last_lat = df.iloc[total_points]["lat"]
    last_lon = df.iloc[total_points]["lng"]
    first_point = Point(first_lon, first_lat)
    last_point = Point(last_lon, last_lat)
    first_point_found = False
    last_point_found = False

    # checking for first point
    intersections = sdf.sindex.intersection(first_point.bounds)
    for index in intersections:
        polygon = sdf.loc[index, "geometry"]
        # Check for intersection
        if first_point.intersects(polygon):
            first_point_found = True
            break

    # checking for last point
    intersections = sdf.sindex.intersection(last_point.bounds)
    for index in intersections:
        polygon = sdf.loc[index, "geometry"]
        # Check for intersection
        if last_point.intersects(polygon):
            last_point_found = True
            break

    if first_point_found and last_point_found:
        ar = [2] * df.shape[0]
        return ar
    elif first_point_found or last_point_found:
        ar = [1] * df.shape[0]
        return ar
    else:
        ar = [0] * df.shape[0]
        return ar


def checkIfAtGrrenSpace(df: pd.DataFrame, sdf: gpd.GeoDataFrame) -> List[int]:
    """
    Description:
        Checks if the first point of a trip is within a green space area.
        If it is, returns 1 for all points in the trip; otherwise, returns 0.
        Also checks if the last point of a trip is within a green space area.
        If it is, returns 1 for all points in the trip; otherwise, returns 0.
        If both the first and last points are within a green space area, returns 2 for all points in the trip.

    Parameters:
        df (pd.DataFrame): DataFrame containing trip data with latitude and longitude columns.
        sdf (gpd.GeoDataFrame): GeoDataFrame containing green space polygons.

    Returns:
        List[int]: A list of integers indicating whether the first and/or last points are within a green space area.

    Example:
        >>> checkIfAtGrrenSpace(df, sdf)
        [1]
    """

    count = 0
    for i in range(df.shape[0]):
        lat = df.iloc[0]["lat"]
        lon = df.iloc[0]["lng"]
        coord_point = Point(lon, lat)
        point_found_at_gs = False
        intersections = sdf.sindex.intersection(coord_point.bounds)
        for index in intersections:
            polygon = sdf.loc[index, "geometry"]
            if coord_point.intersects(polygon):
                point_found_at_gs = True
                break
        if point_found_at_gs is True:
            count += 1
            if count == 5:
                break
    if count >= 5:
        return [1] * df.shape[0]
    else:
        return [0] * df.shape[0]


def calculateStraightnessIndex(df: pd.DataFrame) -> List[float]:
    """
    Description:
        Calculates the straightness index of a trip based on the distance covered and the straight line distance.

    Parameters:
        df (pd.DataFrame): DataFrame containing trip data with latitude and longitude columns.

    Returns:
        List[float]: A list of floats representing the straightness index for each point in the trip.

    Example:
        >>> df = pd.DataFrame({"lat": [12.9716, 13.0827], "lng": [77.5946, 80.2707]})
        >>> calculateStraightnessIndex(df)
        [0.5]
    """
    # Calculate the length of the actual path
    path_length = df["distance_covered"].sum()
    if path_length == 0 or np.isnan(path_length):
        return [np.nan] * df.shape[0]  # Avoid division error
    total_points = df.shape[0] - 1
    first_lat = df.iloc[0]["lat"]
    first_lon = df.iloc[0]["lng"]
    last_lat = df.iloc[total_points]["lat"]
    last_lon = df.iloc[total_points]["lng"]

    # Calculate the length of the shortest possible straight line
    straight_line_length = haversine(
        (first_lat, first_lon), (last_lat, last_lon), unit="m"
    )

    # Calculate the straightness index
    straightness_index = straight_line_length / path_length
    return [straightness_index] * df.shape[0]


def generateTrajStats(df: pd.DataFrame) -> pd.DataFrame:
    progress_bar = tqdm(total=25)

    temp_df = df.copy()
    if "transport_mode" not in temp_df.columns:
        temp_df["transport_mode"] = np.nan

    progress_bar.update(1)
    temp_df["speed_median"] = temp_df.groupby(["uid", "trip_id"])[
        "new_speed"
    ].transform(lambda x: x.median())
    progress_bar.update(1)
    temp_df["speed_pct_95"] = temp_df.groupby(["uid", "trip_id"])[
        "new_speed"
    ].transform(lambda x: np.percentile(x, 95))
    progress_bar.update(1)
    temp_df["speed_std"] = temp_df.groupby(["uid", "trip_id"])["new_speed"].transform(
        lambda x: np.std(x)
    )
    progress_bar.update(1)
    temp_df["acceleration_median"] = temp_df.groupby(["uid", "trip_id"])[
        "accelaration"
    ].transform(lambda x: np.nanmedian(x))
    progress_bar.update(1)
    temp_df["acceleration_pct_95"] = temp_df.groupby(["uid", "trip_id"])[
        "accelaration"
    ].transform(lambda x: np.nanpercentile(x, 95))
    progress_bar.update(1)
    temp_df["acceleration_std"] = temp_df.groupby(["uid", "trip_id"])[
        "accelaration"
    ].transform(lambda x: np.nanstd(x))
    progress_bar.update(1)
    temp_df["jerk_median"] = temp_df.groupby(["uid", "trip_id"])["jerk"].transform(
        lambda x: np.nanmedian(x)
    )
    progress_bar.update(1)
    temp_df["jerk_pct_95"] = temp_df.groupby(["uid", "trip_id"])["jerk"].transform(
        lambda x: np.nanpercentile(x, 95)
    )
    progress_bar.update(1)
    temp_df["jerk_std"] = temp_df.groupby(["uid", "trip_id"])["jerk"].transform(
        lambda x: np.nanstd(x)
    )
    progress_bar.update(1)
    temp_df["angular_dev_median"] = temp_df.groupby(["uid", "trip_id"])[
        "angular_deviation"
    ].transform(lambda x: np.nanmedian(x))
    progress_bar.update(1)
    temp_df["angular_dev_pct_95"] = temp_df.groupby(["uid", "trip_id"])[
        "angular_deviation"
    ].transform(lambda x: np.nanpercentile(x, 95))
    progress_bar.update(1)
    temp_df["angular_dev_std"] = temp_df.groupby(["uid", "trip_id"])[
        "angular_deviation"
    ].transform(lambda x: np.nanstd(x))
    progress_bar.update(1)
    temp_df["straightness_index"] = temp_df.groupby(["uid", "trip_id"])[
        "straightness_index"
    ].transform(lambda x: x.values[0])
    progress_bar.update(1)
    temp_df["distance_covered"] = temp_df.groupby(["uid", "trip_id"])[
        "distance_covered"
    ].transform(lambda x: sum(x) / 1000)
    progress_bar.update(1)
    temp_df["start_end_at_bus_stop"] = temp_df.groupby(["uid", "trip_id"])[
        "start_end_at_bus_stop"
    ].transform(lambda x: x.values[0])
    progress_bar.update(1)
    temp_df["start_end_at_train_stop"] = temp_df.groupby(["uid", "trip_id"])[
        "start_end_at_train_stop"
    ].transform(lambda x: x.values[0])
    progress_bar.update(1)
    temp_df["start_end_at_metro_stop"] = temp_df.groupby(["uid", "trip_id"])[
        "start_end_at_metro_stop"
    ].transform(lambda x: x.values[0])
    progress_bar.update(1)
    temp_df["found_at_green_space"] = temp_df.groupby(["uid", "trip_id"])[
        "found_at_green_space"
    ].transform(lambda x: x.values[0])
    progress_bar.update(1)
    # temp_df['temperature']=temp_df.groupby(['uid','trip_id'])['t'].transform(lambda x: x.mean())
    progress_bar.update(1)
    # temp_df['visibility']=temp_df.groupby(['uid','trip_id'])['v'].transform(lambda x: x.mean())
    progress_bar.update(1)
    # temp_df['wind_speed']=temp_df.groupby(['uid','trip_id'])['s'].transform(lambda x: x.mean())
    progress_bar.update(1)
    temp_df["is_weekend"] = temp_df.groupby(["uid", "trip_id"])["is_weekend"].transform(
        lambda x: x.values[0]
    )
    progress_bar.update(1)
    temp_df["hour_category"] = temp_df.groupby(["uid", "trip_id"])[
        "hour_category"
    ].transform(lambda x: x.values[0])
    progress_bar.update(1)
    temp_df = temp_df[
        [
            "datetime",
            "uid",
            "trip_id",
            "speed_median",
            "speed_pct_95",
            "speed_std",
            "acceleration_median",
            "acceleration_pct_95",
            "acceleration_std",
            "jerk_median",
            "jerk_pct_95",
            "jerk_std",
            "angular_dev_median",
            "angular_dev_pct_95",
            "angular_dev_std",
            "straightness_index",
            "distance_covered",
            "start_end_at_bus_stop",
            "start_end_at_train_stop",
            "start_end_at_metro_stop",
            "found_at_green_space",
            "is_weekend",
            "hour_category",
            "transport_mode",
        ]
    ]
    progress_bar.update(1)
    return temp_df


def featureEngineering(
    df_collection: List[pd.DataFrame],
    shape_files: List[gpd.GeoDataFrame],
    cores: int = max(1, int(cpu_count() // 2)),
) -> pd.DataFrame:
    args = [(df, shape_files) for df in df_collection]  # Wrap each df in a tuple
    with Pool(cores) as p:
        tdf = p.starmap(processData, args)
    return pd.concat(tdf, ignore_index=True)


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
    bus_stops = gpd.GeoDataFrame.from_file(bus_stops_shape_file)
    bus_stops.sindex

    print(f"{datetime.now()}: Reading Train Stops Shape File")
    train_stops = gpd.GeoDataFrame.from_file(train_stops_shape_file)
    train_stops.sindex

    print(f"{datetime.now()}: Reading Metro Stops Shape File")
    metro_stops = gpd.GeoDataFrame.from_file(metro_stops_shape_file)
    metro_stops.sindex

    print(f"{datetime.now()}: Reading Green Space Shape File")
    green_space_df = gpd.GeoDataFrame.from_file(gspace_file)
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
        trip_data_dir = f"{os.getenv('TRIP_DATA_DIR')}/{city}/{year}"
        print(f"{datetime.now()}: Reading raw data")
        raw_df = readRawData(raw_data_dir, CORES)
        print(f"{datetime.now()}: Reading trip & NA flow data")
        trip_df = readTripData(year, city, trip_data_dir)
        exit()
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
        trip_df = featureEngineering(df_collection, CORES, shape_files)
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
