import pandas as pd
import geopandas as gpd
import ast
import os
import numpy as np
from ReadJson import readJsonFiles
from multiprocessing import Pool
from datetime import datetime
from haversine import haversine, Unit
from scipy import stats
from tqdm import tqdm
from shapely.geometry import Point


def readTripData(year: int, city: str) -> pd.DataFrame:
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

    trip_file_path = f"U:/Projects/Huq/Faraz/final_OD_work_v2/{city}/{year}/trip_points"
    trip_df = pd.read_csv(trip_file_path + f"/trip_points_500m_{year}.csv")
    trip_df["trip_points"] = trip_df["trip_points"].apply(ast.literal_eval)
    trip_df = trip_df.explode("trip_points")
    trip_df.dropna(subset=["trip_points"], inplace=True)
    trip_df[["lat", "lng"]] = pd.DataFrame(
        trip_df["trip_points"].tolist(), index=trip_df.index
    )
    trip_df.drop(columns=["trip_points"], inplace=True)
    na_flows_file_path = (
        f"U:/Projects/Huq/Faraz/final_OD_work_v2/{city}/{year}/na_flows"
    )
    tdf = pd.read_csv(na_flows_file_path + f"/na_flows_500m_{year}.csv")
    trip_df = trip_df.merge(
        tdf[["uid", "trip_id", "org_lat", "org_lng", "dest_lat", "dest_lng"]],
        on=["uid", "trip_id"],
        how="left",
    )
    return trip_df


def readRawData(year: int, city: str) -> pd.DataFrame:
    """
    Description:
    Reads and compiles raw JSON data files for a given year and city by parallel processing
    multiple monthly files.

    Parameters:
    - year (int): The year of the data to be read.
    - city (str): The name of the city for which the raw data is being retrieved.

    Returns:
    - pd.DataFrame: A DataFrame containing compiled raw data from all monthly files.

    Example:
    >>> df = readRawData(2023, "London")
    >>> print(df.head())
    """

    root = f"U:/Operations/SCO/Faraz/huq_compiled/{city}/{year}"
    month_files = os.listdir(root)
    args = [(root, mf) for mf in month_files]
    with Pool(6) as p:
        df = p.starmap(readJsonFiles, args)
    return pd.concat(df, ignore_index=True)


def processData(df, shape_files):
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
    temp_df["trip_group"] = group.grouper.group_info[0]
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


def removeOutlier(group):

    group_speed = group["speed"]
    group_z_score = group["speed_z_score"]
    # group_speed_mean = np.mean(group_speed[group_z_score < 3])
    group_median_speed = np.median(group_speed)

    # group_median_speed=np.median(group_speed[np.isfinite(group_speed)])
    group_speed[group_z_score >= 3] = group_median_speed
    # group_speed=group_speed.replace([np.inf,-np.inf],group_median_speed)
    group["speed"] = group_speed
    return group


def calculateBearing(lat1, lon1, lat2, lon2):
    """Calculates the bearing between two GPS coordinates."""

    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    bearing = np.rad2deg(bearing)
    bearing = (bearing + 360) % 360
    return bearing


def getLoadBalancedBuckets(tdf: pd.DataFrame, bucket_size: int) -> list:
    """
    Description:
        Multiprocessing is being used for processing the data for Stop node detection and flow generation.
        This Funcition devides the data based on the UID and Number of Impressions in a way that load on
        every processor core being used is well-balanced.

    Parameters:
        tdf (pd.DataFrame): Trajectory DataFrame containing the data to be processed.
        bucket_size (int): Number of CPU Cores to be used for processing the data.

    Returns:
        list: List of Trajectory DataFrames. Each DataFrame will be processed in a seperate core as a seperate process.

    Example:
        >>> getLoadBalancedBuckets(tdf,bucket_size=8)
        [df1,df2,df3,df4,df5,df6,df7,df8]
    """

    print(f"{datetime.now()}: Getting unique users")
    unique_users = tdf["uid"].unique()  # Getting Unique Users in the data
    print(f"{datetime.now()}: Number of unique users: {len(unique_users)}")
    print(f"{datetime.now()}: Creating sets")
    num_impr_df = (
        pd.DataFrame(tdf.groupby("uid").size(), columns=["num_impressions"])
        .reset_index()
        .sort_values(by=["num_impressions"], ascending=False)
    )  # Creating a DataFrame containing Unique UID and Total number of impressions that Unique UID has in the data.
    buckets = (
        {}
    )  # A dictionary containing buckets of UIDs. Each bucket represent the CPU core. This dictionary tells how many users' data will be process on which core. For example, if bucket 1 contains 10 UIDs, data of those 10 UIDs will be processed on the core 1.
    bucket_sums = {}  # A flag dictionary to keep the track of load on each bucket.

    for i in range(1, bucket_size + 1):
        buckets[i] = []  # Initializing empty buckets
        bucket_sums[i] = 0  # Load in each bucket is zero initially

    # Allocate users to buckets
    for _, row in num_impr_df.iterrows():
        user, impressions = (
            row["uid"],
            row["num_impressions"],
        )  # Getting the UID and the number of impressions of that UID
        # Find the bucket with the minimum sum of impressions
        min_bucket = min(
            bucket_sums, key=bucket_sums.get
        )  # Getting the bucket with the minimum load. Initially, all the buckets have zero load.
        # Add user to this bucket
        buckets[min_bucket].append(user)  # Adding UID to the minimum bucket
        # Update the sum of impressions for this bucket
        bucket_sums[
            min_bucket
        ] += impressions  # Updating the load value of the bucket. For example, UID 1 was added to Bucket 1 and UID 1 had 1000 impressions (records). So, load of bucket 1 is 1000 now.

    print(f"{datetime.now()}: Creating seperate dataframes")
    tdf_collection = (
        []
    )  # List of collection of DataFrames. This list will contain the number of DataFrames=number of CPU Cores. Each DataFrame will be processed in a seperate core as a seperate process.
    for i in range(1, bucket_size + 1):
        tdf_collection.append(tdf[tdf["uid"].isin(buckets[i])].copy())
    return tdf_collection


def checkIfNearStop(df, sdf):
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


def checkIfAtGrrenSpace(df, sdf):
    count = 0
    for i in range(df.shape[0]):
        lat = df.iloc[0]["lat"]
        lon = df.iloc[0]["lng"]
        coord_point = Point(lon, lat)
        point_found_at_gs = False
        intersections = sdf.sindex.intersection(coord_point.bounds)
        for index in intersections:
            polygon = sdf.loc[index, "geometry"]
            # Check for intersection
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


def calculateStraightnessIndex(df):
    # Calculate the length of the actual path
    path_length = df["distance_covered"].sum()
    if path_length == 0 or np.isnan(path_length):
        return [np.nan] * df.shape[0]  # Avoid division error
    total_points = df.shape[0] - 1
    # first_lat=df.iloc[0,4]
    # first_lon=df.iloc[0,5]
    # last_lat=df.iloc[total_points,4]
    # last_lon=df.iloc[total_points,5]

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


def generateHuqStats(df):

    progress_bar = tqdm(total=25)

    temp_df = df.copy()
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
        ]
    ]
    progress_bar.update(1)
    return temp_df


def featureEngineering(df_collection, cores, shape_files):
    args = [(df, shape_files) for df in df_collection]  # Wrap each df in a tuple
    with Pool(cores) as p:
        tdf = p.starmap(processData, args)
    return pd.concat(tdf, ignore_index=True)


if __name__ == "__main__":

    print(f"{datetime.now()}: Starting Process...")
    city = "Glasgow"
    years = [2020, 2021, 2022, 2022, 2023]
    CORES = 8
    print(f"{datetime.now()}: Reading this Bus Stops Shape File")
    bus_stops_shape_file = "D:/Mobile Device Data/TMD_repo/travel_mode_detection/bus_stops/bus_stops_shape_file/output.shp"
    bus_stops = gpd.GeoDataFrame.from_file(bus_stops_shape_file)
    bus_stops.sindex

    print(f"{datetime.now()}: Reading Train Stops Shape File")
    train_stops_shape_file = "D:/Mobile Device Data/TMD_repo/travel_mode_detection/train_station_locations/train_station_locations.shp"
    train_stops = gpd.GeoDataFrame.from_file(train_stops_shape_file)
    train_stops.sindex

    print(f"{datetime.now()}: Reading Metro Stops Shape File")
    metro_stops_shape_file = "D:/Mobile Device Data/TMD_repo/travel_mode_detection/metro_station_locations/metro_station_locations.shp"
    metro_stops = gpd.GeoDataFrame.from_file(metro_stops_shape_file)
    metro_stops.sindex

    print(f"{datetime.now()}: Reading Green Space Shape File")
    gspace_file = "D:/Mobile Device Data/TMD_repo/Green_Space_ShapeFile/green_spaces/Glasgow_filtered.shp"
    green_space_df = gpd.GeoDataFrame.from_file(gspace_file)
    if green_space_df.crs is not None and green_space_df.crs != "EPSG:4326":
        green_space_df = green_space_df.to_crs(epsg=4326)
    green_space_df.sindex
    print(f"{datetime.now()}: Finished Reading Shape Files")

    shape_files = [bus_stops, train_stops, metro_stops, green_space_df]
    # shape_files = [bus_stops, train_stops, metro_stops]
    for year in years:
        print(
            f"""
        City: {city}
        Year: {year}
        """
        )
        print(f"{datetime.now()}: Reading raw data")
        raw_df = readRawData(year, "Glasgow")
        print(f"{datetime.now()}: Reading trip & na-flows data")
        trip_df = readTripData(year, "Glasgow")
        print(f"{datetime.now()}: Merging raw data with trip data to get datetime")
        trip_df = trip_df.merge(
            raw_df[["uid", "datetime", "lat", "lng"]],
            on=["uid", "lat", "lng"],
            how="left",
        )
        assert trip_df["datetime"].isna().sum() == 0
        print(f"{datetime.now()}: Validation Done")
        del raw_df
        print(f"{datetime.now()}: Converting datetime to datetime object")
        trip_df["datetime"] = pd.to_datetime(trip_df["datetime"])
        print(f"{datetime.now()}: Get Load Balanced Buckets")
        df_collection = getLoadBalancedBuckets(trip_df, 6)
        del trip_df
        print(f"{datetime.now()}: Feature Engineering")
        trip_df = featureEngineering(df_collection, CORES, shape_files)
        del df_collection
        print(f"{datetime.now()}: Saving Processed Data")
        os.makedirs(
            f"U:/Projects/Huq/Faraz/travel_mode_detection/{city}/{year}", exist_ok=True
        )
        trip_df.to_csv(
            f"U:/Projects/Huq/Faraz/travel_mode_detection/{city}/{year}/processed_trip_points_data.csv",
            index=False,
        )
        print(f"{datetime.now()}: Generating Huq Stats")
        huq_stats = generateHuqStats(trip_df)
        output_file_path = f"U:/Projects/Huq/Faraz/travel_mode_detection/{city}/{year}"
        os.makedirs(output_file_path, exist_ok=True)
        huq_stats.to_csv(
            f"{output_file_path}/huq_stats_df_for_ml.csv",
            index=False,
        )

    print(f"{datetime.now()}: Finished.")
