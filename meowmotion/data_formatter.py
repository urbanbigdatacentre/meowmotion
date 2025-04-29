import ast
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


def processTripData(
    trip_point_df: pd.DataFrame, na_flow_df: pd.DataFrame, raw_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Processes trip-level data by expanding stored trip-point coordinates, merging
    in origin-destination flows, and then attaching timestamps from a raw dataset.
    The result is a single DataFrame containing trip points (latitude, longitude, and
    timestamps) and corresponding origin/destination information.

    Key Steps:
        1. Expands list-based trip points in `trip_point_df` into individual rows
           for each (lat, lng) point.
        2. Joins the expanded trip points to `na_flow_df` to retrieve origin,
           destination, and timing fields.
        3. Filters trips to ensure total travel time does not exceed 24 hours.
        4. Merges `raw_df` to add precise timestamps for each (lat, lng) point and
           ensures each point is within the trip's time window.

    Args:
        trip_point_df (pd.DataFrame):
            Contains user IDs, trip IDs, and a column of list-based trip points.
            Must have columns `["uid", "trip_id", "trip_points"]`.
        na_flow_df (pd.DataFrame):
            Non-aggregated OD flow data containing origin/destination coordinates
            and timestamps (`org_leaving_time`, `dest_arival_time`, etc.).
        raw_df (pd.DataFrame):
            The raw dataset with columns `["uid", "datetime", "lat", "lng"]`, used
            to match each trip point to a specific timestamp.

    Returns:
        pd.DataFrame: A cleaned and merged DataFrame with columns for each trip's
        user ID, trip ID, origin/destination coordinates, and per-point latitude,
        longitude, and timestamps.

    Example:
        >>> # Suppose you already have three DataFrames: trip_point_df, na_flow_df, raw_df
        >>> result_df = processTripData(trip_point_df, na_flow_df, raw_df)
        >>> print(result_df.head())
          uid  trip_id       lat       lng       datetime  org_lat  org_lng  ...
        0   1       10  12.9716  77.59460  2023-01-01 ...   12.970  77.5940  ...
        1   1       10  12.9720  77.59470  2023-01-01 ...   12.970  77.5940  ...
        ...

    """

    # trip_file_path = f"{data_dir}/{city}/{year}/trip_points"
    # trip_point_df = pd.read_csv(f"{trip_file_path}/trip_points_500m_{year}.csv")
    trip_point_df["trip_points"] = trip_point_df["trip_points"].apply(ast.literal_eval)
    trip_point_df = trip_point_df.explode("trip_points")
    trip_point_df.dropna(subset=["trip_points"], inplace=True)
    trip_point_df[["lat", "lng"]] = pd.DataFrame(
        trip_point_df["trip_points"].tolist(), index=trip_point_df.index
    )
    trip_point_df.drop(columns=["trip_points"], inplace=True)

    # na_flows_file_path = f"{data_dir}/{city}/{year}/na_flows"
    # tdf = pd.read_csv(na_flows_file_path + f"/na_flows_500m_{year}.csv")

    na_flow_df["org_arival_time"] = pd.to_datetime(na_flow_df["org_arival_time"])
    na_flow_df["org_leaving_time"] = pd.to_datetime(na_flow_df["org_leaving_time"])
    na_flow_df["dest_arival_time"] = pd.to_datetime(na_flow_df["dest_arival_time"])
    trip_point_df = trip_point_df.merge(
        na_flow_df[
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
    trip_point_df = trip_point_df[
        (
            trip_point_df["dest_arival_time"] - trip_point_df["org_leaving_time"]
        ).dt.total_seconds()
        / 3600
        <= 24
    ]

    print(f"{datetime.now()}: Merging raw data with trip data to get datetime")
    trip_point_df = trip_point_df.merge(
        raw_df[["uid", "datetime", "lat", "lng"]],
        on=["uid", "lat", "lng"],
        how="left",
    )

    print(f"{datetime.now()}: Converting datetime to datetime object")
    trip_point_df["datetime"] = pd.to_datetime(trip_point_df["datetime"])
    trip_point_df = trip_point_df[
        trip_point_df["datetime"].between(
            trip_point_df["org_leaving_time"], trip_point_df["dest_arival_time"]
        )
    ].reset_index(drop=True)
    assert trip_point_df["datetime"].isna().sum() == 0
    print(f"{datetime.now()}: Validation Done")

    return trip_point_df


def readRawData(
    data_dir: str, cpu_cores: int = max(1, cpu_count() // 2)
) -> pd.DataFrame:
    """
    Reads and compiles raw JSON data files for a given year and city by parallel processing
    multiple monthly files.

    Args:
        cpu_cores (int): The number of CPU cpu_cores to be used for parallel processing. By default, it uses half of the available cpu_cores.
        data_dir (str): The directory where the raw data files are stored.

    Returns:
        pd.DataFrame: A DataFrame containing compiled raw data from all monthly files.

    Example:
        >>> df = readRawData(2023, "path_to_root/city/year")
        >>> print(df.head())
    """
    root = data_dir
    month_files = os.listdir(root)
    args = [(root, mf) for mf in month_files]
    with Pool(cpu_cores) as p:
        df = p.starmap(readJsonFiles, args)
    return pd.concat(df, ignore_index=True)


def processData(df: pd.DataFrame, shape_files: List[gpd.GeoDataFrame]) -> pd.DataFrame:
    """
    Cleans and enriches raw trip-point data with motion-related features
    (speed, acceleration, jerk, bearing, angular deviation, straightness
    index) and contextual flags indicating proximity to public-transport
    stops or green spaces.

    The function operates **per trip** (``uid``–``trip_id``):
      1. Removes duplicate timestamps and drops trips with fewer than
         five distinct observations (``num_of_impressions`` < 5).
      2. Computes time deltas, inter-point distance (haversine), speed,
         speed z-scores, acceleration, and jerk, replacing extreme
         speed outliers (|z| ≥ 3) with the median speed.
      3. Derives temporal attributes—calendar month, hour of day,
         weekend flag, and a four-level ``hour_category``
         (0 Night, 1 Morning, 2 Afternoon, 3 Evening).
      4. For each trip, determines whether the first and/or last point
         lies within
         • a **bus stop** (shape_files[0])
         • a **train station** (shape_files[1])
         • a **metro station** (shape_files[2])
         and whether **≥ 5 points** fall inside a **green space**
         polygon (shape_files[3]).
      5. Calculates a straightness index (straight-line ÷ actual path
         length) and removes trips with an index > 1 (spurious data).

    Args:
        df (pd.DataFrame): Point-level trip data containing at least
            ``["uid", "trip_id", "lat", "lng", "datetime"]``.
        shape_files (List[gpd.GeoDataFrame]): A list of four
            GeoDataFrames **in this order**:
            ``[bus_stops_gdf, train_stops_gdf, metro_stops_gdf,
            green_space_gdf]``.  Each must use CRS EPSG 4326.

    Returns:
        pd.DataFrame: The cleaned and feature-rich DataFrame, one row
        per retained point, including new columns such as

        * ``num_of_impressions`` • ``time_taken`` • ``distance_covered``
        * ``speed`` • ``speed_z_score`` • ``new_speed``
        * ``accelaration`` • ``jerk``
        * ``bearing`` • ``angular_deviation``
        * ``month`` • ``hour`` • ``is_weekend`` • ``hour_category``
        * ``start_end_at_bus_stop`` / ``train_stop`` / ``metro_stop``
        * ``found_at_green_space`` • ``straightness_index``

    Example:
        >>> processed = processData(raw_trip_df, [
        ...     bus_stops_gdf, train_stops_gdf, metro_stops_gdf,
        ...     green_space_gdf
        ... ])
        >>> processed.head()
    """
    temp_df = df.copy()

    # In some trips, for very same timestamp, we observed multiple datapoints. To deal with that, dropping all the duplicate timestamps and keeping the first one
    temp_df = (
        temp_df.groupby(["uid", "trip_id"], group_keys=True)
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
    Filters outliers in the `speed` column by replacing high z-score values (≥ 3) with
    the median speed. This function is typically applied to each group within a larger
    grouped DataFrame (e.g., a single trip trajectory).

    How It Works:
        1. Calculates the median speed within the group.
        2. Identifies rows where `speed_z_score` is ≥ 3.
        3. Replaces those outlier `speed` values with the median speed.
        4. Returns the modified group DataFrame.

    Args:
        group (pd.DataFrame): Subset of a larger DataFrame, typically representing
            one trip. Must contain at least:
            - `speed`: The speed values to check.
            - `speed_z_score`: The corresponding z-score values for speed.

    Returns:
        pd.DataFrame: The same DataFrame with outlier speeds replaced by the median.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = {
        ...     'speed': [5.0, 120.0, 6.0],
        ...     'speed_z_score': [0.2, 3.5, 0.3]
        ... }
        >>> df = pd.DataFrame(data)
        >>> print(df)
             speed  speed_z_score
        0     5.0            0.20
        1   120.0            3.50
        2     6.0            0.30

        >>> cleaned = removeOutlier(df)
        >>> print(cleaned)
             speed  speed_z_score
        0     5.0            0.20
        1     5.5            3.50  # replaced with median (5.5)
        2     6.0            0.30
    """

    group_speed = group["speed"]
    group_z_score = group["speed_z_score"]
    group_median_speed = np.median(group_speed)
    group_speed[group_z_score >= 3] = group_median_speed
    group["speed"] = group_speed
    return group


def calculateBearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the initial bearing (forward azimuth) from one geographic coordinate to another.
    This bearing is measured clockwise from true north and returned as a value between 0 and 360 degrees.

    How It Works:
        1. Converts both starting (lat1, lon1) and ending (lat2, lon2) coordinates to radians.
        2. Uses the difference in longitudes (dlon) and trigonometric functions to calculate
           the bearing in radians.
        3. Converts the bearing from radians to degrees.
        4. Normalizes the result to ensure it falls within the range of [0, 360).

    Args:
        lat1 (float): Latitude of the start location (in decimal degrees).
        lon1 (float): Longitude of the start location (in decimal degrees).
        lat2 (float): Latitude of the end location (in decimal degrees).
        lon2 (float): Longitude of the end location (in decimal degrees).

    Returns:
        float: The initial bearing in degrees, between 0 and 360.

    Example:
        >>> bearing = calculateBearing(12.9716, 77.5946, 13.0827, 80.2707)
        >>> print(bearing)
        76.123456789  # Example output
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
    Determines whether the first and/or last point of a trip lies within a given
    polygon area (e.g., bus stop, train station, or metro station). It returns a list
    of length equal to `df.shape[0]`, with each element indicating whether:

    - Both the first and last points intersect the polygon(s): 2
    - Only one of the points intersects the polygon(s): 1
    - Neither the first nor the last point intersects the polygon(s): 0

    Args:
        df (pd.DataFrame): DataFrame containing trip data, including `lat` and `lng`
            columns for each point in the trip.
        sdf (gpd.GeoDataFrame): GeoDataFrame representing polygons for stops or stations
            (e.g., bus stops, train stations, metro stations).

    Returns:
        List[int]: A list of integers (2, 1, or 0) indicating the presence of the
        first/last point in the polygon(s).

    Example:
        >>> import pandas as pd
        >>> import geopandas as gpd
        >>> from shapely.geometry import Polygon
        >>> # Example DataFrame with two points
        >>> df = pd.DataFrame({
        ...     'lat': [12.9716, 12.9760],
        ...     'lng': [77.5946, 77.5950]
        ... })
        >>> # Example GeoDataFrame representing a stop
        >>> polygons = [Polygon([(77.5940, 12.9710), (77.5950, 12.9710),
        ...                      (77.5950, 12.9720), (77.5940, 12.9720)])]
        >>> sdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
        >>> result = checkIfNearStop(df, sdf)
        >>> print(result)
        [1, 1]  # Only the first point intersects the polygon
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
    Checks whether a trip has at least five data points that fall within the specified
    green space polygons. If it does, the entire trip is marked with 1 for every row.
    Otherwise, it returns 0 for each row.

    Note:
        - This function uses a threshold of five detections by default, but you can
          customize this threshold as needed.

    How It Works:
        1. Iterates through all points in `df` (each representing a trajectory point in the trip).
        2. For each point, checks if it intersects any of the polygons in `sdf`.
        3. If at least five points from the trip are found in a green space, returns a list
           of 1s (one per row in `df`).
        4. Otherwise, returns a list of 0s.

    Args:
        df (pd.DataFrame): DataFrame containing trip data with at least `lat` and `lng`
            columns for each point in the trip.
        sdf (gpd.GeoDataFrame): GeoDataFrame representing one or more green space polygons.

    Returns:
        List[int]: A list of integers (1 or 0) for each row in `df`.
            - `[1, 1, ..., 1]` if at least five points are within green space
            - `[0, 0, ..., 0]` otherwise

    Example:
        >>> df = pd.DataFrame({
        ...     "lat": [12.9716, 12.9780, 12.9825, 12.9850, 12.9900],
        ...     "lng": [77.5946, 77.5949, 77.5953, 77.5960, 77.5965]
        ... })
        >>> green_spaces = gpd.read_file("greenspaces.shp")  # Example file
        >>> result = checkIfAtGrrenSpace(df, green_spaces)
        >>> print(result)
        [1, 1, 1, 1, 1]   # indicates at least 5 points are inside a green space
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
    Calculates a trip’s "straightness index" by comparing the total distance traveled
    to the straight-line distance between the first and last points. The resulting ratio
    (straight-line distance ÷ actual path distance) measures how directly a traveler
    moved from start to end.

    How It Works:
        1. Summarizes the total distance covered (`distance_covered`).
        2. Calculates the straight-line (haversine) distance between the first and last
           coordinates in the trip.
        3. Divides the straight-line distance by the actual path length.
        4. Returns that value for every row in the DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame representing a single trip, containing at least
            - `lat`: Latitude coordinates
            - `lng`: Longitude coordinates
            - `distance_covered`: The distance between consecutive points (in meters)

    Returns:
        List[float]: A list (the same length as `df`) with the same straightness index
        repeated for each row. If the path length is 0 or NaN, returns `[np.nan] * len(df)`.

    Example:
        >>> import pandas as pd
        >>> from haversine import haversine
        >>> df = pd.DataFrame({
        ...     "lat": [12.9716, 13.0827],
        ...     "lng": [77.5946, 80.2707],
        ...     "distance_covered": [0, 35000]  # for example
        ... })
        >>> result = calculateStraightnessIndex(df)
        >>> print(result)
        [0.5, 0.5]  # Indicates the path is half as direct as a straight line
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
    """
    Aggregates and summarizes key trip-level statistics (e.g., median/percentile speeds,
    accelerations, jerk, angular deviation, distance) from enhanced trip data. This function
    operates on data that has already undergone feature engineering (e.g., via `featureEngineering`),
    and creates consolidated columns reflecting various trip metrics. A progress bar is displayed
    during calculation.

    Args:
        df (pd.DataFrame): A DataFrame containing enhanced trip data, including columns such as
            `uid`, `trip_id`, `new_speed`, `accelaration`, `jerk`, and `angular_deviation`.
            It may also contain flags for stops and green spaces.

    Returns:
        pd.DataFrame: A DataFrame containing aggregated statistics for each trip, including:
            - Speed median, 95th percentile, and standard deviation
            - Acceleration median, 95th percentile, and standard deviation
            - Jerk median, 95th percentile, and standard deviation
            - Angular deviation median, 95th percentile, and standard deviation
            - Straightness index
            - Total distance covered (km)
            - Indicators for whether the trip starts/ends near specific transport stops or green spaces
            - Weekend/hour categories
            - A placeholder for `transport_mode`

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     "uid": [1, 1, 1, 2, 2],
        ...     "trip_id": [10, 10, 10, 20, 20],
        ...     "new_speed": [3.0, 5.5, 2.0, 4.0, 4.5],
        ...     "accelaration": [0.1, 0.2, 0.3, 0.1, 0.05],
        ...     "jerk": [0.01, 0.02, 0.03, 0.01, 0.02],
        ...     "angular_deviation": [5, 10, 15, 3, 4],
        ... }
        >>> df = pd.DataFrame(data)
        >>> result = generateTrajStats(df)
        >>> result.head()
           datetime  uid  trip_id  speed_median  ...  hour_category  transport_mode
        0       NaT    1       10           3.5  ...             0             NaN
        1       NaT    1       10           3.5  ...             0             NaN
        2       NaT    1       10           3.5  ...             0             NaN
        3       NaT    2       20           4.25 ...             0             NaN
        4       NaT    2       20           4.25 ...             0             NaN

    """

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
    trip_df: pd.DataFrame,
    shape_files: List[gpd.GeoDataFrame],
    cpu_cores: int = max(1, int(cpu_count() // 2)),
) -> pd.DataFrame:
    """
    Performs feature engineering on raw trip data by partitioning it and processing each partition
    in parallel. This includes calculating advanced trip features such as speed, acceleration,
    angular deviation, and straightness index, as well as identifying whether a trip starts or ends
    near transport stops or green spaces.

    This function distributes work across the specified number of CPU cores, calls the `processData`
    child function on each chunk, and then merges all processed chunks into a single DataFrame.

    Args:
        trip_df (pd.DataFrame): A DataFrame containing raw trip information, including columns for
            user ID, trip ID, latitude (`lat`), longitude (`lng`), and timestamps (`datetime`).
        shape_files (List[gpd.GeoDataFrame]): A list of GeoDataFrames representing various geographic
            layers (e.g., bus stops, train stops, metro stops, green spaces). These are used to check
            if trips start/end near these points or areas.
        cpu_cores (int, optional): Number of CPU cores to use for parallel processing. Defaults to half
            of the available cores.

    Returns:
        pd.DataFrame: A concatenated DataFrame containing the enhanced feature set for all trips.
        Features include:
            - Speed, acceleration, jerk, and angular deviation
            - Straightness index
            - Indicators for whether a trip begins or ends near transport stops or in green spaces
            - Filtered trips based on minimum impressions

    Example:
        >>> # Suppose 'trip_df' is a DataFrame of trips and 'shapes' is a list of GeoDataFrames
        >>> from your_module import featureEngineering
        >>> enhanced_df = featureEngineering(trip_df, shapes, cores=4)
        >>> print(enhanced_df.head())
    """

    print(f"{datetime.now()}: Get Load Balanced Buckets")
    df_collection = getLoadBalancedBuckets(trip_df, cpu_cores)
    args = [(df, shape_files) for df in df_collection]  # Wrap each df in a tuple
    del df_collection
    with Pool(cpu_cores) as p:
        tdf = p.starmap(processData, args)
    return pd.concat(tdf, ignore_index=True)


if __name__ == "__main__":

    print(f"{datetime.now()}: Starting Process...")
    # city = os.getenv("CITY")  # "Glasgow"
    # years = json.loads(os.getenv("YEARS"))  # [2022, 2023]
    # CORES = int(os.getenv("CORES"))
    # bus_stops_shape_file = os.getenv("SHAPE_FILE_BUS")
    # train_stops_shape_file = os.getenv("SHAPE_FILE_TRAIN")
    # metro_stops_shape_file = os.getenv("SHAPE_FILE_METRO")
    # gspace_file = os.getenv("SHAPE_FILE_GREEN_SPACES")
    # output_dir = os.getenv("OUTPUT_DIR")

    # print(f"{datetime.now()}: Reading Bus Stops Shape File")
    # bus_stops = gpd.GeoDataFrame.from_file(bus_stops_shape_file)
    # bus_stops.sindex

    # print(f"{datetime.now()}: Reading Train Stops Shape File")
    # train_stops = gpd.GeoDataFrame.from_file(train_stops_shape_file)
    # train_stops.sindex

    # print(f"{datetime.now()}: Reading Metro Stops Shape File")
    # metro_stops = gpd.GeoDataFrame.from_file(metro_stops_shape_file)
    # metro_stops.sindex

    # print(f"{datetime.now()}: Reading Green Space Shape File")
    # green_space_df = gpd.GeoDataFrame.from_file(gspace_file)
    # if green_space_df.crs is not None and green_space_df.crs != "EPSG:4326":
    #     green_space_df = green_space_df.to_crs(epsg=4326)
    # green_space_df.sindex
    # print(f"{datetime.now()}: Finished Reading Shape Files")

    # shape_files = [bus_stops, train_stops, metro_stops, green_space_df]
    # for year in years:
    #     print(
    #         f"""
    #     City: {city}
    #     Year: {year}
    #     """
    #     )
    #     raw_data_dir = f"{os.getenv('RAW_DATA_DIR')}/{city}/{year}"
    #     trip_data_dir = f"{os.getenv('TRIP_DATA_DIR')}/{city}/{year}"
    #     print(f"{datetime.now()}: Reading raw data")
    #     raw_df = readRawData(raw_data_dir, CORES)
    #     print(f"{datetime.now()}: Reading trip & NA flow data")
    #     trip_df = readTripData(year, city, trip_data_dir)
    #     exit()
    #     print(f"{datetime.now()}: Merging raw data with trip data to get datetime")
    #     trip_df = trip_df.merge(
    #         raw_df[["uid", "datetime", "lat", "lng"]],
    #         on=["uid", "lat", "lng"],
    #         how="left",
    #     )
    #     print(f"{datetime.now()}: Converting datetime to datetime object")
    #     trip_df["datetime"] = pd.to_datetime(trip_df["datetime"])
    #     trip_df = trip_df[
    #         trip_df["datetime"].between(
    #             trip_df["org_leaving_time"], trip_df["dest_arival_time"]
    #         )
    #     ].reset_index(drop=True)
    #     assert trip_df["datetime"].isna().sum() == 0
    #     print(f"{datetime.now()}: Validation Done")
    #     del raw_df
    #     print(f"{datetime.now()}: Get Load Balanced Buckets")
    #     df_collection = getLoadBalancedBuckets(trip_df, CORES)
    #     del trip_df
    #     print(f"{datetime.now()}: Feature Engineering")
    #     trip_df = featureEngineering(df_collection, CORES, shape_files)
    #     del df_collection
    #     print(f"{datetime.now()}: Saving Processed Data")
    #     os.makedirs(f"{output_dir}/{city}/{year}", exist_ok=True)
    #     trip_df.to_csv(
    #         f"{output_dir}/{city}/{year}/processed_trip_points_data.csv",
    #         index=False,
    #     )
    #     print(f"{datetime.now()}: Generating Huq Stats")
    #     huq_stats = generateTrajStats(trip_df)
    #     huq_stats.to_csv(
    #         f"{output_dir}/{city}/{year}/huq_stats_df_for_ml.csv",
    #         index=False,
    #     )

    print(f"{datetime.now()}: Finished.")
