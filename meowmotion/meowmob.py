import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from skmob import TrajDataFrame
from skmob.preprocessing import detection
from tqdm import tqdm

from meowmotion.process_data import getLoadBalancedBuckets, saveFile, spatialJoin


def getStopNodes(
    tdf: TrajDataFrame,
    time_th: Optional[int] = 5,
    radius: Optional[int] = 500,
    cpu_cores: Optional[int] = max(1, int(cpu_count() / 2)),
) -> pd.DataFrame:
    """
    Detect stop nodes from trajectory data in parallel using scikit-mobility's stay_locations.

    This function splits the input TrajDataFrame across multiple CPU cores (via getLoadBalancedBuckets),
    detects stops on each chunk using the stay_locations function, then merges the results back together.
    After detection, latitude and longitude columns are renamed to "org_lat" and "org_lng" in the final
    returned DataFrame.

    Args:
        tdf (TrajDataFrame):
            Input trajectory data with columns at least ["uid", "datetime", "lat", "lng"].
        time_th (int, optional):
            Time threshold (in minutes) used by stay_locations to detect a stop. Defaults to 5.
        radius (int, optional):
            Spatial radius (in meters) within which points are considered part of the same stop. Defaults to 500.
        cpu_cores (int, optional):
            Number of CPU cores to use for parallel processing. Defaults to half the available cores (at least 1).

    Returns:
        pd.DataFrame:
            A DataFrame representing the detected stop nodes. The main columns include:
            "uid", "org_lat", "org_lng", "datetime" (representing arrival time), "leaving_datetime",
            and any additional columns returned by stay_locations.

    Example:
        >>> from meowmotion.meowmob import getStopNodes
        >>> stops_df = getStopNodes(tdf, time_th=10, radius=1000, cpu_cores=4)
        >>> print(stops_df.head())
    """
    tdf = tdf.reset_index(drop=True)
    tdf_collection = getLoadBalancedBuckets(tdf, cpu_cores)
    print(f"{datetime.now()}: Stop Node Detection Started")
    args = [(df, time_th, radius) for df in tdf_collection if not df.empty]
    with Pool(cpu_cores) as pool:
        results = pool.starmap(stopNodes, args)

    del tdf_collection  # Deleting the data to free up the memory
    stdf = pd.concat([*results])
    del results  # Deleting the results to free up the memory
    stdf.rename(columns={"lat": "org_lat", "lng": "org_lng"}, inplace=True)
    stdf = pd.DataFrame(stdf)
    stdf = stdf.drop(columns=["impression_acc"])
    print(f"{datetime.now()} Stop Node Detection Completed\n")
    return stdf


def stopNodes(tdf: TrajDataFrame, time_th: int, radius: int) -> TrajDataFrame:
    return detection.stay_locations(
        tdf,
        minutes_for_a_stop=time_th,
        spatial_radius_km=(radius / 1000),
        leaving_time=True,
    )


def processFlowGeneration(
    stdf: pd.DataFrame,
    raw_df: pd.DataFrame,
    cpu_cores: int = max(1, int(cpu_count() / 2)),
) -> pd.DataFrame:
    """
    Generate flow data from stop nodes using parallel processing.

    This function takes two data sources:
     1. `stdf`: A DataFrame of stop nodes, which must contain columns such as
         "uid", "datetime", "org_lat", "org_lng", and the "dest_*" fields added here.
     2. `raw_df`: The underlying trajectory data (with columns like "uid", "datetime",
         "lat", "lng") from which the detailed trip segments and stay points are extracted.

    The function first prepares `stdf` by assigning "dest_at", "dest_lat", and "dest_lng"
    (the next stop in sequence for each user), then uses `getLoadBalancedBuckets` to split
    the DataFrame for multiprocessing. For each split/bucket, it calls `flowGenration(...)`
    in parallel to build the trip segments and stay details from the raw data. Finally,
    it concatenates the partial results and returns a single DataFrame of flow data.

    Columns in the final flow DataFrame typically include:
      - "uid"
      - "org_lat", "org_lng", "org_arival_time", "org_leaving_time"
      - "dest_lat", "dest_lng", "dest_arival_time"
      - "stay_points", "trip_points"
      - "trip_time", "stay_duration", "observed_stay_duration"
      (and any other columns you choose to include in `flowGenration`)

    Args:
        stdf (pd.DataFrame): DataFrame containing stop nodes. Must have columns such as
            "uid", "datetime", "org_lat", "org_lng". Additional columns will be created
            or renamed (e.g., "dest_lat", "dest_lng", "dest_at").
        raw_df (pd.DataFrame): The raw trajectory data with columns like
            "uid", "datetime", "lat", "lng". Used to extract trip details.
        cpu_cores (int, optional): Number of CPU cores for multiprocessing.
            Defaults to half of available cores, at minimum 1.

    Returns:
        pd.DataFrame: A concatenation of all flows generated in parallel.
        Each row represents a trip between one stop node and the next.

    Example:
        >>> from meowmotion.meowmob import getStopNodes, processFlowGeneration
        >>> stop_nodes_df = getStopNodes(traj_df)
        >>> flow_data = processFlowGeneration(stop_nodes_df, raw_df, cpu_cores=4)
        >>> print(flow_data.head())
    """

    stdf["dest_at"] = stdf.groupby("uid")["datetime"].transform(lambda x: x.shift(-1))
    stdf["dest_lat"] = stdf.groupby("uid")["org_lat"].transform(lambda x: x.shift(-1))
    stdf["dest_lng"] = stdf.groupby("uid")["org_lng"].transform(lambda x: x.shift(-1))
    print(stdf.head())  # Printing the first 5 rows of the stop nodes data
    stdf = stdf.dropna(subset=["dest_lat"])
    tdf_collection = getLoadBalancedBuckets(stdf, cpu_cores)
    print(f"{datetime.now()}: Generating args")
    args = []
    for tdf in tdf_collection:
        temp_raw_df = raw_df[raw_df["uid"].isin(tdf["uid"].unique())].copy()
        temp_raw_df.set_index(["uid", "datetime"], inplace=True)
        temp_raw_df.sort_index(inplace=True)
        args.append((tdf, temp_raw_df))
    del tdf_collection
    print(f"{datetime.now()}: args Generation Completed")
    print(f"{datetime.now()}: Flow Generation Started\n\n")
    with Pool(cpu_cores) as pool:
        results = pool.starmap(flowGenration, args)

    flow_df = pd.concat(
        [*results]
    )  # Concatinating the flow data from all the processes
    del results  # Deleting the results to free up the memory
    print(f"{datetime.now()} Flow Generation Completed\n")
    return flow_df


def flowGenration(stdf: pd.DataFrame, raw_df: TrajDataFrame) -> pd.DataFrame:

    flow_df = []
    cols = [
        "uid",
        "org_lat",
        "org_lng",
        "org_arival_time",
        "org_leaving_time",
        "dest_lat",
        "dest_lng",
        "dest_arival_time",
        "stay_points",
        "trip_points",
        "trip_time",
        "stay_duration",
        "observed_stay_duration",
    ]
    for ind, row in tqdm(stdf.iterrows()):
        flow_df.append(fetchDataFromRaw(row, raw_df))

    # Converting list to Dataframe
    flow_df = pd.DataFrame(flow_df, columns=cols)

    return flow_df


def fetchDataFromRaw(record: pd.Series, raw_df: TrajDataFrame) -> list:
    org_at = record["datetime"]  # Origin arriving time
    org_lt = record["leaving_datetime"]  # origin leaving time
    dest_at = record["dest_at"]  # destination arriving time

    stay_points = json.dumps(
        raw_df.loc[(record["uid"], org_at) : (record["uid"], org_lt)][["lat", "lng"]]
        .values[0:-1]
        .tolist()
    )  # points inside the stop nodes= all the points between the time of first point inside cluster and time of first point outside the cluster
    trip_points = json.dumps(
        raw_df.loc[(record["uid"], org_lt) : (record["uid"], dest_at)][["lat", "lng"]]
        .values[0:-1]
        .tolist()
    )  # points in between two stop nodes= all the points between the time of first point outside the origin stay node and the time of first point inside the destination stop node

    stay_duration = round(
        (org_lt - org_at).total_seconds() / 60
    )  # How long stay inside the stop node (calculated using the leaving time generated by Scikit-mob)= The time of the first point outside the stop node - the time of the first point inside the stop node
    org_last_point_time = raw_df.loc[(record["uid"], org_at) : (record["uid"], org_lt)][
        ["lat", "lng"]
    ].index.tolist()[-2][
        1
    ]  # Time of the last point inside the stop node
    observed_stay_duration = (
        org_last_point_time - org_at
    ).total_seconds() / 60  # observed stay= The time of the last point inside the stop node - the time of the first point inside the stop node

    trip_time = round(
        (dest_at - org_lt).total_seconds() / 60
    )  # Trip time= destination arrival time - orgin leaving time

    temp = [
        record["uid"],
        record["org_lat"],  # org_lat,
        record["org_lng"],  # org_lng,
        org_at,
        org_lt,
        record["dest_lat"],  # dst_lat,
        record["dest_lng"],  # dst_lng,
        dest_at,
        stay_points,
        trip_points,
        trip_time,
        stay_duration,
        observed_stay_duration,
    ]
    return temp


def getActivityStats(
    df: pd.DataFrame, output_dir: str, cpu_cores: int = max(1, int(cpu_count() / 2))
) -> pd.DataFrame:
    """
    Compute per-month user activity (number of active days) in parallel and save to disk.

    This function partitions the input DataFrame into load-balanced buckets (based on
    unique users), processes each bucket in parallel, and then combines the results.
    Each row in the returned DataFrame corresponds to a specific user and month, with
    a column indicating how many days that user was active during that month.

    Key Points:
    - Requires at least the columns "uid" and "datetime" in the input DataFrame.
    - Uses multiprocessing to handle large datasets efficiently, controlled by `cpu_cores`.
    - Saves the final aggregated statistics to "activity_stats.csv" in the provided output directory.
    - The returned DataFrame has columns:
        * "uid"
        * "month"
        * "total_active_days"  (number of unique days in that month with at least one record)
    - Designed to produce monthly-level stats from typically yearly data. If you need
      a yearly total, aggregate "total_active_days" across all months per user before
      using these stats in any further steps (like OD generation).

    Args:
        df (pd.DataFrame):
            The input DataFrame containing at least the columns "uid" and "datetime".
        output_dir (str):
            Path where the resulting "activity_stats.csv" file will be saved.
        cpu_cores (int, optional):
            Number of CPU cores to use for multiprocessing. Defaults to half of the
            available cores (at least 1).

    Returns:
        pd.DataFrame:
            A DataFrame of monthly user activity counts, with columns ["uid", "month",
            "total_active_days"].

    Example:
        >>> from meowmotion.meowmob import getActivityStats
        >>> # Suppose df has columns: uid, datetime, lat, lng, etc.
        >>> activity_df = getActivityStats(df, output_dir="./stats", cpu_cores=4)
        >>> activity_df.head()
           uid  month  total_active_days
        0    1      1                 10
        1    1      2                 12
        2    2      1                  8
    """
    print(f"{datetime.now()}: Generating Activity Stats")
    init_unique_users = df["uid"].nunique()
    tdf_collection = getLoadBalancedBuckets(df, cpu_cores)
    with Pool(cpu_cores) as p:
        df = p.map(activityStats, tdf_collection)

    df = pd.concat(df, ignore_index=True)
    df = df.reset_index(drop=True)
    final_unique_users = df["uid"].nunique()
    assert (
        init_unique_users == final_unique_users
    ), "Something is wrong..data Loss in Activity Stats Generation"
    print(f"{datetime.now()}: Activity Stats generated.")
    print(f"{datetime.now()}: Saving Activity Stats")
    saveFile(path=f"{output_dir}/activity_stats", fname="activity_stats.csv", df=df)
    return df


def activityStats(df: pd.DataFrame) -> pd.DataFrame:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df = (
        df.drop_duplicates(subset=["uid", "month", "day"], keep="first")
        .groupby(["uid", "month"])["day"]
        .size()
        .reset_index()
        .rename(columns={"day": "total_active_days"})
    )
    return df


def generateOD(
    trip_df: pd.DataFrame,
    shape: gpd.GeoDataFrame,
    active_day_df: pd.DataFrame,
    hldf: pd.DataFrame,
    adult_population: pd.DataFrame,
    output_dir: str,
    org_loc_cols: Optional[Tuple[str, str]] = ("org_lng", "org_lat"),  # (lng, lat)
    dest_loc_cols: Optional[Tuple[str, str]] = ("dest_lng", "dest_lat"),  # (lng, lat)
    cpu_cores: Optional[int] = max(1, cpu_count() // 2),
    save_drived_products: Optional[bool] = True,
    od_type: Optional[List[str]] = ["type3"],
) -> List[pd.DataFrame]:
    """
    Generate weighted Origin-Destination (OD) matrices from trip-level data, using
    spatial joins, demographic weights, and user activity data. This function
    leverages multiprocessing to handle large datasets efficiently and can produce
    multiple types of OD matrices in a single pass.

    Key Steps:
    1. **Shape File Preparation**:
       - Ensures the provided `shape` GeoDataFrame uses EPSG:4326.
       - Pre-builds a spatial index for quicker joins.

    2. **Spatial Joins**:
       - Splits `trip_df` into load-balanced buckets (via `getLoadBalancedBuckets`)
         for parallel processing.
       - Spatially joins origins and destinations against the `shape` to label each
         trip with "origin_geo_code" and "destination_geo_code".

    3. **Filtering**:
       - Removes trips longer than 24 hours and stay durations over 3600 minutes.
       - Drops records without valid origin or destination geo-codes.

    4. **Disclosure Analysis**:
       - Aggregates trip counts by origin-destination pairs and user IDs to help
         identify any potential risk of user-level data disclosure.
       - Saves results in "disclosure_analysis_.csv".

    5. **Trip ID & Metrics**:
       - Assigns incremental `trip_id`s per user.
       - Computes total trips per user and merges with `active_day_df` to
         calculate "trips per active day" (TPAD).

    6. **Adding Demographic Data**:
       - Merges each record with user-level IMD quintiles and council info
         from `hldf`.
       - Adds placeholder columns for travel mode if needed.

    7. **Optional Saving of Intermediate Products** (if `save_drived_products=True`):
       - Saves non-aggregated flows, aggregated flows, stay points, and trip points
         in separate CSV files for further analysis.

    8. **Final OD Matrix Generation**:
       - Filters out infrequent or low-activity users based on active days and TPAD.
       - For each OD type in `od_type` (e.g., "type1", "type2", "type3", "type4"),
         selects trips matching the time-of-day/week criteria.
       - Applies weighting (`getWeights`) to scale user trip counts to
         population-level estimates.
       - Aggregates trips, then calculates weighted trips with different weighting
         factors (activity, council, IMD) for each origin-destination pair.
       - Saves the resulting OD matrix as a CSV (e.g., "type3_od.csv") and collects
         it in a list of OD DataFrames to be returned.

    Args:
        trip_df (pd.DataFrame):
            The main trip-level DataFrame. Must contain columns indicating user IDs,
            timestamps (arrivals/departures), plus the origin/destination lat-lng
            pairs (specified by `org_loc_cols` and `dest_loc_cols`).
        shape (gpd.GeoDataFrame):
            A GeoDataFrame containing the geographic boundaries (e.g., MSOA or LSOA).
            Must have a valid geometry column. This is used for spatial joins.
        active_day_df (pd.DataFrame):
            DataFrame with columns ["uid", "total_active_days"], representing how
            many days each user was active.
        hldf (pd.DataFrame):
            DataFrame mapping user IDs to home council and IMD quintile info.
        adult_population (pd.DataFrame):
            Contains population counts broken down by council and IMD quintile.
        output_dir (str):
            Directory path where all output files will be saved.
        org_loc_cols (Tuple[str, str], optional):
            Column names for the origin's (longitude, latitude).
            Defaults to ("org_lng", "org_lat").
        dest_loc_cols (Tuple[str, str], optional):
            Column names for the destination's (longitude, latitude).
            Defaults to ("dest_lng", "dest_lat").
        cpu_cores (int, optional):
            Number of CPU cores to use for parallel processing. Defaults to half
            of available cores (at least 1).
        save_drived_products (bool, optional):
            Whether to save intermediate or "derived" datasets (e.g., stay points).
            Defaults to True.
        od_type (List[str], optional):
            Which OD matrix types to produce. Recognized values:
            - "type1": AM Peak Weekdays (7am–10am)
            - "type2": PM Peak Weekdays (4pm–7pm)
            - "type3": All Trips (default)
            - "type4": All Trips excluding type1 + type2
            Passing multiple values produces multiple OD DataFrames. Defaults to ["type3"].

    Returns:
        List[pd.DataFrame]:
            A list of OD matrix DataFrames, one for each type listed in `od_type`.
            Each DataFrame has columns like:
            - "origin_geo_code", "destination_geo_code"
            - "trips" (unweighted)
            - "activity_weighted_trips"
            - "council_weighted_trips"
            - "act_cncl_weighted_trips" (combined weighting)
            - "percentage" (percentage share of total trips)

    Example:
        >>> from meowmotion.meowmob import generateOD
        >>> od_matrices = generateOD(
                trip_df=trip_data,
                shape=lsoa_shapes,
                active_day_df=active_days,
                hldf=home_locations,
                adult_population=population_stats,
                org_loc_cols=('org_lng', 'org_lat'),
                dest_loc_cols=('dest_lng', 'dest_lat'),
                output_dir='./output',
                cpu_cores=4,
                od_type=["type3", "type1"]
            )
        >>> print(od_matrices[0].head())  # OD matrix for "type3"
    """

    print(f"{datetime.now()}: Current CRS {shape.crs}")
    shape = shape.to_crs("EPSG:4326")
    print(f"{datetime.now()}: CRS after Conversion: {shape.crs}")
    print(f"{datetime.now()}: Indexing Shape File")
    shape.sindex

    #############################################################
    #                                                           #
    #                   Spatial Join for Origin                 #
    #                                                           #
    #############################################################

    df_collection = getLoadBalancedBuckets(trip_df, cpu_cores)
    print(f"{datetime.now()}: Spatial Join for Origin Started")
    # args=[(tdf, shape, 'org_lng', 'org_lat', 'origin') for tdf in df_collection]
    args = [
        (tdf, shape, org_loc_cols[0], org_loc_cols[1], "origin")
        for tdf in df_collection
    ]
    with Pool(cpu_cores) as pool:
        results = pool.starmap(spatialJoin, args)
    df_collection = [*results]
    print(f"{datetime.now()}: Spatial Join for Origin Finished")

    #############################################################
    #                                                           #
    #                  Spatial Join for Destination             #
    #                                                           #
    #############################################################

    print(f"{datetime.now()}: Spatial Join for Destination Started")
    # args=[(tdf, shape, 'dest_lng', 'dest_lat', 'destination') for tdf in df_collection]
    args = [
        (tdf, shape, dest_loc_cols[0], dest_loc_cols[1], "destination")
        for tdf in df_collection
    ]
    with Pool(cpu_cores) as pool:
        results = pool.starmap(spatialJoin, args)
    geo_df = pd.concat([*results])
    del results
    print(f"{datetime.now()}: Spatial Join for Destination Finished")
    #############################################################
    #                                                           #
    # Filtering trips based on travel time and stay duration    #
    #                                                           #
    #############################################################

    print(f"{datetime.now()}: Filtering on Travel Time and Stay Duration")
    geo_df = geo_df[
        (geo_df["dest_arival_time"] - geo_df["org_leaving_time"]).dt.total_seconds()
        / 3600
        <= 24
    ]
    geo_df = geo_df[geo_df["stay_duration"] <= 3600]
    nusers = geo_df["uid"].nunique()
    print(f"{datetime.now()}: Total Unique Users: {nusers}")
    geo_df["origin_geo_code"] = geo_df["origin_geo_code"].fillna("Others")
    geo_df["destination_geo_code"] = geo_df["destination_geo_code"].fillna("Others")
    geo_df = geo_df[geo_df["origin_geo_code"] != "Others"]
    geo_df = geo_df[geo_df["destination_geo_code"] != "Others"]
    print(f"{datetime.now()}: Filtering Completed")

    #############################################################
    #                                                           #
    #                   Disclosure Analysis                     #
    #                                                           #
    #############################################################

    print(f"{datetime.now()}: Generating file for disclosure analysis")
    analysis_df = (
        geo_df.groupby(["origin_geo_code", "destination_geo_code"])
        .agg(
            total_trips=pd.NamedAgg(column="uid", aggfunc="count"),
            num_users=pd.NamedAgg(column="uid", aggfunc="nunique"),
        )
        .reset_index()
    )

    print(f"{datetime.now()}: Saving disclosure analysis file")
    saveFile(
        path=f"{output_dir}/disclosure_analysis",
        fname="disclosure_analysis_.csv",
        df=analysis_df,
    )
    print(f"{datetime.now()}: Saved disclosure analysis file")

    ############################################################
    #                                                          #
    #                   Adding Trip ID                         #
    #                                                          #
    ############################################################

    print(f"{datetime.now()}: Adding Trip ID")
    geo_df = geo_df.assign(
        trip_id=lambda df: df.groupby(["uid"])["trip_time"].transform(
            lambda x: [i for i in range(1, len(x) + 1)]
        )
    )

    # first_cols = ['uid', 'trip_id']
    # other_cols = [col for col in geo_df.columns if col not in first_cols]
    # geo_df = geo_df[first_cols + other_cols]

    geo_df = geo_df[
        [
            "uid",
            "trip_id",
            "org_lat",
            "org_lng",
            "org_arival_time",
            "org_leaving_time",
            "dest_lat",
            "dest_lng",
            "dest_arival_time",
            "stay_points",
            "trip_points",
            "trip_time",
            "stay_duration",
            "observed_stay_duration",
            "origin_geo_code",
            "origin_name",
            "destination_geo_code",
            "destination_name",
        ]
    ]
    print(f"{datetime.now()}: Trip ID Added")

    #############################################################
    #                                                           #
    #                    Calculate Total Trips/User             #
    #                                                           #
    #############################################################

    print(f"{datetime.now()}: Calculating Total Trips/User")
    # geo_df["month"] = geo_df["org_leaving_time"].dt.month
    geo_df = geo_df.assign(
        total_trips=lambda df: df.groupby("uid")["trip_id"].transform(lambda x: len(x))
    )
    # geo_df = geo_df.drop(columns=["month"])
    print(f"{datetime.now()}: Trips/User Calculated")

    #############################################################
    #                                                           #
    #                   Add Trips/Active Day                    #
    #                                                           #
    #############################################################

    print(f"{datetime.now()}: Calculating TPAD")
    geo_df = geo_df.merge(active_day_df, how="left", on="uid").assign(
        tpad=lambda tdf: tdf["total_trips"] / tdf["total_active_days"]
    )
    print(f"{datetime.now()}: TPAD Calculated")

    #############################################################
    #                                                           #
    #                       Add IMD Level                       #
    #                                                           #
    #############################################################

    print(f"{datetime.now()}: Adding IMD")
    geo_df = geo_df.merge(
        hldf[["uid", "council_name", "imd_quintile"]], on="uid", how="left"
    )[
        [
            "uid",
            "council_name",
            "imd_quintile",
            "trip_id",
            "org_lat",
            "org_lng",
            "org_arival_time",
            "org_leaving_time",
            "dest_lat",
            "dest_lng",
            "origin_geo_code",
            "destination_geo_code",
            "dest_arival_time",
            "stay_points",
            "trip_points",
            "trip_time",
            "stay_duration",
            "observed_stay_duration",
            "total_trips",
            "total_active_days",
            "tpad",
        ]
    ]

    #############################################################
    #                                                           #
    #                Add Travel Mode Placeholder                #
    #                                                           #
    #############################################################

    geo_df = geo_df.assign(travel_mode=np.nan)

    if save_drived_products:

        #############################################################
        #                                                           #
        #              Save Aggregated Flow                         #
        #                                                           #
        #############################################################

        print(f"{datetime.now()}: Saving Non-Aggregated OD Flow")
        saveFile(
            path=f"{output_dir}/na_flows",
            fname="na_flows.csv",
            df=geo_df[
                [
                    "uid",
                    "imd_quintile",
                    "trip_id",
                    "org_lat",
                    "org_lng",
                    "org_arival_time",
                    "org_leaving_time",
                    "dest_lat",
                    "dest_lng",
                    "dest_arival_time",
                    "total_trips",
                    "total_active_days",
                    "tpad",
                    "travel_mode",
                ]
            ],
        )
        print(f"{datetime.now()}: Non-Aggregated OD Flow Saved")

        #############################################################
        #                                                           #
        #              Save Aggregated Flow                         #
        #                                                           #
        #############################################################

        print(f"{datetime.now()}: Saving Aggregated OD Flow")

        saveFile(
            path=f"{output_dir}/agg_stay_points",
            fname="agg_stay_points.csv",
            df=geo_df[
                [
                    "origin_geo_code",
                    "destination_geo_code",
                    "org_arival_time",
                    "org_leaving_time",
                    "dest_arival_time",
                    "travel_mode",
                ]
            ],
        )
        print(f"{datetime.now()}: Aggregated OD Flow Saved")

        #############################################################
        #                                                           #
        #              Save Non Aggregated Stay Points              #
        #                                                           #
        #############################################################

        print(f"{datetime.now()}: Saving Non-Aggragated Stay Points")

        saveFile(
            path=f"{output_dir}/non_agg_stay_points",
            fname="non_agg_stay_points.csv",
            df=geo_df[
                [
                    "uid",
                    "imd_quintile",
                    "stay_points",
                    "org_arival_time",
                    "org_leaving_time",
                    "stay_duration",
                    "org_lat",
                    "org_lng",
                    "total_active_days",
                ]
            ].rename(
                columns={
                    "org_lat": "centroid_lat",  # Changing the name to centroid because stay points don't have origin and destination
                    "org_lng": "centroid_lng",
                    "org_arival_time": "stop_node_arival_time",
                    "org_leaving_time": "stop_node_leaving_time",
                }
            ),
        )

        print(f"{datetime.now()}: Non-Aggragated Stay Points Saved")

        #############################################################
        #                                                           #
        #                  Save Aggregated Stay Points              #
        #                                                           #
        #############################################################

        print(f"{datetime.now()}: Saving Aggragated Stay Points")

        saveFile(
            path=f"{output_dir}/agg_stay_points",
            fname="agg_stay_points.csv",
            df=geo_df[
                [
                    "imd_quintile",
                    "origin_geo_code",
                    "org_arival_time",
                    "org_leaving_time",
                    "stay_duration",
                ]
            ].rename(
                columns={
                    "org_arival_time": "stop_node_arival_time",
                    "org_leaving_time": "stop_node_leaving_time",
                    "origin_geo_code": "stop_node_geo_code",
                }
            ),
        )

        print(f"{datetime.now()}: Aggragated Stay Points Saved")

        #############################################################
        #                                                           #
        #                      Save Trip Points                     #
        #                                                           #
        #############################################################

        print(f"{datetime.now()}: Saving Trips Points")

        saveFile(
            path=f"{output_dir}/trip_points",
            fname="trip_points.csv",
            df=geo_df[
                [
                    "uid",
                    "imd_quintile",
                    "trip_id",
                    "trip_points",
                    "total_active_days",
                    "travel_mode",
                ]
            ],
        )

        print(f"{datetime.now()}: Trips Points Saved")

    ##################################################################################
    #                                                                                #
    #                           OD Generation                                        #
    #                                                                                #
    ##################################################################################

    print(f"{datetime.now()}: OD Calculation Started")
    geo_df = geo_df[
        (geo_df["total_active_days"] >= 7) & (geo_df["tpad"] >= 0.2)
    ]  # Filtering based on number of active days and trips/active day

    print(f"{datetime.now()}: Total Trips: {len(geo_df)}")
    print(f'{datetime.now()}: Total Users: {len(geo_df["uid"].unique())}')
    print(f'{datetime.now()}: TPAD Stats:\n{geo_df["tpad"].describe()}')
    od_trip_df = pd.DataFrame(
        geo_df.groupby(["uid", "origin_geo_code", "destination_geo_code"]).apply(
            lambda x: len(x)
        ),
        columns=["trips"],
    ).reset_index()  # Get number of Trips between orgins and destination for individual users
    od_trip_df = od_trip_df.merge(
        active_day_df, how="left", left_on="uid", right_on="uid"
    ).assign(tpad=lambda tdf: tdf["trips"] / tdf["total_active_days"])

    print(f"{datetime.now()}: Calculating Weights")
    weighted_trips = getWeights(
        geo_df,
        hldf,
        adult_population,
        "origin_geo_code",
        "destination_geo_code",
        active_day_df,
    )
    weighted_trips = weighted_trips[
        ["uid", "imd_weight", "council_weight", "activity_weight"]
    ]
    weighted_trips = weighted_trips.drop_duplicates(subset="uid", keep="first")
    print(f"{datetime.now()}: Weights Calculated")
    data_population = len(geo_df["uid"].unique())  # Total number of users in the data
    adult_population = adult_population["Total"].sum()  # Total population

    # Producing 5 Type of OD Matrices
    # Type 1: AM peak weekdays (7am-10am)
    # Type 2: PM peak weekdays (4 pm-7 pm)
    # Type 3: Everything
    # Type 4: Type 3 - (Type 1 + Type 2)

    type_meta = {
        "type1": "AM Peak Weekdays (7am-10am)",
        "type2": "PM Peak Weekdays (4 pm-7 pm)",
        "type3": "All (Everything)",
        "type4": "All - (AM Peak + PM Peak)",
    }
    return_ods = []
    for typ in od_type:
        print(f"{datetime.now()}: Generating {type_meta[typ]} OD Matrix")
        if typ == "type1":
            geo_df_filtered = geo_df[
                (geo_df["org_leaving_time"].dt.hour >= 7)
                & (geo_df["org_leaving_time"].dt.hour <= 10)
                & (geo_df["org_leaving_time"].dt.dayofweek < 5)
            ]
        elif typ == "type2":
            geo_df_filtered = geo_df[
                (geo_df["org_leaving_time"].dt.hour >= 16)
                & (geo_df["org_leaving_time"].dt.hour <= 19)
                & (geo_df["org_leaving_time"].dt.dayofweek < 5)
            ]
        elif typ == "type3":
            geo_df_filtered = geo_df.copy()  # No filtering for type3
        elif typ == "type4":
            geo_df_filtered = geo_df[
                ~(
                    (geo_df["org_leaving_time"].dt.hour >= 7)
                    & (geo_df["org_leaving_time"].dt.hour <= 10)
                    & (geo_df["org_leaving_time"].dt.dayofweek < 5)
                )
            ]
            geo_df = geo_df[
                ~(
                    (geo_df["org_leaving_time"].dt.hour >= 16)
                    & (geo_df["org_leaving_time"].dt.hour <= 19)
                    & (geo_df["org_leaving_time"].dt.dayofweek < 5)
                )
            ]
        else:
            raise ValueError(f"Invalid OD type: {typ}. Must be one of {type_meta}.")

        print(f"{datetime.now()}: Generating OD trip DF")
        od_trip_df = pd.DataFrame(
            geo_df_filtered.groupby(
                ["uid", "origin_geo_code", "destination_geo_code"]
            ).apply(lambda x: len(x)),
            columns=["trips"],
        ).reset_index()  # Get number of Trips between orgins and destination for individual users
        print(f"{datetime.now()}: Adding weights to OD trips")
        od_trip_df = od_trip_df.merge(
            weighted_trips[["uid", "activity_weight", "imd_weight", "council_weight"]],
            how="left",
            on="uid",
        )
        od_trip_df["imd_weight"] = od_trip_df["imd_weight"].fillna(1)
        od_trip_df["council_weight"] = od_trip_df["council_weight"].fillna(1)
        od_trip_df.reset_index(drop=True, inplace=True)
        print(f"{datetime.now()}: Aggregating trips")
        agg_od_df = (
            od_trip_df.groupby(["origin_geo_code", "destination_geo_code"])
            .agg(
                trips=("trips", "sum"),
                activity_weighted_trips=(
                    "trips",
                    lambda x: (
                        (x * od_trip_df.loc[x.index, "activity_weight"]).sum()
                        / data_population
                    )
                    * adult_population,
                ),
                council_weighted_trips=(
                    "trips",
                    lambda x: (
                        (
                            x
                            * od_trip_df.loc[x.index, "imd_weight"]
                            * od_trip_df.loc[x.index, "council_weight"]
                        ).sum()
                        / data_population
                    )
                    * adult_population,
                ),
                act_cncl_weighted_trips=(
                    "trips",
                    lambda x: (
                        (
                            x
                            * od_trip_df.loc[x.index, "activity_weight"]
                            * od_trip_df.loc[x.index, "imd_weight"]
                            * od_trip_df.loc[x.index, "council_weight"]
                        ).sum()
                        / data_population
                    )
                    * adult_population,
                ),
            )
            .reset_index()
        )

        agg_od_df = agg_od_df[agg_od_df["origin_geo_code"] != "Others"]
        agg_od_df = agg_od_df[agg_od_df["destination_geo_code"] != "Others"]

        print(f"{datetime.now()}: OD Generation Completed")
        print(f"{datetime.now()}: Saving OD")
        agg_od_df["percentage"] = (
            agg_od_df["act_cncl_weighted_trips"]
            / agg_od_df["act_cncl_weighted_trips"].sum()
        ) * 100

        agg_od_df = agg_od_df.rename(
            columns={"act_cncl_weighted_trips": "trips_weighted"}
        )
        agg_od_df = agg_od_df[
            [
                "origin_geo_code",
                "destination_geo_code",
                "trips",
                "trips_weighted",
                "percentage",
            ]
        ]

        saveFile(path=f"{output_dir}/od_matrix", fname=f"{typ}_od.csv", df=agg_od_df)
        return_ods.append(agg_od_df)

    return return_ods


def getWeights(
    geo_df: pd.DataFrame,
    hldf: pd.DataFrame,
    adult_population: pd.DataFrame,
    origin_col: str,
    destination_col: str,
    active_day_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes activity-based, IMD-level, and council-level weights for users to scale
    observed trips to population-level estimates.

    Args:
        geo_df (pd.DataFrame): Geo-tagged trip DataFrame containing user ID and trip counts.
        hldf (pd.DataFrame): Home location and demographic info including IMD and council.
        adult_population (pd.DataFrame): Population statistics broken down by IMD and council.
        origin_col (str): Name of the column containing origin geo code.
        destination_col (str): Name of the column containing destination geo code.
        active_day_df (pd.DataFrame): DataFrame with total number of active days per user.

    Returns:
        pd.DataFrame: DataFrame with user-level weights including:
            - `imd_weight`
            - `council_weight`
            - `activity_weight`

    Example:
        >>> weighted_df = getWeights(
                geo_df=geo_enriched_data,
                hldf=home_locations,
                adult_population=population_stats,
                origin_col="origin_geo_code",
                destination_col="destination_geo_code",
                active_day_df=active_days
            )
        >>> print(weighted_df[['uid', 'activity_weight', 'imd_weight']].head())
    """
    od_trip_df = pd.DataFrame(
        geo_df.groupby(["uid", origin_col, destination_col]).apply(lambda x: len(x)),
        columns=["trips"],
    ).reset_index()  # Get number of Trips between orgins and destination for individual users
    od_trip_df = od_trip_df.merge(active_day_df, how="left", on="uid").assign(
        tpad=lambda tdf: tdf["trips"] / tdf["total_active_days"]
    )
    od_trip_df = pd.merge(
        od_trip_df, hldf[["uid", "council_name", "imd_quintile"]], how="left", on="uid"
    )
    od_trip_df = od_trip_df.rename(columns={"council_name": "user_home_location"})

    # Calculating Weights Based in Adult Population and data Population

    annual_users = (
        od_trip_df.dropna(subset=["imd_quintile"])
        .groupby(["user_home_location", "imd_quintile"])
        .agg(users=("uid", "nunique"))
        .reset_index()
        .merge(
            adult_population,
            left_on=["user_home_location", "imd_quintile"],
            right_on=["council", "imd_quintile"],
            how="left",
        )
        .groupby("user_home_location", group_keys=True)
        .apply(
            lambda group: group.assign(
                data_percent=group["users"] / group["users"].sum()
            )
        )
        .reset_index(drop=True)
        .assign(imd_weight=lambda df: df["Percentage"] / df["data_percent"])
        .groupby("user_home_location", group_keys=True)
        .apply(
            lambda group: group.assign(
                total_pop=group["Total"].sum(), data_pop=group["users"].sum()
            )
        )
        .reset_index(drop=True)
        .assign(
            council_weight=lambda df: (df["total_pop"] / df["Total"].sum())
            / (df["data_pop"] / df["users"].sum())
        )
    )

    annual_users = annual_users[  # Rearranging Columns
        [
            "council",
            "imd_quintile",
            "users",
            "Total",
            "Percentage",
            "data_percent",
            "total_pop",
            "data_pop",
            "imd_weight",
            "council_weight",
        ]
    ]

    annual_users = annual_users.rename(
        columns={
            "users": "data_user_imd_level",
            "Total": "adult_pop_imd_level",
            "percentage": "adult_pop_percentage_imd_level",
            "data_percent": "data_users_percentage_imd_level",
            "total_pop": "adult_pop_council_level",
            "data_pop": "data_users_council_level",
        }
    )

    od_trip_df = od_trip_df.merge(
        annual_users[["council", "imd_quintile", "imd_weight", "council_weight"]],
        how="left",
        left_on=["user_home_location", "imd_quintile"],
        right_on=["council", "imd_quintile"],
        suffixes=["_od", "_anu"],
    )
    od_trip_df["imd_weight"] = od_trip_df["imd_weight"].fillna(1)
    od_trip_df["council_weight"] = od_trip_df["council_weight"].fillna(1)
    od_trip_df["activity_weight"] = (
        365 / od_trip_df["total_active_days"]
    )  # Activity weight = 365 (total days in a year) / number of active days
    return od_trip_df
