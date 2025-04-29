import gzip
import io
import json
import os
import zipfile
from datetime import datetime
from multiprocessing import Pool, cpu_count
from os.path import join
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from skmob import TrajDataFrame
from skmob.preprocessing import filtering


def readJsonFiles(root: str, month_file: str) -> pd.DataFrame:
    """
    Load a month-worth of impression records stored as *gzipped JSON-Lines*
    inside a ZIP archive and return them as a tidy DataFrame.

    Data-at-Rest Format
    -------------------
    The function expects the following directory / file structure:

    ``root/
        2023-01.zip              # <- month_file argument
            2023-01-01-00.json.gz
            2023-01-01-01.json.gz
            ...
            2023-01-31-23.json.gz
    ```

    * Each ``.json.gz`` file is a **JSON-Lines** file (one JSON object per line).
    * Every JSON object is expected to contain at least these keys:

      - ``impression_acc``  (float) – GNSS accuracy (metres)
      - ``device_iid_hash`` (str)   – Anonymised user or device ID
      - ``impression_lng``  (float) – Longitude in WGS-84
      - ``impression_lat``  (float) – Latitude  in WGS-84
      - ``timestamp``       (str/int) – ISO-8601 string *or* Unix epoch (ms)

    The loader iterates through each ``.json.gz`` in the archive, parses every
    line, and extracts the subset of fields listed above.

    Args:
        root (str):
            Path to the directory that contains *month_file* (e.g.
            ``"/data/impressions"``).
        month_file (str):
            Name of the ZIP archive to read
            (e.g. ``"2023-01.zip"`` or ``"london_2024-06.zip"``).

    Returns:
        pandas.DataFrame:
            Columns → ``["impression_acc", "uid", "lng", "lat", "datetime"]``
            One row per JSON object across all ``.json.gz`` files in the archive.

    Example:
        >>> df = readJsonFiles("/data/impressions", "2023-01.zip")
        >>> df.head()
             impression_acc             uid        lng        lat             datetime
        0              6.5  a1b2c3d4e5f6g7h8  -0.12776   51.50735   2023-01-01T00:00:10Z
        1              4.8  h8g7f6e5d4c3b2a1  -0.12800   51.50720   2023-01-01T00:00:11Z
        ...
    """

    print(f"{datetime.now()}: Processing {month_file}")

    data = []
    month_zip_file = f"{root}/{month_file}"

    with zipfile.ZipFile(month_zip_file, "r") as zf:
        for gz_file in zf.namelist():
            print(f"{datetime.now()}: Processing {gz_file}")
            with zf.open(gz_file) as f, gzip.GzipFile(fileobj=f, mode="r") as g:
                for line in io.TextIOWrapper(g, encoding="utf-8"):
                    json_obj = json.loads(line.strip())
                    data.append(
                        {
                            "impression_acc": json_obj.get("impression_acc"),
                            "uid": json_obj.get("device_iid_hash"),
                            "lng": json_obj.get("impression_lng"),
                            "lat": json_obj.get("impression_lat"),
                            "datetime": json_obj.get("timestamp"),
                        }
                    )

    df = pd.DataFrame(data)
    print(f"{datetime.now()}: {month_file} processed.")

    return df


def getFilteredData(
    df: pd.DataFrame,
    impr_acc: Optional[int] = 100,
    cpu_cores: Optional[int] = max(1, int(cpu_count() / 2)),
) -> TrajDataFrame:
    """
    Parallel, two–stage cleansing of raw impression data that

    1. **drops points whose GNSS accuracy** (``impression_acc``) exceeds the
       user-specified threshold, and
    2. **removes physically implausible jumps** using scikit-mob’s
       :pyfunc:`skmob.preprocessing.filtering.filter`
       (``max_speed_kmh=200`` by default).

    The work is split into load-balanced buckets and processed concurrently
    with :pyclass:`multiprocessing.Pool`.

    Args:
        df (pd.DataFrame):
            Point-level impressions with *at least* the columns
            ``["uid", "lat", "lng", "datetime", "impression_acc"]``
            (plus any additional attributes you want to keep).
        impr_acc (int, optional):
            Maximum allowed GNSS accuracy in **metres**. Points with a larger
            ``impression_acc`` are discarded. Defaults to ``100``.
        cpu_cores (int, optional):
            Number of CPU cores to devote to multiprocessing. By default, half
            of the available logical cores (but at least 1).

    Returns:
        TrajDataFrame:
            A scikit-mob ``TrajDataFrame`` containing only points that pass
            both the accuracy and speed filters, with its original columns
            preserved.

    Example:
        >>> clean_traj = getFilteredData(raw_df, impr_acc=50, cpu_cores=8)
        >>> print(clean_traj.shape)
    """

    print(f"{datetime.now()}: Filtering data based on impression accuracy={impr_acc}")
    print(f"{datetime.now()}: Creating buckets for multiprocessing")
    tdf_collection = getLoadBalancedBuckets(df, cpu_cores)
    args = [(tdf, impr_acc) for tdf in tdf_collection if not tdf.empty]
    print(f"{datetime.now()}: Filtering Started...")
    with Pool(cpu_cores) as pool:
        results = pool.starmap(
            filterData, args
        )  # Filtering the data based on Impression Accuracy and Speed between GPS points

    del tdf_collection  # Deleting the data to free up the memory
    traj_df = pd.concat(
        [*results]
    )  # Concatinating the filtered data from all the processes
    del results  # Deleting the results to free up the memory
    print(f"{datetime.now()}: Filtering Finished\n\n\n")

    return traj_df


def filterData(df: pd.DataFrame, impr_acc: int) -> TrajDataFrame:

    traj_df = TrajDataFrame(
        df, latitude="lat", longitude="lng", user_id="uid", datetime="datetime"
    )
    print(f"Filtering based on impression accuracy={impr_acc}")
    bf = traj_df.shape[0]
    traj_df = traj_df[traj_df["impression_acc"] <= impr_acc]
    af = traj_df.shape[0]

    try:
        print(
            f"""
        Records before impression accuracy filtering: {bf}
        Records after impression accuracy filtering: {af}
        Difference: {bf-af}
        Percentage of deleted record after impression accuracy filter: {round(((bf-af)/bf)*100)}%
        """
        )
    except ZeroDivisionError:
        raise ValueError(
            """
            Cannot calculate percentage of deleted records as no records were found before impression accuracy filtering.
            """
        )

    # Filtering based on the speed

    print("Filtering based on the speed in between two consecutive GPS points...")

    traj_df = filtering.filter(traj_df, max_speed_kmh=200)

    try:
        print(
            f"""
        Records before speed filtering: {af}
        Records after speed filtering: {traj_df.shape[0]}
        Difference: {af-traj_df.shape[0]}
        Percentage of deleted record after speed filtering: {round(((af-traj_df.shape[0])/af)*100,2)}%
        """
        )
    except ZeroDivisionError:
        raise ValueError(
            """
            Cannot calculate percentage of deleted records as no records were found before speed filtering.
            """
        )
    return traj_df


def getLoadBalancedBuckets(tdf: pd.DataFrame, bucket_size: int) -> list:
    """
    Partition a user-level DataFrame into *bucket_size* sub-DataFrames whose
    total row counts (i.e. number of “impressions”) are as evenly balanced as
    possible.  Each bucket can then be processed in parallel on its own CPU
    core.

    Algorithm
    ---------
    1. Count the number of rows (“impressions”) for every unique ``uid``.
    2. Sort users in descending order of impression count.
    3. Greedily assign each user to the bucket that currently has the
       **smallest** total number of impressions (*load-balancing heuristic*).
    4. Build one DataFrame per bucket containing only the rows for the users
       assigned to that bucket.
    5. Return the list of non-empty bucket DataFrames.

    Args:
        tdf (pd.DataFrame):
            A DataFrame that **must contain a ``"uid"`` column** plus any other
            fields.  Each row represents one GPS impression or point.
        bucket_size (int):
            The desired number of buckets—typically equal to the number of CPU
            cores you plan to use with :pyclass:`multiprocessing.Pool`.

    Returns:
        list[pd.DataFrame]:
            A list whose length is **≤ *bucket_size***.  Each element is a
            DataFrame containing a disjoint subset of users such that the
            cumulative row counts across buckets are approximately balanced.
            Empty buckets are omitted.

    Example:
        >>> buckets = getLoadBalancedBuckets(raw_points_df, bucket_size=8)
        >>> for i, bucket_df in enumerate(buckets, start=1):
        ...     print(f"Bucket {i}: {len(bucket_df):,} rows "
        ...           f"({bucket_df['uid'].nunique()} users)")

    Note:
        The function is designed for *embarrassingly parallel* workloads where
        each user’s data can be processed independently (e.g. feature
        extraction or filtering).
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
        # tdf_collection.append(tdf[tdf["uid"].isin(buckets[i])].copy())
        tdf[tdf["uid"].isin(buckets[i])].copy()
        if not tdf.empty:
            tdf_collection.append(tdf[tdf["uid"].isin(buckets[i])].copy())
    return tdf_collection


def saveFile(path: str, fname: str, df: pd.DataFrame) -> None:
    """
    Write a pandas DataFrame to a **CSV** file, creating the target directory
    if it does not already exist.

    Args:
        path (str): Folder in which to store the file
            (e.g. ``"outputs/predictions"``).
        fname (str): Name of the CSV file to create
            (e.g. ``"trip_points.csv"``).
        df (pd.DataFrame): The DataFrame to be saved.

    Returns:
        None

    Example:
        >>> saveFile("outputs", "clean_points.csv", clean_df)
        # → file written to outputs/clean_points.csv
    """

    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(join(path, fname), index=False)
    return None


def spatialJoin(
    df: pd.DataFrame,
    shape: gpd.GeoDataFrame,
    lng_col: str,
    lat_col: str,
    loc_type: str,  # 'origin' or 'destination'
):
    """
    Spatially joins point data (supplied in *df*) to polygon features
    (supplied in *shape*) and appends the polygon’s code and name as new
    columns that are prefixed with the provided *loc_type*.

    Workflow:
        1. Convert each ``(lng_col, lat_col)`` pair into a Shapely
           :class:`Point` and wrap *df* into a GeoDataFrame (CRS = EPSG 4326).
        2. Perform a left, *intersects*-based spatial join with *shape*.
        3. Rename ``"geo_code" → f"{loc_type}_geo_code"`` and
           ``"name" → f"{loc_type}_name"``.
        4. Drop internal join artefacts (``index_right`` and the point
           ``geometry``) and return a plain pandas DataFrame.

    Args:
        df (pd.DataFrame):
            Point-level DataFrame containing longitude and latitude columns
            specified by *lng_col* and *lat_col*.
        shape (gpd.GeoDataFrame):
            Polygon layer with at least the columns
            ``["geo_code", "name", "geometry"]`` (e.g. LSOAs, census tracts).
            Must be in CRS WGS-84 (EPSG 4326) or convertible as such.
        lng_col (str): Name of the longitude column in *df*.
        lat_col (str): Name of the latitude column in *df*.
        loc_type (str): Prefix for the new columns—commonly ``"origin"`` or
            ``"destination"``.

    Returns:
        pd.DataFrame: A copy of *df* with two new columns:

        * ``f"{loc_type}_geo_code"``
        * ``f"{loc_type}_name"``

        Rows that do not intersect any polygon will contain ``NaN`` in these
        columns.

    Example:
        >>> enriched_df = spatialJoin(
        ...     df=trip_points,
        ...     shape=lsoa_gdf,
        ...     lng_col="org_lng",
        ...     lat_col="org_lat",
        ...     loc_type="origin"
        ... )
        >>> enriched_df[["origin_geo_code", "origin_name"]].head()
    """

    geometry = [Point(xy) for xy in zip(df[lng_col], df[lat_col])]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    geo_df.sindex
    geo_df = gpd.sjoin(
        geo_df,
        shape[["geo_code", "name", "geometry"]],
        how="left",
        predicate="intersects",
    )
    col_rename_dict = {
        "geo_code": f"{loc_type}_geo_code",
        "name": f"{loc_type}_name",
    }
    geo_df = geo_df.rename(columns=col_rename_dict)
    geo_df = geo_df.drop(columns=["index_right", "geometry"])
    geo_df = geo_df.reset_index(drop=True)
    geo_df = pd.DataFrame(geo_df)
    return geo_df
