import gzip
import io
import json
import os
import zipfile
from datetime import datetime
from os.path import join

import pandas as pd
from skmob.preprocessing import filtering


def readJsonFiles(root: str, month_file: str) -> pd.DataFrame:
    """
    Reads gzipped JSON files from a zip archive and extracts relevant fields into a DataFrame.

    Parameters:
        root (str): Directory containing the zip file.
        month_file (str): Name of the zip file.

    Returns:
        pd.DataFrame: DataFrame containing the extracted JSON data.
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


def getFilteredData(traj_df, impr_acc):

    # traj_df=traj_df[traj_df['uid'].isin(user_set)]

    print(f"Filtering based on impression accuracy={impr_acc}")
    bf = traj_df.shape[0]
    traj_df = traj_df[traj_df.impression_acc <= impr_acc]
    af = traj_df.shape[0]

    print(
        f"""
    Records before accuracy filtering: {bf}
    Records after accuracy filtering: {af}
    Difference: {bf-af}
    Percentage of deleted record: {round(((bf-af)/bf)*100)}%
    """
    )

    # Filtering based on the speed

    print("Filtering based on the speed in between two consecutive GPS points...")

    traj_df = filtering.filter(traj_df, max_speed_kmh=200)

    print(
        f"""
    Records before speed filtering: {af}
    Records after speed filtering: {traj_df.shape[0]}
    Difference: {af-traj_df.shape[0]}
    Percentage of deleted record: {round(((af-traj_df.shape[0])/af)*100,2)}%
    """
    )
    return traj_df


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


def saveFile(path: str, fname: str, df: pd.DataFrame) -> None:
    """
    Description:
        This function saves the DataFrame into a CSV file. If the directory does not exist,
        it will create the directory.

    Parameters:
        path (str): Path where the file is to be saved.
        fname (str): Name of the file.
        df (pd.DataFrame): DataFrame to be saved.

    Returns:
        None

    Example:
        >>> saveFile('D:\\Mobile Device Data\\OD_calculation_latest_work\\HUQ_OD\\2019\\stop_nodes','huq_stop_nodes_Manchester_2019_1_500m_5min_100m.csv',stdf)
    """

    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(join(path, fname), index=False)
    return None
