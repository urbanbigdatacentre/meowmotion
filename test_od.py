import pandas as pd
from datetime import datetime
import os
from multiprocessing import Pool
from meowmotion.ReadJson import readJsonFiles
from meowmotion.data_formatter import getLoadBalancedBuckets
from skmob import TrajDataFrame


if __name__ == "__main__":

    CITY = "Glasgow"
    YEAR = 2019
    root = f"U:/Operations/SCO/Faraz/huq_compiled/{CITY}/{YEAR}"
    cpu_cores = 12
    impr_acc = 100
    month = 1

    start_time = datetime.now()
    ##################################################################################
    #                                                                                #
    #                           Fetching Data From DB                                #
    #                                                                                #
    ##################################################################################

    print(f"{start_time}: Fetching data from Database")
    print(f"{datetime.now()}: Fetching data from Json Files")
    month_files = [
        f for f in os.listdir(root) if f.split("_")[-1].split(".")[0] == str(month)
    ]  # Getting the files for the specific month
    args = [(root, mf) for mf in month_files]
    with Pool(cpu_cores) as p:
        results = p.starmap(readJsonFiles, args)

    print(f"{datetime.now()}: Data Concatination")
    traj_df = pd.concat(results)  # Concatinating the data fetched from the database
    del results  # Deleting the results to free up the memory
    print(f"{datetime.now()}: Data Concatination Completed")
    print(f"{datetime.now()}: Data fetching completed\n\n")
    print(f"Number of Records: {traj_df.shape[0]}")

    # Converting Raw DataFrame into a Trajectory DataFrame
    traj_df = TrajDataFrame(
        traj_df, latitude="lat", longitude="lng", user_id="uid", datetime="datetime"
    )  # Coverting raw data into a trajectory dataframe
    tdf_collection = getLoadBalancedBuckets(
        traj_df, cpu_cores
    )  # Dividing the data into buckets for multiprocessing
