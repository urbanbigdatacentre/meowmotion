import pandas as pd
from datetime import datetime
import os
from multiprocessing import Pool
from meowmotion.process_data import (
    getLoadBalancedBuckets,
    getFilteredData,
    readJsonFiles,
    getStopNodes,
    processFlowGenration,
    saveFile,
)
from skmob import TrajDataFrame


if __name__ == "__main__":

    city = "Glasgow"
    year = 2019
    root = f"U:/Operations/SCO/Faraz/huq_compiled/{city}/{year}"
    cpu_cores = 12
    impr_acc = 100
    month = 1
    radius = 500
    time_th = 5  # in minutes
    output_dir = "U:/Projects/Huq/Faraz/package_testing"

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

    ##################################################################################
    #                                                                                #
    #                           Filtering Data Based on                              #
    #              Impression Accuracy and Speed Between GPS Points                  #
    #                                                                                #
    ##################################################################################

    print(f"{datetime.now()}: Filtering Started")
    args = [(tdf, impr_acc) for tdf in tdf_collection]
    with Pool(cpu_cores) as pool:
        results = pool.starmap(
            getFilteredData, args
        )  # Filtering the data based on Impression Accuracy and Speed between GPS points

    del tdf_collection  # Deleting the data to free up the memory
    # result1, result2, result3, result4,result5, result6, result7, result8 = results
    # traj_df=pd.concat([result1,result2,result3,result4,result5,result6,result7,result8])
    traj_df = pd.concat(
        [*results]
    )  # Concatinating the filtered data from all the processes
    del results  # Deleting the results to free up the memory
    print(f"{datetime.now()}: Filtering Finished\n\n\n")

    ##################################################################################
    #                                                                                #
    #                           Stope Node Detection                                 #
    #                                                                                #
    ##################################################################################

    print(f"{datetime.now()}: Stop Node Detection Started\n\n")
    print(
        f"Detecting stop nodes for the month: {traj_df.datetime.dt.month.unique().tolist()}"
    )
    print(
        f"Radius: {radius}\nTime Threshold: {time_th}\nImpression Accuracy: {impr_acc}"
    )
    tdf_collection = getLoadBalancedBuckets(traj_df, cpu_cores)
    print(f"{datetime.now()}: Stop Node Detection Started")
    args = [(tdf, time_th, radius) for tdf in tdf_collection]
    with Pool(cpu_cores) as pool:
        results = pool.starmap(getStopNodes, args)

    del tdf_collection  # Deleting the data to free up the memory
    stdf = pd.DataFrame(
        pd.concat([*results])
    )  # Concatinating the stop nodes from all the processes
    del results  # Deleting the results to free up the memory
    print(f"{datetime.now()} Stop Node Detection Completed\n")

    # Saving Stop Nodes
    saveFile(
        path=f"{output_dir}/{city}/{year}/stop_nodes",
        fname=f"huq_stop_nodes_{city}_{year}_{month}_{radius}m_{time_th}min_{impr_acc}m.csv",
        df=stdf,
    )

    ##################################################################################
    #                                                                                #
    #                           Flow Generation                                      #
    #                                                                                #
    ##################################################################################

    stdf.rename(columns={"lat": "org_lat", "lng": "org_lng"}, inplace=True)
    stdf["dest_at"] = stdf.groupby("uid")["datetime"].transform(lambda x: x.shift(-1))
    stdf["dest_lat"] = stdf.groupby("uid")["org_lat"].transform(lambda x: x.shift(-1))
    stdf["dest_lng"] = stdf.groupby("uid")["org_lng"].transform(lambda x: x.shift(-1))
    stdf = stdf.dropna(subset=["dest_lat"])
    tdf_collection = getLoadBalancedBuckets(stdf, cpu_cores)
    print(f"{datetime.now()}: Generating args")
    args = []
    for tdf in tdf_collection:
        temp_raw_df = traj_df[traj_df["uid"].isin(tdf["uid"].unique())].copy()
        temp_raw_df.set_index(["uid", "datetime"], inplace=True)
        temp_raw_df.sort_index(inplace=True)
        args.append((tdf, temp_raw_df))
    del tdf_collection
    print(f"{datetime.now()}: args Generation Completed")
    print(f"{datetime.now()}: Flow Generation Started\n\n")
    with Pool(cpu_cores) as pool:
        results = pool.starmap(processFlowGenration, args)

    flow_df = pd.concat(
        [*results]
    )  # Concatinating the flow data from all the processes
    del results  # Deleting the results to free up the memory
    print(f"{datetime.now()} Flow Generation Completed\n")
    # Saving Flow
    saveFile(
        path=f"{output_dir}\\{city}\\{year}\\trips",
        fname=f"huq_trips_{city}_{year}_{month}_{radius}m_{time_th}min_{impr_acc}m.csv",
        df=flow_df,
    )
    end_time = datetime.now()
    print(f"{end_time}: Process Completed")
    print(f"\n\nTotal Time Taken: {(end_time-start_time).total_seconds()/60} minutes")
