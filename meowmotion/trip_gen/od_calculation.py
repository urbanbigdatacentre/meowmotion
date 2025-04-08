import pandas as pd
from skmob import TrajDataFrame
import os
from os.path import join
from datetime import datetime
from multiprocessing import Pool
import utils

####################################################
#                                                  #
#         Customized Modules                       #
#                                                  #
####################################################
from impression_filtering import filter_data_process
from stop_node_detection import stop_node_process
from flow_generation import generateFlow
from meowmotion.ReadJson import readJsonFiles
from meowmotion.data_formatter import getLoadBalancedBuckets


class ODCalculation:

    db_type = utils.DB_TYPE  # 'postgres' | 'json'
    year = utils.YEAR
    month = (
        utils.MONTH
    )  # [i for i in range(1,13)] # [i for i in range(1,13)] month number | ['all']
    radius = utils.RADIUS  # 500 ; Radius in meters for Stop Node Detection
    time_th = (
        utils.TIME_THRESHOLD
    )  # 5; Time Threshold in minutes for Stop Node Detection
    impr_acc = (
        utils.IMPRESSION_ACCURACY
    )  # 100; Impression Accuracy in meters for filtering the data
    cpu_cores = utils.CPU_CORES  # 8; Cores to be used for multiprocessing
    city = utils.CITY  # 'Edinburgh'
    root = (
        utils.ROOT
    )  # f'U:/Operations/SCO/Faraz/huq_compiled/{city}/{year}'# For Json Data
    output_dir = (
        utils.OUTPUT_DIR
    )  # f'U:\\Projects\\Huq\\Faraz\\\od_validation' # Output Directory

    def __init__(self_):
        return

    def getQueriesForAllYearProcessing(self_, year: int) -> list:
        """
        Description:
            This function generates the queries for fetching data from the database for the whole year.
            The data is being fetched in chunks of two months. For example, the data for January and
            February will be fetched in the first query, March and April in the second query and so on.
            Each query will be processed in a seperate process (CPU core).

        Parameters:
            year (int): Year for which the data is to be fetched.

        Returns:
            list: List of queries. Each query will fetch data for two months.

        Example:
            >>> getQueriesForAllYearProcessing(2019)
            [query1,query2,query3,query4,query5,query6]
        """

        query = []
        for month in range(
            1, 13, 2
        ):  # This loop will run 6 times. Each loop will generate a query for fetching data for two months
            if month == 11:
                query.append(
                    f"""
                    SELECT timestamp as datetime, device_iid_hash as uid, impression_lat as lat, impression_lng as lng, impression_acc
                    FROM by_year.huq_gla_{year}_v1_2
                    WHERE timestamp >= '{year}-{month:02d}-01' and timestamp <'{year+1}-{1:02d}-01'
                    """
                )
            else:
                query.append(
                    f"""
                    SELECT timestamp as datetime, device_iid_hash as uid, impression_lat as lat, impression_lng as lng, impression_acc
                    FROM by_year.huq_gla_{year}_v1_2
                    WHERE timestamp >= '{year}-{month:02d}-01' and timestamp <'{year}-{month+2:02d}-01'
                    """
                )
        return query

    def getQueriesForMonthlyProcessing(self_, year: int, month: int) -> list:
        """
        Description:
            This function generates the queries for fetching data from the database for a specific month.
            The data is being fetched in chunks of 5 days. For example, the data for the whole month will
            be fetched in 6 queries. Each query will be processed in a seperate process (CPU core).
        Parameters:
            year (int): Year for which the data is to be fetched.
            month (int): Month for which the data is to be fetched.
        Returns:
            list: List of queries. Each query will fetch data for 5 days and will be executed on a seperate
            CPU core.
        Example:
            >>> getQueriesForMonthlyProcessing(2019,1)
            [query1,query2,query3,query4,query5,query6]
        """

        query = []
        for day in range(
            1, 31, 5
        ):  # We are using 6 threads to fetch the data. Each month will be devided into 5 date windows. Each Thread will be responsible for fetching data for its own date window.

            if day == 26:
                if month == 12:
                    query.append(
                        f"""
                    SELECT timestamp as datetime, device_iid_hash as uid, impression_lat as lat, impression_lng as lng, impression_acc
                    FROM by_year.huq_gla_{year}_v1_2
                    WHERE timestamp >= '{year}-{month:02d}-{day:02d}' and timestamp <'{year+1}-01-01'
                    """
                    )
                else:
                    query.append(
                        f"""
                    SELECT timestamp as datetime, device_iid_hash as uid, impression_lat as lat, impression_lng as lng, impression_acc
                    FROM by_year.huq_gla_{year}_v1_2
                    WHERE timestamp >= '{year}-{month:02d}-{day:02d}' and timestamp <'{year}-{month+1:02d}-01'
                    """
                    )
            else:
                query.append(
                    f"""
                    SELECT timestamp as datetime, device_iid_hash as uid, impression_lat as lat, impression_lng as lng, impression_acc
                    FROM by_year.huq_gla_{year}_v1_2
                    WHERE timestamp >= '{year}-{month:02d}-{day:02d}' and timestamp <'{year}-{month:02d}-{day+5:02d}'
                    """
                )
        return query

    def saveFile(self_, path: str, fname: str, df: pd.DataFrame) -> None:
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
        return


if __name__ == "__main__":

    obj = ODCalculation()
    for month in obj.month:
        print(
            f"""
        <OD Calculation Parameters>
        City: {obj.city}
        Year: {obj.year}
        Month: {month}
        Radius: {obj.radius}
        \n
        """
        )
        start_time = datetime.now()
        ##################################################################################
        #                                                                                #
        #                           Fetching Data From DB                                #
        #                                                                                #
        ##################################################################################

        print(f"{start_time}: Fetching data from Database")
        print(f"{datetime.now()}: Fetching data from Json Files")
        month_files = [
            f
            for f in os.listdir(obj.root)
            if f.split("_")[-1].split(".")[0] == str(month)
        ]  # Getting the files for the specific month
        args = [(obj.root, mf) for mf in month_files]
        with Pool(obj.cpu_cores) as p:
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
            traj_df, obj.cpu_cores
        )  # Dividing the data into buckets for multiprocessing

        ##################################################################################
        #                                                                                #
        #                           Filtering Data Based on                              #
        #              Impression Accuracy and Speed Between GPS Points                  #
        #                                                                                #
        ##################################################################################

        print(f"{datetime.now()}: Filtering Started")
        args = [(tdf, obj.impr_acc) for tdf in tdf_collection]
        with Pool(obj.cpu_cores) as pool:
            results = pool.starmap(
                filter_data_process, args
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
            f"Radius: {obj.radius}\nTime Threshold: {obj.time_th}\nImpression Accuracy: {obj.impr_acc}"
        )
        tdf_collection = obj.getLoadBalancedBuckets(traj_df, obj.cpu_cores)
        print(f"{datetime.now()}: Stop Node Detection Started")
        args = [(tdf, obj.time_th, obj.radius) for tdf in tdf_collection]
        with Pool(obj.cpu_cores) as pool:
            results = pool.starmap(stop_node_process, args)

        del tdf_collection  # Deleting the data to free up the memory

        # result1, result2, result3, result4,result5, result6, result7, result8 = results
        # stdf=pd.DataFrame(pd.concat([result1,result2,result3,result4,result5,result6,result7,result8]))
        stdf = pd.DataFrame(
            pd.concat([*results])
        )  # Concatinating the stop nodes from all the processes
        del results  # Deleting the results to free up the memory
        print(f"{datetime.now()} Stop Node Detection Completed\n")

        # Saving Stop Nodes
        obj.saveFile(
            # path=f'D:\Mobile Device Data\OD_calculation_latest_work\HUQ_OD\\{obj.year}\\stop_nodes',
            path=f"{obj.output_dir}\\{obj.city}\\{obj.year}\\stop_nodes",
            fname=f"huq_stop_nodes_{obj.city}_{obj.year}_{month}_{obj.radius}m_{obj.time_th}min_{obj.impr_acc}m.csv",
            df=stdf,
        )
        ##################################################################################
        #                                                                                #
        #                           Flow Generation                                      #
        #                                                                                #
        ##################################################################################

        stdf.rename(columns={"lat": "org_lat", "lng": "org_lng"}, inplace=True)
        stdf["dest_at"] = stdf.groupby("uid")["datetime"].transform(
            lambda x: x.shift(-1)
        )
        stdf["dest_lat"] = stdf.groupby("uid")["org_lat"].transform(
            lambda x: x.shift(-1)
        )
        stdf["dest_lng"] = stdf.groupby("uid")["org_lng"].transform(
            lambda x: x.shift(-1)
        )
        stdf = stdf.dropna(subset=["dest_lat"])
        tdf_collection = obj.getLoadBalancedBuckets(stdf, obj.cpu_cores)
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
        with Pool(obj.cpu_cores) as pool:
            results = pool.starmap(generateFlow, args)

        flow_df = pd.concat(
            [*results]
        )  # Concatinating the flow data from all the processes
        del results  # Deleting the results to free up the memory
        print(f"{datetime.now()} Flow Generation Completed\n")
        # Saving Flow
        obj.saveFile(
            path=f"{obj.output_dir}\\{obj.city}\\{obj.year}\\trips",
            fname=f"huq_trips_{obj.city}_{obj.year}_{month}_{obj.radius}m_{obj.time_th}min_{obj.impr_acc}m.csv",
            df=flow_df,
        )
        end_time = datetime.now()
        print(f"{end_time}: Process Completed")
        print(
            f"\n\nTotal Time Taken: {(end_time-start_time).total_seconds()/60} minutes"
        )
        ##################################################################################
        #                                                                                #
        #                           Trips Extrapolation                                  #
        #                                                                                #
        ##################################################################################
