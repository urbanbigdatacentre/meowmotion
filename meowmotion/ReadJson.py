import os
import zipfile
import gzip
import json
import pandas as pd
from multiprocessing import Pool
from datetime import datetime
import io


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


if __name__ == "__main__":

    print(f"{datetime.now()}: Starting...")
    city = "Manchester"
    year = "2021"
    root = f"U:/Operations/SCO/Faraz/huq_compiled/{city}/{year}"
    cores = 5  # os.cpu_count()

    month_files = os.listdir(root)
    # pass root and month_files to the function
    args = [(root, mf) for mf in month_files]

    with Pool(cores) as p:
        df = p.starmap(readJsonFiles, args)
        df = pd.concat(df, ignore_index=True)
    print(f"{datetime.now()}: Finished")
