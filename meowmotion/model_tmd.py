import os
import random
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from meowmotion.data_formatter import generateTrajStats


def processTrainingData(data: pd.DataFrame) -> pd.DataFrame:
    """
    Description:
        This function processes the training data by filtering out invalid records,
        removing outliers, and generating statistics.
        It also splits the data into training and validation sets.

    Parameters:
        data (pd.DataFrame): The input data to be processed.

    Returns:
        stat_df (pd.DataFrame): The processed training data with statistics.
        vald_stat_df (pd.DataFrame): The processed validation data with statistics.
    Example:
        stat_df, vald_stat_df = processTrainingData(data)
    """

    proc_data = data.copy()
    proc_data = proc_data.loc[
        ~(
            (proc_data["transport_mode"] == "car")
            & (proc_data["maximum_match_confidence"] < 0.8)
        )
    ]

    proc_data = proc_data.loc[
        ~(
            (proc_data["transport_mode"] == "walk")
            & (proc_data["maximum_match_confidence"] < 0.8)
        )
    ]

    proc_data = proc_data.loc[
        ~(
            (proc_data["transport_mode"] == "bicycle")
            & (proc_data["maximum_match_confidence"] < 0.6)
        )
    ]

    proc_data = proc_data.loc[
        ~(
            (proc_data["transport_mode"] == "bus")
            & (proc_data["maximum_match_confidence"] < 0.6)
        )
    ]

    proc_data = proc_data.loc[
        ~(
            (proc_data["transport_mode"] == "train")
            & (proc_data["maximum_match_confidence"] < 0.4)
        )
    ]
    proc_data = proc_data.loc[
        ~(
            (proc_data["transport_mode"] == "metro")
            & (proc_data["maximum_match_confidence"] < 0.4)
        )
    ]

    proc_data = proc_data.dropna(subset=["maximum_match_confidence"])
    trip_group = proc_data.groupby(["installation_id", "trip_id", "leg_id"])
    proc_data["trip_group"] = trip_group.grouper.group_info[0]

    # Randomly Generating 33% of the validation/Testing data

    # Define the range and the number of random numbers you want
    lower_bound = 0
    upper_bound = proc_data["trip_group"].max()
    num_of_random_numbers = int(0.33 * upper_bound)  # 4445
    # Generate the distinct random numbers using random.sample
    distinct_random_numbers = random.sample(
        range(lower_bound, upper_bound), num_of_random_numbers
    )
    vald_proc_data = proc_data[proc_data["trip_group"].isin(distinct_random_numbers)]
    proc_data = proc_data.drop(vald_proc_data.index)

    proc_data = proc_data.loc[
        ~(
            (proc_data["transport_mode"] == "car")
            & (proc_data["found_at_green_space"] == 1)
        )
    ]
    proc_data = proc_data.loc[
        ~(
            (proc_data["transport_mode"] == "bus")
            & (proc_data["found_at_green_space"] == 1)
        )
    ]
    proc_data = proc_data.loc[
        ~(
            (proc_data["transport_mode"] == "train")
            & (proc_data["found_at_green_space"] == 1)
        )
    ]
    proc_data = proc_data[
        proc_data.transport_mode.isin(
            ["walk", "bicycle", "car", "bus", "train", "metro"]
        )
    ]

    # Removing extreme outliers
    print(f"{datetime.now()}: Number of Records before Filtering: {proc_data.shape[0]}")
    proc_data = proc_data[proc_data.new_speed <= 40]
    print(f"{datetime.now()} :Number of Records After Filtering: {proc_data.shape[0]}")

    vald_proc_data = vald_proc_data[
        vald_proc_data.transport_mode.isin(
            ["walk", "bicycle", "car", "bus", "train", "metro"]
        )
    ]
    # Removing extreme outliers
    print(
        f"{datetime.now()}: Number of Records before Filtering: {vald_proc_data.shape[0]}"
    )
    vald_proc_data = vald_proc_data[vald_proc_data.new_speed <= 40]
    print(
        f"{datetime.now()}: Number of Records After Filtering: {vald_proc_data.shape[0]}"
    )
    proc_data["accelaration"].fillna(0, inplace=True)
    proc_data["angular_deviation"].fillna(0, inplace=True)
    proc_data["jerk"].fillna(0, inplace=True)
    print(
        f"{datetime.now()}: Number of Records after Filling NaN: {proc_data.shape[0]}"
    )
    print(f"{datetime.now()}: NA Values Summary\n")
    print(proc_data.isna().sum())
    print("\n")

    vald_proc_data["accelaration"].fillna(0, inplace=True)
    vald_proc_data["angular_deviation"].fillna(0, inplace=True)
    vald_proc_data["jerk"].fillna(0, inplace=True)
    print(
        f"{datetime.now()}: Number of Records after Filling NaN: {vald_proc_data.shape[0]}"
    )
    print(f"{datetime.now()}: NA Values Summary\n")
    print(vald_proc_data.isna().sum())
    print("\n")

    # Removing extreme outliers acceleration, jerk and angular_deviation
    proc_data = proc_data[
        (proc_data["accelaration"] >= -7) & (proc_data["accelaration"] <= 7)
    ]
    # https://pdf.sciencedirectassets.com/308315/1-s2.0-S2352146517X00070/1-s2.0-S2352146517307937/main.pdf? \
    # X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFoaCXVzLWVhc3QtMSJHMEUCIAU0wDBQETM6g4KbEu%2Bpf2UF00B6IxSgJenWpXUc65YoAi \
    # EAvgV1EU%2FJrUl6SYWYoXwVsCDZpo0KhNHk4m61VK9lxz8qvAUI0%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjU \
    # iDKzfI9kTW4eyOkQFlSqQBaIjoxj3k%2Bi4XxQtY7I32gLlYt7wMA5epgw03US0nzU1aj64wEtF0cjv7yMClisfXNknekhQpPGhWIQ1pMuF9Az \
    # mFIyTPCua2cT3S4p0pYKEkEwR43rTIXnCfMlb%2FGlmRN2kbMjvqKmugPgEqkM9jUlAqdRojhHFqBDF4ZmtowaTBkz%2FfEQS17SeKaDEwfwGbz \
    # LIv9LZ1%2BYFbQfutRa2%2F4dqtfb%2B0reaGqu5knx9nNvtAUOM5sOZHT8tJigW%2Bx6eXxL22%2B12aKviyc2Q2xOBKbRmstcYdFmErb7j0nJr \
    # FOd2r0tKWry1JQpSXs7z9kjhNNzUkz0YlQoUQTYfhPJV%2FG%2BHeZwNpDWdme%2BvouPU13IJrEKDiHMGJZ9q2k6XmeoMzr2Ce9atwdb9oQr7r3h \
    # cNRRtT2djXz%2FhHYaaMgdVxMdF1ExrnQ41wY8j6JIGJob2ZOu8dKlVr8NTrN%2Fc32HGn%2BoKgccMIjWEoIoZGLxEAW4cHB%2BDgn8xHL9xEwr4i \
    # L%2FYGCNocQuBwCSWjHaERKrBovhYA5EY%2BdcZ0Wza9hu8Al5GZr6IJ8u2bsWLofFv2XAvGBkr3qz1MB%2F%2FGSnvoqRchtwvm3L0B3ri7XmTY7WS \
    # xp8NKhbqniQ%2Bj9%2FlSrlhfTMm0SY%2Fhr%2Fdzej9wXuG9%2Bm1wwH4sRnO3AokQI8XyoyHCLovDAUaimH0jsn2bZviON8mdAN0MlAsuiiJXJDNXo \
    # YDMR%2BywgrsvTftka2xi7CRrd7YCERFnE752y%2BhB2XMTDZMNpXOplgXRSWzTzLDVXLs6N5P1Pk0L0NwmlfSwctGsfhsr5iJdB4Hmr%2FjvV2VTvut \
    # DnXSW%2Fhil67P2ukmt2pcxJ%2FIzi7TkDMVQJNQkWnVb5A7MO%2FpuaUGOrEB%2FkRFKZUN2u6suA1u2qIkQrN8yRaggRxm3I2142qtSk2xK2XXnx6yC \
    # ybkjte9EC0uXUVoFngEPePiverUnKaTnNR5reisksXGJ7HTylbDQ7MZTtBbBztdPqHFyDkoYPWwKJa%2BWEfmwr%2Fov3aIppu%2B7rwTxjwUhjZJP9JPCY \
    # %2FB4QWkqCYeW9IaJQ1%2FjoCnBMQQdXKIN7AC9SZd1sBIu1kXZ0LMaDDAfVu%2F%2B9X4NRpSpO%2FP&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz- \
    # Date=20230712T095527Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYUQKYG5EE%2F20230712%2Fus-ea \
    # st-1%2Fs3%2Faws4_request&X-Amz-Signature=76c7fc38436931a3d9c0192a069388486d9130d958e3b2994acb779c5ee74351&hash=6a5ef828673 \
    # a2fa4b5e146070ce2a43e35de41966f8b027cb6f9d6490ba1c26e&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61 \
    # &pii=S2352146517307937&tid=spdf-6ea3e761-01b4-49d4-85a3-438ad5ad03b3&sid=b8dd4ae7949db8401e9abfd-f11e02af27f2gxrqb&type=clie \
    # nt&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0204520a515a575c5003&rr=7e58691aed2a075a&cc=gb
    vald_proc_data = vald_proc_data[
        (vald_proc_data["accelaration"] >= -7) & (vald_proc_data["accelaration"] <= 7)
    ]

    proc_data = proc_data.drop(columns=["trip_id", "leg_id"])
    proc_data = proc_data.rename(
        columns={
            "trip_group": "trip_id",
            "installation_id": "uid",
            "timestamp": "datetime",
        }
    )
    proc_data["trip_id"] = proc_data["trip_id"].astype("int32")

    vald_proc_data = vald_proc_data.drop(columns=["trip_id", "leg_id"])
    vald_proc_data = vald_proc_data.rename(
        columns={
            "trip_group": "trip_id",
            "installation_id": "uid",
            "timestamp": "datetime",
        }
    )
    vald_proc_data["trip_id"] = vald_proc_data["trip_id"].astype("int32")
    trip_points_df = proc_data.groupby(["uid", "trip_id", "transport_mode"])[
        ["uid"]
    ].apply(lambda x: x.count())
    trip_points_df.rename(columns={"uid": "total_points"}, inplace=True)
    trip_points_df.reset_index(inplace=True)
    trip_points_df = trip_points_df.groupby(["transport_mode"])[["total_points"]].apply(
        lambda x: round(x.mean())
    )

    stat_df = generateTrajStats(proc_data)
    stat_df = stat_df.drop_duplicates(
        subset=[col for col in stat_df.columns if col != "datetime"], keep="first"
    )
    stat_df["datetime"] = pd.to_datetime(stat_df["datetime"])
    stat_df["month"] = stat_df.datetime.dt.month
    stat_df = stat_df.astype({"is_weekend": "int32"})

    print(f"{datetime.now()}: Generating Validation Stats")
    # Generating Stats for Validation Data
    vald_stat_df = generateTrajStats(vald_proc_data)
    vald_stat_df = vald_stat_df.drop_duplicates(
        subset=[col for col in vald_stat_df.columns if col != "datetime"], keep="first"
    )
    vald_stat_df["datetime"] = pd.to_datetime(vald_stat_df["datetime"])
    vald_stat_df["month"] = vald_stat_df.datetime.dt.month
    vald_stat_df = vald_stat_df.astype({"is_weekend": "int32"})

    ##########################################################################################################
    #      For Training Data
    ##########################################################################################################
    stat_df = stat_df.loc[
        ~((stat_df["transport_mode"] == "walk") & (stat_df["speed_median"] >= 6.9))
    ]
    stat_df = stat_df.loc[
        ~((stat_df["transport_mode"] == "walk") & (stat_df["speed_pct_95"] >= 12.22))
    ]
    stat_df = stat_df.loc[
        ~((stat_df["transport_mode"] == "bicycle") & (stat_df["speed_median"] >= 15))
    ]
    stat_df = stat_df.loc[
        ~((stat_df["transport_mode"] == "bicycle") & (stat_df["speed_pct_95"] < 1))
    ]
    stat_df = stat_df.loc[
        ~((stat_df["transport_mode"] == "bicycle") & (stat_df["speed_pct_95"] >= 22))
    ]
    stat_df = stat_df.loc[
        ~((stat_df["transport_mode"] == "car") & (stat_df["speed_pct_95"] < 5.5))
    ]
    stat_df = stat_df.loc[
        ~((stat_df["transport_mode"] == "bus") & (stat_df["speed_pct_95"] < 3))
    ]
    stat_df = stat_df.loc[
        ~((stat_df["transport_mode"] == "train") & (stat_df["speed_pct_95"] < 5.5))
    ]
    stat_df = stat_df.loc[
        ~((stat_df["transport_mode"] == "metro") & (stat_df["speed_pct_95"] < 5.5))
    ]

    ###################################################################################################
    #         For Testing Data
    ###################################################################################################

    vald_stat_df = vald_stat_df.loc[
        ~(
            (vald_stat_df["transport_mode"] == "walk")
            & (vald_stat_df["speed_median"] >= 6.9)
        )
    ]
    vald_stat_df = vald_stat_df.loc[
        ~(
            (vald_stat_df["transport_mode"] == "walk")
            & (vald_stat_df["speed_pct_95"] >= 12.22)
        )
    ]

    vald_stat_df = vald_stat_df.loc[
        ~(
            (vald_stat_df["transport_mode"] == "bicycle")
            & (vald_stat_df["speed_median"] >= 15)
        )
    ]
    vald_stat_df = vald_stat_df.loc[
        ~(
            (vald_stat_df["transport_mode"] == "bicycle")
            & (vald_stat_df["speed_pct_95"] < 1)
        )
    ]
    vald_stat_df = vald_stat_df.loc[
        ~(
            (vald_stat_df["transport_mode"] == "bicycle")
            & (vald_stat_df["speed_pct_95"] >= 22)
        )
    ]
    vald_stat_df = vald_stat_df.loc[
        ~(
            (vald_stat_df["transport_mode"] == "car")
            & (vald_stat_df["speed_pct_95"] < 5.5)
        )
    ]
    vald_stat_df = vald_stat_df.loc[
        ~(
            (vald_stat_df["transport_mode"] == "bus")
            & (vald_stat_df["speed_pct_95"] < 3)
        )
    ]
    vald_stat_df = vald_stat_df.loc[
        ~(
            (vald_stat_df["transport_mode"] == "train")
            & (vald_stat_df["speed_pct_95"] < 5.5)
        )
    ]
    vald_stat_df = vald_stat_df.loc[
        ~(
            (vald_stat_df["transport_mode"] == "metro")
            & (vald_stat_df["speed_pct_95"] < 5.5)
        )
    ]

    return stat_df, vald_stat_df


def trainMLModel(
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    model_name: str,
    output_dir: Optional[str] = None,
) -> None:
    print(f"{datetime.now()}: Training ML Model")
    ml_df = df_tr.copy()
    val_ml_df = df_val.copy()
    print(f"{datetime.now()}: Encoding Class Labels")
    le = LabelEncoder()
    ml_df["class_label"] = le.fit_transform(ml_df.transport_mode)
    val_ml_df["class_label"] = le.fit_transform(val_ml_df.transport_mode)
    tr_cols = [
        "month",
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

    x = ml_df[tr_cols]
    y = ml_df["class_label"].values
    val_x = val_ml_df[tr_cols]
    val_y = val_ml_df["class_label"].values

    print(f"{datetime.now()}: Oversampling the data")
    oversample = SMOTE()
    x_train, y_train = oversample.fit_resample(x, y)
    print(f"{datetime.now()}: Oversampling Completed")

    print(f"{datetime.now()}: Training {model_name} Model")
    if model_name == "DecisionTree":
        model = trainDecisionTree(x_train, y_train, val_x, val_y, le)
    elif model_name == "RandomForest":
        model = trainRandomForest(x_train, y_train, val_x, val_y, le)

    if output_dir is not None:
        print(f"{datetime.now()}: Saving Model")
        os.makedirs(f"{output_dir}/artifacts", exist_ok=True)
        joblib.dump(model, f"{output_dir}/artifacts/{model_name}_model.joblib")
        print(f"{datetime.now()}: Saving Label Encoder")
        joblib.dump(le, f"{output_dir}/artifacts/label_encoder.joblib")


def trainDecisionTree(
    x_train: pd.DataFrame,
    y_train: np.array,
    val_x: pd.DataFrame,
    val_y: np.array,
    le: LabelEncoder,
) -> DecisionTreeClassifier:
    """
    Description:
        This function trains a Decision Tree Classifier using the provided training data.
        It also evaluates the model on the validation data and prints the precision, recall, accuracy,
        and confusion matrix.

    Parameters:
        x_train (pd.DataFrame): The training features.
        y_train (np.array): The training labels.
        val_x (pd.DataFrame): The validation features.
        val_y (np.array): The validation labels.

    Returns:
        dt (DecisionTreeClassifier): The trained Decision Tree model.

    Example:
        dt_model = trainDecisionTree(x_train, y_train, val_x, val_y)
    """
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(x_train, y_train)
    dt_pred = dt.predict(val_x)
    dt_precision, dt_recall, dt_fscore, _ = precision_recall_fscore_support(
        val_y, dt_pred
    )
    dt_acc = accuracy_score(val_y, dt_pred)
    dt_precision = np.round(dt_precision * 100, 2)
    dt_recall = np.round(dt_recall * 100, 2)
    dt_acc = np.round(dt_acc * 100, 2)
    cm = confusion_matrix(val_y, dt_pred, labels=dt.classes_)
    print(f"Precision:{dt_precision}\nRecall:{dt_recall}\nAcc:{dt_acc}")
    print(f"Confusion Matrix:\n{le.inverse_transform(dt.classes_)}\n{cm}")
    return dt


def trainRandomForest(
    x_train: pd.DataFrame,
    y_train: np.array,
    val_x: pd.DataFrame,
    val_y: np.array,
    le: LabelEncoder,
) -> RandomForestClassifier:
    """
    Description:
        This function trains a Random Forest Classifier using the provided training data.
        It also evaluates the model on the validation data and prints the precision, recall, accuracy,
        and confusion matrix.

    Parameters:
        x_train (pd.DataFrame): The training features.
        y_train (np.array): The training labels.
        val_x (pd.DataFrame): The validation features.
        val_y (np.array): The validation labels.

    Returns:
        rf (RandomForestClassifier): The trained Random Forest model.

    Example:
        rf_model = trainRandomForest(x_train, y_train, val_x, val_y)
    """

    rf = RandomForestClassifier(n_estimators=200, max_depth=200, max_features=None)
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(val_x)
    rf_precision, rf_recall, rf_fscore, _ = precision_recall_fscore_support(
        val_y, rf_pred
    )
    rf_acc = accuracy_score(val_y, rf_pred)
    rf_precision = np.round(rf_precision * 100, 2)
    rf_recall = np.round(rf_recall * 100, 2)
    rf_acc = np.round(rf_acc * 100, 2)
    cm = confusion_matrix(val_y, rf_pred, labels=rf.classes_)
    print(f"Precision:{rf_precision}\nRecall:{rf_recall}\nAcc:{rf_acc}")
    print(f"Confusion Matrix:\n{le.inverse_transform(rf.classes_)}\n{cm}")
    return rf
