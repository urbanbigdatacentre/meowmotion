import pandas as pd

from meowmotion.model_tmd import processTrainingData

proc_data = pd.read_csv(
    "D:/Mobile Device Data/TMD_repo/ML_model/data/final_processed_data_ready_to_use.csv"
)
stat_df, val_stat_df = processTrainingData(proc_data)

print(stat_df.head())
print(val_stat_df.head())
