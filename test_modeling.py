import pandas as pd

from meowmotion.model_tmd import trainMLModel

# proc_data = pd.read_csv(
#     "D:/Mobile Device Data/TMD_repo/ML_model/data/final_processed_data_ready_to_use.csv"
# )
# stat_df, val_stat_df = processTrainingData(proc_data)
# stat_df.to_csv('data/stat_df.csv', index=False)
# val_stat_df.to_csv('data/val_stat_df.csv', index=False)

stat_df = pd.read_csv("data/stat_df.csv")
val_stat_df = pd.read_csv("data/val_stat_df.csv")

acc, prec, recall, cm = trainMLModel(
    df_tr=stat_df, df_val=val_stat_df, model_name="RandomForest"
)
print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {recall}")
print(f"Confusion Matrix:\n{cm}")
