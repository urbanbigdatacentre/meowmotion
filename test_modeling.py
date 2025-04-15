from meowmotion.model_tmd import modePredict

year = 2019
artifacts_dir = ""
model_file_name = ""
le_file_name = ""
processed_non_agg_data = ""
stats_agg_data = ""
shape_file = shape_file = ""
output_dir = ""

op_df, agg_op_df = modePredict(
    processed_non_agg_data=processed_non_agg_data,
    stats_agg_data=stats_agg_data,
    artifacts_dir=artifacts_dir,
    model_file_name=model_file_name,
    le_file_name=le_file_name,
    shape_file=shape_file,
    output_dir=output_dir,
)
