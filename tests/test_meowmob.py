from pathlib import Path

import pandas as pd

from meowmotion.meowmob import getActivityStats


def test_get_activity_stats(tmp_path):
    data = {
        "uid": [1, 1, 1, 2, 2, 2, 3],
        "datetime": [
            "2023-01-01",
            "2023-01-01",
            "2023-01-03",  # User 1: 2 unique days in Jan
            "2023-02-01",
            "2023-02-02",
            "2023-02-02",  # User 2: 2 unique days in Feb
            "2023-01-01",  # User 3: 1 day in Jan
        ],
    }
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    output_path = tmp_path
    result = getActivityStats(df, output_dir=str(output_path))

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"uid", "month", "total_active_days"}

    result_dict = result.set_index(["uid", "month"])["total_active_days"].to_dict()
    expected = {
        (1, 1): 2,
        (2, 2): 2,
        (3, 1): 1,
    }
    assert result_dict == expected

    # Check if file was saved
    output_file = Path(output_path) / "activity_stats.csv"
    assert output_file.exists()
    saved_df = pd.read_csv(output_file)
    assert not saved_df.empty
