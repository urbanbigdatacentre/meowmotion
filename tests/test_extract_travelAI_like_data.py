import pytest
import pandas as pd
import numpy as np
from travel_mode_detection.extract_travelAI_like_data import removeOutlier


@pytest.fixture
def sample_dataframe():
    """Creates a test DataFrame simulating grouped speed and z-score values"""
    data = {
        "uid": [1, 1, 1, 2, 2, 2],  # Two trips: one for each user
        "trip_id": [101, 101, 101, 102, 102, 102],  # Trip identifiers
        "speed": [10, 15, 100, 20, 25, 30],  # In trip 101, 100 is an outlier
        "speed_z_score": [0.5, 1.2, 3.5, 0.8, 1.0, 0.4],  # 3.5 indicates an outlier
    }
    df = pd.DataFrame(data)
    return df


def test_removeOutlier(sample_dataframe):
    """Tests `removeOutlier` in a groupby-apply context"""

    # Call removeOutlier as it is used in processData:
    processed_df = sample_dataframe.groupby(["uid", "trip_id"], group_keys=False).apply(
        removeOutlier
    )

    # Expected median speeds based on the current implementation:
    # For trip 101, np.median([10, 15, 100]) is 15
    expected_median_101 = np.median([10, 15, 100])

    # Check that the outlier in trip 101 is replaced with the computed median (15)
    assert (
        processed_df.loc[2, "speed"] == expected_median_101
    ), f"Expected {expected_median_101}, got {processed_df.loc[2, 'speed']}"

    # Verify non-outlier speeds remain unchanged:
    assert processed_df.loc[0, "speed"] == 10
    assert processed_df.loc[1, "speed"] == 15
    assert processed_df.loc[3, "speed"] == 20
    assert processed_df.loc[4, "speed"] == 25
    assert processed_df.loc[5, "speed"] == 30

    print("Test passed: removeOutlier correctly replaces outliers!")
