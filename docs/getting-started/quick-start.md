# 🚀 Quick Start

Welcome to **MeowMotion**! This quick start guide walks you through detecting trips and predicting transport modes using sample GPS data — all in just a few lines of code.

> ⚠️ Make sure you've followed the [Installation Guide](https://faraz-m-awan.github.io/meowmotion/getting-started/installation/) before starting.

---

## 📂 Step 1: Prepare Your Data

Ensure you have a GPS data file (e.g., `sample_gps_data.csv`) with the following minimum columns:

| Column         | Description                      |
|----------------|----------------------------------|
| uid            | Unique identifier for each user  |
| datetime       | UTC timestamp of the GPS point   |
| lat            | Latitude                         |
| lng            | Longitude                        |
| impression_acc |  GPS point accuracy in meters    |

📌 **Example snippet (Microsoft Research Asia's [Geolife GPS Trajectory Dataset](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)):**

```csv
uid,datetime,lat,lng,impression_acc
000,2008-10-23 02:53:04,39.984702,116.318417,99
000,2008-10-23 02:53:10,39.984683,116.31845,5
000,2008-10-23 02:53:15,39.984686,116.318417,99
000,2008-10-23 02:53:20,39.984688,116.318385,99
000,2008-10-23 02:53:25,39.984655,116.318263,99
000,2008-10-23 02:53:30,39.984611,116.318026,5
```

## 🧹 Step 2: Filter the Data

```python
from meowmotion.process_data import getFilteredData

# Filter based on impression accuracy and speed
raw_df_filtered = getFilteredData(
    raw_df,
    impr_acc=impr_acc,
    cpu_cores=cpu_cores
)
```
This step removes noisy and low-quality points to prepare the data for stop detection.

## 🛑 Step 3: Detect Stop Nodes

```python
from meowmotion.meowmob import getStopNodes
from meowmotion.process_data import saveFile

# Detect significant stop locations
stdf = getStopNodes(
    tdf=raw_df_filtered,
    time_th=time_th,
    radius=radius,
    cpu_cores=cpu_cores
)

# Save to disk
saveFile(output_dir, 'stop_nodes.csv', stdf)
```
This identifies meaningful places where users stayed for a period of time.

## 🧭 Step 4: Generate Trips from Stop Nodes

```python
from meowmotion.meowmob import processFlowGenration

# Create trips between stop nodes
trip_df = processFlowGenration(
    stdf=stdf,
    raw_df=raw_df_filtered,
    cpu_cores=cpu_cores
)

# Save trip data
saveFile(output_dir, 'trip_data.csv', trip_df)

```

## 📊 Step 5: Calculate Activity Statistics

```python
from meowmotion.meowmob import getActivityStats

# Compute user activity summary
activity_df = getActivityStats(
    df=raw_df,
    output_dir=output_dir,
    cpu_cores=cpu_cores
)

# Save to disk
saveFile(output_dir, 'activity_stats.csv', activity_df)
```
This helps weight the users' trips based on their active status in the data.

## 🗺️ Step 6: Generate OD Matrices

```python
from meowmotion.meowmob import generateOD
import geopandas as gpd
import pandas as pd

# Load supporting data
shape = gpd.read_file(shape_file)
hldf = pd.read_csv(hldf_file)
adult_population_df = pd.read_csv(adult_population_file)

# Generate 5 types of OD matrices with scaling
generateOD(
    trip_df=trip_df,
    shape=shape,
    active_day_df=activity_df,
    hldf=hldf,
    adult_population=adult_population_df,
    org_loc_cols=org_loc_cols,
    dest_loc_cols=dest_loc_cols,
    output_dir=output_dir,
    cpu_cores=cpu_cores,
)
```
This produces four types of OD matrices using demographic and activity-based weights:
 - Type 1: AM peak (7–10am)
 - Type 2: PM peak (4–7pm)
 - Type 3: All-day
 - Type 4: Non-peak (Type 3 − Type 1 & 2)

## ✅ You're Done!
🎉 You've successfully completed the MeowMotion core pipeline!

You now have:
 - Cleaned and filtered GPS data
 - Detected stop nodes
 - Generated trip-level flows
 - Scaled OD matrices for advanced mobility analysis
