# 🚀 Quick Start

Welcome to **MeowMotion**! This quick start guide walks you through detecting trips and predicting transport modes using sample GPS data, all in just a few lines of code.

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
shape = gpd.read_file(shape_file) #
hldf = pd.read_csv(hldf_file) # Detected home location data of the users in the data
adult_population_df = pd.read_csv(adult_population_file)

# Generate 4 types of OD matrices with scaling
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

 - **Type 1:** AM peak (7–10am)
 - **Type 2:** PM peak (4–7pm)
 - **Type 3:** All-day
 - **Type 4:** Non-peak (Type 3 − Type 1 & 2)

## 📌 Notes on Required Input Files

### 🧭 1. Shapefile

The shapefile must include the following mandatory columns:

| Column    | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| geo_code  | Unique identifier for each geographic area (e.g., LSOA, MSOA, data zone)   |
| name      | Human-readable name for the geographic area                                |
| geometry  | Polygon geometry representing the spatial boundary                         |

> 📌 The coordinate reference system (CRS) **must be EPSG:4326 (WGS84)**.


This shapefile defines the spatial resolution for OD matrix generation. You can choose to calculate OD matrices at different [geographic levels](https://www.ons.gov.uk/methodology/geography/ukgeographies/statisticalgeographies), including:

 - **Local level:** data zones, LSOA
 - **Intermediate level:** MSOA, intermediate zones
 - **Regional level:** councils, municipalities

🗂️ **Sample Shapefile Preview**

| geo_code   | name                                      | geometry (EPSG:4326)               |
|------------|-------------------------------------------|------------------------------------|
| S02001902  | Garrowhill West                           | POLYGON ((-4.11936 55.85619, ...)) |
| S02001903  | Garrowhill East and Swinton               | POLYGON ((-4.09793 55.85989, ...)) |
| S02001908  | Barlanark                                 | POLYGON ((-4.13333 55.86491, ...)) |
| S02001907  | North Barlanark and Easterhouse South     | POLYGON ((-4.11959 55.86862, ...)) |
| S02001927  | Dennistoun North                          | POLYGON ((-4.21574 55.86692, ...)) |

> ✅ Ensure geometries are valid and CRS is correctly set to **EPSG:4326 (WGS84)** for spatial operations to succeed.



### 🏠 2. Home Location File

The **Home Location file** contains information about the detected home locations of users in the GPS dataset. These locations are identified using a [**novel home detection method**](https://www.sciencedirect.com/science/article/pii/S0143622823001285) that combines:

- **Active evening presence thresholds**, and
- **UK residential building data**

This hybrid approach yields more accurate home location detection compared to traditional methods that rely solely on evening activity.

> ℹ️ **Note:** The current version of MeowMotion **does not generate** this file.
> You can request the home location dataset from the **UBDC Data Service**:
>
> 📧 `ubdc-dataservice@glasgow.ac.uk`

---

#### 🗂️ Required Columns in the Home Location File

| Column                   | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `uid`                    | Unique identifier for the user                                              |
| `home_datazone` / `lsoa` | The data zone or LSOA where the user's home is located                      |
| `msoa` / `intzone_code`  | MSOA or intermediate zone code                                              |
| `msoa_name`              | Name of the MSOA or intermediate zone                                       |
| `council_code`           | Unique code for the local authority or council area                         |
| `council_name`           | Name of the local authority or council                                      |
| `imd_quintile`           | Index of Multiple Deprivation quintile (1 = most deprived, 5 = least)       |

> ✅ Ensure that the `uid` column matches the user IDs in your GPS dataset for consistent joining.

#### 📋 Sample Home Location Data

| uid | home_datazone/lsoa | msoa/intzone_code | msoa/intzone_name | council_code | council_name  | imd_quintile |
|-----|---------------------|-------------------|--------------------|---------------|----------------|---------------|
| 0   | 001                 | S01009758         | S02001842          | Darnley East  | S12000046      | Glasgow City  | 2             |
| 1   | 002                 | S01009758         | S02001842          | Darnley East  | S12000046      | Glasgow City  | 2             |
| 2   | 003                 | S01009758         | S02001842          | Darnley East  | S12000046      | Glasgow City  | 2             |
| 3   | 004                 | S01009758         | S02001842          | Darnley East  | S12000046      | Glasgow City  | 2             |
| 4   | 005                 | S01009759         | S02001842          | Darnley East  | S12000046      | Glasgow City  | 1             |


### 🧮 3. Adult Population File

The **Adult Population file** contains information about the total number of adults in each **Index of Multiple Deprivation (IMD) quintile** within a given council area. The **proportional share** of each quintile can be calculated as a percentage of the total population within the corresponding city or region.

This data is publicly available from:

- [Office of National Statistics (ONS)](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/adhocs/13773populationsbyindexofmultipledeprivationimddecileenglandandwales2020)
- [National Records of Scotland (NRS)](https://www.nrscotland.gov.uk/publications/population-estimates-by-scottish-index-of-multiple-deprivation-simd/)

---

#### 🗂️ Required Columns in the Adult Population File

| Column         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `council`      | Name of the local authority or council area                                 |
| `imd_quintile` | IMD quintile (1 = most deprived, 5 = least deprived)                        |
| `Total`        | Total adult population in that IMD quintile within the council              |
| `Percentage`   | Proportion of total population this IMD quintile represents (e.g., 0.43 = 43%) |

> ✅ Ensure the `council` values match those in the Home Location file for accurate merging.

---

#### 📋 Sample Adult Population Data

| council       | imd_quintile | Total  | Percentage |
|---------------|---------------|--------|------------|
| Glasgow City  | 1             | 229597 | 0.43       |
| Glasgow City  | 2             | 93635  | 0.17       |
| Glasgow City  | 3             | 73942  | 0.14       |
| Glasgow City  | 4             | 67347  | 0.13       |
| Glasgow City  | 5             | 70641  | 0.13       |


## ✅ You're Done!
🎉 You've successfully completed the MeowMotion core pipeline!

You now have:

 - Cleaned and filtered GPS data
 - Detected stop nodes
 - Generated trip-level flows
 - Scaled OD matrices for advanced mobility analysis
