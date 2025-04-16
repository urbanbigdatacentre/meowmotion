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

raw_df = readData() # Reading raw GPS data
impr_acc = 100 # Setting impression accuracy (GPS accuracy) to at least 100m
cpu_cores = 12 # Using 12 cores of processor

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

output_dir = 'path/to/output/directory'
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
This step identifies user stop locations based on temporal and spatial clustering of GPS points.

#### 📋 Output: `stop_nodes.csv` Schema

The output file contains one row per detected stay (stop) location. Each row includes:

| Column             | Description                                                             |
|--------------------|-------------------------------------------------------------------------|
| `uid`              | Unique identifier for the user                                          |
| `org_lng`          | Longitude of the centroid of the detected stay location                |
| `org_lat`          | Latitude of the centroid of the detected stay location                 |
| `datetime`         | Arrival time, when the user first arrived at the stay location        |
| `leaving_datetime` | Departure time, when the user left the stay location                  |

> ✅ These stop locations are later used to generate trip flows between consecutive stops.



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

#### 📋 Output: `trip_data.csv` Schema

The output file contains one row per detected trip between two stay locations. Each row includes:

| Column                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `uid`                   | Unique identifier for the user                                              |
| `org_lat`               | Latitude of the origin stay location centroid                               |
| `org_lng`               | Longitude of the origin stay location centroid                              |
| `org_arival_time`       | Time when the user arrived at the origin stay location                      |
| `org_leaving_time`      | Time when the user left the origin stay location                            |
| `dest_lat`              | Latitude of the destination stay location centroid                          |
| `dest_lng`              | Longitude of the destination stay location centroid                         |
| `dest_arival_time`      | Time when the user arrived at the destination stay location                 |
| `stay_points`           | All GPS points within the origin stay location cluster                      |
| `trip_points`           | Trajectory points generated during the trip between two stay locations      |
| `trip_time`             | Total duration of the trip                                                  |
| `stay_duration`         | Duration the user stayed at the origin location (detected using scikit-mobility) |
| `observed_stay_duration`| Duration inferred based on GPS points within the stay location              |

> 🧭 These trips are the basis for later mode classification and OD matrix generation.


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
#### 📋 Output: `activity_stats.csv` Schema

The output file contains activity statistics per user, aggregated by month. Each row includes:

| Column              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `uid`              | Unique identifier for the user                                              |
| `month`            | Month in `YYYY-MM` format                                                   |
| `total_active_days`| Total number of days the user was observed active in that month             |

> 📊 This information is later used to weight users' trip contributions when generating OD matrices.


## 🗺️ Step 6: Generate OD Matrices

```python
from meowmotion.meowmob import generateOD
import geopandas as gpd
import pandas as pd

# Load supporting data
city_shape = gpd.read_file(city_shape_file_path) # Shapefile of the city
hldf = pd.read_csv(hl_file_path) # Detected home location data of the users in the data
adult_population_df = pd.read_csv(adult_population_file_path) # Adult population of the city W.R.T. to IMD

# Generate 4 types of OD matrices with scaling
generateOD(
    trip_df=trip_df,
    shape=city_shape,
    active_day_df=activity_df,
    hldf=hldf,
    adult_population=adult_population_df,
    output_dir=output_dir,
    cpu_cores=cpu_cores,
)
```
This produces four types of OD matrices using demographic and activity-based weights:

 - **Type 1:** AM peak (7–10am)
 - **Type 2:** PM peak (4–7pm)
 - **Type 3:** All-day
 - **Type 4:** Non-peak (Type 3 − Type 1 & 2)

#### 📋 Output: `od_matrix_type_X.csv` Schema

The output file contains Origin-Destination (OD) pairs with associated trip counts and scaled values. Each row represents a unique OD pair for a given time window (e.g., AM peak, PM peak, etc.).

| Column                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `origin_geo_code`       | Geographic code of the origin area (e.g., data zone, LSOA, MSOA)            |
| `destination_geo_code`  | Geographic code of the destination area                                     |
| `trips`                 | Number of detected trips in the raw GPS data                                |
| `activity_weighted`     | Trips scaled to population using activity-based user weighting              |
| `council_weighted_trips`| Trips scaled using council-level and IMD adult population weights                   |
| `act_cncl_weighted_trips`| Trips scaled using both activity-based and council-level weights           |
| `percentage`            | Share of trips for this OD pair relative to all trips in the region         |

> 📌 Multiple OD matrix files are generated (AM, PM, all-day, non-peak), each following this schema.

#### 📦 Additional Outputs from `generateOD()`

In addition to OD matrices, the `generateOD()` function produces the following five datasets by default:

---

##### 1. `trip_points.csv`

This file contains detailed trajectory points for each detected trip and includes:

| Column              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `uid`              | Unique identifier for the user                                              |
| `imd_quintile`     | IMD quintile of the user's home location                                    |
| `trip_id`          | Unique identifier for the trip                                              |
| `trip_points`      | List of GPS points forming the trajectory between origin and destination    |
| `total_active_days`| Number of days the user was active in the dataset                           |
| `travel_mode`      | Placeholder column (mode not yet detected at this stage)                    |

---

##### 2. `non_agg_stay_points.csv`

This file lists all GPS points within the detected stay location clusters for each user:

| Column                  | Description                                                            |
|--------------------------|------------------------------------------------------------------------|
| `uid`                   | Unique identifier for the user                                         |
| `imd_quintile`          | IMD quintile of the user's home location                               |
| `stay_points`           | List of GPS points within the stay location cluster                    |
| `stop_node_arival_time` | Time when the user arrived at the stay location                        |
| `stop_node_leaving_time`| Time when the user left the stay location                              |
| `stay_duration`         | Duration of stay at the location                                       |
| `centroid_lat`          | Latitude of the stay location centroid                                 |
| `centroid_lng`          | Longitude of the stay location centroid                                |
| `total_active_days`     | Number of active days for the user                                     |

---

##### 3. `na_flows.csv`

Unlike the trip flows from Step 4, this dataset includes additional user-level attributes:

| Column              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `uid`              | Unique identifier for the user                                              |
| `imd_quintile`     | IMD quintile of the user's home location                                    |
| `trip_id`          | Unique trip ID                                                              |
| `org_lat`          | Latitude of the origin stay location                                        |
| `org_lng`          | Longitude of the origin stay location                                       |
| `org_arival_time`  | Time of arrival at the origin stay location                                 |
| `org_leaving_time` | Time of departure from the origin stay location                             |
| `dest_lat`         | Latitude of the destination stay location                                   |
| `dest_lng`         | Longitude of the destination stay location                                  |
| `dest_arival_time` | Time of arrival at the destination stay location                            |
| `total_trips`      | Total number of trips detected for the user                                 |
| `total_active_days`| Number of active days for the user                                          |
| `tpad`             | Trips per active day (`total_trips / total_active_days`)                    |
| `travel_mode`      | Placeholder column (mode not yet detected)                                  |

---

##### 4. `agg_stay_points.csv`

This abstracted version of stay point data assigns each detected stop to a geographic zone (`geo_code`), making it less sensitive:

| Column                  | Description                                                             |
|--------------------------|-------------------------------------------------------------------------|
| `imd_quintile`          | IMD quintile of the user's home location                                |
| `stop_node_geo_code`    | Geographic zone code where the stop was detected                        |
| `stop_node_arival_time` | Time of arrival at the stay location                                    |
| `stop_node_leaving_time`| Time of departure from the stay location                                |
| `stay_duration`         | Duration of stay at the location                                        |


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




## 🚦 Quick Start: Travel Mode Detection

MeowMotion supports **machine learning–based travel mode detection** using features derived from GPS traces, public transport infrastructure, and movement patterns.

---

### 🧱 Step 1: Prepare the Trip Data

```python
from meowmotion.data_formatter import processTripData
from datetime import datetime
import pandas as pd

print(f"{datetime.now()}: Reading raw data")
raw_df = readData() # This is the raw GPS data. Read it the way you are most comfortable with

print(f"{datetime.now()}: Reading Trip Point Data")
tp_df = pd.read_csv(trip_point_data_file) # Trip points data generated by 'generateOD() above.'

print(f"{datetime.now()}: Reading NA-flow Data")
naf_df = pd.read_csv(na_flow_data_file) # na_flows data generated by 'generateOD() above.'

# Format the data for modeling
trip_df = processTripData(trip_point_df=tp_df, na_flow_df=naf_df, raw_df=raw_df)
```

---

### 🧠 Step 2: Feature Engineering

```python
from meowmotion.data_formatter import featureEngineering
import geopandas as gpd

# Read shape files
bus_stops = gpd.read_file('path/to/bus_stops/shape_file.shp')
train_stops = gpd.read_file('path/to/train_stations/shape_file.shp')
metro_stops = gpd.read_file('path/to/metro_stations/shape_file.shp')
green_space_df = gpd.read_file('path/to/green_spaces/shape_file.shp')

shape_files = [bus_stops, train_stops, metro_stops, green_space_df] # Create list of shapefiles. Keep it in the same order. It will be passed as a parameter to featureEngineering

# Enrich trip data with contextual features
trip_df = featureEngineering(
    trip_df=trip_df, shape_files=shape_files, cpu_cores=cpu_cores
)

# Save enriched data
saveFile(f"{output_dir}/tmd", "processed_trip_points_data.csv", trip_df)
```

---

### 📊 Step 3: Generate Trip Statistics

```python
from meowmotion.data_formatter import generateTrajStats

# Extract movement stats for each trip
trip_stats_df = generateTrajStats(trip_df)

# Save the results
saveFile(f"{output_dir}/tmd", "trip_stats_data.csv", trip_stats_df)
```

---

### 🤖 Step 4: Predict Travel Mode

```python
from meowmotion.model_tmd import modePredict

op_df, agg_op_df = modePredict(
    processed_non_agg_data=processed_non_agg_data,       # processed_data dataFrame
    stats_agg_data=stats_agg_data,                       # stats_data dataFrame
    artifacts_dir="path/to/artifacts", # Create a folder artifacts and keep model and label encoder in it.
    model_file_name="model.pkl",
    le_file_name="label_encoder.joblib",
    shape_file="path/to/shape_file.shp",
    output_dir="path/to/output_dir"
)
```

---

### ✅ Output





## ✅ You're Done!
🎉 You've successfully completed the MeowMotion core pipeline!

You now have:

 - Cleaned and filtered GPS data
 - Detected stop nodes
 - Generated trip-level flows
 - Scaled OD matrices for advanced mobility analysis
