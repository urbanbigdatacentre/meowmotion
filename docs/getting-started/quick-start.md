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

📌 **Example snippet (from Geolife Data):**

```csv
uid,datetime,lat,lng,impression_acc
000,2008-10-23 02:53:04,39.984702,116.318417,99
000,2008-10-23 02:53:10,39.984683,116.31845,5
000,2008-10-23 02:53:15,39.984686,116.318417,99
000,2008-10-23 02:53:20,39.984688,116.318385,99
000,2008-10-23 02:53:25,39.984655,116.318263,99
000,2008-10-23 02:53:30,39.984611,116.318026,5
```


## 🧭 Step 2: Stop Nodes and Trip Generation

```bash
import pandas as pd
from meowmotion.trip_gen import TripDetector

# Load GPS data
df = pd.read_csv("sample_gps_data.csv")

# Detect trips
trip_detector = TripDetector()
trips = trip_detector.run(df)

```
