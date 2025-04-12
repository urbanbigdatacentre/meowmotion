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

Example row:

```csv
uid,datetime,lat,lng,impression_acc
123,2023-05-01T08:00:00Z,51.5074,-0.1278,55.5
