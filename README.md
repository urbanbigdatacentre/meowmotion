<p align="center">
  <img src="docs/assets/meowmotion_logo.png" alt="MeowMotion Logo" width="250"/><br>
  <strong>Detecting Trips, OD Matrices, and Transport Modes from GPS Data</strong><br>
  <em>A Python package for processing geolocation data at scale</em><br>
  <em>Developer & Author: Dr. Faraz M. Awan</em>
</p>

<p align="center">
  <a href="https://urbanbigdatacentre.github.io/meowmotion/">📖 Documentation</a> •
  <a href="#installation">🛠 Installation</a> •
  <a href="#quick-start">🚀 Quick Start</a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-311/">
    <img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11">
  </a>
  <a href="https://urbanbigdatacentre.github.io/meowmotion/">
    <img src="https://img.shields.io/badge/docs-online-brightgreen.svg" alt="Documentation">
  </a>
  <a href="https://github.com/faraz-m-awan/meowmotion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT">
  </a>
  <a href="https://www.ubdc.ac.uk/">
    <img src="https://img.shields.io/badge/developed%20by-UBDC-blueviolet" alt="Developed by UBDC">
  </a>
  <a href="https://doi.org/10.5281/zenodo.15346203">
    <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15346203-blue.svg" alt="DOI">
  </a>
  <!-- Optional: Build Status -->
  <!--
  <a href="https://github.com/faraz-m-awan/meowmotion/actions">
    <img src="https://github.com/faraz-m-awan/meowmotion/actions/workflows/main.yml/badge.svg" alt="Build Status">
  </a>
  -->
</p>


---

## 🐾 What is MeowMotion?

**MeowMotion** is a Python package for processing raw GPS data to detect trips, classify transport modes, and generate Origin-Destination (OD) matrices using scalable and modular methods. Originally developed at the Urban Big Data Centre (UBDC), it supports advanced functionality such as:

- Stay point and trip detection
- Activity-based and demographic-based trip scaling
- Generation of 4 OD matrix types (AM, PM, All-day, and Non-peak)
- Machine learning-based travel mode classification

It’s an ideal tool for urban mobility researchers, transport planners, and geospatial data scientists working with mobile GPS traces or passive location data.

---

## 📖 Full Documentation

👉 **Read the full docs here**: [https://urbanbigdatacentre.github.io/meowmotion/](https://urbanbigdatacentre.github.io/meowmotion/)

The documentation includes:

- Installation instructions
- Data format requirements
- End-to-end usage examples
- Input file specifications
- Details on OD matrix generation and model predictions

---

## 🛠 Installation

MeowMotion is a Poetry-based project.

```bash
git clone https://github.com/faraz-m-awan/meowmotion.git
cd meowmotion
poetry install
```
> 🔧 For compatibility tips and alternative setups (e.g., using uv), see the [Installation Guide](https://urbanbigdatacentre.github.io/meowmotion/getting-started/installation/).

## 🚀 Quick Start

Here’s a minimal pipeline example (see full [Quick Start](https://urbanbigdatacentre.github.io/meowmotion/getting-started/quick-start/) guide):

```python
from meowmotion.meowmob import getStopNodes, processFlowGenration, getActivityStats, generateOD
from meowmotion.process_data import getFilteredData
import pandas as pd

raw_df = readData()  # Load your GPS data
filtered_df = getFilteredData(raw_df, impr_acc=10, cpu_cores=4)
stdf = getStopNodes(filtered_df, time_th=300, radius=150, cpu_cores=4)
trip_df = processFlowGenration(stdf, raw_df, cpu_cores=4)
activity_df = getActivityStats(df=raw_df, output_dir='outputs', cpu_cores=4)

# Load shapefile and support files, then generate OD matrices
generateOD(
    trip_df=trip_df,
    shape=shape,
    active_day_df=activity_df,
    hldf=hldf,
    adult_population=adult_population_df,
    org_loc_cols=["origin_geo"],
    dest_loc_cols=["dest_geo"],
    output_dir="outputs",
    cpu_cores=4,
)

```
### 📬 Feedback & Support
For questions, suggestions, or collaborations, feel free to reach out via Issues or contact:

📧 ubdc-dataservice@glasgow.ac.uk

📄 License
This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).
