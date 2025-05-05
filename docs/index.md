<p align="center">
  <img src="assets/meowmotion_logo.png" alt="MeowMotion Logo" width="250"/>
</p>

<p align="center">
  <strong>MeowMotion</strong><br>
  <em>Detecting Trips, OD Matrices, and Transport Mode from GPS Data</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-311/">
    <img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11">
  </a>
  <a href="https://faraz-m-awan.github.io/meowmotion/">
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


# Welcome to MeowMotion 🐾

**MeowMotion** is a Python package designed for processing and analyzing geolocation data to detect trips and infer transport modes. Originally developed at the Urban Big Data Centre (UBDC), MeowMotion provides a complete pipeline from raw GPS data ingestion and preprocessing to machine learning-based travel mode detection.
In addition to trip detection and classification, MeowMotion now includes advanced functionality for **trip scaling and OD matrix generation**. By applying **demographic-based** and **activity-based weighting**, it can scale trip data to better represent population-wide mobility. The package produces **five types of Origin-Destination (OD) matrices**, including peak period flows and residual non-peak movement, making it especially useful for transport modeling, demand analysis, and policy evaluation.
Whether you're working with mobile phone app data or other location sources, MeowMotion helps you clean, structure, and analyze the movement patterns of individuals, offering both granular trip-level outputs and high-level aggregated summaries. It's an ideal tool for researchers, analysts, and developers interested in urban mobility, transportation planning, or smart city applications.

## Key Features

- **Data Ingestion**
  Reads raw geolocation data from supported formats.

- **Filtering and Cleaning**
  Applies preprocessing to remove noise and prepare the data for analysis.

- **Stay Point Detection**
  Identifies and saves user stay points based on spatial and temporal thresholds.

- **Trip Generation**
  Creates trip data using detected stay points.

- **Trip Scaling & OD Matrix Generation**
Reads generated trips and applies:
  - **Demographic-based user weighting** using external population/sample profiles.
  - **Novel activity-based weighting** based on users' activity in the data.

Scales trips accordingly and produces **four different types of Origin-Destination (OD) matrices**, offering robust representations of mobility flows.

    - Type 1: AM peak weekdays (7 AM – 10 AM)
    - Type 2: PM peak weekdays (4 PM – 7 PM)
    - Type 3: All-day / All trips
    - Type 4: Type 3 minus (Type 1 + Type 2), i.e. non-peak OD flows

In addition to OD matrices, the process also outputs:

    - `trip_points.csv`: Raw GPS trajectories for each trip (with placeholder mode).
    - `non_agg_stay_points.csv`: All GPS points within detected stay clusters.
    - `na_flows.csv`: Enhanced trip flows with user-level context (trips per active day, etc.).
    - `agg_stay_points.csv`: Abstracted stay location data linked to geographic zones.


- **Trajectory Processing**
  Analyzes raw and trip data to produce:
    - Granular trip data with trajectory points.
    - Aggregated trip statistics.

- **Feature Engineering**
  Performs data engineering on aggregated trip statistics to prepare inputs for machine learning models.

- **Machine Learning for Travel Mode Detection**
    - Trains two built-in ML models: Decision Tree and Random Forest.
    - Provides functionality to save (pickle) trained models.
    - Includes a prediction endpoint to load models and classify the travel mode of unseen trip data.

- **Custom Model Training Support**
  While MeowMotion includes built-in models, it also supports retraining with custom datasets.

- **Prediction Output**
  Generates two output datasets:
    - **Non-aggregated trip data** with origin geo code, destination geo code, and predicted travel mode for each trip.
    - **Aggregated summary** showing trip counts between each origin-destination pair, split by transport modes (Walk, Bicycle, Car, Bus, Train, Metro).

---

## 📑 Citation
If you use MeowMotion in your research, please cite:
```bibtex
@software{MeowMotion,
  author       = {Faraz M. Awan},
  title        = {{MeowMotion: Detecting Trips, OD Matrices, and Transport Modes from GPS Data}},
  affiliation  = {Urban Big Data Centre, University of Glasgow},
  publisher    = {Zenodo},
  year         = {2024},
  doi          = {10.5281/zenodo.15346203},
  url          = {https://doi.org/10.5281/zenodo.15346203}
}
```
