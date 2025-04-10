<img src="../assets/meowmotion_logo.png" alt="MeowMotion Logo" width="250"/><br>

# Welcome to MeowMotion 🐾

**MeowMotion** is a Python package designed for processing and analyzing geolocation data to detect trips and infer transport modes. Originally developed at the Urban Big Data Centre (UBDC), MeowMotion provides a complete pipeline from raw GPS data ingestion and preprocessing to machine learning-based travel mode detection. Whether you're working with mobile phone app data or other location sources, MeowMotion helps you clean, structure, and analyze the movement patterns of individuals, offering both granular trip-level outputs and high-level aggregated summaries. It's an ideal tool for researchers, analysts, and developers interested in urban mobility, transportation planning, or smart city applications.

## Key Features

- **Data Ingestion**
  Reads raw geolocation data from supported formats.

- **Filtering and Cleaning**
  Applies preprocessing to remove noise and prepare the data for analysis.

- **Stay Point Detection**
  Identifies and saves user stay points based on spatial and temporal thresholds.

- **Trip Generation**
  Creates trip data using detected stay points.

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

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
