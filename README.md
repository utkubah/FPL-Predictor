# FPL Point Predictor

## Abstract
This project aims to predict the total points a Fantasy Premier League (FPL) player will score in a given gameweek. It explores different machine learning models, including Recurrent Neural Networks (RNNs) and Transformers, trained on historical FPL player performance data. The project involves data preprocessing, feature engineering, model training, and evaluation to identify effective strategies for FPL point prediction.

## Table of Contents
1.  [Introduction](#introduction)
2.  [Features](#features)
3.  [Setup](#setup)
4.  [Usage](#usage)
    *   [Data Preparation](#data-preparation)
    *   [Training Models](#training-models)
    *   [Evaluating Models](#evaluating-models)
5.  [Data Source](#data-source)
6.  [Future Work](#future-work)

## Introduction
Fantasy Premier League (FPL) is a popular fantasy sports game. This project attempts to leverage machine learning techniques to forecast player points for upcoming gameweeks. The core idea is to use historical player data, including past performance and fixture context, to train models that can learn patterns indicative of future FPL scores.

The primary models explored are sequence-based, such as LSTMs (a type of RNN) and Transformers, to capture temporal dependencies in player performance. Variations including log-transformed targets and custom weighted loss functions are also investigated to handle the skewed distribution of FPL points and the impact of underprediction.

## Features
*   **Data Preprocessing & Feature Engineering:** Scripts and notebooks for cleaning raw FPL data and creating lagged, rolling window, and contextual features.
*   **Sequence Modeling:** Utilizes RNN (LSTM) and Transformer encoder architectures.
*   **Custom Loss Functions:** Includes a `WeightedSmoothL1Loss` to penalize underpredictions more heavily (used in LogRNN and LogTransformer in your provided model code).
*   **Target Transformation:** Explores `log(1+y)` transformation for the target variable (`total_points`).
*   **Model Training:** Script (`train_all_models.py`) for training specified model variants using PyTorch Lightning.
*   **Model Evaluation:** Script (`evaluate_model.py`) for evaluating trained models on a test set and specific player data (e.g., Salah), generating metrics and visualizations.
*   **Visualization:** Generates plots for training progress (via CSVLogger), prediction vs. actuals, residuals, binned predictions, and cumulative accuracy.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/utkubah/FPL-Predictor.git
    cd FPL-Predictor
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created based on the project's needs.)*

4.  **Data Preparation Workflow:**
    *   **Step 1 (EDA & Initial Processing):** Run the `notebooks/eda.ipynb` notebook. This will perform exploratory data analysis, feature engineering, and save `data/processed_data.csv` (which should include `total_points` and `log1p_total_points`).
    *   **Step 2 (Sequence Generation - Handled by Training/Evaluation Scripts):** The `src/train_all_models.py` and `src/evaluate_model.py` scripts directly call the `prepare_player_sequences` function (assumed to be in `data/make_dataset.py`). This function uses `data/processed_data.csv` to generate the necessary sequence arrays on the fly. No separate script is needed to pre-generate `.npy` files if this is the setup.

**Creating `requirements.txt`:**
If you don't have one, create it by listing necessary packages. Key packages will likely include:
Use code with caution.
torch
pytorch-lightning
numpy
pandas
scikit-learn
matplotlib
seaborn
pyyaml
joblib
You can also generate it from your active environment after installing all packages: `pip freeze > requirements.txt`.

## Usage

All scripts are expected to be run from the project's root directory (e.g., `FPL-Predictor/`). Paths within the scripts are relative to `src/` (using `../data`, `../configs`, etc.) or should be adjusted.

### Data Preparation
As outlined in the "Setup" section:
1.  Run `notebooks/eda.ipynb` to generate `data/processed_data.csv`. This file is the input for the training and evaluation scripts.
2.  The sequence generation (creating numerical sequences from `processed_data.csv`) is handled internally by `src/train_all_models.py` and `src/evaluate_model.py` via the `prepare_player_sequences` function.

### Training Models

The main training script is `src/train_all_models.py`. It trains a predefined set of model variants (RNN, Transformer, LogRNN, LogTransformer).

1.  **Configure Hyperparameters (Optional):**
    *   Base hyperparameters for RNN and Transformer are in `configs/best_params.yaml`.
    *   Training settings (epochs, batch size) are constants at the top of `src/train_all_models.py`.

2.  **Run Training:**
    ```bash
    python src/train_all_models.py
    ```
    This script will:
    *   Load `data/processed_data.csv`.
    *   Call `prepare_player_sequences` for original targets and log-transformed targets.
    *   Fit and save/load a `MaxAbsScaler` to `saved_models/fitted_scaler.joblib`.
    *   Iterate through model configurations (RNN, Transformer, LogRNN, LogTransformer).
    *   For each model:
        *   Create appropriate DataLoaders.
        *   Initialize the model with hyperparameters from `configs/best_params.yaml`.
        *   Train using PyTorch Lightning.
        *   Save the best model checkpoint (e.g., in `saved_models/LogTransformer/`) and a training summary JSON.
        *   Run evaluation on the test set and save results.

### Evaluating Models

The script `src/evaluate_model.py` loads trained model checkpoints for detailed evaluation.

1.  **Ensure Models are Trained:** Model checkpoints (`.ckpt`) and their `_summary.json` files (containing `best_checkpoint_path`) must be present in `saved_models/`, generated by `train_all_models.py`. The `fitted_scaler.joblib` must also be in `saved_models/`.

2.  **Run Evaluation:**
    ```bash
    python src/evaluate_model.py
    ```
    This script will:
    *   Load `data/processed_data.csv` and the scaler.
    *   Iterate through the predefined model variants.
    *   For each:
        *   Load its best saved checkpoint.
        *   Evaluate on the general test set.
        *   Generate and save plots: Residual plot, Binned Prediction vs. Actual, Cumulative Accuracy into a model-specific subdirectory within `saved_models/model_name/evaluation_plots_general_test/`.
        *   Evaluate on specific player data (`data/salah.csv`) and active players data (`data/active_players.csv`).
    *   Save a combined plot comparing all models for the specific player (Salah) in `saved_models/evaluation_plots_specific_player/`.
    *   Save a bar chart of MAEs on the active players set in `saved_models/evaluation_plots_active_players/`.
    *   Save overall evaluation metrics and specific player predictions in JSON format in `saved_models/`.

## Data Source
The primary historical FPL dataset used in this project was sourced from Vaastav Anand's Fantasy-Premier-League GitHub repository.
```bash
@misc{anand2016fantasypremierleague,
title = {{FPL Historical Dataset}},
author = {Anand, Vaastav},
year = {2022},
howpublished = {Retrieved August 2022 from \url{https://github.com/vaastav/Fantasy-Premier-League/}}
}
```

Data for `active_players.csv` and `salah.csv` are subsets or specific preparations based on the main dataset.

## Future Work
*   More extensive hyperparameter tuning (e.g., using Optuna or Ray Tune).
*   Advanced feature selection and engineering techniques.
*   Exploration of different Transformer architectures (e.g., including a decoder for multi-step ahead prediction if desired).
*   Ensemble modeling by combining predictions from the best-performing variants.
*   Refining the handling of rare events or player roles.
*   Investigating the impact of different data splitting strategies.

