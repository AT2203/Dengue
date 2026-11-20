# ============================================================
# DENGUEGUARD BANGLADESH
# ML-Based Dengue Forecasting & Division Growth Prediction
#
# Dataset:
#   data(1).xlsx
#
# Target:
#   Next month's dengue infected cases
#
# Main task:
#   Predict future dengue cases for each division and rank
#   divisions according to predicted growth.
#
# Models:
#   1. Naive baseline
#   2. Random Forest Regressor
#   3. XGBoost Regressor
#
# Explainability:
#   SHAP
#
# Outputs:
#   outputs/
#       model_comparison.csv
#       division_growth_ranking.csv
#       test_predictions.csv
#       feature_importance.csv
#       latest_forecast.csv
#       latest_forecast.json
#       best_model.joblib
#       model_features.json
#       prediction_plot.png
#       growth_ranking.png
#       shap_summary.png
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import json
import warnings
import joblib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# SHAP is optional.
# The rest of the program will still work if SHAP is unavailable.
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_FILE = "data(1).xlsx"

OUTPUT_DIR = "outputs"

# We will train on 2020-2024
# and test on 2025.
TEST_START_DATE = "2025-01-01"

RANDOM_STATE = 42


# ============================================================
# 3. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """
    Prints a visually clear section heading.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics.
    """

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


def calculate_growth(current, predicted):
    """
    Calculate percentage growth.

    Growth % =
        ((Predicted - Current) / Current) * 100

    If current value is zero, return NaN.
    """

    current = np.asarray(current, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    growth = np.where(
        current != 0,
        ((predicted - current) / current) * 100,
        np.nan
    )

    return growth


def risk_level(growth):
    """
    Simple presentation-level risk categorization.

    IMPORTANT:
    These thresholds are NOT medical or WHO thresholds.
    They are only for the prototype dashboard.
    """

    if pd.isna(growth):
        return "UNKNOWN"

    if growth >= 30:
        return "HIGH"

    elif growth >= 15:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# 5. LOAD DATASET
# ============================================================

print_section("STEP 1 - LOADING DATASET")

if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        f"\nCould not find '{DATA_FILE}'.\n"
        f"Make sure the Excel file is in the same folder "
        f"as this Python script."
    )


df = pd.read_excel(DATA_FILE)

print(f"Dataset shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 6. STANDARDIZE COLUMN NAMES
# ============================================================

print_section("STEP 2 - CLEANING COLUMN NAMES")

# Remove accidental spaces from column names.
df.columns = df.columns.str.strip()

print("Cleaned columns:")
print(df.columns.tolist())


# ============================================================
# 7. FORWARD-FILL DIVISION AND YEAR
# ============================================================

print_section("STEP 3 - FIXING DIVISION AND YEAR")

# In your dataset Division and Year appear only at the
# beginning of each block.
#
# Example:
#
# Division      Year
# Dhaka         2020
# NaN           NaN
# NaN           NaN
# ...
#
# Forward fill restores these values.

df["Division"] = df["Division"].ffill()

df["Year"] = df["Year"].ffill()

print("\nDivision counts:")
print(df["Division"].value_counts())

print("\nYears:")
print(sorted(df["Year"].unique()))


# ============================================================
# 8. CLEAN NUMERIC COLUMNS
# ============================================================

print_section("STEP 4 - CLEANING NUMERIC FEATURES")


# ------------------------------------------------------------
# Temperature
# ------------------------------------------------------------

df["temperature"] = (
    df["temperature"]
    .astype(str)
    .str.replace("°C", "", regex=False)
    .str.replace("\xa0", "", regex=False)
    .str.strip()
)

df["temperature"] = pd.to_numeric(
    df["temperature"],
    errors="coerce"
)


# ------------------------------------------------------------
# Pressure
# ------------------------------------------------------------

df["Pressure"] = (
    df["Pressure"]
    .astype(str)
    .str.replace("mbar", "", regex=False)
    .str.replace("\xa0", "", regex=False)
    .str.strip()
)

df["Pressure"] = pd.to_numeric(
    df["Pressure"],
    errors="coerce"
)


# ------------------------------------------------------------
# Other numeric columns
# ------------------------------------------------------------

numeric_columns = [
    "Infected",
    "Death",
    "Humidity"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


print("\nData types after cleaning:")

print(
    df[
        [
            "Infected",
            "Death",
            "temperature",
            "Humidity",
            "Pressure"
        ]
    ].dtypes
)


# ============================================================
# 9. CREATE MONTH NUMBER
# ============================================================

print_section("STEP 5 - CREATING TIME FEATURES")

month_map = {

    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,

    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}


df["Month_Num"] = df["Month"].map(month_map)


# Check for invalid months.

if df["Month_Num"].isna().any():

    print("\nWARNING:")
    print("Some month values could not be converted.")

    print(
        df.loc[
            df["Month_Num"].isna(),
            "Month"
        ].unique()
    )


# ============================================================
# 10. CREATE DATE COLUMN
# ============================================================

df["Date"] = pd.to_datetime(
    dict(
        year=df["Year"].astype(int),
        month=df["Month_Num"].astype(int),
        day=1
    ),
    errors="coerce"
)


# ============================================================
# 11. SORT CHRONOLOGICALLY
# ============================================================

df = (
    df
    .sort_values(
        ["Division", "Date"]
    )
    .reset_index(drop=True)
)


print("\nDate range:")

print(
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# ============================================================
# 12. BASIC DATA QUALITY CHECK
# ============================================================

print_section("STEP 6 - DATA QUALITY CHECK")

print("\nMissing values:")

print(
    df.isna().sum()
)


print("\nDuplicate rows:")

print(
    df.duplicated().sum()
)


# Remove exact duplicate rows if any.

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print(
    f"\nRemoved {before_duplicates - after_duplicates} duplicate rows."
)


# ============================================================
# 13. CREATE LAG FEATURES
# ============================================================

print_section("STEP 7 - CREATING DENGUE LAG FEATURES")


# Previous month

df["Infected_Lag_1"] = (
    df
    .groupby("Division")["Infected"]
    .shift(1)
)


# Two months ago

df["Infected_Lag_2"] = (
    df
    .groupby("Division")["Infected"]
    .shift(2)
)


# Three months ago

df["Infected_Lag_3"] = (
    df
    .groupby("Division")["Infected"]
    .shift(3)
)


# Same month previous year

df["Infected_Lag_12"] = (
    df
    .groupby("Division")["Infected"]
    .shift(12)
)


# ============================================================
# 14. CREATE ROLLING FEATURES
# ============================================================

print_section("STEP 8 - CREATING ROLLING FEATURES")


# 3-month average.
#
# IMPORTANT:
# shift(1) ensures the current month's infection
# value is NOT included.

df["Infected_Rolling_3"] = (
    df
    .groupby("Division")["Infected"]
    .transform(
        lambda x:
        x.shift(1)
         .rolling(
             window=3,
             min_periods=3
         )
         .mean()
    )
)


# 6-month average

df["Infected_Rolling_6"] = (
    df
    .groupby("Division")["Infected"]
    .transform(
        lambda x:
        x.shift(1)
         .rolling(
             window=6,
             min_periods=6
         )
         .mean()
    )
)


# ============================================================
# 15. CREATE NEXT-MONTH TARGET
# ============================================================

print_section("STEP 9 - CREATING FORECAST TARGET")


# This is the most important line in the entire model.
#
# Current row:
#
#   January 2025
#
# Target:
#
#   February 2025 infection count
#
# Therefore the model learns:
#
#   Current information -> next month's dengue cases

df["Target_Next_Month"] = (
    df
    .groupby("Division")["Infected"]
    .shift(-1)
)


print("\nTarget preview:")

print(
    df[
        [
            "Division",
            "Date",
            "Infected",
            "Target_Next_Month"
        ]
    ].head(20)
)


# ============================================================
# 16. REMOVE ROWS WITH MISSING MODEL VALUES
# ============================================================

print_section("STEP 10 - PREPARING MODEL DATA")


# We need 12 months of history for Lag_12,
# so early observations will contain NaN.

df_model = df.dropna(
    subset=[
        "Infected_Lag_1",
        "Infected_Lag_2",
        "Infected_Lag_3",
        "Infected_Lag_12",
        "Infected_Rolling_3",
        "Infected_Rolling_6",
        "Target_Next_Month"
    ]
).copy()


print(
    f"Original rows: {len(df)}"
)

print(
    f"Rows available for modeling: {len(df_model)}"
)


# ============================================================
# 17. DEFINE FEATURES
# ============================================================

print_section("STEP 11 - DEFINING FEATURES")


numeric_features = [

    "Infected_Lag_1",

    "Infected_Lag_2",

    "Infected_Lag_3",

    "Infected_Lag_12",

    "Infected_Rolling_3",

    "Infected_Rolling_6",

    "Death",

    "temperature",

    "Humidity",

    "Pressure",

    "Month_Num"
]


categorical_features = [

    "Division",

    "Pollution",

    "Density"
]


target_column = "Target_Next_Month"


# Keep Date separately.

model_features_df = df_model[
    numeric_features + categorical_features
].copy()


y = df_model[
    target_column
].copy()


# ============================================================
# 18. ONE-HOT ENCODE CATEGORICAL FEATURES
# ============================================================

print_section("STEP 12 - ENCODING CATEGORICAL FEATURES")


X = pd.get_dummies(
    model_features_df,
    columns=categorical_features,
    drop_first=False
)


# Make sure all values are numeric.

X = X.astype(float)


print(
    f"Number of model features: {X.shape[1]}"
)

print("\nFeature names:")

for feature in X.columns:

    print(" -", feature)


# ============================================================
# 19. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

print_section("STEP 13 - CHRONOLOGICAL TRAIN / TEST SPLIT")


train_mask = (
    df_model["Date"]
    < pd.Timestamp(TEST_START_DATE)
)


test_mask = ~train_mask


X_train = X.loc[
    train_mask
].copy()


X_test = X.loc[
    test_mask
].copy()


y_train = y.loc[
    train_mask
].copy()


y_test = y.loc[
    test_mask
].copy()


print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows: {len(X_test)}"
)


print(
    "\nTraining period:"
)

print(
    df_model.loc[
        train_mask,
        "Date"
    ].min()
)

print(
    "to"
)

print(
    df_model.loc[
        train_mask,
        "Date"
    ].max()
)


print(
    "\nTesting period:"
)

print(
    df_model.loc[
        test_mask,
        "Date"
    ].min()
)

print(
    "to"
)

print(
    df_model.loc[
        test_mask,
        "Date"
    ].max()
)


# ============================================================
# 20. BASELINE MODEL
# ============================================================

print_section("STEP 14 - NAIVE BASELINE")


# A simple forecasting baseline:
#
# Next month = current month's infections.
#
# This is extremely important.
#
# Our ML model should beat this baseline.

baseline_predictions = (
    df_model.loc[
        test_mask,
        "Infected"
    ].values
)


baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)


baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)


baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)


print(
    f"Baseline MAE : {baseline_mae:.4f}"
)

print(
    f"Baseline RMSE: {baseline_rmse:.4f}"
)

print(
    f"Baseline R2  : {baseline_r2:.4f}"
)


# ============================================================
# 21. TRAIN RANDOM FOREST
# ============================================================

print_section("STEP 15 - TRAINING RANDOM FOREST")


rf_model = RandomForestRegressor(

    n_estimators=500,

    max_depth=12,

    min_samples_leaf=2,

    random_state=RANDOM_STATE,

    n_jobs=-1
)


rf_model.fit(
    X_train,
    y_train
)


rf_predictions = rf_model.predict(
    X_test
)


rf_mae, rf_rmse, rf_r2 = calculate_metrics(
    y_test,
    rf_predictions
)


print("\nRandom Forest Results:")

print(
    f"MAE : {rf_mae:.4f}"
)

print(
    f"RMSE: {rf_rmse:.4f}"
)

print(
    f"R2  : {rf_r2:.4f}"
)


# ============================================================
# 22. TRAIN XGBOOST
# ============================================================

print_section("STEP 16 - TRAINING XGBOOST")


xgb_model = XGBRegressor(

    n_estimators=500,

    learning_rate=0.03,

    max_depth=5,

    min_child_weight=2,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    eval_metric="mae",

    random_state=RANDOM_STATE,

    n_jobs=-1
)


xgb_model.fit(
    X_train,
    y_train
)


xgb_predictions = xgb_model.predict(
    X_test
)


xgb_mae, xgb_rmse, xgb_r2 = calculate_metrics(
    y_test,
    xgb_predictions
)


print("\nXGBoost Results:")

print(
    f"MAE : {xgb_mae:.4f}"
)

print(
    f"RMSE: {xgb_rmse:.4f}"
)

print(
    f"R2  : {xgb_r2:.4f}"
)


# ============================================================
# 23. MODEL COMPARISON
# ============================================================

print_section("STEP 17 - MODEL COMPARISON")


baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)


results = pd.DataFrame({

    "Model": [

        "Naive Baseline",

        "Random Forest",

        "XGBoost"

    ],

    "MAE": [

        baseline_mae,

        rf_mae,

        xgb_mae

    ],

    "RMSE": [

        baseline_rmse,

        rf_rmse,

        xgb_rmse

    ],

    "R2": [

        baseline_r2,

        rf_r2,

        xgb_r2

    ]

})


print()

print(
    results.to_string(
        index=False
    )
)


results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 24. SELECT BEST MODEL
# ============================================================

# We use MAE as the primary selection metric.
#
# Lower MAE = better.

if xgb_mae < rf_mae:

    best_model = xgb_model

    best_model_name = "XGBoost"

    best_predictions = xgb_predictions

    best_mae = xgb_mae

    best_rmse = xgb_rmse

    best_r2 = xgb_r2

else:

    best_model = rf_model

    best_model_name = "Random Forest"

    best_predictions = rf_predictions

    best_mae = rf_mae

    best_rmse = rf_rmse

    best_r2 = rf_r2


print_section("BEST MODEL")

print(
    f"Selected model: {best_model_name}"
)

print(
    f"MAE : {best_mae:.4f}"
)

print(
    f"RMSE: {best_rmse:.4f}"
)

print(
    f"R2  : {best_r2:.4f}"
)


# ============================================================
# 25. CREATE TEST PREDICTION DATAFRAME
# ============================================================

print_section("STEP 18 - CREATING TEST PREDICTIONS")


test_results = df_model.loc[
    test_mask,
    [
        "Division",
        "Date",
        "Infected",
        "Target_Next_Month"
    ]
].copy()


test_results.rename(

    columns={
        "Infected":
            "Current_Infected",

        "Target_Next_Month":
            "Actual_Next_Month"

    },

    inplace=True
)


test_results[
    "Predicted_Next_Month"
] = best_predictions


# Negative predictions are not meaningful
# for infection counts.

test_results[
    "Predicted_Next_Month"
] = test_results[
    "Predicted_Next_Month"
].clip(lower=0)


# ============================================================
# 26. CALCULATE PREDICTED GROWTH
# ============================================================

test_results[
    "Predicted_Growth_%"
] = calculate_growth(

    test_results[
        "Current_Infected"
    ],

    test_results[
        "Predicted_Next_Month"
    ]

)


# ============================================================
# 27. ADD RISK LEVEL
# ============================================================

test_results[
    "Risk_Level"
] = test_results[
    "Predicted_Growth_%"
].apply(
    risk_level
)


# Save all test predictions.

test_results.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "test_predictions.csv"
    ),

    index=False
)


# ============================================================
# 28. DIVISION GROWTH RANKING
# ============================================================

print_section("STEP 19 - DIVISION GROWTH RANKING")


division_ranking = (

    test_results

    .groupby("Division")

    .agg(

        Average_Current_Infected=(
            "Current_Infected",
            "mean"
        ),

        Average_Predicted_Next_Month=(
            "Predicted_Next_Month",
            "mean"
        ),

        Average_Predicted_Growth=(
            "Predicted_Growth_%",
            "mean"
        )

    )

    .reset_index()

)


# Sort highest expected growth first.

division_ranking = (

    division_ranking

    .sort_values(
        "Average_Predicted_Growth",
        ascending=False
    )

    .reset_index(drop=True)

)


# Add ranking.

division_ranking.insert(

    0,

    "Rank",

    range(
        1,
        len(division_ranking) + 1
    )

)


# Add risk.

division_ranking[
    "Risk_Level"
] = division_ranking[
    "Average_Predicted_Growth"
].apply(
    risk_level
)


print()

print(
    division_ranking.to_string(
        index=False
    )
)


# Save.

division_ranking.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "division_growth_ranking.csv"
    ),

    index=False
)


# ============================================================
# 29. FEATURE IMPORTANCE
# ============================================================

print_section("STEP 20 - FEATURE IMPORTANCE")


feature_importance = pd.DataFrame({

    "Feature":
        X_train.columns,

    "Importance":
        best_model.feature_importances_

})


feature_importance = (

    feature_importance

    .sort_values(
        "Importance",
        ascending=False
    )

    .reset_index(drop=True)

)


print()

print(
    feature_importance.head(20).to_string(
        index=False
    )
)


feature_importance.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "feature_importance.csv"
    ),

    index=False
)


# ============================================================
# 30. PLOT FEATURE IMPORTANCE
# ============================================================

plt.figure(
    figsize=(12, 8)
)

top_features = feature_importance.head(15)

sns.barplot(

    data=top_features,

    x="Importance",

    y="Feature"

)

plt.title(
    f"Top Features - {best_model_name}"
)

plt.tight_layout()

plt.savefig(

    os.path.join(
        OUTPUT_DIR,
        "feature_importance.png"
    ),

    dpi=300
)

plt.close()


# ============================================================
# 31. ACTUAL VS PREDICTED PLOT
# ============================================================

print_section("STEP 21 - CREATING ACTUAL VS PREDICTED PLOT")


plt.figure(
    figsize=(12, 6)
)


plt.plot(

    test_results["Date"],

    test_results["Actual_Next_Month"],

    label="Actual"

)


plt.plot(

    test_results["Date"],

    test_results["Predicted_Next_Month"],

    label="Predicted"

)


plt.title(
    f"Actual vs Predicted Dengue Cases - {best_model_name}"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Dengue Cases"
)

plt.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        OUTPUT_DIR,
        "prediction_plot.png"
    ),

    dpi=300
)

plt.close()


# ============================================================
# 32. GROWTH RANKING PLOT
# ============================================================

print_section("STEP 22 - CREATING GROWTH RANKING PLOT")


plt.figure(
    figsize=(12, 7)
)


sns.barplot(

    data=division_ranking,

    x="Average_Predicted_Growth",

    y="Division"

)


plt.axvline(
    0,
    linestyle="--"
)


plt.title(
    "Predicted Dengue Growth by Division"
)

plt.xlabel(
    "Predicted Growth (%)"
)

plt.ylabel(
    "Division"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        OUTPUT_DIR,
        "growth_ranking.png"
    ),

    dpi=300
)

plt.close()


# ============================================================
# 33. SHAP EXPLAINABILITY
# ============================================================

print_section("STEP 23 - SHAP EXPLAINABILITY")


if SHAP_AVAILABLE:

    try:

        print(
            "Calculating SHAP values..."
        )

        # Use XGBoost / tree explainer.

        explainer = shap.TreeExplainer(
            best_model
        )


        shap_values = explainer.shap_values(
            X_test
        )


        # SHAP summary plot.

        plt.figure()

        shap.summary_plot(
            shap_values,
            X_test,
            show=False
        )


        plt.tight_layout()


        plt.savefig(

            os.path.join(
                OUTPUT_DIR,
                "shap_summary.png"
            ),

            dpi=300,

            bbox_inches="tight"
        )


        plt.close()


        print(
            "SHAP summary saved successfully."
        )


        # ----------------------------------------------------
        # SHAP feature importance
        # ----------------------------------------------------

        mean_abs_shap = np.abs(
            shap_values
        ).mean(axis=0)


        shap_importance = pd.DataFrame({

            "Feature":
                X_test.columns,

            "Mean_Absolute_SHAP":
                mean_abs_shap

        })


        shap_importance = (

            shap_importance

            .sort_values(
                "Mean_Absolute_SHAP",
                ascending=False
            )

            .reset_index(drop=True)

        )


        shap_importance.to_csv(

            os.path.join(
                OUTPUT_DIR,
                "shap_feature_importance.csv"
            ),

            index=False
        )


        print("\nTop SHAP features:")

        print(

            shap_importance
            .head(15)
            .to_string(index=False)

        )


    except Exception as shap_error:

        print(
            "\nSHAP could not be calculated."
        )

        print(
            f"Reason: {shap_error}"
        )

else:

    print(
        "SHAP is not installed."
    )

    print(
        "Install it using:"
    )

    print(
        "pip install shap"
    )


# ============================================================
# 34. SAVE BEST MODEL
# ============================================================

print_section("STEP 24 - SAVING MODEL")


model_path = os.path.join(

    OUTPUT_DIR,

    "best_model.joblib"

)


joblib.dump(

    best_model,

    model_path

)


print(
    f"Model saved to: {model_path}"
)


# ============================================================
# 35. SAVE MODEL FEATURES
# ============================================================

feature_list = X.columns.tolist()


feature_path = os.path.join(

    OUTPUT_DIR,

    "model_features.json"

)


with open(
    feature_path,
    "w"
) as f:

    json.dump(
        feature_list,
        f,
        indent=4
    )


print(
    f"Feature list saved to: {feature_path}"
)


# ============================================================
# 36. CREATE LATEST FORECAST
# ============================================================

print_section("STEP 25 - GENERATING LATEST FORECAST")


# The latest available row for every division.

latest_rows = (

    df_model

    .sort_values(
        ["Division", "Date"]
    )

    .groupby("Division")

    .tail(1)

    .copy()

)


latest_X = X.loc[
    latest_rows.index
].copy()


latest_predictions = best_model.predict(
    latest_X
)


latest_predictions = np.clip(
    latest_predictions,
    a_min=0,
    a_max=None
)


latest_forecast = pd.DataFrame({

    "Division":
        latest_rows["Division"].values,

    "Latest_Date":
        latest_rows["Date"].values,

    "Current_Infected":
        latest_rows["Infected"].values,

    "Predicted_Next_Month":
        latest_predictions

})


# Calculate growth.

latest_forecast[
    "Predicted_Growth_%"
] = calculate_growth(

    latest_forecast[
        "Current_Infected"
    ],

    latest_forecast[
        "Predicted_Next_Month"
    ]

)


# Add risk.

latest_forecast[
    "Risk_Level"
] = latest_forecast[
    "Predicted_Growth_%"
].apply(
    risk_level
)


# Sort by growth.

latest_forecast = (

    latest_forecast

    .sort_values(
        "Predicted_Growth_%",
        ascending=False
    )

    .reset_index(drop=True)

)


# Add rank.

latest_forecast.insert(

    0,

    "Rank",

    range(
        1,
        len(latest_forecast) + 1
    )

)


print()

print(
    latest_forecast.to_string(
        index=False
    )
)


# ============================================================
# 37. SAVE LATEST FORECAST CSV
# ============================================================

latest_csv_path = os.path.join(

    OUTPUT_DIR,

    "latest_forecast.csv"

)


latest_forecast.to_csv(

    latest_csv_path,

    index=False
)


# ============================================================
# 38. SAVE LATEST FORECAST JSON
# ============================================================

latest_json_path = os.path.join(

    OUTPUT_DIR,

    "latest_forecast.json"

)


# Convert timestamps to strings
# because JSON cannot directly serialize
# pandas timestamps.

json_data = latest_forecast.copy()


json_data[
    "Latest_Date"
] = json_data[
    "Latest_Date"
].astype(str)


json_records = json_data.to_dict(
    orient="records"
)


with open(
    latest_json_path,
    "w"
) as f:

    json.dump(
        json_records,
        f,
        indent=4
    )


print(
    f"\nLatest forecast JSON saved to:"
)

print(
    latest_json_path
)


# ============================================================
# 39. PRINT FINAL RESULT
# ============================================================

print_section("FINAL RESULT")


top_division = latest_forecast.iloc[0]


print(
    f"""
BEST MODEL
----------
{best_model_name}

MODEL PERFORMANCE
-----------------
MAE  : {best_mae:.4f}
RMSE : {best_rmse:.4f}
R2   : {best_r2:.4f}


HIGHEST PREDICTED GROWTH
------------------------
Division:
{top_division['Division']}

Current cases:
{top_division['Current_Infected']:.0f}

Predicted next-month cases:
{top_division['Predicted_Next_Month']:.0f}

Predicted growth:
{top_division['Predicted_Growth_%']:.2f}%

Risk level:
{top_division['Risk_Level']}
"""
)


# ============================================================
# 40. OUTPUT FILE SUMMARY
# ============================================================

print_section("GENERATED FILES")


output_files = [

    "model_comparison.csv",

    "division_growth_ranking.csv",

    "test_predictions.csv",

    "feature_importance.csv",

    "feature_importance.png",

    "prediction_plot.png",

    "growth_ranking.png",

    "best_model.joblib",

    "model_features.json",

    "latest_forecast.csv",

    "latest_forecast.json"

]


if SHAP_AVAILABLE:

    output_files.append(
        "shap_summary.png"
    )

    output_files.append(
        "shap_feature_importance.csv"
    )


for filename in output_files:

    full_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(full_path):

        print(
            f"[OK] {filename}"
        )


print()
print("=" * 70)
print("DENGUE FORECASTING PIPELINE COMPLETED")
print("=" * 70)