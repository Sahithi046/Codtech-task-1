"""
etl_pipeline.py
===============

A simple, self‑contained ETL (Extract‑Transform‑Load) pipeline.

* Input  : sample_customer_data.csv  (raw data)
* Output : processed_customer_data.csv (cleaned & transformed)

Steps
-----
1. Extract  – read raw CSV into a pandas DataFrame.
2. Transform – basic preprocessing:
      • Fill missing numerical values with median.
      • Fill missing categorical values with mode.
      • One‑hot encode categorical columns.
      • Scale numerical columns with StandardScaler.
3. Load     – save the transformed DataFrame to a new CSV.

Run
---
$ python etl_pipeline.py

Dependencies
------------
pandas, scikit‑learn (install via `pip install pandas scikit-learn`)
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

RAW_FILE  = "sample_customer_data.csv"
PROCESSED_FILE = "processed_customer_data.csv"

def extract(file_path: str) -> pd.DataFrame:
    """Read raw data from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    df = pd.read_csv(file_path)
    print(f"✅ Extract: loaded {len(df)} rows from {file_path}")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess and transform the DataFrame.

    1. Handle missing values:
       • Numerical → median
       • Categorical → mode
    2. One‑hot encode categorical features.
    3. Scale numerical features.
    """
    # Identify column types
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    # Fill missing values
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # Build transformers
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # remainder="passthrough" keeps other columns (the one‑hot encoder will be applied automatically by pandas get_dummies)
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, num_cols)],
        remainder="passthrough",
    )

    # Apply transformations
    df_transformed = preprocessor.fit_transform(df)

    # ColumnTransformer returns NumPy array; convert back to DataFrame
    # Retrieve generated column names for numerical features (same names)
    num_features_scaled = num_cols
    # One‑hot encode categorical columns separately to retain column names
    cat_encoded = pd.get_dummies(df[cat_cols], drop_first=True)

    # Combine numerical (scaled) + one‑hot encoded categorical
    import numpy as np
    df_numeric_scaled = pd.DataFrame(
        df_transformed[:, : len(num_cols)],
        columns=num_features_scaled,
    )
    df_processed = pd.concat([df_numeric_scaled, cat_encoded], axis=1)

    print("✅ Transform: preprocessing complete")
    return df_processed


def load(df: pd.DataFrame, output_path: str) -> None:
    """Save the processed DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    print(f"✅ Load: saved cleaned data to {output_path}")


def main() -> None:
    """Run the ETL pipeline end‑to‑end."""
    raw_df = extract(RAW_FILE)
    processed_df = transform(raw_df)
    load(processed_df, PROCESSED_FILE)


if __name__ == "__main__":
    main()
