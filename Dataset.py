
# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import zipfile
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold


# ==========================================
# 2. DEFINE FILE PATHS
# ==========================================

# Get the folder containing this Python file
script_dir = Path(__file__).resolve().parent

# Locate the original ZIP file
zip_file_path = script_dir.parent / "raw" / "rt-iot2022.zip"

# Define where processed data will be saved
save_path = script_dir / "rt_iot2022_model_data.npz"

# Define where preprocessing objects will be saved
objects_path = script_dir / "preprocessing_objects.joblib"


# Check whether the original ZIP file exists
if not zip_file_path.exists():
    raise FileNotFoundError(
        f"Dataset ZIP file not found: {zip_file_path}"
    )


# ==========================================
# 3. LOAD THE DATASET
# ==========================================

# Supported file extensions
supported_extensions = {
    '.csv',
    '.txt',
    '.tsv',
    '.data',
    '.xlsx',
    '.xls',
    '.json',
    '.parquet'
}

df = None
file_to_open = None
failed_files = []

# Open the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:

    # Collect all files, excluding directories
    all_files = [
        file for file in zip_ref.infolist()
        if not file.is_dir()
    ]

    # Display every file found inside the ZIP
    print("\n--- FILES FOUND INSIDE ZIP ---")

    for file in all_files:
        print(
            f"File: {file.filename} | "
            f"Extension: {Path(file.filename).suffix or 'No extension'}"
        )

    # Identify files with supported extensions.
    # Files without extensions are also attempted.
    candidate_files = [
        file for file in all_files
        if (
            Path(file.filename).suffix.lower()
            in supported_extensions
            or not Path(file.filename).suffix
        )
    ]

    # Try reading each candidate until one succeeds
    for file_info in candidate_files:

        file_name = file_info.filename
        extension = Path(file_name).suffix.lower()

        print(f"\nAttempting to read: {file_name}")

        try:

            # Read the file's contents into memory
            with zip_ref.open(file_info) as file:
                file_bytes = file.read()

            # Load according to the file extension
            if extension == '.csv':
                temp_df = pd.read_csv(BytesIO(file_bytes))

            elif extension == '.tsv':
                temp_df = pd.read_csv(
                    BytesIO(file_bytes),
                    sep='\t'
                )

            elif extension in ['.txt', '.data', '']:
                # Try common delimiters
                try:
                    temp_df = pd.read_csv(
                        BytesIO(file_bytes),
                        sep=None,
                        engine='python'
                    )
                except Exception:
                    temp_df = pd.read_csv(
                        BytesIO(file_bytes),
                        sep=','
                    )

            elif extension in ['.xlsx', '.xls']:
                temp_df = pd.read_excel(BytesIO(file_bytes))

            elif extension == '.json':
                temp_df = pd.read_json(BytesIO(file_bytes))

            elif extension == '.parquet':
                temp_df = pd.read_parquet(BytesIO(file_bytes))

            else:
                continue

            # Basic check to avoid accepting an empty or unusable table
            if temp_df.empty or len(temp_df.columns) < 2:
                raise ValueError(
                    "File is empty or does not contain enough columns."
                )

            # Dataset successfully loaded
            df = temp_df
            file_to_open = file_name

            print("\nDataset loaded successfully!")
            print("Selected file:", file_to_open)

            break

        except Exception as error:

            failed_files.append((file_name, str(error)))

            print("Could not read this file.")
            print("Reason:", error)


# Stop if no file could be loaded
if df is None:

    print("\n--- FILES THAT COULD NOT BE READ ---")

    for file_name, error in failed_files:
        print(f"{file_name}: {error}")

    raise FileNotFoundError(
        "No supported tabular dataset could be loaded from the ZIP. "
        "Check the file listing and errors printed above."
    )


# ==========================================
# 4. INSPECT THE DATASET
# ==========================================

# Display the first five rows
print("\n--- FIRST 5 ROWS ---")
print(df.head())

# Display dataset dimensions
print("\n--- DATASET SHAPE ---")
print(df.shape)

# Display column names and data types
print("\n--- DATASET INFO ---")
df.info()

# Display target class distribution
print("\n--- TARGET CLASS DISTRIBUTION ---")
print(df.iloc[:, -1].value_counts())


# ==========================================
# 5. CHECK DUPLICATES AND INDEX COLUMN
# ==========================================

# Count completely duplicated rows
print("Duplicate rows:", df.duplicated().sum())

# Check whether index-like column exists
if 'Unnamed: 0' in df.columns:

    # Display the first ten index-like values
    print(df['Unnamed: 0'].head(10))

    # Count unique index values
    print(
        "Unique index values:",
        df['Unnamed: 0'].nunique()
    )

else:
    print("No 'Unnamed: 0' column found.")


# ==========================================
# 6. BASIC DATA CLEANING
# ==========================================

# Remove unnecessary index-like column if it exists
df_clean = df.drop(
    columns=['Unnamed: 0'],
    errors='ignore'
)


# ==========================================
# 7. SEPARATE FEATURES (X) AND TARGET (y)
# ==========================================

# Verify that the expected target column exists
if 'Attack_type' not in df_clean.columns:
    raise KeyError(
        "Expected target column 'Attack_type' was not found. "
        "Available columns: " + str(list(df_clean.columns))
    )

# X contains input features
X = df_clean.drop(columns=['Attack_type'])

# y contains the correct attack categories
y = df_clean['Attack_type']

# Display dimensions
print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# 8. SPLIT DATA INTO TRAINING AND TESTING SETS
# ==========================================

# Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display dimensions
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

# Display training class distribution
print("\nTraining class distribution:")
print(y_train.value_counts())


# ==========================================
# 9. IDENTIFY CATEGORICAL FEATURES
# ==========================================

# Identify categorical columns
categorical_cols = X_train.select_dtypes(
    include=['object', 'str']
).columns

# Identify numerical columns
numeric_cols = X_train.select_dtypes(
    exclude=['object', 'str']
).columns

print("Categorical columns:", list(categorical_cols))
print("Numerical feature count:", len(numeric_cols))


# ==========================================
# 10. ENCODE CATEGORICAL FEATURES
# ==========================================

# Create OneHotEncoder
encoder = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

# Learn categories from training data
X_train_encoded = encoder.fit_transform(
    X_train[categorical_cols]
)

# Apply the same encoding to testing data
X_test_encoded = encoder.transform(
    X_test[categorical_cols]
)

# Display encoded dimensions
print("Training encoded shape:", X_train_encoded.shape)
print("Testing encoded shape:", X_test_encoded.shape)


# ==========================================
# 11. COMBINE NUMERICAL AND ENCODED FEATURES
# ==========================================

# Preserve original numerical feature names
numeric_feature_names = list(numeric_cols)

# Preserve generated categorical feature names
encoded_feature_names = list(
    encoder.get_feature_names_out(categorical_cols)
)

# Combine numerical and encoded features
X_train_final = np.concatenate([
    X_train[numeric_cols].values,
    X_train_encoded
], axis=1)

# Perform the same combination for testing data
X_test_final = np.concatenate([
    X_test[numeric_cols].values,
    X_test_encoded
], axis=1)

# Store all feature names in the same column order
all_feature_names = (
    numeric_feature_names + encoded_feature_names
)

print("Final training shape:", X_train_final.shape)
print("Final testing shape:", X_test_final.shape)


# ==========================================
# 12. DATA QUALITY AUDIT
# ==========================================

# Check for rows containing missing or infinite values
print(
    "Rows with missing or infinite values:",
    (~np.isfinite(X_train_final)).any(axis=1).sum()
)

# Count infinite values
print(
    "Infinite values:",
    np.isinf(X_train_final).sum()
)

# Check maximum absolute feature value
print(
    "Maximum absolute feature value:",
    np.nanmax(np.abs(X_train_final))
)

# Check feature data type
print("Feature data type:", X_train_final.dtype)

# Check target labels
print("Missing training labels:", y_train.isna().sum())
print("Missing testing labels:", y_test.isna().sum())

# Check feature dimensions
print(
    "Feature dimensions match:",
    X_train_final.shape[1] == X_test_final.shape[1]
)


# ==========================================
# 13. HANDLE MISSING AND INFINITE VALUES
# ==========================================

# Replace infinite values with NaN
X_train_final = np.where(
    np.isfinite(X_train_final),
    X_train_final,
    np.nan
)

X_test_final = np.where(
    np.isfinite(X_test_final),
    X_test_final,
    np.nan
)

# Create median imputer
imputer = SimpleImputer(strategy='median')

# Learn medians from training data
X_train_final = imputer.fit_transform(X_train_final)

# Apply the same medians to testing data
X_test_final = imputer.transform(X_test_final)

print(
    "Remaining missing values:",
    np.isnan(X_train_final).sum()
)


# ==========================================
# 14. REMOVE CONSTANT FEATURES
# ==========================================

# Create variance selector
selector = VarianceThreshold(threshold=0)

# Learn selected features from training data
X_train_final = selector.fit_transform(X_train_final)

# Apply the same selection to testing data
X_test_final = selector.transform(X_test_final)

print("Training shape after cleaning:", X_train_final.shape)
print("Testing shape after cleaning:", X_test_final.shape)


# ==========================================
# 15. PRESERVE FINAL FEATURE NAMES
# ==========================================

# Get the Boolean mask of retained features
selected_features = selector.get_support()

# Keep names of features that survived selection
final_feature_names = np.array(
    all_feature_names
)[selected_features]

print(
    "Number of preserved feature names:",
    len(final_feature_names)
)


# ==========================================
# 16. FINAL DATA VALIDATION
# ==========================================

# Check that all feature values are finite
assert np.isfinite(X_train_final).all()
assert np.isfinite(X_test_final).all()

# Check that target labels are not missing
assert not y_train.isna().any()
assert not y_test.isna().any()

# Check matching feature dimensions
assert X_train_final.shape[1] == X_test_final.shape[1]

# Check feature names match final feature dimensions
assert len(final_feature_names) == X_train_final.shape[1]

print("\nDataset passed final validation!")


# ==========================================
# 17. SAVE PROCESSED DATA
# ==========================================

# Save training features, testing features and labels
np.savez_compressed(
    save_path,
    X_train_final=X_train_final,
    X_test_final=X_test_final,
    y_train=y_train.to_numpy(),
    y_test=y_test.to_numpy(),
    feature_names=final_feature_names
)

print("\nProcessed data saved successfully!")
print("Saved at:", save_path)


# ==========================================
# 18. SAVE PREPROCESSING OBJECTS
# ==========================================

# Store fitted preprocessing components and metadata
preprocessing_objects = {
    "encoder": encoder,
    "imputer": imputer,
    "selector": selector,
    "numeric_cols": list(numeric_cols),
    "categorical_cols": list(categorical_cols),
    "all_feature_names": all_feature_names,
    "feature_names": final_feature_names
}

# Save fitted preprocessing objects
joblib.dump(
    preprocessing_objects,
    objects_path
)

print("\nPreprocessing objects saved successfully!")
print("Saved at:", objects_path)


# ==========================================
# 19. FINAL SUMMARY
# ==========================================

print("\n========== FINAL DATASET SUMMARY ==========")

print("Source file:", file_to_open)

print("Training features:", X_train_final.shape)
print("Testing features:", X_test_final.shape)

print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)

print("Total final features:", len(final_feature_names))

print("\nAll files saved successfully!")
