import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import re
import os

def function_clean_data(data):
    """
    Clean and standardize wastewater taxonomy data from raw collection files.

    This function performs a series of preprocessing steps to convert the raw 
    wastewater data into a clean, analysis-ready format. It:
      1. Renames 'collection.date' to 'date'.
      2. Splits the 'wwtp' column (e.g., "San Jose, CA") into separate 'city'
         and 'state' columns.
      3. Extracts facility names from the 'wrf' column (e.g., "WRF D" → "D").
      4. Identifies taxonomy-related columns ('realm', 'phylum', 'class',
         'order', 'family', 'subfamily', 'genus') with suffixes '(Reference)'
         or '(prediction)'.
      5. Cleans these taxonomy columns by:
         - Trimming whitespace,
         - Removing substrings following '| nan',
         - Replacing "nan" or "NaN" values with proper missing values ('np.nan').
        6. Drops all '(prediction)' columns.

    Parameters
    ----------
    data: pandas.DataFrame. The raw dataset to clean.

    Returns
    -------
    data: pandas.DataFrame. The cleaned dataset.
    """
    data = data.rename(columns={"sample.name": "name"})
    data = data.rename(columns={'collection.date': 'date'})

    # Split the 'wwtp' column into city and state
    data[['city', 'state']] = data['wwtp'].str.split(r',\s*', expand=True)
    data = data.drop(columns=['wwtp'])

    # Extract facility name from 'wrf' column
    data['facility'] = data['wrf'].str.extract(r'^(?:WRRF|WRF)\s*([^,]+)')
    data = data.drop(columns=['wrf'])

    # Identify all taxonomy-related columns
    head_pattern = re.compile(\
        r'^(realm|phylum|class|order|family|subfamily|genus) \((Reference|prediction)\)$')
    tax_cols = [c for c in data.columns if head_pattern.match(c)]

    # Clean those taxonomy columns
    for col in tax_cols:
        # Convert to string and strip spaces
        data[col] = data[col].astype(str).str.strip()
        # Remove parts like "| nan..." at the end
        data[col] = data[col].str.replace(r'\|\s*nan\b.*$', '', regex=True)
        # Replace 'nan' (case-insensitive) with real NaN
        data[col] = data[col].apply(\
            lambda x: np.nan if str(x).lower() == 'nan' else x)
    
    return data

def function_filter_data_by_reference(data):
    """
    Keep only rows where the 'Reference' column indicates a TRUE value.

    Handles cases where 'Reference' is:
      - string ('TRUE', 'True', ' true ')
      - boolean (True)
    """
    if "Reference" not in data.columns:
        raise KeyError("Column 'Reference' not found in the dataset.")

    # Normalize to lowercase strings and strip spaces
    ref_clean = data["Reference"].astype(str).str.strip().str.lower()

    # Keep rows that equal 'true' or boolean True
    mask = (ref_clean == "true") | (data["Reference"] == True)

    filtered_data = data[mask].reset_index(drop=True)

    print(f"Filtered dataset: {len(filtered_data)} rows kept (out of {len(data)}).")
    return filtered_data

def function_select_relevant_columns(data):
    """
    Keep only columns needed for neural network training.
    Inputs: pore.size, GenomeName, RefSeqID, Proteins, Size (Kb)
    Targets: realm–genus (Reference)
    """
    cols_to_keep = [
        "pore.size", "GenomeName", "RefSeqID", "Proteins", "Size (Kb)",
        "realm (Reference)", "phylum (Reference)", "class (Reference)",
        "order (Reference)", "family (Reference)", "subfamily (Reference)",
        "genus (Reference)"
    ]
    return data[cols_to_keep]


def function_handle_missing_values(data, fill_with="Unknown"):
    """
    Handle missing values.
    Options:
      - fill_with="Unknown": replace NaN with "Unknown"
      - fill_with=None: drop rows with NaN in target columns
    """
    target_cols = [
        "realm (Reference)", "phylum (Reference)", "class (Reference)",
        "order (Reference)", "family (Reference)", "subfamily (Reference)",
        "genus (Reference)"
    ]
    if fill_with is None:
        data = data.dropna(subset=target_cols)
    else:
        data = data.fillna(fill_with)
    return data

def function_encode_features(data):
    """
    Encode categorical and numeric input features for NN.
    """
    # Encode categorical features
    le_genome = LabelEncoder()
    le_refseq = LabelEncoder()
    le_pore = LabelEncoder()

    data["GenomeName"] = le_genome.fit_transform(data["GenomeName"].astype(str))
    data["RefSeqID"] = le_refseq.fit_transform(data["RefSeqID"].astype(str))
    data["pore.size"] = le_pore.fit_transform(data["pore.size"].astype(str))

    # Scale numeric features
    scaler = StandardScaler()
    data[["Proteins", "Size (Kb)"]] = scaler.fit_transform(data[["Proteins", "Size (Kb)"]])

    return data


def function_encode_targets(data):
    """
    Encode each taxonomy level (realm-genus) into numeric labels.
    """
    y_encoders = {}
    for col in [
        "realm (Reference)", "phylum (Reference)", "class (Reference)",
        "order (Reference)", "family (Reference)", "subfamily (Reference)",
        "genus (Reference)"
    ]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        y_encoders[col] = le
    return data, y_encoders

def function_save_label_encoders(encoders_dict, output_dir=None):
    """
    Save each LabelEncoder used for target taxonomy levels as a .pkl file.
    """
    if output_dir is None:
        output_dir = os.path.join("data", "processed_data", "label_encoders")
    os.makedirs(output_dir, exist_ok=True)
    for col, encoder in encoders_dict.items():
        filename = os.path.join(output_dir, f"{col.replace(' ', '_')}_encoder.pkl")
        with open(filename, "wb") as f:
            pickle.dump(encoder, f)
    print(f"Saved {len(encoders_dict)} target label encoders to '{output_dir}'")


def main():
    print("Current working directory:", os.getcwd())
    raw_data_path = os.path.join("data", "raw_data")

    files_list = sorted([f for f in os.listdir(raw_data_path) if f.endswith(".csv")])

    # Load and clean all CSVs
    dataframes = []
    for file in files_list:
        df = pd.read_csv(os.path.join(raw_data_path, file))
        df = function_clean_data(df)
        dataframes.append(df)

    # Merge all cleaned data
    merged_data = pd.concat(dataframes, ignore_index=True)
    print(f"Merged dataset shape: {merged_data.shape}")

    # Apply processing pipeline
    merged_data = function_filter_data_by_reference(merged_data)
    merged_data = function_select_relevant_columns(merged_data)
    merged_data = function_handle_missing_values(merged_data, fill_with="Unknown")
    merged_data = function_encode_features(merged_data)
    merged_data, y_encoders = function_encode_targets(merged_data)

    # Save cleaned data and encoders

    processed_path = "data/processed_data/processed_for_training.csv"
    merged_data.to_csv(processed_path, index=False)
    print(f"Processed dataset saved to '{processed_path}' ({merged_data.shape[0]} rows).")

    function_save_label_encoders(y_encoders)

    print("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()