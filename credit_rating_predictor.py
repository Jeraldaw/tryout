import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import os
import sys

# --- Define the directory where model components are saved ---
SAVE_DIR = 'credit_rating_model'

# --- Global variables for model components, loaded once ---
_model = None
_scaler = None
_imputer = None
_label_encoder = None
_features = None
_target = None
_merge_key_left = None
_merge_key_right = None
_merged_df_for_lookup = None # To store the merged data for company lookup


def _load_model_components():
    """
    Loads the trained model and preprocessing objects.
    This function is designed to be called once.
    """
    global _model, _scaler, _imputer, _label_encoder, _features, _target, _merge_key_left, _merge_key_right

    if _model is not None: # Already loaded
        return

    try:
        _model = keras.models.load_model(os.path.join(SAVE_DIR, 'credit_rating_model.h5'))
        _scaler = joblib.load(os.path.join(SAVE_DIR, 'scaler.pkl'))
        _imputer = joblib.load(os.path.join(SAVE_DIR, 'imputer.pkl'))
        _label_encoder = joblib.load(os.path.join(SAVE_DIR, 'label_encoder.pkl'))
        _features = joblib.load(os.path.join(SAVE_DIR, 'features.pkl'))
        _target = joblib.load(os.path.join(SAVE_DIR, 'target.pkl'))
        _merge_key_left = joblib.load(os.path.join(SAVE_DIR, 'merge_key_left.pkl'))
        _merge_key_right = joblib.load(os.path.join(SAVE_DIR, 'merge_key_right.pkl'))

        print("[credit_rating_predictor] Model and preprocessing objects loaded successfully.")
    except FileNotFoundError:
        print(f"[credit_rating_predictor] CRITICAL ERROR: Model files not found in '{SAVE_DIR}/'.")
        print("Please ensure you have run 'train_model.py' first to train and save the model.")
        sys.exit(1)
    except Exception as e:
        print(f"[credit_rating_predictor] CRITICAL ERROR loading model or preprocessing objects: {e}")
        sys.exit(1)

def _load_lookup_data():
    """
    Re-loads and merges the original data for company lookup.
    This function is designed to be called once.
    """
    global _merged_df_for_lookup

    if _merged_df_for_lookup is not None: # Already loaded
        return

    try:
        # Load components first to get merge keys
        _load_model_components()

        # --- IMPORTANT: Adjust this line based on your actual sentiment file type (.csv or .xlsx) ---
        # If your sentiment file is cnbc_company_sentiment.xlsx:
        corporate_rating_df_temp = pd.read_csv('corporate_rating.csv', encoding='latin1') # Include encoding if needed
        sentiment_df_temp = pd.read_excel('cnbc_company_sentiment.xlsx') # <--- Use read_excel for .xlsx

        # If your sentiment file is part5test_companies_sentiment.csv:
        # corporate_rating_df_temp = pd.read_csv('corporate_rating.csv', encoding='latin1') # Include encoding if needed
        # sentiment_df_temp = pd.read_csv('part5test_companies_sentiment.csv', encoding='latin1') # <--- Use read_csv for .csv

        # Ensure merge keys are string type for consistency
        corporate_rating_df_temp[_merge_key_left] = corporate_rating_df_temp[_merge_key_left].astype(str)
        sentiment_df_temp[_merge_key_right] = sentiment_df_temp[_merge_key_right].astype(str)

        _merged_df_for_lookup = pd.merge(corporate_rating_df_temp, sentiment_df_temp,
                                         left_on=_merge_key_left, right_on=_merge_key_right, how='inner')
        print("[credit_rating_predictor] Original data for company lookup loaded successfully.")
    except FileNotFoundError:
        print("[credit_rating_predictor] WARNING: Original data files not found for company lookup.")
        _merged_df_for_lookup = pd.DataFrame() # Set to empty to avoid errors
    except Exception as e:
        print(f"[credit_rating_predictor] WARNING: Error loading original data for company lookup: {e}")
        _merged_df_for_lookup = pd.DataFrame() # Set to empty to avoid errors


def predict_rating(input_data: dict):
    """
    Predicts the credit rating for a given dictionary of features.
    Args:
        input_data (dict): A dictionary with feature names as keys and their values.
                           E.g., {'Avg_Compound': 0.5, 'netProfitMargin': 0.1, ...}
    Returns:
        tuple: (predicted_rating_string, probabilities_dict)
    """
    _load_model_components() # Ensure components are loaded

    # Create a DataFrame from the input data to ensure consistent column order for imputer/scaler
    input_df = pd.DataFrame([input_data])
    # Ensure columns are in the same order as '_features'
    input_df = input_df[_features]

    # Preprocess
    input_imputed = _imputer.transform(input_df)
    input_scaled = _scaler.transform(input_imputed)

    # Predict
    prediction_probabilities = _model.predict(input_scaled, verbose=0)[0]
    predicted_class_index = np.argmax(prediction_probabilities)
    predicted_rating = _label_encoder.inverse_transform([predicted_class_index])[0]

    probabilities_dict = dict(zip(_label_encoder.classes_, prediction_probabilities.round(4)))

    return predicted_rating, probabilities_dict

def get_company_names():
    """Returns a list of company names available in the lookup data."""
    _load_lookup_data() # Ensure lookup data is loaded
    if not _merged_df_for_lookup.empty:
        return _merged_df_for_lookup[_merge_key_left].tolist()
    return []

def get_company_data(company_name: str):
    """Retrieves feature data for a given company name from the lookup data."""
    _load_lookup_data() # Ensure lookup data is loaded
    if not _merged_df_for_lookup.empty:
        company_row = _merged_df_for_lookup[_merged_df_for_lookup[_merge_key_left] == company_name].iloc[0]
        return company_row[_features].to_dict()
    return None

def get_feature_defaults():
    """Returns median defaults for features from the imputer."""
    _load_model_components()
    defaults = {}
    for i, feature_name in enumerate(_features):
        if i < len(_imputer.statistics_): # Ensure index exists
            defaults[feature_name] = _imputer.statistics_[i]
        else: # Fallback if feature not in imputer stats (e.g., categorical or first feature)
            defaults[feature_name] = 0.0 # Placeholder, adjust as needed
    return defaults

# Initialize components and lookup data when this module is imported
_load_model_components()
_load_lookup_data()
