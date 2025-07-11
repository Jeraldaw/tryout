import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib # For loading scikit-learn objects
import os # For checking file existence
import sys # For exiting on critical errors

# --- Load the saved model and preprocessing objects ---
save_dir = 'credit_rating_model'

try:
    model = keras.models.load_model(os.path.join(save_dir, 'credit_rating_model.h5'))
    scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
    imputer = joblib.load(os.path.join(save_dir, 'imputer.pkl'))
    label_encoder = joblib.load(os.path.join(save_dir, 'label_encoder.pkl'))
    features = joblib.load(os.path.join(save_dir, 'features.pkl')) # Load feature names
    target = joblib.load(os.path.join(save_dir, 'target.pkl'))     # Load target name
    merge_key_left = joblib.load(os.path.join(save_dir, 'merge_key_left.pkl'))
    merge_key_right = joblib.load(os.path.join(save_dir, 'merge_key_right.pkl'))

    print("\n--- Model and preprocessing objects loaded successfully ---")
except FileNotFoundError:
    print(f"\n--- CRITICAL ERROR: Model files not found in '{save_dir}/'. ---")
    print("Please ensure you have run 'train_model.py' first to train and save the model.")
    sys.exit(1)
except Exception as e:
    print(f"\n--- CRITICAL ERROR loading model or preprocessing objects: {e} ---")
    sys.exit(1)

# Re-load the original merged_df for company lookup in prediction
# This assumes the original data files are still accessible.
merged_df = None
try:
    corporate_rating_df_temp = pd.read_csv('corporate_rating.csv')
    # CHANGED: Reading sentiment file as CSV
    sentiment_df_temp = pd.read_csv('cnbc_company_sentiment.csv')

    # Ensure merge keys are string type for consistency
    corporate_rating_df_temp[merge_key_left] = corporate_rating_df_temp[merge_key_left].astype(str)
    sentiment_df_temp[merge_key_right] = sentiment_df_temp[merge_key_right].astype(str)

    merged_df = pd.merge(corporate_rating_df_temp, sentiment_df_temp, left_on=merge_key_left, right_on=merge_key_right, how='inner')
    print("Original data for company lookup loaded successfully.")
except FileNotFoundError:
    print("\n--- WARNING: Original data files not found for company lookup. ---")
    print("Option 2 ('Select a company from the existing dataset') will not work.")
    print("Please ensure 'corporate_rating.csv' and 'cnbc_company_sentiment.csv' are in the same directory.")
except Exception as e:
    print(f"\n--- WARNING: Error loading original data for company lookup: {e} ---")
    print("Option 2 ('Select a company from the existing dataset') might not work correctly.")


def get_user_input(prompt_text, feature_name):
    """
    Prompts the user for a numerical input, allowing them to skip.
    If skipped, uses the median value for that feature from the training data.
    """
    while True:
        user_input = input(f"{prompt_text} (Press Enter to use median default): ").strip()
        if user_input == '':
            try:
                feature_index = features.index(feature_name)
                default_val = imputer.statistics_[feature_index]
                print(f"Using median default for {feature_name}: {default_val:.4f}")
                return default_val
            except ValueError:
                print(f"Error: Feature '{feature_name}' not found in trained features for default. Please input a value.")
                continue
            except AttributeError:
                print("Error: Imputer not loaded. Please input a value.")
                continue
        try:
            value = float(user_input)
            return value
        except ValueError:
            print("Invalid input. Please enter a number or press Enter.")

def predict_credit_rating_interactive():
    """
    Interactively prompts the user for features and predicts credit rating.
    Allows choosing to input data or use a company from the existing dataset.
    """
    print("\n--- Predict Credit Rating ---")
    print("Choose an option:")
    print("1. Enter new financial and sentiment data.")
    print("2. Select a company from the existing dataset (uses its data).")
    print("3. Exit.")

    choice = input("Enter your choice (1, 2, or 3): ").strip()

    if choice == '1':
        print("\nEnter new data. For any field, press Enter to use the median value from the training data.")
        try:
            sentiment_score = get_user_input("Enter Average Compound Sentiment (e.g., -1.0 to 1.0)", 'Avg_Compound')
            net_profit_margin = get_user_input("Enter Net Profit Margin (e.g., 0.1 for 10%)", 'netProfitMargin')
            debt_equity_ratio = get_user_input("Enter Debt to Equity Ratio (e.g., 1.5)", 'debtEquityRatio')
            current_ratio = get_user_input("Enter Current Ratio (e.g., 2.0)", 'currentRatio')

            financial_data_input = {
                "Avg_Compound": sentiment_score,
                "netProfitMargin": net_profit_margin,
                "debtEquityRatio": debt_equity_ratio,
                "currentRatio": current_ratio
            }

            # Create a DataFrame from the input data to ensure consistent column order for imputer/scaler
            input_df = pd.DataFrame([financial_data_input])
            # Ensure columns are in the same order as 'features'
            input_df = input_df[features]

            input_imputed = imputer.transform(input_df)
            input_scaled = scaler.transform(input_imputed)

            prediction_probabilities = model.predict(input_scaled, verbose=0)[0]
            predicted_class_index = np.argmax(prediction_probabilities)
            predicted_rating = label_encoder.inverse_transform([predicted_class_index])[0]

            print(f"\n--- Prediction Results ---")
            print(f"Input Sentiment Score: {sentiment_score:.4f}")
            print(f"Input Financial Data: {financial_data_input}")
            print(f"Predicted Credit Rating: {predicted_rating}")
            print(f"Prediction Probabilities: {dict(zip(label_encoder.classes_, prediction_probabilities.round(3)))}")
            return predicted_rating

        except Exception as e:
            print(f"An error occurred during input or prediction: {e}")
            return None

    elif choice == '2':
        if merged_df is None or merged_df.empty:
            print("\nCannot select from existing dataset: Original data files were not loaded or merged successfully.")
            return None

        company_name_query = input("Enter part of the company name to search (e.g., 'Boeing'): ").strip()
        # Use the dynamically determined merge_key_left for searching
        matching_companies = merged_df[merged_df[merge_key_left].str.contains(company_name_query, case=False, na=False)]

        if matching_companies.empty:
            print(f"No companies found matching '{company_name_query}'.")
            return None
        elif len(matching_companies) > 1:
            print("Multiple companies found. Please be more specific or choose one:")
            for i, row in matching_companies.iterrows():
                print(f"{i+1}. {row[merge_key_left]}") # Use merge_key_left for display
            try:
                selection_index = int(input("Enter the number of the company to select: ")) - 1
                if 0 <= selection_index < len(matching_companies):
                    selected_company_name = matching_companies.iloc[selection_index][merge_key_left]
                else:
                    print("Invalid selection.")
                    return None
            except ValueError:
                print("Invalid input. Please enter a number.")
                return None
        else:
            selected_company_name = matching_companies.iloc[0][merge_key_left]

        company_row = merged_df[merged_df[merge_key_left] == selected_company_name].iloc[0]

        # Extract features using the actual column names from the merged_df
        data_for_prediction = company_row[features].to_dict()

        # Prepare input for the model in the correct order of 'features' list
        input_data_array = np.array([[data_for_prediction[f] for f in features]])

        input_imputed = imputer.transform(input_data_array)
        input_scaled = scaler.transform(input_imputed)

        prediction_probabilities = model.predict(input_scaled, verbose=0)[0]
        predicted_class_index = np.argmax(prediction_probabilities)
        predicted_rating = label_encoder.inverse_transform([predicted_class_index])[0]

        print(f"\n--- Prediction Results for {selected_company_name} (from dataset) ---")
        print(f"Sentiment Score: {data_for_prediction['Avg_Compound']:.4f}")
        print(f"Financial Data: {data_for_prediction}")
        print(f"Predicted Credit Rating: {predicted_rating}")
        print(f"Prediction Probabilities: {dict(zip(label_encoder.classes_, prediction_probabilities.round(3)))}")
        return predicted_rating

    elif choice == '3':
        print("Exiting prediction tool.")
        return "Exit"
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
        return None

# --- Main interactive loop ---
if __name__ == "__main__":
    print("\n--- Starting Interactive Prediction ---")
    while True:
        result = predict_credit_rating_interactive()
        if result == "Exit":
            break
        elif result is not None:
            print("\nPrediction complete. Ready for next prediction.")
        else:
            print("\nPrediction failed or invalid input. Please try again.")

        if input("\nDo you want to make another prediction? (yes/no): ").lower() != 'yes':
            break
    print("Program finished.")