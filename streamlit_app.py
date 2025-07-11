import streamlit as st
import credit_rating_predictor as crp # Import our prediction module
import pandas as pd # Just for general use, already in crp but good to be explicit

st.set_page_config(page_title="Credit Rating Predictor", layout="centered")

st.title("📊 Credit Rating Predictor")
st.markdown("""
Welcome to the interactive credit rating prediction tool!
You can either input financial and sentiment data manually or select an existing company to see its predicted credit rating.
""")

# Use Streamlit's caching for efficient loading of model components and lookup data
# This decorator ensures the function runs only once per app session (or when inputs change)
@st.cache_resource
def load_all_resources():
    crp._load_model_components() # Load model and preprocessing objects
    crp._load_lookup_data()      # Load data for company lookup
    return True # Indicate success

load_all_resources() # Load resources at app startup

# Get feature defaults (medians) from the loaded imputer
feature_defaults = crp.get_feature_defaults()

# --- User Input Method Selection ---
prediction_method = st.radio(
    "Choose your prediction method:",
    ("Enter New Data", "Select Existing Company from Dataset")
)

# --- Prediction Logic based on method ---
if prediction_method == "Enter New Data":
    st.header("📈 Enter New Financial & Sentiment Data")

    # Use default values from imputer for number inputs
    avg_compound = st.number_input(
        "Average Compound Sentiment (e.g., -1.0 to 1.0):",
        min_value=-1.0, max_value=1.0, value=float(feature_defaults.get('Avg_Compound', 0.0)), step=0.01
    )
    net_profit_margin = st.number_input(
        "Net Profit Margin (e.g., 0.1 for 10%):",
        min_value=-10.0, max_value=10.0, value=float(feature_defaults.get('netProfitMargin', 0.05)), format="%.4f"
    )
    debt_equity_ratio = st.number_input(
        "Debt to Equity Ratio (e.g., 1.5):",
        min_value=0.0, max_value=100.0, value=float(feature_defaults.get('debtEquityRatio', 1.0)), format="%.4f"
    )
    current_ratio = st.number_input(
        "Current Ratio (e.g., 2.0):",
        min_value=0.0, max_value=100.0, value=float(feature_defaults.get('currentRatio', 1.5)), format="%.4f"
    )

    if st.button("Predict Credit Rating"):
        input_data = {
            'Avg_Compound': avg_compound,
            'netProfitMargin': net_profit_margin,
            'debtEquityRatio': debt_equity_ratio,
            'currentRatio': current_ratio
        }
        predicted_rating, probabilities = crp.predict_rating(input_data)

        st.subheader("Prediction Results:")
        st.success(f"**Predicted Credit Rating: {predicted_rating}**")
        st.write("Confidence per Rating Class:")
        st.json(probabilities) # Display probabilities as JSON for readability

elif prediction_method == "Select Existing Company from Dataset":
    st.header("🏢 Select an Existing Company")

    company_names = crp.get_company_names()

    if not company_names:
        st.warning("No company data available for lookup. Please ensure 'corporate_rating.csv' and 'part5test_companies_sentiment.csv' are in the root directory and contain matching company names.")
    else:
        selected_company = st.selectbox(
            "Select a Company:",
            options=['-- Select a Company --'] + sorted(company_names)
        )

        if selected_company != '-- Select a Company --':
            company_data = crp.get_company_data(selected_company)
            if company_data:
                st.subheader(f"Data for {selected_company}:")
                # Display the raw data used for prediction
                st.json(company_data)

                if st.button(f"Predict Rating for {selected_company}"):
                    predicted_rating, probabilities = crp.predict_rating(company_data)

                    st.subheader("Prediction Results:")
                    st.info(f"**Predicted Credit Rating for {selected_company}: {predicted_rating}**")
                    st.write("Confidence per Rating Class:")
                    st.json(probabilities)
            else:
                st.error("Could not retrieve data for the selected company.")

st.markdown("---")
st.markdown("Developed with Streamlit and TensorFlow/Keras.")