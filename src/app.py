
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import datetime

# Page Config
st.set_page_config(page_title="Walmart Sales Predictor", layout="wide", page_icon="🛒")

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #0071dc;
        color: white;
        border-radius: 5px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1, h2, h3 {
        color: #0071dc;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load Data and Resources
@st.cache_resource
def load_resources():
    # Load Model
    model = pickle.load(open('data/outputs/model.pkl', 'rb'))
    
    # Load Data
    X_test = pd.read_csv('data/validation_data/test_data.csv')
    y_test = pd.read_csv('data/validation_data/test_labels.csv')
    
    # Load Store Means
    store_means = pickle.load(open('data/outputs/store_means.pkl', 'rb'))
    
    return model, X_test, y_test, store_means

try:
    model, X_test, y_test, store_means = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# Reconstruct Store ID and Date in X_test for lookup
# Reverse mapping mean -> store_id
# Note: precision issues might make exact matching hard, so we use tolerance or round
mean_to_store = {round(v, 6): k for k, v in store_means.items()}
X_test['Store_Target_Mean_Rounded'] = X_test['Store_Target_Mean'].round(6)
X_test['Store'] = X_test['Store_Target_Mean_Rounded'].map(mean_to_store)

# Create Date column
def get_date_from_year_week(row):
    # Year, Week are available. Month is also available.
    # Simple approx: ISO week date
    try:
        return datetime.date.fromisocalendar(int(row['Year']), int(row['Week']), 5) # Friday
    except:
        return None

X_test['Date'] = X_test.apply(get_date_from_year_week, axis=1)

# Title
st.title("Walmart Weekly Sales Predictor")
st.markdown("### AI-Powered Forecasting with XGBoost")

# --- SECTION A: The "Smart" Predictor ---
st.header("Interactive Demo")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    # Store Selector
    available_stores = sorted([k for k in store_means.keys() if k in X_test['Store'].unique()])
    selected_store = st.selectbox("Select Store", available_stores, format_func=lambda x: f"Store {x}")

with col2:
    # Date Selector (Limit to range in test data)
    min_date = X_test['Date'].min()
    max_date = X_test['Date'].max()
    selected_date = st.date_input("Select Date", min_value=min_date, max_value=max_date, value=min_date)

# Find corresponding row
# Filter X_test
mask = (X_test['Store'] == selected_store) & (X_test['Date'] <= selected_date + datetime.timedelta(days=3)) & (X_test['Date'] >= selected_date - datetime.timedelta(days=3))
# Just picking the closest week row if exact match fails, or exact match on Week/Year
# Better: Match Year and Week
selected_year = selected_date.year
selected_week = selected_date.isocalendar()[1]
row_mask = (X_test['Store'] == selected_store) & (X_test['Year'] == selected_year) & (X_test['Week'] == selected_week)

row = X_test[row_mask]

if not row.empty:
    input_row = row.iloc[0].copy()
    actual_sales = y_test.iloc[row.index[0]].values[0] if not y_test.empty else None
    
    with col3:
        # Holiday Override
        is_holiday = st.checkbox("Is this a Holiday?", value=bool(input_row['Holiday_Flag']))
        input_row['Holiday_Flag'] = 1 if is_holiday else 0
    
    # Prepare input for prediction (Drop helper columns)
    features_for_model = input_row.drop(['Store', 'Date', 'Store_Target_Mean_Rounded']).to_frame().T
    features_for_model = features_for_model.astype(float)
    
    # Ensure correct column order as expected by model (using X_test columns excluding helpers)
    model_features = X_test.columns.drop(['Store', 'Date', 'Store_Target_Mean_Rounded'])
    
    # Predict
    prediction = np.exp(model.predict(features_for_model[model_features])[0]) #taking anti-log cause we log-transformed the values for that store sales are on scale for each store
    
    st.subheader("Prediction Result")
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label="Predicted Weekly Sales", value=f"${prediction:,.2f}")
    with metric_col2:
        if actual_sales:
            delta = prediction - actual_sales
            st.metric(label="Actual Sales", value=f"${actual_sales:,.2f}", delta=f"{delta:,.2f}", delta_color="inverse")
        else:
            st.info("Actual sales data not available for strict comparison.")

    # --- SECTION B: Explain It To Me ---
    st.markdown("---")
    st.header("SHAP Interpretation")
    
    if st.button("Generate Explanation"):
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(features_for_model[model_features])
            
            # Transform SHAP values from log-scale to original scale (dollars) to maintain additivity in the plot
            log_base = shap_values.base_values[0]
            log_values = shap_values.values[0]
            actual_base = np.exp(log_base)
            actual_pred = np.exp(log_base + log_values.sum())
            
            # Distribute the total difference in original scale proportionally to log-scale contributions
            if abs(log_values.sum()) > 1e-6:
                shap_values.values[0] = log_values * (actual_pred - actual_base) / log_values.sum()
            
            # Waterfall Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False, max_display=10)
            st.pyplot(fig)
            
            # Narrative
            # Find top contributing features
            values = shap_values[0].values
            feature_names = model_features
            
            top_idx = np.argmax(np.abs(values))
            top_feature = feature_names[top_idx]
            top_contribution = values[top_idx]
            
            impact = "increase" if top_contribution > 0 else "decrease"
            # st.markdown(f"####Insight:")
            # st.write(f"Sales are predicted to **{impact}** significantly because **'{top_feature}'** has a value of **{features_for_model[top_feature].values[0]:.2f}** (Impact: ${top_contribution:,.0f}).")

    
# --- SECTION C: Performance Dashboard ---
st.markdown("---")
st.header("Performance Dashboard")

# Calculate metrics on the fly for the test set
# Calculate metrics on the fly for the test set
model_features_full = X_test.columns.drop(['Store', 'Date', 'Store_Target_Mean_Rounded'])
preds_full = np.exp(model.predict(X_test[model_features_full]))
# Note: XGBoost model from pickle might not have .features_names attribute accessible this way easily depending on version, 
# but we know columns match X_test minus helpers.
# Actually, if I trained with sklearn API, it preserves feature names. 
# Let's assume columns match.

# Align indices
if len(preds_full) == len(y_test):
    y_true = y_test.iloc[:, 0].values # Assuming first col is target
    
    r2 = r2_score(y_true, preds_full)
    mae = mean_absolute_error(y_true, preds_full)
    rmse = np.sqrt(mean_squared_error(y_true, preds_full))
    mape = mean_absolute_percentage_error(y_true, preds_full)

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{r2:.4f}")
    m2.metric("MAPE", f"{mape:.4%}")
    m3.metric("RMSE", f"${rmse:,.0f}")
    
    st.markdown("#### Actual vs Predicted")
    fig_perf, ax_perf = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=y_true, y=preds_full, alpha=0.5, ax=ax_perf)
    ax_perf.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax_perf.set_xlabel("Actual Sales")
    ax_perf.set_ylabel("Predicted Sales")
    st.pyplot(fig_perf)
else:
    st.error("Mismatch in Test Data and Labels length.")

# --- SECTION D: Under the Hood ---
st.markdown("---")
st.header("Under the Hood")

st.markdown("""
### Model Architecture
This application uses **XGBoost (Extreme Gradient Boosting)**, a state-of-the-art machine learning algorithm known for its performance on structured data.

### Secret Sauce
1.  **Target Encoding**: I encode the `Store` categorical variable using `Store_Target_Mean` (mean sales per store) to capture store-level performance baselines.
2.  **Lag Features**: I use past performance (e.g., `Lag_Sales_1w`, `Rolling_Mean_Sales_4w`) as powerful predictors for future sales. This allows the model to "remember" recent trends.
3.  **Holiday Handling**: Custom features like `Weeks_To_Next_Holiday` allow the model to anticipate sales spikes before major events.
4. **Handeliing scale differences across stores :** I trained the model on `log transformsed weekly sales`, effectively optimising relative error while retaing squared error staibility
""")

