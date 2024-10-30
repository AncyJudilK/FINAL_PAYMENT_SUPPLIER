import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.graph_objects as go
from sklearn.model_selection import KFold

# Set up Streamlit page configuration
st.set_page_config(page_title="Payment Forecasting", page_icon="📈", layout="wide")
st.title('Payment Forecasting Application')

# Path to the sample data
sample_excel_path = "final_payment.xlsx"

# Checkbox to use sample data
use_sample_data = st.checkbox("Use Sample Data")

# Load and preview sample data if selected
if use_sample_data:
    with st.spinner('Loading sample data...'):
        time.sleep(2)
        data = pd.read_excel(sample_excel_path)
        st.success('Sample data loaded successfully!')

    # Show preview of sample data
    st.write("Sample Excel Preview:")
    st.dataframe(data)

    # Download button for sample data
    st.download_button(
        label="Download Sample Excel",
        data=data.to_csv(index=False),
        file_name="sample_payment_data.csv",
        mime="text/csv"
    )

# Otherwise, upload an Excel file
else:
    uploaded_file = st.file_uploader("Upload your Excel file for forecasting", type=["xlsx"])
    if uploaded_file is not None:
        with st.spinner('Loading your data...'):
            time.sleep(2)
            data = pd.read_excel(uploaded_file)
        st.success('Your data has been loaded successfully!')

# Proceed if data is available
if 'data' in locals():
    # Define features and target, excluding "Batch ID"
    input_features = [
        "Quarterly Revenue (USD)", "Production Costs (USD)", "gross margin (revenue - production) = (C - E)",
        "Logistics Costs (USD)", "R&D Costs (USD)", "Net Profit Margin (%)", "Commodity Price Index (Cobalt)",
        "Battery Recycling Volume (Metric Tons)", "Carbon Credits Earned (USD)"
    ]
    target_column = 'FINAL_PAYMENT_SUPPLIER'

    # Prepare data for training
    X = data[input_features]
    y = data[target_column]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define hyperparameter grid for Gradient Boosting
    param_grid_gb = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_features': ['auto', 'sqrt', 'log2'],
        'subsample': [0.8, 1.0]
    }

    # Gradient Boosting model
    gb_model = GradientBoostingRegressor(random_state=42)

    # K-Fold Cross-Validation Setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Use RandomizedSearchCV for broader exploration of hyperparameters
    gb_random_search = RandomizedSearchCV(estimator=gb_model, param_distributions=param_grid_gb,
                                          n_iter=100, cv=cv, n_jobs=-1, verbose=1, random_state=42)
    with st.spinner("Fine-tuning the Gradient Boosting model..."):
        gb_random_search.fit(X_train, y_train)

    # Best model from RandomizedSearchCV
    best_gb_model = gb_random_search.best_estimator_

    # Make predictions with the best model
    y_pred_gb = best_gb_model.predict(X_test)

    # Calculate evaluation metrics for Gradient Boosting
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mse_gb)
    r2_gb = r2_score(y_test, y_pred_gb)
    mape_gb = mean_absolute_percentage_error(y_test, y_pred_gb)

    # Display evaluation metrics for Gradient Boosting
    st.subheader("Gradient Boosting Model Evaluation Metrics:")
    st.write(f"**R-squared (R²):** {r2_gb:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse_gb:.4f}")
    st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape_gb:.4%}")

    # Creating a DataFrame for forecast results
    forecast_df_gb = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred_gb})
    st.subheader("Actual vs Predicted Payment Table (Gradient Boosting)")
    st.dataframe(forecast_df_gb)

    st.download_button(
        label="Download Gradient Boosting Forecasted Data",
        data=forecast_df_gb.to_csv(index=False),
        file_name="forecasted_payment_gb.csv",
        mime="text/csv"
    )

    # Forecast results visualization for Gradient Boosting
    st.subheader("Forecast Visualization (Gradient Boosting)")
    forecast_fig_gb = go.Figure()

    # Bar chart for Actual Values
    forecast_fig_gb.add_trace(go.Bar(
        x=forecast_df_gb.index,
        y=forecast_df_gb['Actual'],
        name='Actual Values',
        marker_color='blue'
    ))

    # Bar chart for Predicted Values
    forecast_fig_gb.add_trace(go.Bar(
        x=forecast_df_gb.index,
        y=forecast_df_gb['Predicted'],
        name='Predicted Values',
        marker_color='orange'
    ))

    # Updating layout
    forecast_fig_gb.update_layout(
        title="Actual vs Predicted Payment (Gradient Boosting)",
        xaxis_title="Index",
        yaxis_title="Payment",
        barmode='group',
        legend_title="Legend",
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        hovermode="x unified"
    )
    st.plotly_chart(forecast_fig_gb)
