import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="Demand Analysis", layout="wide")

## Business Demand Analyzer
st.title("📊 Demand & Lead Time Analysis")

# --- Inputs ---
with st.sidebar:
    st.header("Settings")
    user_type = st.radio("Entity Type", ["Retailer", "Distributor"])
    lead_time = st.number_input("Lead Time (Days/Periods)", min_value=1, value=7)
    offset = st.number_input("Offset Window", min_value=0, value=0)
    service_level = st.slider("Service Level (%)", 70.0, 99.0, 95.0) / 100

uploaded_file = st.file_uploader("Upload Demand CSV (columns: date, demand)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # --- 1. Create Lead Time Demand (Rolling Sum) FIRST ---
    # This captures the true total demand in every window, zeros included.
    df['lt_demand'] = df['demand'].rolling(window=lead_time).sum().shift(-offset)
    
    # --- 2. Filter Zeros for Distributors AFTER aggregation ---
    # This removes the "idle" windows where no demand occurred.
    processed_df = df.dropna().copy()
    if user_type == "Distributor":
        processed_df = processed_df[processed_df['lt_demand'] > 0]
        st.info("💡 Distributor Mode: Removed windows with zero total lead-time demand.")

    # --- 3. Seasonal Decomposition ---
    # We decompose the aggregated demand to see trend vs. noise
    try:
        # Determine a reasonable period for decomposition
        period_guess = 7 if len(processed_df) > 14 else 1
        analysis = seasonal_decompose(processed_df['lt_demand'], model='additive', period=period_guess)
        
        processed_df['trend'] = analysis.trend
        processed_df['seasonal'] = analysis.seasonal
        processed_df['residual'] = analysis.resid
    except Exception as e:
        st.error(f"Decomposition failed: {e}. Using raw aggregated demand.")
        processed_df['residual'] = processed_df['lt_demand']

    # --- 4. Histogram & Service Level ---
    st.subheader("Lead Time Demand Distribution")
    
    # Calculate demand requirement based on the chosen percentile
    target_value = processed_df['lt_demand'].quantile(service_level)

    fig = px.histogram(
        processed_df, 
        x="lt_demand", 
        nbins=40,
        title=f"Distribution of Demand over {lead_time}-period Lead Time",
        color_discrete_sequence=['#1f77b4'] # Classic blue
    )
    
    # Add Service Level Line
    fig.add_vline(x=target_value, line_dash="dash", line_color="red")
    fig.add_annotation(x=target_value, text=f"Service Level ({service_level*100}%): {target_value:.2f}", showarrow=True)

    st.plotly_chart(fig, use_container_width=True)

    # --- 5. Final Metrics ---
    cols = st.columns(3)
    cols[0].metric("Suggested Stock Level", f"{target_value:.2f}")
    cols[1].metric("Avg. LT Demand", f"{processed_df['lt_demand'].mean():.2f}")
    cols[2].metric("Max LT Demand", f"{processed_df['lt_demand'].max():.2f}")

else:
    st.write("Please upload a CSV to see the analysis.")
