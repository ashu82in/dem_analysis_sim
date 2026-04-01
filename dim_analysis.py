import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from io import BytesIO

st.set_page_config(page_title="Lead Time Demand Intelligence", layout="wide")

# --- 1. SIDEBAR ---
with st.sidebar:
    st.header("🎯 Strategy Settings")
    user_type = st.radio("Business Model", ["Retailer", "Distributor"])
    lead_time = st.number_input("Lead Time (Window Size)", min_value=1, value=7)
    offset = st.number_input("Offset", min_value=0, value=0)
    service_level = st.slider("Service Level (%)", 70.0, 99.9, 95.0) / 100

    st.divider()
    if st.button("✨ Generate Demo Data"):
        dates = pd.date_range(start="2025-01-01", periods=365, freq='D')
        base = np.linspace(50, 150, 365) + (25 * np.sin(2 * np.pi * np.arange(365) / 7))
        mask = np.random.random(365) > 0.85
        demo_demand = np.where(mask, base * 3, 0).astype(int)
        st.session_state['df'] = pd.DataFrame({'ds': dates, 'y': demo_demand})

# --- 2. DATA LOADING ---
uploaded_file = st.file_uploader("Upload Excel/CSV", type=["csv", "xlsx"])
if uploaded_file:
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    data.columns = [c.lower().strip() for c in data.columns]
    data = data.rename(columns={'date': 'ds', 'demand': 'y'})
    st.session_state['df'] = data

# --- 3. CORE ANALYSIS ---
if 'df' in st.session_state:
    df = st.session_state['df'].copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')

    # Aggregating into the Lead Time Window FIRST
    df['lt_demand'] = df['y'].rolling(window=lead_time).sum().shift(-offset)
    
    analysis_df = df.dropna().copy()
    if user_type == "Distributor":
        analysis_df = analysis_df[analysis_df['lt_demand'] > 0]

    try:
        # Decomposition of the 7-day Window
        decomp = seasonal_decompose(analysis_df['lt_demand'], model='additive', period=7)
        analysis_df['trend'] = decomp.trend
        analysis_df['seasonal'] = decomp.seasonal
        analysis_df['residual'] = decomp.resid
        
        st.subheader(f"🔍 Decomposition of {lead_time}-Day Lead Time Demand")
        fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                                   subplot_titles=("Raw LT Demand", "Trend (Window Growth)", "Seasonality (Window Cycle)", "Residuals (Window Uncertainty)"))
        
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="Raw", line=dict(color='#A0AEC0')), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['trend'], name="Trend", line=dict(color='#3182CE')), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['seasonal'], name="Seasonality", line=dict(color='#805AD5')), row=3, col=1)
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['residual'], name="Residuals", mode='markers', marker=dict(color='#E53E3E', size=4)), row=4, col=1)
        fig_decomp.update_layout(height=700, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_decomp, use_container_width=True)

        # STEP 4: Prophet Forecasting (On the Windowed Data)
        st.divider()
        st.header("🔮 100-Day Windowed Forecast")
        
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        # We train on 'lt_demand' directly
        m.fit(analysis_df[['ds', 'lt_demand']].rename(columns={'lt_demand': 'y'}))
        
        future = m.make_future_dataframe(periods=100)
        forecast = m.predict(future)
        f_part = forecast.tail(100).copy()

        # Forecasting Zones
        avg_f = f_part['yhat'].mean()
        def get_zone(val):
            if val < avg_f * 0.9: return "Zone 1 (Stable)"
            elif val < avg_f * 1.1: return "Zone 2 (Mid)"
            else: return "Zone 3 (High Surge)"
        f_part['Zone'] = f_part['yhat'].apply(get_zone)

        # Visual: Forecast
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="Actual LT Demand", line=dict(color="#A0AEC0")))
        fig_f.add_trace(go.Scatter(x=f_part['ds'], y=f_part['yhat'], name="Forecasted LT Demand", line=dict(color="#3182CE", width=3)))
        fig_f.update_layout(template="plotly_dark", title="Expected Demand per Lead Time Window")
        st.plotly_chart(fig_f, use_container_width=True)

        # --- NEW: Forecast Table ---
        st.subheader("📅 Detailed Forecast Table")
        display_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'Zone']
        formatted_table = f_part[display_cols].rename(columns={
            'ds': 'Date', 'yhat': 'Expected Demand', 
            'yhat_lower': 'Min Range', 'yhat_upper': 'Max Range'
        })
        st.dataframe(formatted_table.style.format({
            'Expected Demand': '{:.0f}', 'Min Range': '{:.0f}', 'Max Range': '{:.0f}'
        }), use_container_width=True)

        # Safety Stock Calculation
        noise = analysis_df['residual'].dropna()
        safety_buffer = noise.quantile(service_level)
        last_baseline = analysis_df['trend'].dropna().iloc[-1] + analysis_df['seasonal'].dropna().iloc[-1]
        total_req = last_baseline + safety_buffer

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Stock Requirement", f"{total_req:.0f} units")
            st.write(f"Safety Buffer: **{safety_buffer:.1f}**")
        with c2:
            st.write("**Zone Summary**")
            st.table(f_part.groupby('Zone')['yhat'].agg(['count', 'mean']).rename(columns={'count':'Days', 'mean':'Avg Demand'}))

    except Exception as e:
        st.error(f"Error: {e}")
