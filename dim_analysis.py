import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from io import BytesIO

st.set_page_config(page_title="Demand Intelligence Pro", layout="wide")

# --- 1. SIDEBAR: Parameters ---
with st.sidebar:
    st.header("🎯 Strategy Settings")
    user_type = st.radio("Business Model", ["Retailer", "Distributor"])
    
    col_lt, col_off = st.columns(2)
    with col_lt:
        lead_time = st.number_input("Lead Time", min_value=1, value=7)
    with col_off:
        offset = st.number_input("Offset", min_value=0, value=0)
        
    service_level = st.slider("Service Level (%)", 70.0, 99.9, 95.0) / 100

    st.divider()
    if st.button("✨ Generate Demo Data"):
        dates = pd.date_range(start="2025-01-01", periods=365, freq='D')
        # Base with Growth + Weekly Heartbeat
        base = np.linspace(40, 120, 365) + (15 * np.sin(2 * np.pi * np.arange(365) / 7))
        # Distributor Lumpy Pattern
        mask = np.random.random(365) > 0.85
        demo_demand = np.where(mask, base * 4, 0).astype(int)
        st.session_state['df'] = pd.DataFrame({'date': dates, 'demand': demo_demand})

# --- 2. DATA LOADING ---
uploaded_file = st.file_uploader("Upload Excel/CSV", type=["csv", "xlsx"])
if uploaded_file:
    st.session_state['df'] = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

# --- 3. CORE ANALYSIS & FORECASTING ---
if 'df' in st.session_state:
    df = st.session_state['df'].copy()
    df.columns = [c.lower().strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # STEP 1: Rolling Sum (Lead Time Demand)
    df['lt_demand'] = df['demand'].rolling(window=lead_time).sum().shift(-offset)
    
    # STEP 2: Filter Zeros for Distributor (POST-Aggregation)
    analysis_df = df.dropna().copy()
    if user_type == "Distributor":
        analysis_df = analysis_df[analysis_df['lt_demand'] > 0]
        st.caption("✅ Distributor Mode: Zero-demand windows removed.")

    # STEP 3: Decomposition
    try:
        decomp = seasonal_decompose(analysis_df['lt_demand'], model='additive', period=7)
        analysis_df['trend'] = decomp.trend
        analysis_df['seasonal'] = decomp.seasonal
        analysis_df['residual'] = decomp.resid
        
        # --- VISUAL A: 4-Row Decomposition ---
        st.subheader("🔍 Demand Decomposition Breakout")
        fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                                   subplot_titles=("Raw Lead Time Demand", "Trend (Growth)", "Seasonality (Cycle)", "Residuals (Unpredictable Noise)"))
        
        fig_decomp.add_trace(go.Scatter(x=analysis_df['date'], y=analysis_df['lt_demand'], name="Raw", line=dict(color='#A0AEC0')), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=analysis_df['date'], y=analysis_df['trend'], name="Trend", line=dict(color='#3182CE')), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(x=analysis_df['date'], y=analysis_df['seasonal'], name="Seasonality", line=dict(color='#805AD5')), row=3, col=1)
        fig_decomp.add_trace(go.Scatter(x=analysis_df['date'], y=analysis_df['residual'], name="Residuals", mode='markers', marker=dict(color='#E53E3E', size=4)), row=4, col=1)
        fig_decomp.update_layout(height=700, showlegend=False, template="plotly_dark")
        st.plotly_chart(fig_decomp, use_container_width=True)

        # STEP 4: SARIMA Forecast (Next 100 Days)
        st.divider()
        st.header("🔮 100-Day SARIMA Forecast & Zones")
        
        series = analysis_df.set_index('date')['lt_demand']
        sarima_model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
        forecast_res = sarima_model.get_forecast(steps=100)
        forecast_df = forecast_res.summary_frame()
        forecast_df.index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=100)

        # Zoning Logic
        avg_f = forecast_df['mean'].mean()
        def get_zone(val):
            if val < avg_f * 0.85: return "Zone 1 (Low)"
            elif val < avg_f * 1.15: return "Zone 2 (Normal)"
            else: return "Zone 3 (Peak)"
        forecast_df['Zone'] = forecast_df['mean'].apply(get_zone)

        # --- VISUAL B: Forecast Chart ---
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=series.index[-30:], y=series[-30:], name="History", line=dict(color="#A0AEC0")))
        fig_f.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], name="Forecast", line=dict(color="#3182CE", width=3)))
        fig_f.add_trace(go.Scatter(
            x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
            y=forecast_df['mean_ci_upper'].tolist() + forecast_df['mean_ci_lower'].tolist()[::-1],
            fill='toself', fillcolor='rgba(49, 130, 206, 0.2)', line=dict(color='rgba(255,255,255,0)'), name="95% Confidence"
        ))
        st.plotly_chart(fig_f, use_container_width=True)

        # STEP 5: Smart Safety Stock (Residuals)
        noise = analysis_df['residual'].dropna()
        safety_buffer = noise.quantile(service_level)
        current_req = (analysis_df['trend'].iloc[-1] + analysis_df['seasonal'].iloc[-1]) + safety_buffer

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Order Requirement", f"{current_req:.0f} units")
            st.write(f"Safety Buffer: {safety_buffer:.1f} units")
        with col2:
            st.table(forecast_df.groupby('Zone')['mean'].agg(['count', 'mean', 'max']).rename(columns={'count':'Days', 'mean':'Avg Demand'}))

    except Exception as e:
        st.error(f"Computation Error: {e}")
