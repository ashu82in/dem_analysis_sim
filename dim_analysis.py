import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from io import BytesIO

st.set_page_config(page_title="Demand Strategy Pro", layout="wide")

# --- 1. SIDEBAR: Parameters ---
with st.sidebar:
    st.header("🎯 Strategy Settings")
    user_type = st.radio("Business Model", ["Retailer", "Distributor"])
    
    col_lt, col_off = st.columns(2)
    with col_lt:
        lead_time = st.number_input("Lead Time (Days)", min_value=1, value=7)
    with col_off:
        offset = st.number_input("Offset", min_value=0, value=0)
        
    service_level = st.slider("Service Level (%)", 70.0, 99.9, 95.0) / 100

    st.divider()
    if st.button("✨ Generate Demo Data"):
        dates = pd.date_range(start="2025-01-01", periods=365, freq='D')
        # Strong Growth + Weekly Heartbeat
        base = np.linspace(50, 180, 365) + (20 * np.sin(2 * np.pi * np.arange(365) / 7))
        # Lumpy Distributor Orders
        mask = np.random.random(365) > 0.85
        demo_demand = np.where(mask, base * 3.5, 0).astype(int)
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

    # STEP 1: Rolling Sum (Windowed Lead Time Demand)
    # This is the demand for the NEXT lead_time days starting from 'ds'
    df['lt_demand'] = df['y'].rolling(window=lead_time).sum().shift(-offset)
    
    analysis_df = df.dropna().copy()
    if user_type == "Distributor":
        analysis_df = analysis_df[analysis_df['lt_demand'] > 0]

    try:
        # STEP 2: Decomposition (Capturing current state)
        decomp = seasonal_decompose(analysis_df['lt_demand'], model='additive', period=7)
        analysis_df['trend'] = decomp.trend
        analysis_df['seasonal'] = decomp.seasonal
        analysis_df['residual'] = decomp.resid
        
        st.subheader(f"🔍 Lead Time Window ({lead_time}-Day) Decomposition")
        fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                                   subplot_titles=("Raw Aggregated Demand", "Trend (Window Volume)", "Seasonality (Weekly Wave)", "Residuals (Uncertainty)"))
        
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="Raw", line=dict(color='#A0AEC0')), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['trend'], name="Trend", line=dict(color='#3182CE')), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['seasonal'], name="Seasonality", line=dict(color='#805AD5')), row=3, col=1)
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['residual'], name="Residuals", mode='markers', marker=dict(color='#E53E3E', size=4)), row=4, col=1)
        fig_decomp.update_layout(height=700, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_decomp, use_container_width=True)

        # STEP 3: Prophet Forecasting (The Macro Trend & Forecast)
        st.divider()
        st.header("🔮 100-Day Business Forecast & Zones")
        
        # Prepare data for Prophet
        prophet_df = analysis_df[['ds', 'lt_demand']].rename(columns={'lt_demand': 'y'})
        prophet_df['floor'] = 0 # Prevent negatives
        
        # changepoint_prior_scale=0.01 creates the 'straight-line' effect
        m = Prophet(growth='linear', yearly_seasonality=True, weekly_seasonality=True, 
                    changepoint_prior_scale=0.01)
        m.fit(prophet_df)
        
        future = m.make_future_dataframe(periods=100)
        future['floor'] = 0
        forecast = m.predict(future)
        
        # Enforce non-negativity across all forecast results
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            forecast[col] = forecast[col].clip(lower=0)
            
        f_part = forecast.tail(100).copy()

        # Define Dynamic Zones
        f_mean = f_part['yhat'].mean()
        def get_zone(val):
            if val < f_mean * 0.9: return "Zone 1 (Stable)"
            elif val < f_mean * 1.1: return "Zone 2 (Mid)"
            else: return "Zone 3 (High Surge)"
        f_part['Zone'] = f_part['yhat'].apply(get_zone)

        # Forecast Chart
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="History", line=dict(color="#A0AEC0")))
        fig_f.add_trace(go.Scatter(x=f_part['ds'], y=f_part['yhat'], name="Forecast", line=dict(color="#3182CE", width=3)))
        fig_f.add_trace(go.Scatter(
            x=f_part['ds'].tolist() + f_part['ds'].tolist()[::-1],
            y=f_part['yhat_upper'].tolist() + f_part['yhat_lower'].tolist()[::-1],
            fill='toself', fillcolor='rgba(49, 130, 206, 0.2)', line=dict(color='rgba(255,255,255,0)'), name="Risk Range"
        ))
        fig_f.update_layout(template="plotly_dark", title="Future Windowed Demand")
        st.plotly_chart(fig_f, use_container_width=True)

        # STEP 4: Forecast Table
        st.subheader("📅 Detailed Strategy Table")
        st.dataframe(f_part[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'Zone']].rename(columns={
            'ds':'Date', 'yhat':'Expected Demand', 'yhat_lower':'Lower Risk', 'yhat_upper':'Upper Risk'
        }).style.format(precision=0), use_container_width=True)

        # STEP 5: Strategic Metrics
        noise = analysis_df['residual'].dropna()
        safety_buffer = noise.quantile(service_level)
        
        # Safe Baseline Extraction
        last_trend = analysis_df['trend'].dropna().iloc[-1]
        last_seasonal = analysis_df['seasonal'].dropna().iloc[-1]
        total_req = max(0, last_trend + last_seasonal + safety_buffer)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Order Requirement", f"{total_req:.0f} units")
            st.write(f"Safety Buffer: **{safety_buffer:.1f}**")
            st.caption("Buffer covers the 'Window Residuals' (Unpredictability).")
        with c2:
            st.write("**Future Zone Analysis**")
            st.table(f_part.groupby('Zone')['yhat'].agg(['count', 'mean']).rename(columns={'count':'Days', 'mean':'Avg Demand'}))

    except Exception as e:
        st.error(f"Computation Error: {e}")
