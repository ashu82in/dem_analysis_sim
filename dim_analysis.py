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
        # Macro Growth + Weekly Heartbeat
        base = np.linspace(60, 200, 365) + (25 * np.sin(2 * np.pi * np.arange(365) / 7))
        # Distributor Lumpy Orders
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

# --- 3. CORE ANALYSIS & FORECASTING ---
if 'df' in st.session_state:
    df = st.session_state['df'].copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')

    # STEP 1: Rolling Window Aggregation (Cumulative Risk)
    df['lt_demand'] = df['y'].rolling(window=lead_time).sum().shift(-offset)
    
    analysis_df = df.dropna().copy()
    if user_type == "Distributor":
        # Remove zeros POST-aggregation to keep Trend from collapsing
        analysis_df = analysis_df[analysis_df['lt_demand'] > 0]

    try:
        # STEP 2: Prophet Modeling (For Straight-Line Trend & Forecast)
        prophet_df = analysis_df[['ds', 'lt_demand']].rename(columns={'lt_demand': 'y'})
        prophet_df['floor'] = 0
        
        # changepoint_prior_scale=0.001 forces the trend to be a series of straight lines
        m = Prophet(growth='linear', yearly_seasonality=True, weekly_seasonality=True, 
                    changepoint_prior_scale=0.001)
        m.fit(prophet_df)
        
        # Get historical trend and 100-day forecast
        future = m.make_future_dataframe(periods=100)
        future['floor'] = 0
        forecast = m.predict(future)
        
        # Non-negative Clipping
        for col in ['yhat', 'yhat_lower', 'yhat_upper', 'trend']:
            forecast[col] = forecast[col].clip(lower=0)
            
        # Split forecast into historical fit and future prediction
        hist_forecast = forecast[forecast['ds'] <= analysis_df['ds'].max()]
        future_forecast = forecast[forecast['ds'] > analysis_df['ds'].max()]

        # STEP 3: Visual Decomposition (Using Prophet's Straight Trend)
        # We still use statsmodels for the Seasonal/Residual breakout
        decomp = seasonal_decompose(analysis_df['lt_demand'], model='additive', period=7)
        analysis_df['seasonal'] = decomp.seasonal
        analysis_df['residual'] = decomp.resid
        
        st.subheader(f"🔍 Lead Time Window ({lead_time}-Day) Decomposition")
        fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                                   subplot_titles=("Raw Aggregated Demand", "Macro Trend (Business Direction)", 
                                                   "Seasonality (Weekly Wave)", "Residuals (Unpredictable Noise)"))
        
        # Row 1: Raw
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="Raw", line=dict(color='#A0AEC0')), row=1, col=1)
        # Row 2: STRAIGHT LINE TREND (from Prophet)
        fig_decomp.add_trace(go.Scatter(x=hist_forecast['ds'], y=hist_forecast['trend'], name="Trend", line=dict(color='#3182CE', width=3)), row=2, col=1)
        # Row 3: Seasonality
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['seasonal'], name="Seasonality", line=dict(color='#805AD5')), row=3, col=1)
        # Row 4: Residuals
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['residual'], name="Residuals", mode='markers', marker=dict(color='#E53E3E', size=4)), row=4, col=1)
        
        fig_decomp.update_layout(height=800, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_decomp, use_container_width=True)

        # STEP 4: 100-Day Forecast & Zones
        st.divider()
        st.header("🔮 100-Day Business Forecast & Zones")
        
        # Zoning Logic
        f_mean = future_forecast['yhat'].mean()
        def get_zone(val):
            if val < f_mean * 0.9: return "Zone 1 (Stable)"
            elif val < f_mean * 1.1: return "Zone 2 (Mid)"
            else: return "Zone 3 (High Surge)"
        future_forecast['Zone'] = future_forecast['yhat'].apply(get_zone)

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="History", line=dict(color="#A0AEC0")))
        fig_f.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], name="Forecast", line=dict(color="#3182CE", width=3)))
        fig_f.add_trace(go.Scatter(
            x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1],
            y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1],
            fill='toself', fillcolor='rgba(49, 130, 206, 0.2)', line=dict(color='rgba(255,255,255,0)'), name="Risk Range"
        ))
        fig_f.update_layout(template="plotly_dark", title="Future Windowed Demand Forecast")
        st.plotly_chart(fig_f, use_container_width=True)

        # STEP 5: Detailed Forecast Table
        st.subheader("📅 Detailed Strategy Table")
        st.dataframe(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'Zone']].rename(columns={
            'ds':'Date', 'yhat':'Expected Demand', 'yhat_lower':'Lower Risk', 'yhat_upper':'Upper Risk'
        }).style.format(precision=0), use_container_width=True)

        # STEP 6: Safety Stock calculation from Residuals
        noise = analysis_df['residual'].dropna()
        safety_buffer = noise.quantile(service_level)
        
        # Baseline = Last Stiff Trend + Last Seasonal point
        last_trend = hist_forecast['trend'].iloc[-1]
        last_seasonal = analysis_df['seasonal'].dropna().iloc[-1]
        total_req = max(0, last_trend + last_seasonal + safety_buffer)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Order Requirement", f"{total_req:.0f} units")
            st.write(f"Safety Buffer: **{safety_buffer:.1f}**")
            st.caption("Buffer is calculated purely on 'Window Residuals' to cover the blind spots.")
        with c2:
            st.write("**Future Zone Analysis**")
            st.table(future_forecast.groupby('Zone')['yhat'].agg(['count', 'mean']).rename(columns={'count':'Days', 'mean':'Avg Demand'}))

    except Exception as e:
        st.error(f"Computation Error: {e}")
