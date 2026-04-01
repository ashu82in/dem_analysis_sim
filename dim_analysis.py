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
        # Macro Growth + Yearly Cycle
        base = np.linspace(100, 300, 365) + (50 * np.sin(2 * np.pi * np.arange(365) / 365))
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

# --- 3. CORE ANALYSIS & FORECASTING ---
if 'df' in st.session_state:
    df = st.session_state['df'].copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')

    # STEP 1: Overlapping Rolling Window
    # We aggregate first to capture cumulative risk over the lead time
    df['lt_demand'] = df['y'].rolling(window=lead_time).sum().shift(-offset)
    
    analysis_df = df.dropna().copy()
    if user_type == "Distributor":
        # Remove zeros POST-aggregation so the Trend doesn't collapse
        analysis_df = analysis_df[analysis_df['lt_demand'] > 0]

    try:
        # STEP 2: Prophet Modeling (Clean Macro Focus)
        prophet_df = analysis_df[['ds', 'lt_demand']].rename(columns={'lt_demand': 'y'})
        prophet_df['floor'] = 0
        
        # We disable weekly seasonality because the 7-day rolling window already absorbs it
        m = Prophet(
            growth='linear', 
            yearly_seasonality=True, 
            weekly_seasonality=False, 
            daily_seasonality=False,
            changepoint_prior_scale=0.001 # Forces the "Stiff" Trend Line
        )
        m.fit(prophet_df)
        
        # Generate 100-day forecast
        future = m.make_future_dataframe(periods=100)
        future['floor'] = 0
        forecast = m.predict(future)
        
        # Enforce non-negativity
        for col in ['yhat', 'yhat_lower', 'yhat_upper', 'trend', 'yearly']:
            if col in forecast.columns:
                forecast[col] = forecast[col].clip(lower=0)
            
        hist_forecast = forecast[forecast['ds'] <= analysis_df['ds'].max()]
        future_forecast = forecast[forecast['ds'] > analysis_df['ds'].max()].copy()

        # STEP 3: Visual Decomposition (Stiff Trend + Macro Cycles)
        # Using a 30-day period for statsmodels to capture monthly/macro patterns 
        # instead of the noisy weekly one.
        decomp = seasonal_decompose(analysis_df['lt_demand'], model='additive', period=30)
        analysis_df['residual'] = decomp.resid
        
        st.subheader(f"🔍 {lead_time}-Day Window Macro Analysis")
        fig_decomp = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                   subplot_titles=("1. Lead Time Demand (Aggregated)", 
                                                   "2. Macro Trend (Business Growth Direction)", 
                                                   "3. Planning Uncertainty (Residuals)"))
        
        # Row 1: Raw Aggregated
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="Aggregated", line=dict(color='#A0AEC0')), row=1, col=1)
        # Row 2: STRAIGHT LINE TREND (from Prophet)
        fig_decomp.add_trace(go.Scatter(x=hist_forecast['ds'], y=hist_forecast['trend'], name="Stiff Trend", line=dict(color='#3182CE', width=3)), row=2, col=1)
        # Row 3: Residuals
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['residual'], mode='markers', marker=dict(color='#E53E3E', size=4)), row=3, col=1)
        
        fig_decomp.update_layout(height=700, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_decomp, use_container_width=True)

        # STEP 4: 100-Day Forecast & Strategy Zones
        st.divider()
        st.header("🔮 100-Day Business Forecast & Zones")
        
        f_mean = future_forecast['yhat'].mean()
        def get_zone(val):
            if val < f_mean * 0.9: return "Zone 1 (Stable)"
            elif val < f_mean * 1.1: return "Zone 2 (Mid)"
            else: return "Zone 3 (High Surge)"
        future_forecast['Zone'] = future_forecast['yhat'].apply(get_zone)

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="Actual", line=dict(color="#A0AEC0")))
        fig_f.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], name="Forecast", line=dict(color="#3182CE", width=3)))
        fig_f.add_trace(go.Scatter(
            x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1],
            y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1],
            fill='toself', fillcolor='rgba(49, 130, 206, 0.2)', line=dict(color='rgba(255,255,255,0)'), name="Safety Buffer Range"
        ))
        fig_f.update_layout(template="plotly_dark")
        st.plotly_chart(fig_f, use_container_width=True)

        # STEP 5: Detailed Forecast Table
        st.subheader("📅 Strategy Breakdown Table")
        cols_to_show = ['ds', 'yhat', 'trend', 'yhat_lower', 'yhat_upper', 'Zone']
        if 'yearly' in future_forecast.columns:
            cols_to_show.insert(3, 'yearly')

        table_rename = {
            'ds': 'Date', 'yhat': 'Total Forecast', 'trend': 'Macro Trend (Base)',
            'yearly': 'Yearly Cycle', 'yhat_lower': 'Floor Risk', 'yhat_upper': 'Peak Risk'
        }
        st.dataframe(future_forecast[cols_to_show].rename(columns=table_rename).style.format(precision=0), use_container_width=True)

        # STEP 6: Strategic Metrics
        noise = analysis_df['residual'].dropna()
        safety_buffer = noise.quantile(service_level)
        
        # Last Trend Point
        last_trend = hist_forecast['trend'].iloc[-1]
        total_req = max(0, last_trend + safety_buffer)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Order Requirement", f"{total_req:.0f} units")
            st.write(f"Safety Buffer: **{safety_buffer:.1f}**")
            st.caption("Buffer is calculated on residuals of the 7-day window to mitigate cumulative risk.")
        with c2:
            st.write("**Zone Summary**")
            st.table(future_forecast.groupby('Zone')['yhat'].agg(['count', 'mean']).rename(columns={'count':'Days', 'mean':'Avg Demand'}))

    except Exception as e:
        st.error(f"Error: {e}")
