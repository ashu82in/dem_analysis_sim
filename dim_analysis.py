import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet

st.set_page_config(page_title="3-Year Strategy Horizon", layout="wide")

# --- 1. SIDEBAR ---
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
    if st.button("✨ Generate 3-Year Demo Data"):
        dates = pd.date_range(start="2023-01-01", periods=1095, freq='D')
        x = np.linspace(0, 15, 1095)
        # Up-Down-Up pattern with growth
        pattern = 150 + 100 * np.sin(x/2) + 0.5 * np.arange(1095) 
        mask = np.random.random(1095) > 0.88
        demo_demand = np.where(mask, pattern * 4, 0).astype(int)
        st.session_state['df'] = pd.DataFrame({'ds': dates, 'y': np.clip(demo_demand, 0, None)})
        st.success("3 years of history generated!")

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

    # STEP 1: Aggregation (Rolling Window)
    df['lt_demand'] = df['y'].rolling(window=lead_time).sum().shift(-offset)
    analysis_df = df.dropna().copy()
    if user_type == "Distributor":
        analysis_df = analysis_df[analysis_df['lt_demand'] > 0]

    try:
        # STEP 2: Prophet Modeling (Pivoting Trend)
        prophet_df = analysis_df[['ds', 'lt_demand']].rename(columns={'lt_demand': 'y'})
        m = Prophet(growth='linear', yearly_seasonality=True, weekly_seasonality=False, 
                    daily_seasonality=False, changepoint_prior_scale=0.05)
        m.fit(prophet_df)
        
        # Forecast 3 years
        future = m.make_future_dataframe(periods=1095)
        forecast = m.predict(future)
        
        # Enforce non-negativity
        for col in ['yhat', 'yhat_lower', 'yhat_upper', 'trend', 'yearly']:
            if col in forecast.columns:
                forecast[col] = forecast[col].clip(lower=0)
            
        hist_forecast = forecast[forecast['ds'] <= analysis_df['ds'].max()].copy()
        future_forecast = forecast[forecast['ds'] > analysis_df['ds'].max()].copy()

        # STEP 3: Decomposition
        decomp = seasonal_decompose(analysis_df['lt_demand'], model='additive', period=365 if len(analysis_df) > 730 else 30)
        analysis_df['yearly_cycle'] = decomp.seasonal
        analysis_df['residual'] = decomp.resid

        # --- VISUAL BREAKDOWN ---
        st.subheader("🔍 3-Year Macro Strategy Breakdown")
        fig_macro = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                   subplot_titles=("1. Historical Demand", "2. Piecewise Trend (Path)", 
                                                   "3. Yearly Cycle (Wave)", "4. Uncertainty (Blind Spots)"))

        fig_macro.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="Raw", line=dict(color='#A0AEC0', width=1)), row=1, col=1)
        fig_macro.add_trace(go.Scatter(x=hist_forecast['ds'], y=hist_forecast['trend'], name="Trend", line=dict(color='#3182CE', width=3)), row=2, col=1)
        fig_macro.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['yearly_cycle'], name="Seasonality", line=dict(color='#805AD5', width=2)), row=3, col=1)
        fig_macro.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['residual'], mode='markers', marker=dict(color='#E53E3E', size=3)), row=4, col=1)
        fig_macro.update_layout(height=800, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_macro, use_container_width=True)

        # --- HISTORICAL DATA TABLE ---
        with st.expander("📊 View Historical Component Breakdown (Trend & Seasonality)"):
            hist_table = analysis_df[['ds', 'lt_demand']].copy()
            # Merge with Prophet trend for accuracy
            hist_table = hist_table.merge(hist_forecast[['ds', 'trend', 'yearly']], on='ds')
            hist_table = hist_table.rename(columns={
                'ds': 'Date', 'lt_demand': 'Actual Demand (Window)', 
                'trend': 'Base Trend', 'yearly': 'Annual Seasonality'
            })
            st.dataframe(hist_table.style.format(precision=0), use_container_width=True)

        # STEP 4: 3-Year Forecast
        st.divider()
        st.header("🔮 3-Year Strategic Growth Forecast")
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="History", line=dict(color="#A0AEC0", width=1)))
        fig_f.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], name="Forecast", line=dict(color="#3182CE", width=2)))
        fig_f.add_trace(go.Scatter(
            x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1],
            y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1],
            fill='toself', fillcolor='rgba(49, 130, 206, 0.1)', line=dict(color='rgba(255,255,255,0)'), name="Risk Range"
        ))
        fig_f.update_layout(template="plotly_dark")
        st.plotly_chart(fig_f, use_container_width=True)

        # --- FORECAST DATA TABLE ---
        st.subheader("📅 3-Year Strategic Forecast Breakdown")
        f_table = future_forecast[['ds', 'yhat', 'trend', 'yearly', 'yhat_upper']].copy()
        f_table = f_table.rename(columns={
            'ds': 'Date', 'yhat': 'Total Forecast', 'trend': 'Projected Trend', 
            'yearly': 'Projected Seasonality', 'yhat_upper': 'Peak Risk'
        })
        st.dataframe(f_table.style.format(precision=0), use_container_width=True)

        # STEP 5: Metrics
        noise = analysis_df['residual'].dropna()
        safety_buffer = noise.quantile(service_level)
        total_req = hist_forecast['trend'].iloc[-1] + safety_buffer

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Immediate Order Target", f"{total_req:.0f} units")
            st.write(f"Safety Buffer (Blind Spot Protection): **{safety_buffer:.0f}**")
        with c2:
            st.info("**Strategic Insight:** Use the Historical Table to see how 'Base Trend' and 'Annual Wave' combined to create your past peaks. The Forecast Table uses that same DNA to predict your next 3 years.")

    except Exception as e:
        st.error(f"Computation Error: {e}")
