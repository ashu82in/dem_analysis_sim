import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from io import BytesIO

st.set_page_config(page_title="3-Year Demand Strategy Pro", layout="wide")

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
    # UPDATED: Generates 3 years of data (1095 days)
    if st.button("✨ Generate 3 Years of Demo Data"):
        dates = pd.date_range(start="2023-01-01", periods=1095, freq='D')
        
        # 1. Base Linear Growth (The "Stiff" part)
        growth = np.linspace(50, 250, 1095) 
        
        # 2. Yearly Seasonality (Repeating Sine Wave over 365 days)
        yearly_pattern = 60 * np.sin(2 * np.pi * np.arange(1095) / 365)
        
        # 3. Weekly Noise (Distributor "Lumpiness")
        mask = np.random.random(1095) > 0.88
        demo_demand = np.where(mask, (growth + yearly_pattern) * 4, 0).astype(int)
        
        # Ensure no negatives in raw data
        demo_demand = np.clip(demo_demand, 0, None)
        
        st.session_state['df'] = pd.DataFrame({'ds': dates, 'y': demo_demand})
        st.success("Generated 1,095 days of historical data!")

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
    df['lt_demand'] = df['y'].rolling(window=lead_time).sum().shift(-offset)
    
    analysis_df = df.dropna().copy()
    if user_type == "Distributor":
        analysis_df = analysis_df[analysis_df['lt_demand'] > 0]

    try:
        # STEP 2: Prophet Modeling (Aggressive Stiffness for 3-Year Projection)
        prophet_df = analysis_df[['ds', 'lt_demand']].rename(columns={'lt_demand': 'y'})
        prophet_df['floor'] = 0
        
        m = Prophet(
            growth='linear', 
            yearly_seasonality=True, # Critical for 3-year outlook
            weekly_seasonality=False, 
            daily_seasonality=False,
            changepoint_prior_scale=0.001 # Keep the trend stiff/straight
        )
        m.fit(prophet_df)
        
        # FORECAST: Projecting another 3 years into the future
        future = m.make_future_dataframe(periods=1095)
        future['floor'] = 0
        forecast = m.predict(future)
        
        # Non-negativity Clipping
        for col in ['yhat', 'yhat_lower', 'yhat_upper', 'trend', 'yearly']:
            if col in forecast.columns:
                forecast[col] = forecast[col].clip(lower=0)
            
        hist_forecast = forecast[forecast['ds'] <= analysis_df['ds'].max()]
        future_forecast = forecast[forecast['ds'] > analysis_df['ds'].max()].copy()

        # STEP 3: Visual Decomposition (Macro-Trend focus)
        # Using 30-day period to hide daily noise
        decomp = seasonal_decompose(analysis_df['lt_demand'], model='additive', period=30)
        analysis_df['residual'] = decomp.resid
        
        st.subheader("🔍 Long-Term Macro Decomposition")
        fig_decomp = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                   subplot_titles=("1. Historical Demand (7-Day Windows)", 
                                                   "2. Stiff Macro Trend (Business Path)", 
                                                   "3. Historical Planning Uncertainty"))
        
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="Aggregated", line=dict(color='#A0AEC0', width=1)), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=hist_forecast['ds'], y=hist_forecast['trend'], name="Trend", line=dict(color='#3182CE', width=3)), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['residual'], mode='markers', marker=dict(color='#E53E3E', size=3)), row=3, col=1)
        
        fig_decomp.update_layout(height=600, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_decomp, use_container_width=True)

        # STEP 4: 3-Year Strategic Forecast Plot
        st.divider()
        st.header("🔮 3-Year Strategic Growth Forecast")
        
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="History", line=dict(color="#A0AEC0", width=1)))
        fig_f.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], name="3-Year Forecast", line=dict(color="#3182CE", width=2)))
        
        # Uncertainty Shading (Confidence Interval)
        fig_f.add_trace(go.Scatter(
            x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1],
            y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1],
            fill='toself', fillcolor='rgba(49, 130, 206, 0.1)', line=dict(color='rgba(255,255,255,0)'), name="Uncertainty Range"
        ))
        
        # Year Markers
        for year in range(future_forecast['ds'].dt.year.min(), future_forecast['ds'].dt.year.max() + 1):
            fig_f.add_vline(x=f"{year}-01-01", line_dash="dot", line_color="gray", opacity=0.3)

        fig_f.update_layout(template="plotly_dark", title="Expected Demand vs Planning Buffer Over 3 Years")
        st.plotly_chart(fig_f, use_container_width=True)

        # STEP 5: Yearly Strategic Breakdown
        st.subheader("📊 Annual Inventory Impact Summary")
        future_forecast['Year'] = future_forecast['ds'].dt.year
        yearly_summary = future_forecast.groupby('Year')['yhat'].agg(['mean', 'max', 'sum']).rename(columns={
            'mean': 'Avg LT Demand', 'max': 'Peak Single Window', 'sum': 'Total Annual Throughput'
        })
        st.table(yearly_summary.style.format(precision=0))

        # STEP 6: Immediate Metrics
        noise = analysis_df['residual'].dropna()
        safety_buffer = noise.quantile(service_level)
        last_trend = hist_forecast['trend'].iloc[-1]
        total_req = last_trend + safety_buffer

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Immediate Stock Target", f"{total_req:.0f} units")
            st.write(f"Safety Buffer: **{safety_buffer:.0f}**")
        with c2:
            st.info(f"""
            **Presentation Insight:**
            Over the next 3 years, your 'Peak Demand Windows' are forecasted to grow from 
            **{yearly_summary['Peak Demand Window'].iloc[0]:.0f}** to **{yearly_summary['Peak Demand Window'].iloc[-1]:.0f}**. 
            
            The widening blue shade shows that while we can track the trend, the 'leaks' in the tank become harder to predict the further we look.
            """)

    except Exception as e:
        st.error(f"Computation Error: {e}")
