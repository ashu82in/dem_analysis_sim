import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet

st.set_page_config(page_title="Strategic Demand Factor Model", layout="wide")

# --- 1. SIDEBAR: Parameters & Data Generation ---
with st.sidebar:
    st.header("🎯 Factor Settings")
    user_type = st.radio("Business Model", ["Retailer", "Distributor"])
    
    col_lt, col_off = st.columns(2)
    with col_lt:
        lead_time = st.number_input("Lead Time (Days)", min_value=1, value=7)
    with col_off:
        offset = st.number_input("Offset", min_value=0, value=0)
        
    service_level = st.slider("Service Level (%)", 70.0, 99.9, 95.0) / 100

    st.divider()
    st.write("**Seasonality Significance**")
    threshold = st.slider("Neutral Range (Units +/-)", 0, 50, 10)

    if st.button("✨ Generate 3-Year Historical Data"):
        dates = pd.date_range(start="2023-01-01", periods=1095, freq='D')
        x = np.linspace(0, 15, 1095)
        # Up-Down-Up pattern: Base(200) + Growth + Annual Sine Wave
        pattern = 200 + 120 * np.sin(x/2) + 0.6 * np.arange(1095) 
        mask = np.random.random(1095) > 0.85
        demo_demand = np.where(mask, pattern * 4, 0).astype(int)
        st.session_state['df'] = pd.DataFrame({'ds': dates, 'y': np.clip(demo_demand, 0, None)})
        st.success("3 Years of history generated!")

# --- 2. DATA LOADING ---
uploaded_file = st.file_uploader("Upload Excel/CSV", type=["csv", "xlsx"])
if uploaded_file:
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    data.columns = [c.lower().strip() for c in data.columns]
    data = data.rename(columns={'date': 'ds', 'demand': 'y'})
    st.session_state['df'] = data

# --- 3. CORE FACTOR ANALYSIS ---
if 'df' in st.session_state:
    df = st.session_state['df'].copy()
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Aggregation (Rolling Window)
    df['lt_demand'] = df['y'].rolling(window=lead_time).sum().shift(-offset)
    analysis_df = df.dropna().copy()
    if user_type == "Distributor":
        analysis_df = analysis_df[analysis_df['lt_demand'] > 0]

    try:
        # STEP 1: Prophet Modeling
        m = Prophet(growth='linear', yearly_seasonality=True, weekly_seasonality=False, 
                    daily_seasonality=False, changepoint_prior_scale=0.05)
        m.fit(analysis_df[['ds', 'lt_demand']].rename(columns={'lt_demand': 'y'}))
        
        # Forecast 3 Years
        future = m.make_future_dataframe(periods=1095)
        forecast = m.predict(future)
        forecast[['yhat', 'trend']] = forecast[['yhat', 'trend']].clip(lower=0)

        # STEP 2: Historical Factor Bifurcation
        hist_forecast = forecast[forecast['ds'] <= analysis_df['ds'].max()].copy()
        base_val = hist_forecast['trend'].iloc[0]
        
        hist_factors = analysis_df[['ds', 'lt_demand']].merge(hist_forecast[['ds', 'trend', 'yearly']], on='ds')
        hist_factors['Factor_Base'] = base_val
        hist_factors['Factor_Trend'] = hist_factors['trend'] - base_val
        hist_factors['Factor_Seasonality'] = hist_factors['yearly']
        
        # Seasonality State Logic
        def get_season_state(val, t):
            if val > t: return "Positive (Tailwind)"
            elif val < -t: return "Negative (Headwind)"
            else: return "Neutral (No Effect)"
        hist_factors['Season_Effect'] = hist_factors['Factor_Seasonality'].apply(lambda x: get_season_state(x, threshold))
        
        hist_factors['Baseline'] = hist_factors['Factor_Base'] + hist_factors['Factor_Trend'] + hist_factors['Factor_Seasonality']
        hist_factors['Residual'] = hist_factors['lt_demand'] - hist_factors['Baseline']

        # --- VISUAL 1: MACRO STRATEGY DECOMPOSITION ---
        st.subheader("🔍 Macro Strategic Breakdown")
        fig_macro = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                   subplot_titles=("1. Historical Demand", "2. Pivoting Trend Factor", "3. Stochastic Residuals (The Spikes)"))
        fig_macro.add_trace(go.Scatter(x=hist_factors['ds'], y=hist_factors['lt_demand'], name="Raw", line=dict(color='#A0AEC0', width=1)), row=1, col=1)
        fig_macro.add_trace(go.Scatter(x=hist_factors['ds'], y=hist_factors['trend'], name="Trend", line=dict(color='#3182CE', width=3)), row=2, col=1)
        fig_macro.add_trace(go.Scatter(x=hist_factors['ds'], y=hist_factors['Residual'], mode='markers', marker=dict(color='#E53E3E', size=3)), row=3, col=1)
        fig_macro.update_layout(height=650, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_macro, use_container_width=True)

        # --- VISUAL 2: STACKED FACTOR MODEL ---
        st.subheader("📊 Historical Factor Stacked Area")
        st.write("Base and Trend Growth are stacked; Seasonality Wave is overlaid to show corrections.")
        fig_stack = go.Figure()
        fig_stack.add_trace(go.Scatter(x=hist_factors['ds'], y=[base_val]*len(hist_factors), name="Fixed Base", stackgroup='one', fillcolor='#4A5568', line=dict(width=0)))
        fig_stack.add_trace(go.Scatter(x=hist_factors['ds'], y=hist_factors['Factor_Trend'], name="Trend Growth", stackgroup='one', fillcolor='#3182CE', line=dict(width=0)))
        # Seasonality (Overlaid to show +/-)
        fig_stack.add_trace(go.Scatter(x=hist_factors['ds'], y=hist_factors['Factor_Seasonality'] + hist_factors['trend'], name="Seasonal Adjusted Path", line=dict(color='#805AD5', width=2)))
        fig_stack.add_trace(go.Scatter(x=hist_factors['ds'], y=hist_factors['lt_demand'], name="Actual Demand", line=dict(color='white', width=1, dash='dot')))
        fig_stack.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_stack, use_container_width=True)

        # --- VISUAL 3: RISK & SEASONAL STATES ---
        st.subheader("🎯 Risk Zones & Seasonal Impact")
        res_std = hist_factors['Residual'].std()
        hist_factors['Zone'] = hist_factors['Residual'].apply(lambda x: "Zone 3 (Surge)" if x > res_std else ("Zone 1 (Under)" if x < -res_std else "Zone 2 (Stable)"))
        
        c1, c2 = st.columns(2)
        with c1:
            fig_dist = px.histogram(hist_factors, x="Residual", color="Zone", nbins=50, title="Variability Distribution",
                                   color_discrete_map={"Zone 1 (Under)": "#4FD1C5", "Zone 2 (Stable)": "#63B3ED", "Zone 3 (Surge)": "#F56565"})
            st.plotly_chart(fig_dist, use_container_width=True)
        with c2:
            fig_pie = px.pie(hist_factors, names='Season_Effect', title="Seasonal State Breakdown", color='Season_Effect',
                            color_discrete_map={"Positive (Tailwind)": "#805AD5", "Negative (Headwind)": "#E53E3E", "Neutral (No Effect)": "#A0AEC0"})
            st.plotly_chart(fig_pie, use_container_width=True)

        # --- HISTORICAL AUDIT TABLE ---
        st.subheader("📅 Historical Factor Audit Table")
        st.dataframe(hist_factors[['ds', 'lt_demand', 'Factor_Base', 'Factor_Trend', 'Factor_Seasonality', 'Season_Effect', 'Residual', 'Zone']].rename(columns={'ds': 'Date', 'Factor_Seasonality': 'Seasonality (+/-)'}).style.format(precision=0), use_container_width=True)

        # --- FORECAST SECTION ---
        st.divider()
        st.header("🔮 3-Year Strategic Horizon Forecast")
        future_forecast = forecast[forecast['ds'] > analysis_df['ds'].max()].copy()
        
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="Past", line=dict(color="#A0AEC0", width=1)))
        fig_f.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], name="Forecast", line=dict(color="#3182CE", width=2)))
        fig_f.add_trace(go.Scatter(x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1], 
                                   y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1], 
                                   fill='toself', fillcolor='rgba(49, 130, 206, 0.1)', line=dict(color='rgba(255,255,255,0)'), name="Confidence Range"))
        fig_f.update_layout(template="plotly_dark")
        st.plotly_chart(fig_f, use_container_width=True)

        # Final Summary Table
        st.subheader("📅 3-Year Future Strategy Breakdown")
        st.dataframe(future_forecast[['ds', 'yhat', 'trend', 'yearly', 'yhat_upper']].rename(columns={'ds': 'Date', 'yhat': 'Forecast', 'yearly': 'Seasonality (+/-)'}).style.format(precision=0), use_container_width=True)

    except Exception as e:
        st.error(f"Computation Error: {e}")
