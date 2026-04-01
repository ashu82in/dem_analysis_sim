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
        # Up-Down-Up pattern
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

# --- 3. CORE ANALYSIS ---
if 'df' in st.session_state:
    df = st.session_state['df'].copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')

    # STEP 1: Rolling Window
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
        
        future = m.make_future_dataframe(periods=1095)
        forecast = m.predict(future)
        
        for col in ['yhat', 'yhat_lower', 'yhat_upper', 'trend', 'yearly']:
            if col in forecast.columns:
                forecast[col] = forecast[col].clip(lower=0)
            
        hist_forecast = forecast[forecast['ds'] <= analysis_df['ds'].max()].copy()
        future_forecast = forecast[forecast['ds'] > analysis_df['ds'].max()].copy()

        # STEP 3: Historical Zone & Distribution Logic
        hist_table = analysis_df[['ds', 'lt_demand']].merge(hist_forecast[['ds', 'trend', 'yearly']], on='ds')
        hist_table['Baseline'] = hist_table['trend'] + hist_table['yearly']
        hist_table['Variance_Ratio'] = hist_table['lt_demand'] / hist_table['Baseline']

        def assign_zone(ratio):
            if ratio < 0.9: return "Zone 1 (Stable)"
            elif ratio < 1.1: return "Zone 2 (Normal)"
            else: return "Zone 3 (High Surge)"
        
        hist_table['Zone'] = hist_table['Variance_Ratio'].apply(assign_zone)

        # --- GRAPH 1: MACRO STRATEGY BREAKDOWN ---
        st.subheader("🔍 Macro Strategic Breakdown (The Truth in Layers)")
        fig_macro = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                   subplot_titles=("1. Historical Demand (7-Day Windows)", 
                                                   "2. Pivoting Macro Trend (The Direction)", 
                                                   "3. Planning Uncertainty (The Blind Spots)"))

        fig_macro.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="Raw", line=dict(color='#A0AEC0', width=1)), row=1, col=1)
        fig_macro.add_trace(go.Scatter(x=hist_forecast['ds'], y=hist_forecast['trend'], name="Trend", line=dict(color='#3182CE', width=3)), row=2, col=1)
        fig_macro.add_trace(go.Scatter(x=hist_table['ds'], y=hist_table['lt_demand'] - hist_table['Baseline'], mode='markers', name="Noise", marker=dict(color='#E53E3E', size=3)), row=3, col=1)
        fig_macro.update_layout(height=700, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_macro, use_container_width=True)

        # --- GRAPH 2: RISK DISTRIBUTION ---
        st.subheader("📊 Historical Risk Distribution")
        col_hist, col_pie = st.columns([2, 1])
        with col_hist:
            fig_dist = px.histogram(hist_table, x="Variance_Ratio", color="Zone", 
                                   title="Actual vs. Baseline Variance",
                                   color_discrete_map={"Zone 1 (Stable)": "#4FD1C5", "Zone 2 (Normal)": "#63B3ED", "Zone 3 (High Surge)": "#F56565"})
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_pie:
            fig_pie = px.pie(hist_table, names='Zone', title="Historical Zone Split",
                            color='Zone', color_discrete_map={"Zone 1 (Stable)": "#4FD1C5", "Zone 2 (Normal)": "#63B3ED", "Zone 3 (High Surge)": "#F56565"})
            st.plotly_chart(fig_pie, use_container_width=True)

        # --- HISTORICAL DATA TABLE ---
        st.subheader("📅 Historical Component & Zone Breakdown")
        st.dataframe(hist_table[['ds', 'lt_demand', 'trend', 'yearly', 'Baseline', 'Zone']].rename(columns={
            'ds': 'Date', 'lt_demand': 'Actual Demand', 'trend': 'Base Trend', 'yearly': 'Annual Wave'
        }).style.format(precision=0), use_container_width=True)

        # --- GRAPH 3: 3-YEAR STRATEGIC FORECAST ---
        st.divider()
        st.header("🔮 3-Year Strategic Growth Forecast")
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="Past", line=dict(color="#A0AEC0", width=1)))
        fig_f.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], name="Forecast", line=dict(color="#3182CE", width=2)))
        fig_f.add_trace(go.Scatter(
            x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1],
            y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1],
            fill='toself', fillcolor='rgba(49, 130, 206, 0.1)', line=dict(color='rgba(255,255,255,0)'), name="Risk Range"
        ))
        fig_f.update_layout(template="plotly_dark", title="Multi-Year Horizon Projection")
        st.plotly_chart(fig_f, use_container_width=True)

        # --- FORECAST DATA TABLE ---
        st.subheader("📅 3-Year Forecast Breakdown")
        future_forecast['Baseline'] = future_forecast['trend'] + future_forecast['yearly']
        future_forecast['Zone'] = (future_forecast['yhat'] / future_forecast['Baseline']).apply(assign_zone)
        st.dataframe(future_forecast[['ds', 'yhat', 'trend', 'yearly', 'yhat_upper', 'Zone']].rename(columns={
            'ds': 'Date', 'yhat': 'Expected Demand', 'trend': 'Projected Trend', 
            'yearly': 'Projected Wave', 'yhat_upper': 'Peak Risk Limit'
        }).style.format(precision=0), use_container_width=True)

        # STEP 5: Metrics
        decomp = seasonal_decompose(analysis_df['lt_demand'], model='additive', period=365 if len(analysis_df) > 730 else 30)
        safety_buffer = decomp.resid.dropna().quantile(service_level)
        total_req = hist_forecast['trend'].iloc[-1] + safety_buffer

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Immediate Order Target", f"{total_req:.0f} units")
            st.write(f"Safety Buffer: **{safety_buffer:.0f}**")
        with c2:
            st.info("The charts and tables above provide a full audit trail from past uncertainty to future growth.")

    except Exception as e:
        st.error(f"Computation Error: {e}")
