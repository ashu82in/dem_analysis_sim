import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet

st.set_page_config(page_title="3-Year Strategic Growth", layout="wide")

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

    # STEP 1: Aggregation
    df['lt_demand'] = df['y'].rolling(window=lead_time).sum().shift(-offset)
    analysis_df = df.dropna().copy()
    if user_type == "Distributor":
        analysis_df = analysis_df[analysis_df['lt_demand'] > 0]

    try:
        # STEP 2: Prophet Modeling
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

        # STEP 3: Historical Zone Analysis
        hist_table = analysis_df[['ds', 'lt_demand']].merge(hist_forecast[['ds', 'trend', 'yearly']], on='ds')
        hist_table['Baseline'] = hist_table['trend'] + hist_table['yearly']
        hist_table['Variance_Ratio'] = hist_table['lt_demand'] / hist_table['Baseline']

        # Define Zone Logic
        def assign_zone(ratio):
            if ratio < 0.9: return "Zone 1 (Stable)"
            elif ratio < 1.1: return "Zone 2 (Normal)"
            else: return "Zone 3 (High Surge)"
        
        hist_table['Zone'] = hist_table['Variance_Ratio'].apply(assign_zone)

        # --- VISUAL 1: Historical Distribution ---
        st.subheader("📊 Historical Demand Distribution (Relative to Plan)")
        col_dist, col_pie = st.columns([2, 1])
        
        with col_dist:
            fig_dist = px.histogram(hist_table, x="Variance_Ratio", color="Zone", 
                                   title="How often did we deviate from the Trend?",
                                   color_discrete_map={"Zone 1 (Stable)": "#4FD1C5", "Zone 2 (Normal)": "#63B3ED", "Zone 3 (High Surge)": "#F56565"},
                                   labels={'Variance_Ratio': 'Actual Demand / Baseline Ratio'})
            fig_dist.add_vline(x=1.0, line_dash="dash", line_color="white", annotation_text="The Plan (Baseline)")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col_pie:
            fig_pie = px.pie(hist_table, names='Zone', title="Time Spent in Each Zone",
                            color='Zone', color_discrete_map={"Zone 1 (Stable)": "#4FD1C5", "Zone 2 (Normal)": "#63B3ED", "Zone 3 (High Surge)": "#F56565"})
            st.plotly_chart(fig_pie, use_container_width=True)

        # --- HISTORICAL BREAKDOWN TABLE ---
        st.subheader("📅 Historical Component & Zone Breakdown")
        hist_display = hist_table.rename(columns={
            'ds': 'Date', 'lt_demand': 'Actual Demand', 'trend': 'Base Trend', 
            'yearly': 'Annual Wave', 'Baseline': 'Expected Baseline'
        })
        st.dataframe(hist_display[['Date', 'Actual Demand', 'Base Trend', 'Annual Wave', 'Expected Baseline', 'Zone']].style.format(precision=0), use_container_width=True)

        # STEP 4: 3-Year Strategic Forecast
        st.divider()
        st.header("🔮 3-Year Strategic Growth Forecast")
        
        # Apply Zone labels to forecast
        future_forecast['Baseline'] = future_forecast['trend'] + future_forecast['yearly']
        # For forecast, we use the 'yhat' (mean prediction) to categorize days
        future_forecast['Zone'] = (future_forecast['yhat'] / future_forecast['Baseline']).apply(assign_zone)

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=analysis_df['ds'], y=analysis_df['lt_demand'], name="Past", line=dict(color="#A0AEC0", width=1)))
        fig_f.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], name="Forecast", line=dict(color="#3182CE", width=2)))
        fig_f.add_trace(go.Scatter(
            x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1],
            y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1],
            fill='toself', fillcolor='rgba(49, 130, 206, 0.1)', line=dict(color='rgba(255,255,255,0)'), name="Risk Range"
        ))
        fig_f.update_layout(template="plotly_dark")
        st.plotly_chart(fig_f, use_container_width=True)

        # --- FORECAST TABLE ---
        st.subheader("📅 3-Year Forecast & Risk Zones")
        f_display = future_forecast[['ds', 'yhat', 'trend', 'yearly', 'yhat_upper', 'Zone']].rename(columns={
            'ds': 'Date', 'yhat': 'Expected Demand', 'trend': 'Projected Trend', 
            'yearly': 'Projected Wave', 'yhat_upper': 'Peak Risk Limit'
        })
        st.dataframe(f_display.style.format(precision=0), use_container_width=True)

        # STEP 5: Strategic Metrics
        decomp = seasonal_decompose(analysis_df['lt_demand'], model='additive', period=365 if len(analysis_df) > 730 else 30)
        noise = decomp.resid.dropna()
        safety_buffer = noise.quantile(service_level)
        total_req = hist_forecast['trend'].iloc[-1] + safety_buffer

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Immediate Order Target", f"{total_req:.0f} units")
            st.write(f"Safety Buffer (Blind Spot Protection): **{safety_buffer:.0f}**")
        with c2:
            st.write("**Zone Risk Profile**")
            st.write("Zone 3 (High Surge) represents the 'Blind Spots' where demand exceeded the plan by more than 10%. Your Safety Buffer is built to survive these days.")

    except Exception as e:
        st.error(f"Computation Error: {e}")
