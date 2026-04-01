import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from io import BytesIO

st.set_page_config(page_title="Demand Intelligence Pro", layout="wide")

# --- 1. SIDEBAR: Strategy & Parameters ---
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
        base = np.linspace(40, 100, 365) + (15 * np.sin(2 * np.pi * np.arange(365) / 7))
        mask = np.random.random(365) > 0.82
        demo_demand = np.where(mask, base * 4, 0).astype(int)
        demo_demand[-20:] += np.random.randint(60, 120, 20)
        st.session_state['df'] = pd.DataFrame({'date': dates, 'demand': demo_demand})

# --- 2. DATA LOADING ---
uploaded_file = st.file_uploader("Upload Excel/CSV", type=["csv", "xlsx"])
if uploaded_file:
    st.session_state['df'] = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

# --- 3. CORE LOGIC ---
if 'df' in st.session_state:
    df = st.session_state['df'].copy()
    df.columns = [c.lower().strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # STEP 1: Rolling Sum (Lead Time Demand)
    df['lt_demand'] = df['demand'].rolling(window=lead_time).sum().shift(-offset)
    
    # STEP 2: Filter Zeros for Distributor
    analysis_df = df.dropna().copy()
    if user_type == "Distributor":
        analysis_df = analysis_df[analysis_df['lt_demand'] > 0]

    # STEP 3: Decomposition
    try:
        decomp = seasonal_decompose(analysis_df['lt_demand'], model='additive', period=7)
        analysis_df['trend'] = decomp.trend
        analysis_df['seasonal'] = decomp.seasonal
        analysis_df['residual'] = decomp.resid
        
        # --- VISUAL A: 4-Row Decomposition Breakout ---
        st.subheader("🔍 Demand Decomposition & Raw Data")
        
        # We now use 4 rows to include the Raw Data at the top
        fig_decomp = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.07,
            subplot_titles=("1. Raw Lead Time Demand (Aggregated)", "2. Trend (Growth)", "3. Seasonality (Cycle)", "4. Residuals (Unpredictable Noise)")
        )

        # Row 1: Raw Data
        fig_decomp.add_trace(go.Scatter(x=analysis_df['date'], y=analysis_df['lt_demand'], name="Raw", line=dict(color='#A0AEC0')), row=1, col=1)
        # Row 2: Trend
        fig_decomp.add_trace(go.Scatter(x=analysis_df['date'], y=analysis_df['trend'], name="Trend", line=dict(color='#3182CE')), row=2, col=1)
        # Row 3: Seasonality
        fig_decomp.add_trace(go.Scatter(x=analysis_df['date'], y=analysis_df['seasonal'], name="Seasonality", line=dict(color='#805AD5')), row=3, col=1)
        # Row 4: Residuals
        fig_decomp.add_trace(go.Scatter(x=analysis_df['date'], y=analysis_df['residual'], name="Residuals", mode='markers', marker=dict(color='#E53E3E', size=4)), row=4, col=1)
        
        fig_decomp.update_layout(height=800, showlegend=False, template="plotly_dark")
        st.plotly_chart(fig_decomp, use_container_width=True)

        # STEP 4: Safety Stock Calculation
        noise_series = analysis_df['residual'].dropna()
        safety_buffer = noise_series.quantile(service_level)
        current_baseline = analysis_df['trend'].iloc[-1] + analysis_df['seasonal'].iloc[-1]
        total_req = current_baseline + safety_buffer

        # --- VISUAL B: Results ---
        st.divider()
        c1, c2 = st.columns([2, 1])

        with c1:
            st.subheader("📊 Lead Time Uncertainty (Residuals)")
            fig_hist = px.histogram(noise_series, x=noise_series, nbins=35, title="Safety Stock focused only on 'Noise'")
            fig_hist.add_vline(x=safety_buffer, line_dash="dash", line_color="#E53E3E")
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            st.subheader("💡 Strategic Recommendation")
            st.metric("Total Order Quantity", f"{total_req:.0f} units")
            st.write(f"**Baseline Demand:** {current_baseline:.0f}")
            st.write(f"**Safety Buffer:** {safety_buffer:.0f}")
            
            output = BytesIO()
            analysis_df.to_excel(output, index=False)
            st.download_button("📥 Download Analysis", output.getvalue(), "demand_report.xlsx")

    except Exception as e:
        st.error(f"Analysis failed: {e}")
else:
    st.info("👋 Upload data or click 'Generate Demo Data' to begin.")
