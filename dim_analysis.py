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
    user_type = st.radio("Business Model", ["Retailer", "Distributor"], 
                         help="Distributors: Zeros are removed AFTER calculating lead time demand.")
    
    col_lt, col_off = st.columns(2)
    with col_lt:
        lead_time = st.number_input("Lead Time", min_value=1, value=7)
    with col_off:
        offset = st.number_input("Offset", min_value=0, value=0)
        
    service_level = st.slider("Service Level (%)", 70.0, 99.9, 95.0) / 100

    st.divider()
    if st.button("✨ Generate Demo Data"):
        dates = pd.date_range(start="2025-01-01", periods=365, freq='D')
        # Trend + Seasonality + Random Noise
        base = np.linspace(40, 100, 365) + (15 * np.sin(2 * np.pi * np.arange(365) / 7))
        # Distributor pattern (Lumpy demand)
        mask = np.random.random(365) > 0.82
        demo_demand = np.where(mask, base * 4, 0).astype(int)
        # Year-end Spike (The "Blind Spot")
        demo_demand[-20:] += np.random.randint(60, 120, 20)
        st.session_state['df'] = pd.DataFrame({'date': dates, 'demand': demo_demand})

# --- 2. DATA LOADING ---
uploaded_file = st.file_uploader("Upload Excel/CSV", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        st.session_state['df'] = pd.read_csv(uploaded_file)
    else:
        st.session_state['df'] = pd.read_excel(uploaded_file)

# --- 3. CORE LOGIC & VISUALS ---
if 'df' in st.session_state:
    df = st.session_state['df'].copy()
    df.columns = [c.lower().strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # STEP 1: Rolling Sum (Demand During Lead Time) - Capturing the Volume First
    df['lt_demand'] = df['demand'].rolling(window=lead_time).sum().shift(-offset)
    
    # STEP 2: Filter Zeros for Distributor (Removing Idle Windows)
    analysis_df = df.dropna().copy()
    if user_type == "Distributor":
        analysis_df = analysis_df[analysis_df['lt_demand'] > 0]
        st.caption("✅ Distributor Mode: Idle periods (zero lead-time demand) filtered out.")

    # STEP 3: Decomposition (Trend, Seasonality, Residuals)
    try:
        # Decomposing the 'Active' Lead Time Demand
        decomp = seasonal_decompose(analysis_df['lt_demand'], model='additive', period=7)
        analysis_df['trend'] = decomp.trend
        analysis_df['seasonal'] = decomp.seasonal
        analysis_df['residual'] = decomp.resid
        
        # --- VISUAL A: Triple-Stacked Decomposition ---
        st.subheader("🔍 Time Series Decomposition")
        fig_decomp = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                   subplot_titles=("Trend (Growth)", "Seasonality (Cycle)", "Residuals (Unpredictable)"))

        fig_decomp.add_trace(go.Scatter(x=analysis_df['date'], y=analysis_df['trend'], name="Trend", line=dict(color='#3182CE')), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=analysis_df['date'], y=analysis_df['seasonal'], name="Seasonality", line=dict(color='#805AD5')), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(x=analysis_df['date'], y=analysis_df['residual'], name="Residuals", mode='markers', marker=dict(color='#CBD5E0', size=4)), row=3, col=1)
        
        fig_decomp.update_layout(height=500, showlegend=False, margin=dict(t=30, b=10))
        st.plotly_chart(fig_decomp, use_container_width=True)

        # STEP 4: Smart Safety Stock Calculation (From Residuals)
        noise_series = analysis_df['residual'].dropna()
        safety_buffer = noise_series.quantile(service_level)
        
        # Most recent predictable baseline
        current_baseline = analysis_df['trend'].iloc[-1] + analysis_df['seasonal'].iloc[-1]
        total_req = current_baseline + safety_buffer

        # --- VISUAL B: Results ---
        st.divider()
        c1, c2 = st.columns([2, 1])

        with c1:
            st.subheader("📊 Lead Time Uncertainty (Residuals)")
            fig_hist = px.histogram(noise_series, x=noise_series, nbins=35, 
                                     title="Histogram of the 'Blind Spots' (Residual Errors)")
            fig_hist.add_vline(x=safety_buffer, line_dash="dash", line_color="#E53E3E")
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            st.subheader("💡 Strategic Recommendation")
            st.metric("Total Order Quantity", f"{total_req:.0f} units")
            st.write(f"**Baseline Demand:** {current_baseline:.0f}")
            st.write(f"**Safety Buffer:** {safety_buffer:.0f}")
            st.success(f"Requirement accounts for a {service_level*100:.1f}% Service Level.")
            
            # Export
            output = BytesIO()
            analysis_df.to_excel(output, index=False)
            st.download_button("📥 Download Analysis", output.getvalue(), "demand_report.xlsx")

    except Exception as e:
        st.error(f"Not enough data to decompose: {e}")
else:
    st.info("👋 Upload your demand file or use the demo generator in the sidebar to visualize your business health.")
