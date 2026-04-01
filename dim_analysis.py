import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from io import BytesIO

st.set_page_config(page_title="Demand Analyzer Pro", layout="wide")

# --- 1. Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    user_type = st.radio("Entity Type", ["Retailer", "Distributor"], 
                         help="Distributors often have 'lumpy' demand with many zero-days.")
    
    lead_time = st.number_input("Lead Time (Periods)", min_value=1, value=7)
    offset = st.number_input("Offset Window", min_value=0, value=0)
    service_level = st.slider("Target Service Level (%)", 70.0, 99.9, 95.0) / 100

    st.divider()
    st.write("### Data Source")
    uploaded_file = st.file_uploader("Upload Demand XLSX/CSV", type=["csv", "xlsx"])
    
    # --- INTERNAL DATA GENERATOR ---
    if st.button("✨ Use Demo Sample Data"):
        dates = pd.date_range(start="2025-01-01", periods=365, freq='D')
        # Create a trend + seasonal base
        base = np.linspace(10, 30, 365) + (10 * np.sin(2 * np.pi * np.arange(365) / 7))
        # Make it "Distributor style" (80% zeros, 20% big spikes)
        mask = np.random.random(365) > 0.80
        demo_demand = np.where(mask, base * 5, 0).astype(int)
        # Add a year-end "Blind Spot" spike
        demo_demand[-15:] += np.random.randint(50, 100, 15)
        
        st.session_state['df'] = pd.DataFrame({'date': dates, 'demand': demo_demand})
        st.success("Demo data loaded!")

# --- 2. Data Loading Logic ---
df = None
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.session_state['df'] = df
elif 'df' in st.session_state:
    df = st.session_state['df']

# --- 3. Main Analysis ---
if df is not None:
    df.columns = [c.lower().strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # STEP A: Create Lead Time Demand (Rolling Sum)
    # We sum FIRST to catch the total volume in the window
    df['lt_demand'] = df['demand'].rolling(window=lead_time).sum().shift(-offset)
    
    # STEP B: Remove Zeros for Distributors
    analysis_df = df.dropna().copy()
    if user_type == "Distributor":
        analysis_df = analysis_df[analysis_df['lt_demand'] > 0]

    # STEP C: Seasonal Decomposition
    st.subheader("📈 Trend & Seasonality Analysis")
    try:
        # Use a 7-day period for weekly seasonality
        res = seasonal_decompose(analysis_df['lt_demand'], model='additive', period=7)
        analysis_df['trend'] = res.trend
        
        fig_trend = px.line(analysis_df, x='date', y=['lt_demand', 'trend'], 
                             title="Actual Lead Time Demand vs. Growth Trend",
                             color_discrete_map={"lt_demand": "#CBD5E0", "trend": "#3182CE"})
        st.plotly_chart(fig_trend, use_container_width=True)
    except:
        st.warning("Not enough data points for seasonal decomposition.")

    # STEP D: Histogram & Service Level
    st.divider()
    col_left, col_right = st.columns([2, 1])

    target_qty = analysis_df['lt_demand'].quantile(service_level)

    with col_left:
        st.subheader("📊 Lead Time Demand Distribution")
        fig_hist = px.histogram(analysis_df, x="lt_demand", nbins=30, 
                                 title="How often do we see these demand levels?")
        fig_hist.add_vline(x=target_qty, line_dash="dash", line_color="#E53E3E")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.subheader("💡 Inventory Strategy")
        st.metric("Required Stock Level", f"{target_qty:.0f} units")
        st.write(f"To meet a **{service_level*100:.1f}%** service level, you must hold enough inventory to cover the red line.")
        
        # Download button for the processed data
        output = BytesIO()
        analysis_df.to_excel(output, index=False)
        st.download_button(label="📥 Download Processed Analysis", 
                           data=output.getvalue(), 
                           file_name="demand_analysis.xlsx")

else:
    st.info("👋 Welcome! Please upload your demand file or click 'Use Demo Sample Data' in the sidebar to begin.")
