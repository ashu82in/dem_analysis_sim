import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from scipy import stats

st.set_page_config(page_title="Strategic Demand Factor Model Pro", layout="wide")

# --- 1. SIDEBAR: THE SIMULATION FACTORY ---
with st.sidebar:
    st.header("🧪 Data Generation (The DNA)")
    sim_days = 1095 
    
    st.subheader("Factor Magnitudes")
    base_vol = st.number_input("Starting Base Volume", value=500)
    growth_strength = st.slider("Trend Growth Multiplier", 0.1, 10.0, 4.0)
    season_swing = st.slider("Seasonal Wave Amplitude", 0, 2000, 800)
    
    # Increase this for the "spread" you were looking for
    noise_sigma = st.slider("Stochastic Noise (Chaos Level)", 10, 1000, 350)
    
    st.divider()
    st.subheader("Supply Chain Logic")
    lead_time = st.number_input("Lead Time Window (Days)", min_value=1, value=7)
    service_level = st.slider("Service Level (%)", 70.0, 99.9, 97.7) / 100
    
    st.divider()
    neutral_threshold = st.slider("Neutral Significance Threshold", 0, 200, 50)

    if st.button("🚀 Run 3-Year Stress Test Simulation"):
        t = np.arange(sim_days)
        trend = base_vol + (growth_strength * t) + (400 * np.cos(t/200))
        seasonality = season_swing * np.sin(2 * np.pi * t / 365)
        noise = np.random.normal(0, noise_sigma, sim_days)
        
        y_cont = np.clip(trend + seasonality + noise, 0, None)
        mask = np.random.random(sim_days) > 0.85
        lumpy_y = np.where(mask, y_cont * 6, 0)
        
        st.session_state['df'] = pd.DataFrame({
            'ds': pd.date_range("2023-01-01", periods=sim_days), 
            'y': lumpy_y
        })
        st.success("High-Variability DNA Generated!")

# --- 2. CORE FACTOR ANALYSIS ---
if 'df' in st.session_state:
    df = st.session_state['df'].copy()
    df['lt_demand'] = df['y'].rolling(window=lead_time).sum().fillna(0)
    analysis_df = df[df['lt_demand'] > 0].copy()

    try:
        # STEP 1: Factor Extraction
        m = Prophet(growth='linear', yearly_seasonality=True, weekly_seasonality=False, changepoint_prior_scale=0.05)
        m.fit(analysis_df[['ds', 'lt_demand']].rename(columns={'lt_demand': 'y'}))
        
        future = m.make_future_dataframe(periods=1095)
        forecast = m.predict(future)

        # STEP 2: Logic & Anchors
        hist_f = forecast[forecast['ds'] <= analysis_df['ds'].max()].copy()
        curr_trend = hist_f['trend'].iloc[-1]
        peak_trend, trough_trend = hist_f['trend'].max(), hist_f['trend'].min()
        base_anchor = hist_f['trend'].iloc[0]
        
        hist_table = analysis_df.merge(hist_f[['ds', 'trend', 'yearly']], on='ds')
        
        # Consistent Column Naming to prevent "Not in Index" Errors
        hist_table['Factor_Base'] = base_anchor
        hist_table['Factor_Trend'] = hist_table['trend'] - base_anchor
        hist_table['Factor_Seasonality'] = hist_table['yearly']
        hist_table['Residual_Noise'] = hist_table['lt_demand'] - (hist_table['trend'] + hist_table['yearly'])
        
        def get_season_state(val, t):
            if val > t: return "Positive (Tailwind)"
            elif val < -t: return "Negative (Headwind)"
            else: return "Neutral (Core)"
        hist_table['Season_State'] = hist_table['Factor_Seasonality'].apply(lambda x: get_season_state(x, neutral_threshold))
        
        res_pool = hist_table['Residual_Noise'].dropna()

        # --- VISUAL 1: MACRO BREAKDOWN ---
        st.subheader("🔍 Macro Strategic Breakdown (History)")
        fig_macro = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                   subplot_titles=("Demand", "Pivoting Trend", "Stationary Residuals"))
        fig_macro.add_trace(go.Scatter(x=hist_table['ds'], y=hist_table['lt_demand'], name="Actual"), row=1, col=1)
        fig_macro.add_trace(go.Scatter(x=hist_table['ds'], y=hist_table['trend'], line=dict(color='#3182CE', width=3)), row=2, col=1)
        fig_macro.add_trace(go.Scatter(x=hist_table['ds'], y=hist_table['Residual_Noise'], mode='markers', marker=dict(color='#E53E3E', size=3)), row=3, col=1)
        fig_macro.update_layout(height=650, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_macro, use_container_width=True)

        # --- VISUAL 2: THE TRIPLE SPECTRUM ---
        st.subheader("📊 Strategic Spectrum: Trough vs. Now vs. Peak")
        fig_spec = go.Figure()
        fig_spec.add_trace(go.Histogram(x=res_pool + trough_trend, name="Historical Trough", marker_color='#4FD1C5', opacity=0.4))
        fig_spec.add_trace(go.Histogram(x=res_pool + curr_trend, name="Current Scale (Now)", marker_color='#63B3ED', opacity=0.7))
        fig_spec.add_trace(go.Histogram(x=res_pool + peak_trend, name="Historical Peak", marker_color='#F56565', opacity=0.4))
        fig_spec.update_layout(barmode='overlay', template="plotly_dark")
        st.plotly_chart(fig_spec, use_container_width=True)

        # --- VISUAL 3: NORMALITY AUDIT ---
        st.divider()
        st.header("⚖️ Normality Audit")
        _, p_val = stats.normaltest(res_pool)
        safety_buffer = np.percentile(res_pool, service_level * 100)

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Normality p-value", f"{p_val:.4f}")
        with c2: st.metric("Trend Scale (Now)", f"{curr_trend:.0f}")
        with c3: st.metric("Safety Buffer", f"{safety_buffer:.0f}")

        col_bell, col_qq = st.columns(2)
        with col_bell:
            st.subheader("Current Strategic Distribution")
            fig_bell = px.histogram(hist_table, x=hist_table['Residual_Noise'] + curr_trend, nbins=60, color_discrete_sequence=['#63B3ED'])
            fig_bell.add_vline(x=curr_trend, line_dash="dash", line_color="white")
            fig_bell.add_vline(x=curr_trend + safety_buffer, line_color="#F56565", line_width=3)
            st.plotly_chart(fig_bell, use_container_width=True)
        with col_qq:
            st.subheader("Q-Q Probability Plot")
            res_sorted = np.sort(res_pool)
            norm_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(res_sorted)))
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=norm_q, y=res_sorted, mode='markers', marker=dict(color='#F56565')))
            fig_qq.add_trace(go.Scatter(x=norm_q, y=norm_q * np.std(res_sorted) + np.mean(res_sorted), mode='lines', line=dict(color='white', dash='dash')))
            fig_qq.update_layout(template="plotly_dark")
            st.plotly_chart(fig_qq, use_container_width=True)

        # --- VISUAL 4: AUDIT TABLE (FIXED) ---
        st.subheader("📅 Strategic Factor Audit Table")
        # Ensure only columns created in the hist_table are listed here
        audit_cols = ['ds', 'lt_demand', 'Factor_Base', 'Factor_Trend', 'Factor_Seasonality', 'Season_State', 'Residual_Noise']
        st.dataframe(hist_table[audit_cols].rename(columns={
            'ds': 'Date', 'lt_demand': 'Actual', 'Factor_Seasonality': 'Seasonality (+/-)', 'Residual_Noise': 'Residual'
        }).style.format(precision=0), use_container_width=True)

    except Exception as e:
        st.error(f"Computation Error: {e}")
