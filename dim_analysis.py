import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from scipy import stats

st.set_page_config(page_title="Strategic Demand Factor Model Pro", layout="wide")

# --- 1. SIDEBAR: Control Center ---
with st.sidebar:
    st.header("🎯 Strategy Settings")
    user_type = st.radio("Business Model", ["Retailer", "Distributor"])
    
    col_lt, col_off = st.columns(2)
    with col_lt:
        lead_time = st.number_input("Lead Time Window (Days)", min_value=1, value=7)
    with col_off:
        offset = st.number_input("Offset", min_value=0, value=0)
        
    service_level = st.slider("Service Level (%)", 70.0, 99.9, 95.0) / 100

    st.divider()
    st.write("**Seasonality Significance**")
    threshold = st.slider("Neutral Significance Threshold (Units)", 0, 100, 20)

    if st.button("✨ Generate 3-Year Historical Data"):
        dates = pd.date_range(start="2023-01-01", periods=1095, freq='D')
        x = np.linspace(0, 15, 1095)
        # Simulation: Base + Up-Down-Up Growth + Annual Wave
        pattern = 250 + 150 * np.sin(x/2) + 0.8 * np.arange(1095) 
        mask = np.random.random(1095) > 0.85
        demo_demand = np.where(mask, pattern * 4, 0).astype(int)
        st.session_state['df'] = pd.DataFrame({'ds': dates, 'y': np.clip(demo_demand, 0, None)})
        st.success("3 Years of data generated!")

# --- 2. DATA PROCESSING ---
if 'df' in st.session_state:
    df = st.session_state['df'].copy()
    df['ds'] = pd.to_datetime(df['ds'])
    
    # 7-Day Rolling Window (Cumulative Lead Time Risk)
    df['lt_demand'] = df['y'].rolling(window=lead_time).sum().shift(-offset)
    analysis_df = df.dropna().copy()
    if user_type == "Distributor":
        analysis_df = analysis_df[analysis_df['lt_demand'] > 0]

    try:
        # STEP 1: Prophet Factor Extraction
        m = Prophet(growth='linear', yearly_seasonality=True, weekly_seasonality=False, 
                    daily_seasonality=False, changepoint_prior_scale=0.05)
        m.fit(analysis_df[['ds', 'lt_demand']].rename(columns={'lt_demand': 'y'}))
        
        future = m.make_future_dataframe(periods=1095)
        forecast = m.predict(future)
        forecast[['yhat', 'trend']] = forecast[['yhat', 'trend']].clip(lower=0)

        # STEP 2: Historical Factor Bifurcation
        hist_f = forecast[forecast['ds'] <= analysis_df['ds'].max()].copy()
        base_anchor = hist_f['trend'].iloc[0] 
        curr_trend_scale = hist_f['trend'].iloc[-1]
        
        hist_table = analysis_df[['ds', 'lt_demand']].merge(hist_f[['ds', 'trend', 'yearly']], on='ds')
        
        # Factors
        hist_table['Factor_Base'] = base_anchor
        hist_table['Factor_Trend'] = hist_table['trend'] - base_anchor
        hist_table['Factor_Seasonality'] = hist_table['yearly']
        
        # Residual Calculation (The Stationary Noise)
        hist_table['Planned_Baseline'] = hist_table['trend'] + hist_table['Factor_Seasonality']
        hist_table['Residual_Noise'] = hist_table['lt_demand'] - hist_table['Planned_Baseline']
        
        # Strategic Bell Curve Reconstruction
        hist_table['Strategic_Dist_Units'] = hist_table['Residual_Noise'] + curr_trend_scale

        # --- VISUAL 1: MACRO STRATEGY BREAKDOWN ---
        st.subheader("🔍 Macro Strategic Breakdown (History)")
        fig_macro = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                   subplot_titles=("1. Historical Demand (Rolling Window)", 
                                                   "2. Pivoting Macro Trend (The Direction)", 
                                                   "3. Stationary Residuals (The Pure Noise)"))
        fig_macro.add_trace(go.Scatter(x=hist_table['ds'], y=hist_table['lt_demand'], name="Raw", line=dict(color='#A0AEC0', width=1)), row=1, col=1)
        fig_macro.add_trace(go.Scatter(x=hist_table['ds'], y=hist_table['trend'], name="Trend", line=dict(color='#3182CE', width=3)), row=2, col=1)
        fig_macro.add_trace(go.Scatter(x=hist_table['ds'], y=hist_table['Residual_Noise'], mode='markers', marker=dict(color='#E53E3E', size=3)), row=3, col=1)
        fig_macro.update_layout(height=650, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_macro, use_container_width=True)

        # --- VISUAL 2: STACKED FACTOR MODEL ---
        st.subheader("📊 Historical Factor Components")
        fig_stack = go.Figure()
        fig_stack.add_trace(go.Scatter(x=hist_table['ds'], y=[base_anchor]*len(hist_table), name="Base", stackgroup='one', fillcolor='#4A5568', line=dict(width=0)))
        fig_stack.add_trace(go.Scatter(x=hist_table['ds'], y=hist_table['Factor_Trend'], name="Trend Growth", stackgroup='one', fillcolor='#3182CE', line=dict(width=0)))
        fig_stack.add_trace(go.Scatter(x=hist_table['ds'], y=hist_table['Factor_Seasonality'] + hist_table['trend'], name="Seasonal Adjusted Path", line=dict(color='#805AD5', width=2)))
        fig_stack.add_trace(go.Scatter(x=hist_table['ds'], y=hist_table['lt_demand'], name="Actual Demand", line=dict(color='white', width=1, dash='dot')))
        fig_stack.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_stack, use_container_width=True)

        # --- VISUAL 3: NORMALITY AUDIT & BELL CURVE ---
        st.divider()
        st.header("⚖️ Normality Audit: Testing the 'Pure Noise'")
        res_data = hist_table['Residual_Noise'].dropna()
        stat, p_val = stats.normaltest(res_data)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Normality Test (p-value)", f"{p_val:.4f}")
            st.write("Result: " + ("✅ Normally Distributed" if p_val > 0.05 else "❌ Not Normal (Skewed)"))
        with c2:
            st.metric("Strategic Trend Scale", f"{curr_trend_scale:.0f} units")
        with c3:
            safety_buffer = hist_table['Residual_Noise'].quantile(service_level)
            st.metric("Safety Buffer Required", f"{safety_buffer:.0f} units")

        col_bell, col_qq = st.columns(2)
        with col_bell:
            st.subheader("Strategic Demand Distribution")
            fig_bell = px.histogram(hist_table, x="Strategic_Dist_Units", nbins=60, color_discrete_sequence=['#63B3ED'])
            fig_bell.add_vline(x=curr_trend_scale, line_dash="dash", line_color="white", annotation_text="Trend Center")
            fig_bell.add_vline(x=curr_trend_scale + safety_buffer, line_color="#F56565", line_width=3, annotation_text="Stock Target")
            st.plotly_chart(fig_bell, use_container_width=True)
        with col_qq:
            st.subheader("Q-Q Probability Plot")
            res_sorted = np.sort(res_data)
            norm_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(res_sorted)))
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=norm_q, y=res_sorted, mode='markers', marker=dict(color='#F56565')))
            fig_qq.add_trace(go.Scatter(x=norm_q, y=norm_q * np.std(res_sorted) + np.mean(res_sorted), mode='lines', line=dict(color='white', dash='dash')))
            fig_qq.update_layout(template="plotly_dark", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Residuals")
            st.plotly_chart(fig_qq, use_container_width=True)

        # --- VISUAL 4: 3-YEAR FORECAST ---
        st.divider()
        st.header("🔮 3-Year Strategic Horizon Forecast")
        future_f = forecast[forecast['ds'] > analysis_df['ds'].max()].copy()
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=hist_table['ds'], y=hist_table['lt_demand'], name="Past", line=dict(color="#A0AEC0", width=1)))
        fig_f.add_trace(go.Scatter(x=future_f['ds'], y=future_f['yhat'], name="Forecast", line=dict(color="#3182CE", width=2)))
        fig_f.add_trace(go.Scatter(x=future_f['ds'].tolist() + future_f['ds'].tolist()[::-1], 
                                   y=future_f['yhat_upper'].tolist() + future_f['yhat_lower'].tolist()[::-1], 
                                   fill='toself', fillcolor='rgba(49, 130, 206, 0.1)', line=dict(color='rgba(255,255,255,0)'), name="Risk Corridor"))
        fig_f.update_layout(template="plotly_dark")
        st.plotly_chart(fig_f, use_container_width=True)

        # --- FINAL AUDIT TABLE ---
        st.subheader("📅 Historical Factor Audit Table")
        st.dataframe(hist_table[['ds', 'lt_demand', 'Factor_Base', 'Factor_Trend', 'Factor_Seasonality', 'Residual_Noise']].rename(columns={
            'ds': 'Date', 'Factor_Seasonality': 'Seasonality (+/-)', 'Residual_Noise': 'Residual (Noise)'
        }).style.format(precision=0), use_container_width=True)

    except Exception as e:
        st.error(f"Computation Error: {e}")
