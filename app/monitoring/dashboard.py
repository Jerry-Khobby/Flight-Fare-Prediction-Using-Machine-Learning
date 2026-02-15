import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Model Monitoring", page_icon="📊", layout="wide")
counter = st_autorefresh(interval=5000, limit=None, key="monitor_refresh")
st.write("Refresh count:", counter)
API_URL = "http://localhost:8000"

st.title("📊 Flight Fare Model Monitoring Dashboard")

# Health Check
col1, col2, col3 = st.columns(3)

try:
    health = requests.get(f"{API_URL}/monitoring/health").json()
    
    with col1:
        status_color = "🟢" if health['status'] == 'healthy' else "🟡"
        st.metric("Model Status", f"{status_color} {health['status'].upper()}")
    
    with col2:
      mae_value = health.get("recent_mae") or 0 
      st.metric("Recent MAE",f"৳{mae_value:,.0f}")
    
    with col3:
        drift_rate = health.get('drift_rate', 0)
        st.metric("Drift Rate", f"{drift_rate:.1%}")
    
    if health['warnings']:
        st.warning("⚠️ **Warnings:**\n" + "\n".join(f"- {w}" for w in health['warnings']))
    
except Exception as e:
    st.error(f"Cannot connect to API: {e}")

st.markdown("---")

# Performance Metrics
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Performance Metrics (Last 7 Days)")
    days = st.slider("Days to analyze", 1, 30, 7)
    
    try:
        metrics = requests.get(f"{API_URL}/monitoring/metrics?days={days}").json()
        
        if 'error' not in metrics:
            st.metric("Total Predictions", metrics['total_predictions'])
            st.metric("MAE", f"৳{metrics.get('mae', 0):,.2f}")
            st.metric("RMSE", f"৳{metrics.get('rmse', 0):,.2f}")
            st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
        else:
            st.info(metrics.get('message', 'No data'))
    except:
        st.error("Failed to load metrics")

with col2:
    st.subheader("🔍 Drift Detection Summary")
    
    try:
        drift = requests.get( f"{API_URL}/monitoring/drift?days={days}&_={datetime.now().timestamp()}").json()
        
        st.metric("Total Checks", drift['total_checks'])
        st.metric("Drift Detected", drift['drift_detected'])
        
        if drift['total_checks'] > 0:
            drift_pct = (drift['drift_detected'] / drift['total_checks']) * 100
            st.metric("Drift Rate", f"{drift_pct:.1f}%")
    except:
        st.error("Failed to load drift data")

st.markdown("---")
st.caption("Dashboard refreshes automatically every 30 seconds")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")