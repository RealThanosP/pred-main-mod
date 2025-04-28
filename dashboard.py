import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Predictive Maintenance Dashboard 🚀")

# Load your model outputs
# Assume you have a dataframe 'target_df' from your target_data() function
# Example:
# target_df = your_function()
from data_processing import import_target_data

target_df = import_target_data()
print(target_df)

# Sidebar
st.sidebar.title("Settings")
machine_id = st.sidebar.selectbox("Select Machine ID", options=target_df.index.unique())
view_option = st.sidebar.radio("View Options", ('Overview', 'Detailed Trends'))

# Filtered Data
machine_data = target_df.iloc[machine_id]

# Overview Metrics
st.header(f"Machine ID: {machine_id}")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Health State", int(machine_data['health_state']))
    
with col2:
    st.metric("Predicted RUL (cycles)", int(machine_data['RUL']))
    
with col3:
    maintenance_score_display = round(float(machine_data['maintenance_score']) * 100, 2)
    st.metric("Maintenance Score (%)", f"{maintenance_score_display}%")

st.divider()

# Failure Modes
st.subheader("Failure Mode Probabilities")
st.write(machine_data[['cooler_failure', 'valve_failure', 'pump_failure', 'hydraulic_failure']])

# Plotting Trends
if view_option == 'Detailed Trends':
    st.subheader("Trends Over Time")

    time_series = target_df.copy()

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    axs[0, 0].plot(time_series.index, time_series['maintenance_score'], label="Maintenance Score")
    axs[0, 0].set_title('Maintenance Score Over Time')
    axs[0, 0].set_xlabel('Cycles')
    axs[0, 0].legend()

    axs[0, 1].plot(time_series.index, time_series['RUL'], label="Remaining Useful Life", color='orange')
    axs[0, 1].set_title('RUL Over Time')
    axs[0, 1].set_xlabel('Cycles')
    axs[0, 1].legend()

    axs[1, 0].plot(time_series.index, time_series['health_state'], label="Health State", color='green')
    axs[1, 0].set_title('Health State Evolution')
    axs[1, 0].set_xlabel('Cycles')
    axs[1, 0].legend()

    axs[1, 1].plot(time_series.index, time_series['failure_flag'], label="Failure Flag", color='red')
    axs[1, 1].set_title('Failure Flags')
    axs[1, 1].set_xlabel('Cycles')
    axs[1, 1].legend()

    plt.tight_layout()
    st.pyplot(fig)

# Alerts
st.subheader("Alerts")
if machine_data['failure_flag'] == 1:
    st.error("⚠️ Machine predicted to fail soon! Immediate action required.")
elif machine_data['maintenance_score'] > 0.7:
    st.warning("⚠️ High maintenance score detected. Plan intervention.")
else:
    st.success("✅ Machine is operating normally.")

st.divider()

# Footer
st.caption("Built for Predictive Maintenance Monitoring.")