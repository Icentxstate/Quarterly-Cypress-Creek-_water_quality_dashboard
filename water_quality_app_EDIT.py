import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
import pymannkendall as mk
from sklearn.cluster import KMeans
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Advanced Water Quality Dashboard")

# --- Sidebar Navigation ---
st.sidebar.title("Dashboard Navigation")
with st.sidebar.expander("Manual"):
    st.markdown("""
    - Use this dashboard to select sites and parameters for analysis.
    - View site locations on the map.
    - Perform statistical and advanced analyses.
    """)

with st.sidebar.expander("Site and Parameter Selection"):
    st.subheader("Select Sites and Parameters")
    file_path = r"INPUT.CSV"
    df = pd.read_csv(file_path, encoding='latin1')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['YearMonth'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    
    site_options = df[['Site ID', 'Site Name']].drop_duplicates()
    site_options['Site Display'] = site_options['Site ID'].astype(str) + " - " + site_options['Site Name']
    site_dict = dict(zip(site_options['Site Display'], site_options['Site ID']))
    
    selected_sites_display = st.multiselect("Select Site(s):", site_dict.keys(), default=list(site_dict.keys())[:2])
    selected_sites = [site_dict[label] for label in selected_sites_display]
    
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    selected_parameters = st.multiselect("Select Parameters:", numeric_columns, max_selections=10)
    
    chart_type = st.radio("Select Chart Type:", ["Scatter (Points)", "Line (Connected)"], index=0)

# --- Display Selected Sites and Parameters ---
st.header("Selected Sites and Parameters")
st.write(f"**Selected Sites:** {', '.join(selected_sites_display) if selected_sites else 'None'}")
st.write(f"**Selected Parameters:** {', '.join(selected_parameters) if selected_parameters else 'None'}")

# --- Map Section ---
st.subheader("Map Overview")
locations = pd.DataFrame({
    'Site ID': [12673, 12674, 12675, 12676, 12677, 22109, 22110],
    'Description': [
        'CYPRESS CREEK AT BLANCO RIVER',
        'CYPRESS CREEK AT FM 12',
        'CYPRESS CK - BLUE HOLE CAMPGRD',
        'CYPRESS CREEK AT RR 12',
        'CYPRESS CREEK AT JACOBS WELL',
        'CYPRESS CREEK AT CAMP YOUNG JUDAEA',
        'CYPRESS CREEK AT WOODCREEK DRIVE DAM'
    ],
    'Longitude': [-98.094754, -98.09753, -98.09084, -98.104139, -98.126321, -98.12015, -98.117508],
    'Latitude': [29.991514, 29.996859, 30.002777, 30.012356, 30.034408, 30.02434, 30.020925]
})

st.markdown("### Monitoring Sites Map")
avg_lat = locations['Latitude'].mean()
avg_lon = locations['Longitude'].mean()
m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)
for _, row in locations.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Site ID']}: {row['Description']}",
        tooltip=row['Description'],
        icon=folium.Icon(color='blue')
    ).add_to(m)
st_folium(m, width=700, height=500)

# --- Data Table and Visual Analysis ---
st.subheader("Data Table and Visual Analysis")
data_tab, visual_tab = st.tabs(["Data Table", "Visual Analysis"])

with data_tab:
    st.write("Filtered Data Table")
    if selected_sites and selected_parameters:
        filtered_df = df[df['Site ID'].isin(selected_sites)]
        filtered_df = filtered_df[['Date', 'Site Name', 'Site ID'] + selected_parameters].dropna()
        st.dataframe(filtered_df)

with visual_tab:
    st.write("Visual Analysis")
    for param in selected_parameters:
        st.markdown(f"### {param} Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        for site_id in selected_sites:
            site_data = df[df['Site ID'] == site_id]
            site_name = site_data['Site Name'].iloc[0]
            if chart_type == "Scatter (Points)":
                ax.scatter(site_data['YearMonth'], site_data[param], label=site_name)
            else:
                ax.plot(site_data['YearMonth'], site_data[param], label=site_name)
        ax.set_title(f"{param} Over Time")
        ax.set_xlabel("Year-Month")
        ax.set_ylabel(param)
        ax.legend(title='Site')
        st.pyplot(fig)

# --- Statistical Analysis ---
st.subheader("Statistical Analysis")
stat_tabs = st.tabs([
    "Summary Statistics", "Monthly Averages", 
    "Annual Averages", "Seasonal Averages", 
    "Correlation Matrix", "Trend Analysis"
])

with stat_tabs[0]:
    st.write("Summary Statistics")
    if selected_sites and selected_parameters:
        summary = df[df['Site ID'].isin(selected_sites)].describe().T.round(2)
        st.dataframe(summary)

# --- Advanced Analysis ---
st.subheader("Advanced Analysis")
adv_tabs = st.tabs([
    "Mann-Kendall Trend Test", "Flow vs Parameter", 
    "Water Quality Index (WQI)", "KMeans Clustering", 
    "Time-Spatial Heatmap", "Boxplot by Site", 
    "Normality Test", "PCA", 
    "Autocorrelation (ACF)", "Forecasting (Prophet)"
])

with adv_tabs[0]:
    st.write("Mann-Kendall Trend Test")
    for param in selected_parameters:
        trend_results = []
        for site in selected_sites:
            site_data = df[df['Site ID'] == site][['Date', param]].dropna()
            if len(site_data) > 10:
                result = mk.original_test(site_data[param])
                trend_results.append((site, result.trend, round(result.p, 4), round(result.z, 2)))
        if trend_results:
            st.dataframe(pd.DataFrame(trend_results, columns=["Site ID", "Trend", "P-value", "Z-score"]))
