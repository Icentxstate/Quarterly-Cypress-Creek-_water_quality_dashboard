
# ============================================
# Water Quality Dashboard - Final Full Version
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.colors as mcolors
import zipfile
import io
from sklearn.cluster import KMeans
import pymannkendall as mk
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Water Quality Dashboard", layout="wide")

# --- LOAD DATA ---
file_path = "INPUT.CSV"
df = pd.read_csv(file_path, encoding='latin1')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['YearMonth'] = df['Date'].dt.to_period('M').dt.to_timestamp()
df['Month'] = df['Date'].dt.month
df['Season'] = df['Month'].apply(lambda m: (
    "Winter" if m in [12, 1, 2] else
    "Spring" if m in [3, 4, 5] else
    "Summer" if m in [6, 7, 8] else
    "Fall"
))

# --- SIDEBAR ---
st.sidebar.title("Site and Parameter Selection")
site_options = df[['Site ID', 'Site Name']].drop_duplicates()
site_options['Site Display'] = site_options['Site ID'].astype(str) + " - " + site_options['Site Name']
site_dict = dict(zip(site_options['Site Display'], site_options['Site ID']))
selected_sites_display = st.sidebar.multiselect("Select Site(s):", site_dict.keys(), default=list(site_dict.keys())[:2])
selected_sites = [site_dict[label] for label in selected_sites_display]

numeric_columns = df.select_dtypes(include='number').columns.tolist()
default_params = ['TDS', 'Nitrate (µg/L)', 'Flow (CFS)']
valid_defaults = [p for p in default_params if p in numeric_columns]
selected_parameters = st.sidebar.multiselect("Select Parameters (up to 10):", numeric_columns, default=valid_defaults)
chart_type = st.sidebar.radio("Chart Type:", ["Scatter", "Line"], index=0)

# --- STATIC LOCATIONS ---
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
selected_locations = locations[locations['Site ID'].isin(selected_sites)]
color_palette = sns.color_palette("hsv", len(selected_sites))
site_colors = dict(zip(selected_sites, color_palette))

# --- MAP ---
col1, col2 = st.columns([1, 3])
with col1:
    st.subheader("Monitoring Sites")
    avg_lat = selected_locations['Latitude'].mean() if not selected_locations.empty else locations['Latitude'].mean()
    avg_lon = selected_locations['Longitude'].mean() if not selected_locations.empty else locations['Longitude'].mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)
    for _, row in selected_locations.iterrows():
        color = mcolors.to_hex(site_colors[row['Site ID']])
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['Site ID']}: {row['Description']}",
            tooltip=row['Description'],
            icon=folium.Icon(color='blue', icon_color=color)
        ).add_to(m)
    st_folium(m, width=350, height=500)

with col2:
    st.header("Time Series Plot")
    plot_df = df[df['Site ID'].isin(selected_sites)]
    for param in selected_parameters:
        fig, ax = plt.subplots(figsize=(10, 4))
        for site_id in selected_sites:
            site_data = plot_df[plot_df['Site ID'] == site_id]
            site_name = site_data['Site Name'].iloc[0] if not site_data.empty else f"Site {site_id}"
            color = site_colors[site_id]
            if chart_type == "Scatter":
                ax.scatter(site_data['YearMonth'], site_data[param], label=site_name, color=color, s=30)
            else:
                ax.plot(site_data['YearMonth'], site_data[param], label=site_name, color=color)
        ax.set_title(f"{param} over Time")
        ax.set_xlabel("Year-Month")
        ax.set_ylabel(param)
        ax.legend(title="Site")
        ax.grid(True)
        st.pyplot(fig)

# Note: Remaining blocks (tabs for stats, trend, WQI, clustering, export...) are built and appended in previous messages.
