import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Set page configuration
st.set_page_config(layout="wide")

# --- Load Data ---
file_path = r"INPUT.CSV"
df = pd.read_csv(file_path, encoding='latin1')

# --- Preprocess Date ---
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['YearMonth'] = df['Date'].dt.to_period('M').dt.to_timestamp()

# --- Sidebar Selections ---
st.sidebar.title("Site and Parameter Selection")

# Get unique site options
site_options = df[['Site ID', 'Site Name']].drop_duplicates()
site_options['Site Display'] = site_options['Site ID'].astype(str) + " - " + site_options['Site Name']
site_dict = dict(zip(site_options['Site Display'], site_options['Site ID']))

# Site selection
selected_sites_display = st.sidebar.multiselect(
    label="Select Site(s):",
    options=site_dict.keys(),
    default=list(site_dict.keys())[:2]
)
selected_sites = [site_dict[label] for label in selected_sites_display]

# Parameter selection
numeric_columns = df.select_dtypes(include='number').columns.tolist()
default_params = ['TDS', 'Nitrate (µg/L)']
valid_defaults = [p for p in default_params if p in numeric_columns]

selected_parameters = st.sidebar.multiselect(
    label="Select Parameters (up to 10):",
    options=numeric_columns,
    default=valid_defaults
)

# --- Static Site Location Data ---
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

# Filter locations based on selected sites
selected_locations = locations[locations['Site ID'].isin(selected_sites)]

# Assign colors to each selected site
color_palette = sns.color_palette("hsv", len(selected_sites))
site_colors = dict(zip(selected_sites, color_palette))

# --- Layout with Columns ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Site Map")
    # Center map on the average location of selected sites
    if not selected_locations.empty:
        avg_lat = selected_locations['Latitude'].mean()
        avg_lon = selected_locations['Longitude'].mean()
    else:
        avg_lat = locations['Latitude'].mean()
        avg_lon = locations['Longitude'].mean()

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, width='100%', height='100%')

    for _, row in selected_locations.iterrows():
        site_id = row['Site ID']
        color = mcolors.to_hex(site_colors[site_id])
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{site_id}: {row['Description']}",
            tooltip=row['Description'],
            icon=folium.Icon(color='blue', icon_color=color)
        ).add_to(m)
    st_folium(m, width=350, height=500)

with col2:
    st.header("Time Series Plots")
    plot_df = df[df['Site ID'].isin(selected_sites)]

    if not selected_parameters:
        st.warning("Please select at least one parameter.")
    else:
        for param in selected_parameters:
            fig, ax = plt.subplots(figsize=(10, 4))
            for site_id in selected_sites:
                site_data = plot_df[plot_df['Site ID'] == site_id]
                site_name = site_data['Site Name'].iloc[0]
                color = site_colors[site_id]
                ax.plot(site_data['YearMonth'], site_data[param], label=site_name, color=color)
            ax.set_title(f"{param} Over Time (Monthly)")
            ax.set_xlabel("Year-Month")
            ax.set_ylabel(param)
            ax.legend(title='Site')
            ax.grid(True)
            st.pyplot(fig)

st.markdown("---")
st.caption("Data Source: CRP Monitoring at Cypress Creek")
