# ============================================
# Water Quality Dashboard with Advanced Analysis
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

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Water Quality Dashboard", layout="wide")

# --- LOAD DATA ---
file_path = "INPUT.CSV"
df = pd.read_csv(file_path, encoding='latin1')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['YearMonth'] = df['Date'].dt.to_period('M').dt.to_timestamp()
df['Month'] = df['Date'].dt.month

# Add Season Column
def assign_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"
df['Season'] = df['Month'].apply(assign_season)

# --- SIDEBAR ---
st.sidebar.title("Site and Parameter Selection")
site_options = df[['Site ID', 'Site Name']].drop_duplicates()
site_options['Site Display'] = site_options['Site ID'].astype(str) + " - " + site_options['Site Name']
site_dict = dict(zip(site_options['Site Display'], site_options['Site ID']))
selected_sites_display = st.sidebar.multiselect("Select Site(s):", site_dict.keys(), default=list(site_dict.keys())[:2])
selected_sites = [site_dict[label] for label in selected_sites_display]

numeric_columns = df.select_dtypes(include='number').columns.tolist()
default_params = ['TDS', 'Nitrate (\u00b5g/L)', 'Flow (CFS)']
valid_defaults = [p for p in default_params if p in numeric_columns]
selected_parameters = st.sidebar.multiselect("Select Parameters (up to 10):", numeric_columns, default=valid_defaults)
chart_type = st.sidebar.radio("Chart Type:", ["Scatter", "Line"], index=0)

# --- STATIC LOCATIONS (ADD YOUR REAL VALUES HERE IF NEEDED) ---
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
    if not selected_parameters:
        st.warning("Please select at least one parameter.")
    else:
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

# Continue to next message...
# [... قبلی ادامه دارد ...]

# --- ADVANCED ANALYTICS TABS ---
st.markdown("## Advanced Water Quality Analysis")
adv_tabs = st.tabs([
    "Boxplot", "Seasonal Means", "Trend Analysis", "Flow vs Parameter"
])

analysis_df = df[df['Site ID'].isin(selected_sites)]

# --- Boxplot ---
with adv_tabs[0]:
    st.subheader("Boxplot by Site")
    if selected_parameters:
        for param in selected_parameters:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=analysis_df, x='Site Name', y=param, palette="Set2", ax=ax)
            ax.set_title(f"{param} Distribution by Site")
            ax.set_ylabel(param)
            ax.set_xlabel("Site")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
    else:
        st.info("Please select at least one parameter.")

# --- Seasonal Means ---
with adv_tabs[1]:
    st.subheader("Seasonal Average by Site")
    if selected_parameters:
        for param in selected_parameters:
            seasonal_avg = analysis_df.groupby(['Season', 'Site Name'])[param].mean().unstack()
            st.markdown(f"**{param}**")
            st.dataframe(seasonal_avg.round(2))
            fig, ax = plt.subplots(figsize=(8, 4))
            seasonal_avg.plot(kind='bar', ax=ax)
            ax.set_ylabel(param)
            ax.set_title(f"Seasonal Mean of {param}")
            ax.grid(True)
            st.pyplot(fig)

# --- Mann-Kendall Trend ---
with adv_tabs[2]:
    st.subheader("Mann-Kendall Trend Test")
    if selected_parameters:
        trend_results = []
        for param in selected_parameters:
            st.markdown(f"**Trend for {param}:**")
            mk_results = []
            for site_id in selected_sites:
                site_data = analysis_df[analysis_df['Site ID'] == site_id][["Date", param]].dropna()
                site_data = site_data.sort_values("Date")
                if len(site_data) >= 8:
                    result = mk.original_test(site_data[param])
                    trend = result.trend
                    p = result.p
                    z = result.z
                    mk_results.append((site_id, trend, round(p, 4), round(z, 2)))
            trend_df = pd.DataFrame(mk_results, columns=["Site ID", "Trend", "P-value", "Z-score"])
            st.dataframe(trend_df)

# --- Flow vs Parameter ---
with adv_tabs[3]:
    st.subheader("Flow vs. Parameter Scatterplots")
    if "Flow (CFS)" in analysis_df.columns and selected_parameters:
        for param in selected_parameters:
            if param != "Flow (CFS)":
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=analysis_df, x="Flow (CFS)", y=param, hue="Site Name", ax=ax)
                ax.set_title(f"Flow vs {param}")
                ax.set_xlabel("Flow (CFS)")
                ax.set_ylabel(param)
                ax.grid(True)
                st.pyplot(fig)
    else:
        st.warning("'Flow (CFS)' data or parameters not available.")
# [... قبلی ادامه دارد ...]

# --- WATER QUALITY INDEX (WQI) ---
st.markdown("## Water Quality Index (WQI)")
if selected_parameters:
    # Example weights from literature (customize as needed)
    param_weights = {
        "TDS": 0.2,
        "Nitrate (\u00b5g/L)": 0.2,
        "DO": 0.2,
        "pH": 0.2,
        "Turbidity": 0.2
    }
    available_weights = {p: w for p, w in param_weights.items() if p in selected_parameters}
    if available_weights:
        st.write("Weights used:", available_weights)
        df_wqi = analysis_df.copy()
        norm_df = df_wqi[selected_parameters].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        norm_df = norm_df.fillna(0)
        weighted = pd.DataFrame()
        for p, w in available_weights.items():
            weighted[p] = norm_df[p] * w
        df_wqi['WQI'] = weighted.sum(axis=1)

        wqi_avg = df_wqi.groupby('Site Name')['WQI'].mean().sort_values(ascending=False)
        st.dataframe(wqi_avg.round(2).reset_index())
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=wqi_avg.values, y=wqi_avg.index, ax=ax)
        ax.set_title("Average WQI by Site")
        ax.set_xlabel("WQI")
        st.pyplot(fig)
    else:
        st.warning("No standard weights matched selected parameters for WQI.")

# --- KMEANS CLUSTERING ---
st.markdown("## Cluster Analysis (KMeans)")
if len(selected_parameters) >= 2:
    from sklearn.preprocessing import StandardScaler
    cluster_df = analysis_df[selected_parameters].dropna()
    if len(cluster_df) > 10:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_df)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(X_scaled)
        analysis_df.loc[cluster_df.index, 'Cluster'] = kmeans.labels_

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=analysis_df, x=selected_parameters[0], y=selected_parameters[1], hue='Cluster', palette='Set1')
        ax.set_title("KMeans Clustering")
        st.pyplot(fig)
    else:
        st.info("Not enough valid records for clustering.")
else:
    st.info("Select at least two parameters for clustering.")

# --- TIME-SPATIAL HEATMAP ---
st.markdown("## Heatmap (Monthly Average by Site)")
if selected_parameters:
    for param in selected_parameters:
        heat_df = analysis_df.copy()
        heat_df['MonthYear'] = heat_df['Date'].dt.to_period('M').dt.to_timestamp()
        pivot_heat = heat_df.pivot_table(index='MonthYear', columns='Site Name', values=param, aggfunc='mean')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_heat.T, cmap='YlGnBu', cbar_kws={'label': param})
        ax.set_title(f"Heatmap of {param} by Site and Month")
        st.pyplot(fig)
else:
    st.info("Please select parameters to generate heatmaps.")
st.markdown("---")
st.caption("Dashboard prepared by Icen – Data from Cypress Creek CRP Monitoring")
