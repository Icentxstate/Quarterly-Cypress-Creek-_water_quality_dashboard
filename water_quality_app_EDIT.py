import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import matplotlib.colors as mcolors
import numpy as np
import zipfile
import io

# --- Page Configuration ---
st.set_page_config(layout="wide")

# --- Load Data ---
file_path = r"INPUT.CSV"
df = pd.read_csv(file_path, encoding='latin1')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['YearMonth'] = df['Date'].dt.to_period('M').dt.to_timestamp()

# --- Sidebar Selections ---
st.sidebar.title("Site and Parameter Selection")
site_options = df[['Site ID', 'Site Name']].drop_duplicates()
site_options['Site Display'] = site_options['Site ID'].astype(str) + " - " + site_options['Site Name']
site_dict = dict(zip(site_options['Site Display'], site_options['Site ID']))

selected_sites_display = st.sidebar.multiselect("Select Site(s):", site_dict.keys(), default=list(site_dict.keys())[:2])
selected_sites = [site_dict[label] for label in selected_sites_display]

numeric_columns = df.select_dtypes(include='number').columns.tolist()
default_params = ['TDS', 'Nitrate (\u00b5g/L)']
valid_defaults = [p for p in default_params if p in numeric_columns]

selected_parameters = st.sidebar.multiselect("Select Parameters (up to 10):", numeric_columns, default=valid_defaults)
chart_type = st.sidebar.radio("Select Chart Type:", ["Scatter (Points)", "Line (Connected)"], index=0)

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

selected_locations = locations[locations['Site ID'].isin(selected_sites)]
color_palette = sns.color_palette("hsv", len(selected_sites))
site_colors = dict(zip(selected_sites, color_palette))

# --- Layout ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Site Map")
    avg_lat = selected_locations['Latitude'].mean() if not selected_locations.empty else locations['Latitude'].mean()
    avg_lon = selected_locations['Longitude'].mean() if not selected_locations.empty else locations['Longitude'].mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, width='100%', height='100%')
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
                if chart_type == "Scatter (Points)":
                    ax.scatter(site_data['YearMonth'], site_data[param], label=site_name, color=color, s=30)
                else:
                    ax.plot(site_data['YearMonth'], site_data[param], label=site_name, color=color)
            ax.set_title(f"{param} Over Time (Monthly)")
            ax.set_xlabel("Year-Month")
            ax.set_ylabel(param)
            ax.legend(title='Site')
            ax.grid(True)
            st.pyplot(fig)

# --- Statistical Analysis Tabs ---
st.markdown("## \ud83d\udcca Statistical Analysis")
tabs = st.tabs([
    "\ud83d\udccc Summary Statistics",
    "\ud83d\udcc6 Monthly Averages",
    "\ud83d\udccb Annual Averages",
    "\ud83d\udcc5 Export Data",
    "\ud83d\udcc8 Correlation Matrix"
])

analysis_df = df[df['Site ID'].isin(selected_sites)]

if selected_parameters:
    for param in selected_parameters:
        with tabs[0]:
            st.subheader(f"{param} - Summary Statistics")
            summary = analysis_df.groupby('Site Name')[param].agg(['mean', 'median', 'std']).round(2)
            st.dataframe(summary)

        with tabs[1]:
            st.subheader(f"{param} - Monthly Averages (Across Years)")
            monthly_avg = analysis_df.groupby([analysis_df['Date'].dt.month, 'Site Name'])[param].mean().unstack().round(2)
            st.line_chart(monthly_avg)

        with tabs[2]:
            st.subheader(f"{param} - Annual Averages")
            annual_avg = analysis_df.groupby([analysis_df['Date'].dt.year, 'Site Name'])[param].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=annual_avg, x='Date', y=param, hue='Site Name', ax=ax)
            ax.set_title(f"{param} - Annual Average Comparison")
            ax.set_xlabel("Year")
            ax.set_ylabel(param)
            ax.legend(title="Site")
            st.pyplot(fig)

        with tabs[3]:
            st.subheader("Download Processed Data")
            csv_summary = summary.to_csv().encode('utf-8')
            st.download_button("\ud83d\udcc4 Download Summary Statistics (CSV)", csv_summary, file_name=f"{param}_summary.csv")
            csv_monthly = monthly_avg.to_csv().encode('utf-8')
            st.download_button("\ud83d\udcc6 Download Monthly Averages (CSV)", csv_monthly, file_name=f"{param}_monthly_avg.csv")
            annual_avg_zip = annual_avg.pivot(index='Date', columns='Site Name', values=param).round(2)
            csv_annual = annual_avg_zip.to_csv().encode('utf-8')
            st.download_button("\ud83d\udccb Download Annual Averages (CSV)", csv_annual, file_name=f"{param}_annual_avg.csv")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"{param}_summary.csv", summary.to_csv())
                zf.writestr(f"{param}_monthly_avg.csv", monthly_avg.to_csv())
                zf.writestr(f"{param}_annual_avg.csv", annual_avg_zip.to_csv())
            zip_buffer.seek(0)
            st.download_button("\ud83d\udcc2 Download All as ZIP", data=zip_buffer, file_name=f"{param}_analysis_outputs.zip", mime="application/zip")

    with tabs[4]:
        st.subheader("Correlation Matrix of Selected Parameters")
        if len(selected_parameters) < 2:
            st.info("Please select at least two parameters to compute correlations.")
        else:
            corr_df = analysis_df[selected_parameters].dropna().corr(method='pearson').round(2)
            fig, ax = plt.subplots(figsize=(8, 6))
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap=cmap, center=0, square=True,
                        linewidths=0.5, cbar_kws={"shrink": .8}, annot_kws={"size": 10})
            for i in range(len(corr_df)):
                for j in range(len(corr_df.columns)):
                    value = corr_df.iloc[i, j]
                    if i != j and abs(value) >= 0.8:
                        ax.text(j + 0.5, i + 0.5, f"{value:.2f}", color='red', ha='center', va='center', fontweight='bold')
            st.pyplot(fig)

            corr_pairs = corr_df.where(~np.tril(np.ones(corr_df.shape)).astype(bool)).stack().reset_index()
            corr_pairs.columns = ['Parameter 1', 'Parameter 2', 'Correlation']
            corr_pairs['Abs Correlation'] = corr_pairs['Correlation'].abs()
            top_corr = corr_pairs.sort_values(by='Abs Correlation', ascending=False).head(5)
            st.markdown("### \ud83d\udd1d Top 5 Correlated Parameter Pairs")
            st.dataframe(top_corr[['Parameter 1', 'Parameter 2', 'Correlation']].style
                         .applymap(lambda v: 'color: red; font-weight: bold' if abs(v) >= 0.8 else ''))
else:
    st.warning("Please select at least one parameter to continue.")

st.markdown("---")
st.caption("Data Source: CRP Monitoring at Cypress Creek")
