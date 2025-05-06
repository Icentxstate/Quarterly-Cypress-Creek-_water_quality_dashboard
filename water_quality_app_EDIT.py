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
st.markdown("## Statistical Analysis")
tabs = st.tabs([
    "Summary Statistics",
    "Monthly Averages",
    "Annual Averages",
    "Correlation Matrix",
    "Export Data"
])

analysis_df = df[df['Site ID'].isin(selected_sites)].copy()
analysis_df['Month'] = analysis_df['Date'].dt.month
analysis_df['Season'] = analysis_df['Month'].apply(lambda m: "Winter" if m in [12, 1, 2] else
                                                   "Spring" if m in [3, 4, 5] else
                                                   "Summer" if m in [6, 7, 8] else "Fall")
analysis_df['MonthYear'] = analysis_df['Date'].dt.to_period('M').dt.to_timestamp()
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
            annual_avg = (
                analysis_df.copy()
                .assign(Year=analysis_df['Date'].dt.year)
                .groupby(['Year', 'Site Name'])[param]
                .mean()
                .reset_index()
                .pivot(index='Year', columns='Site Name', values=param)
                .round(2)
            )
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=annual_avg.reset_index().melt(id_vars='Year'), x='Year', y='value', hue='Site Name', ax=ax)
            ax.set_title(f"{param} - Annual Average Comparison")
            ax.set_xlabel("Year")
            ax.set_ylabel(param)
            ax.legend(title="Site")
            st.pyplot(fig)

    with tabs[3]:
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
            st.markdown("Top 5 Correlated Parameter Pairs")
            styled = top_corr[['Parameter 1', 'Parameter 2', 'Correlation']].style
            styled = styled.applymap(lambda v: 'color: red; font-weight: bold' if abs(v) >= 0.8 else '', subset=['Correlation'])
            st.dataframe(styled)

    with tabs[4]:
        st.subheader("Export Processed Data")
        if selected_parameters:
            for param in selected_parameters:
                st.markdown(f"**Parameter:** {param}")
                summary = analysis_df.groupby('Site Name')[param].agg(['mean', 'median', 'std']).round(2)
                monthly_avg = analysis_df.groupby([analysis_df['Date'].dt.month, 'Site Name'])[param].mean().unstack().round(2)
                annual_avg = (
                    analysis_df.copy()
                    .assign(Year=analysis_df['Date'].dt.year)
                    .groupby(['Year', 'Site Name'])[param]
                    .mean()
                    .reset_index()
                    .pivot(index='Year', columns='Site Name', values=param)
                    .round(2)
                )
                st.download_button(f"Download {param} - Summary Statistics", summary.to_csv().encode('utf-8'), file_name=f"{param}_summary.csv")
                st.download_button(f"Download {param} - Monthly Averages", monthly_avg.to_csv().encode('utf-8'), file_name=f"{param}_monthly_avg.csv")
                st.download_button(f"Download {param} - Annual Averages", annual_avg.to_csv().encode('utf-8'), file_name=f"{param}_annual_avg.csv")

            # Download all as ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for param in selected_parameters:
                    summary = analysis_df.groupby('Site Name')[param].agg(['mean', 'median', 'std']).round(2)
                    monthly_avg = analysis_df.groupby([analysis_df['Date'].dt.month, 'Site Name'])[param].mean().unstack().round(2)
                    annual_avg = (
                        analysis_df.copy()
                        .assign(Year=analysis_df['Date'].dt.year)
                        .groupby(['Year', 'Site Name'])[param]
                        .mean()
                        .reset_index()
                        .pivot(index='Year', columns='Site Name', values=param)
                        .round(2)
                    )
                    zf.writestr(f"{param}_summary.csv", summary.to_csv())
                    zf.writestr(f"{param}_monthly_avg.csv", monthly_avg.to_csv())
                    zf.writestr(f"{param}_annual_avg.csv", annual_avg.to_csv())
                # Add raw filtered data
                filtered = df[df['Site ID'].isin(selected_sites)]
                zf.writestr("filtered_data.csv", filtered.to_csv(index=False))
            zip_buffer.seek(0)
            st.download_button("Download All Parameters as ZIP", data=zip_buffer, file_name="all_outputs.zip", mime="application/zip")
else:
    st.warning("Please select at least one parameter to continue.")
# --- Add Season Column ---
df['Month'] = df['Date'].dt.month
df['Season'] = df['Month'].apply(lambda m: "Winter" if m in [12, 1, 2] else
                                             "Spring" if m in [3, 4, 5] else
                                             "Summer" if m in [6, 7, 8] else "Fall")

st.markdown("## Advanced Analysis")

adv_tabs = st.tabs([
    "Seasonal Means", "Mann-Kendall Trend", "Flow vs Parameter", "Water Quality Index", "KMeans Clustering", "Time-Spatial Heatmap"
])

# --- Seasonal Means ---
with adv_tabs[0]:
    st.subheader("Seasonal Averages")
    for param in selected_parameters:
        seasonal_avg = analysis_df.groupby(['Season', 'Site Name'])[param].mean().unstack()
        st.markdown(f"**{param}**")
        st.dataframe(seasonal_avg.round(2))
        fig, ax = plt.subplots(figsize=(8, 4))
        seasonal_avg.plot(kind='bar', ax=ax)
        ax.set_ylabel(param)
        ax.set_title(f"Seasonal Mean of {param}")
        st.pyplot(fig)

# --- Mann-Kendall Trend Test ---
import pymannkendall as mk
with adv_tabs[1]:
    st.subheader("Mann-Kendall Trend Test")
    for param in selected_parameters:
        st.markdown(f"**Trend for {param}:**")
        trend_results = []
        for site_id in selected_sites:
            site_data = analysis_df[analysis_df['Site ID'] == site_id][['Date', param]].dropna().sort_values('Date')
            if len(site_data) >= 8:
                result = mk.original_test(site_data[param])
                trend_results.append((site_id, result.trend, round(result.p, 4), round(result.z, 2)))
        if trend_results:
            trend_df = pd.DataFrame(trend_results, columns=["Site ID", "Trend", "P-value", "Z-score"])
            st.dataframe(trend_df)

# --- Flow vs Parameter ---
with adv_tabs[2]:
    st.subheader("Flow vs. Parameter")
    if "Flow (CFS)" in analysis_df.columns:
        for param in selected_parameters:
            if param != "Flow (CFS)":
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=analysis_df, x="Flow (CFS)", y=param, hue="Site Name", ax=ax)
                ax.set_title(f"Flow vs {param}")
                st.pyplot(fig)
    else:
        st.info("'Flow (CFS)' column not found.")

# --- Water Quality Index (WQI) ---
with adv_tabs[3]:
    st.subheader("Water Quality Index (WQI)")
    param_weights = {
        "TDS": 0.2,
        "Nitrate (\u00b5g/L)": 0.2,
        "DO": 0.2,
        "pH": 0.2,
        "Turbidity": 0.2
    }
    wqi_params = {p: w for p, w in param_weights.items() if p in selected_parameters}
    if wqi_params:
        st.write("Weights used:", wqi_params)
        norm_df = analysis_df[selected_parameters].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        norm_df = norm_df.fillna(0)
        weighted = sum(norm_df[p] * w for p, w in wqi_params.items())
        analysis_df['WQI'] = weighted
        wqi_avg = analysis_df.groupby('Site Name')['WQI'].mean().sort_values(ascending=False)
        st.dataframe(wqi_avg.round(2).reset_index())
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=wqi_avg.values, y=wqi_avg.index, ax=ax)
        ax.set_title("Average WQI by Site")
        st.pyplot(fig)
    else:
        st.warning("No WQI parameters matched.")

# --- KMeans Clustering ---
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
with adv_tabs[4]:
    st.subheader("KMeans Clustering")
    if len(selected_parameters) >= 2:
        cluster_df = analysis_df[selected_parameters].dropna()
        if len(cluster_df) > 10:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(cluster_df)
            kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(scaled)
            analysis_df.loc[cluster_df.index, 'Cluster'] = kmeans.labels_
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=analysis_df, x=selected_parameters[0], y=selected_parameters[1], hue='Cluster', palette='Set1')
            ax.set_title("KMeans Clustering")
            st.pyplot(fig)
        else:
            st.info("Not enough valid data for clustering.")
    else:
        st.info("Please select at least two parameters.")

# --- Time-Spatial Heatmap ---
with adv_tabs[5]:
    st.subheader("Time-Spatial Heatmap (Monthly Average)")
    for param in selected_parameters:
        heat_df = analysis_df.copy()
        heat_df['MonthYear'] = heat_df['Date'].dt.to_period('M').dt.to_timestamp()
        pivot = heat_df.pivot_table(index='MonthYear', columns='Site Name', values=param, aggfunc='mean')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot.T, cmap='YlGnBu', cbar_kws={'label': param})
        ax.set_title(f"Heatmap of {param} by Site and Month")
        st.pyplot(fig)
st.markdown("---")
st.caption("Data Source: CRP Monitoring at Cypress Creek")
