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
import math
from matplotlib.ticker import FuncFormatter
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import shapiro
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.express as px
from prophet import Prophet
import shap
# Part 1-------------------------------------------------------------------------------------------------------
# --- Page Configuration ---
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# --- Custom Page Styling ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f9ff;
    }
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        color: #1a1a1a;
    }
    h1, h2, h3, h4 {
        color: #2E8BC0;
        font-weight: 600;
    }
    .stMetric {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px;
        background-color: #f9f9f9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stRadio > div {
        flex-direction: row;
        gap: 10px;
    }
    .block-container {
        padding: 2rem 3rem;
    }
    </style>
""", unsafe_allow_html=True)
# Part 2--------------------------------------------------------------------------------------------------------
# --- Load Data ---
file_path = r"INPUT.CSV"
df = pd.read_csv(file_path, encoding='latin1')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['YearMonth'] = df['Date'].dt.to_period('M').dt.to_timestamp()

# --- Sidebar Selections ---
st.sidebar.title("🧭 Control Panel")
site_options = df[['Site ID', 'Site Name']].drop_duplicates()
site_options['Site Display'] = site_options['Site ID'].astype(str) + " - " + site_options['Site Name']
site_dict = dict(zip(site_options['Site Display'], site_options['Site ID']))

selected_sites_display = st.sidebar.multiselect(
    "Select Site(s):", 
    site_dict.keys(), 
    default=list(site_dict.keys())[:2]
)
selected_sites = [site_dict[label] for label in selected_sites_display]

# --- Define Numeric Columns and Valid Defaults ---
numeric_columns = df.select_dtypes(include='number').columns.tolist()
default_params = ['TDS', 'Nitrate (\u00b5g/L)']
valid_defaults = [p for p in default_params if p in numeric_columns]

selected_parameters = st.sidebar.multiselect(
    "Select Parameters (up to 10):", numeric_columns, default=valid_defaults
)

chart_type = st.sidebar.radio(
    "Select Chart Type:", 
    ["Scatter (Points)", "Line (Connected)"], 
    index=0
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
selected_locations = locations[locations['Site ID'].isin(selected_sites)]

# --- Stylish Header ---
st.markdown("""
    <div style='text-align: center;'>
        <h1>💧 Cypress Creek Water Quality Dashboard</h1>
        <p style='font-size:18px;'>Monitoring trends, summaries, and site insights</p>
        <hr style='margin-top: 1rem; margin-bottom: 1rem; border: none; height: 2px; background-color: #2E8BC0;'/>
    </div>
""", unsafe_allow_html=True)

# --- Summary Cards (Metrics) ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🧪 Parameters Selected", f"{len(selected_parameters)}")
with col2:
    st.metric("📍 Sites Selected", f"{len(selected_sites)}")
with col3:
    if not df.empty:
        st.metric("🗓️ Date Range", f"{df['Date'].min().date()} → {df['Date'].max().date()}")

st.markdown("---")
# part 3 --------------------------------------------------------------------------------------------------------------------------------------------
# --- Time Series and Summary Tabs ---
tabs_main = st.tabs([
    "📈 Time Series Plots",
    "📑 Summary Table"
])

# --- 📈 Time Series Tab ---
with tabs_main[0]:
    st.markdown("## 📈 Time Series Plots")
    st.markdown("<p style='color: gray;'>Monthly trends per parameter and site</p>", unsafe_allow_html=True)
    st.markdown("---")

    if not selected_parameters:
        st.warning("Please select at least one parameter.")
    else:
        for param in selected_parameters:
            st.markdown(f"### 🔹 {param}")
            plot_df = df[df['Site ID'].isin(selected_sites)].dropna(subset=[param])
            
            if chart_type == "Line (Connected)":
                fig = px.line(
                    plot_df, x='YearMonth', y=param, color='Site Name',
                    markers=True, title=f"{param} Over Time",
                    labels={'YearMonth': 'Date', param: param},
                    template='plotly_white'
                )
            else:
                fig = px.scatter(
                    plot_df, x='YearMonth', y=param, color='Site Name',
                    title=f"{param} Scatter Over Time",
                    labels={'YearMonth': 'Date', param: param},
                    template='plotly_white'
                )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")

# --- 📑 Summary Table Tab ---
with tabs_main[1]:
    st.markdown("## 📑 Summary Statistics by Site")
    st.markdown("<p style='color: gray;'>Basic descriptive statistics for selected parameters</p>", unsafe_allow_html=True)
    st.markdown("---")

    for param in selected_parameters:
        st.markdown(f"### 🔹 {param}")
        summary = (
            df[df['Site ID'].isin(selected_sites)]
            .groupby('Site Name')[param]
            .agg(['mean', 'median', 'std', 'min', 'max', 'count'])
            .round(2)
        )
        st.dataframe(summary)
        st.markdown("---")
#Part 4-----------------------------------------------------------------------------------------------------------------------------------------------------
# --- Add Season Columns for Statistical Analysis ---
df['Month'] = df['Date'].dt.month
df['Season'] = df['Month'].apply(lambda m: "Winter" if m in [12, 1, 2] else
                                 "Spring" if m in [3, 4, 5] else
                                 "Summer" if m in [6, 7, 8] else "Fall")
df['MonthYear'] = df['Date'].dt.to_period('M').dt.to_timestamp()

# --- Statistical Analysis Tabs ---
st.sidebar.subheader("📊 Statistical Analysis")
analysis_options = ["Summary Statistics", "Monthly Averages", "Annual Averages", "Correlation Matrix", "Export Data"]
selected_analysis = st.sidebar.radio("Select Analysis:", analysis_options)

analysis_df = df[df['Site ID'].isin(selected_sites)].copy()

# --- Summary Statistics ---
if selected_analysis == "Summary Statistics":
    st.markdown("## 📋 Summary Statistics")
    st.markdown("<p style='color: gray;'>Descriptive stats per site and parameter</p>", unsafe_allow_html=True)
    st.markdown("---")

    for param in selected_parameters:
        st.markdown(f"### 🔹 {param}")
        summary = (
            analysis_df
            .groupby('Site Name')[param]
            .agg(['mean', 'median', 'std', 'min', 'max', 'count'])
            .round(2)
        )
        st.dataframe(summary)
        st.markdown("---")

# --- Monthly Averages ---
elif selected_analysis == "Monthly Averages":
    st.markdown("## 📅 Monthly Averages")
    st.markdown("<p style='color: gray;'>Average by calendar month across years</p>", unsafe_allow_html=True)
    st.markdown("---")

    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

    for param in selected_parameters:
        st.markdown(f"### 🔹 {param}")
        monthly_avg = (
            analysis_df
            .groupby([analysis_df['Date'].dt.month, 'Site Name'])[param]
            .mean()
            .unstack()
            .round(2)
        )
        monthly_avg.index = monthly_avg.index.map(month_names)

        fig, ax = plt.subplots(figsize=(10, 4))
        monthly_avg.plot(kind='bar', ax=ax)
        ax.set_title(f"Monthly Averages of {param}")
        ax.set_xlabel("Month")
        ax.set_ylabel(param)
        ax.grid(True)
        ax.legend(title="Site")
        st.pyplot(fig)
        st.markdown("---")

# --- Annual Averages ---
elif selected_analysis == "Annual Averages":
    st.markdown("## 📆 Annual Averages")
    st.markdown("<p style='color: gray;'>Grouped by year and site</p>", unsafe_allow_html=True)
    st.markdown("---")

    for param in selected_parameters:
        st.markdown(f"### 🔹 {param}")
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
        sns.barplot(data=annual_avg.reset_index().melt(id_vars='Year'),
                    x='Year', y='value', hue='Site Name', ax=ax)
        ax.set_title(f"Annual Average of {param}")
        ax.set_xlabel("Year")
        ax.set_ylabel(param)
        ax.legend(title="Site")
        st.pyplot(fig)
        st.markdown("---")

# --- Correlation Matrix ---
elif selected_analysis == "Correlation Matrix":
    st.markdown("## 🔗 Correlation Matrix")
    st.markdown("<p style='color: gray;'>Pearson correlation between selected parameters</p>", unsafe_allow_html=True)
    st.markdown("---")

    if len(selected_parameters) < 2:
        st.info("Please select at least two parameters to compute correlations.")
    else:
        corr_df = analysis_df[selected_parameters].dropna().corr(method='pearson').round(2)

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap=cmap, center=0, square=True,
                    linewidths=0.5, cbar_kws={"shrink": .8}, annot_kws={"size": 10})
        
        # Highlight strong correlations
        for i in range(len(corr_df)):
            for j in range(len(corr_df.columns)):
                value = corr_df.iloc[i, j]
                if i != j and abs(value) >= 0.8:
                    ax.text(j + 0.5, i + 0.5, f"{value:.2f}", color='red', ha='center', va='center', fontweight='bold')

        st.pyplot(fig)

        # Top 5 Correlated Pairs
        corr_pairs = corr_df.where(~np.tril(np.ones(corr_df.shape)).astype(bool)).stack().reset_index()
        corr_pairs.columns = ['Parameter 1', 'Parameter 2', 'Correlation']
        corr_pairs['Abs Correlation'] = corr_pairs['Correlation'].abs()
        top_corr = corr_pairs.sort_values(by='Abs Correlation', ascending=False).head(5)

        st.markdown("### 🔝 Top 5 Correlated Pairs")
        st.dataframe(top_corr[['Parameter 1', 'Parameter 2', 'Correlation']])
# Part 5 --------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- Export Data ---
elif selected_analysis == "Export Data":
    st.markdown("## 📥 Export Processed Data")
    st.markdown("<p style='color: gray;'>Download summary statistics, monthly & annual averages per parameter or all at once</p>", unsafe_allow_html=True)
    st.markdown("---")

    if selected_parameters:
        for param in selected_parameters:
            st.markdown(f"### 🔹 {param}")

            summary = (
                analysis_df
                .groupby('Site Name')[param]
                .agg(['mean', 'median', 'std'])
                .round(2)
            )

            monthly_avg = (
                analysis_df
                .groupby([analysis_df['Date'].dt.month, 'Site Name'])[param]
                .mean()
                .unstack()
                .round(2)
            )

            annual_avg = (
                analysis_df.copy()
                .assign(Year=analysis_df['Date'].dt.year)
                .groupby(['Year', 'Site Name'])[param]
                .mean()
                .reset_index()
                .pivot(index='Year', columns='Site Name', values=param)
                .round(2)
            )

            # 📁 Download Buttons
            st.download_button(
                f"⬇️ Download Summary ({param})",
                summary.to_csv().encode('utf-8'),
                file_name=f"{param}_summary.csv"
            )
            st.download_button(
                f"⬇️ Download Monthly Average ({param})",
                monthly_avg.to_csv().encode('utf-8'),
                file_name=f"{param}_monthly_avg.csv"
            )
            st.download_button(
                f"⬇️ Download Annual Average ({param})",
                annual_avg.to_csv().encode('utf-8'),
                file_name=f"{param}_annual_avg.csv"
            )
            st.markdown("---")

        # 🗂 ZIP Export for All
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
                zf.writestr(f"{param}_summary.csv", summary.to_csv(index=True))
                zf.writestr(f"{param}_monthly_avg.csv", monthly_avg.to_csv(index=True))
                zf.writestr(f"{param}_annual_avg.csv", annual_avg.to_csv(index=True))

            filtered = df[df['Site ID'].isin(selected_sites)]
            zf.writestr("filtered_data.csv", filtered.to_csv(index=False))

        zip_buffer.seek(0)
        st.download_button(
            "⬇️ Download All as ZIP",
            data=zip_buffer,
            file_name="all_outputs.zip",
            mime="application/zip"
        )
    else:
        st.warning("Please select at least one parameter to continue.")
# Part6 ------------------------------------------------------------------------------------------------------------------------------------
# part 6-1
# --- Advanced Analysis Sidebar ---
st.sidebar.subheader("🔬 Advanced Analysis")
adv_analysis_options = [
    "Seasonal Means", "Mann-Kendall Trend", "Flow vs Parameter",
    "Water Quality Index", "Boxplot by Site", "Time-Spatial Heatmap",
    "Radar Plot", "KMeans Clustering", "Hierarchical Clustering",
    "PCA Analysis", "Non-linear Correlation", "Normality Test",
    "Rolling Mean & Variance", "Trendline Regression", "Seasonal Decomposition",
    "Autocorrelation (ACF)", "Partial Autocorrelation (PACF)",
    "Forecasting", "AI (XAI + Predictive Modeling)"
]
selected_adv_analysis = st.sidebar.radio("Select Advanced Analysis:", adv_analysis_options)
# Part 6-2
# --- Mann-Kendall Trend ---
import pymannkendall as mk
if selected_adv_analysis == "Mann-Kendall Trend":
    st.markdown("## 📉 Mann-Kendall Trend Test")
    st.markdown("<p style='color: gray;'>Detects monotonic trends over time for each parameter at each site.</p>", unsafe_allow_html=True)
    st.markdown("---")

    for param in selected_parameters:
        st.markdown(f"### 🔹 {param}")
        trend_results = []
        for site_id in selected_sites:
            site_data = analysis_df[analysis_df['Site ID'] == site_id][['Date', param]].dropna().sort_values('Date')
            if len(site_data) >= 8:
                result = mk.original_test(site_data[param])
                trend_results.append((site_id, result.trend, round(result.p, 4), round(result.z, 2)))
        if trend_results:
            trend_df = pd.DataFrame(trend_results, columns=["Site ID", "Trend", "P-value", "Z-score"])
            st.dataframe(trend_df)
        st.markdown("---")


# --- Flow vs Parameter ---
if selected_adv_analysis == "Flow vs Parameter":
    st.markdown("## 🌊 Flow vs. Water Quality Parameter")
    st.markdown("<p style='color: gray;'>Visual relationship between Flow (CFS) and selected parameters.</p>", unsafe_allow_html=True)
    st.markdown("---")

    if "Flow (CFS)" in analysis_df.columns:
        for param in selected_parameters:
            if param != "Flow (CFS)":
                st.markdown(f"### 🔹 {param}")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=analysis_df, x="Flow (CFS)", y=param, hue="Site Name", ax=ax)
                ax.set_title(f"Flow vs {param}")
                ax.set_xlabel("Flow (CFS)")
                ax.set_ylabel(param)
                ax.grid(True)
                st.pyplot(fig)
                st.markdown("---")
    else:
        st.warning("'Flow (CFS)' column not found.")


# --- Water Quality Index (WQI) ---
if selected_adv_analysis == "Water Quality Index":
    st.markdown("## 💧 Water Quality Index (WQI)")
    st.markdown("<p style='color: gray;'>A weighted composite score for water quality per site.</p>", unsafe_allow_html=True)
    st.markdown("---")

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
    st.markdown("---")


# --- Boxplot by Site ---
if selected_adv_analysis == "Boxplot by Site":
    st.markdown("## 📦 Boxplot of Parameters by Site")
    st.markdown("<p style='color: gray;'>Visualize distribution and variability of parameters per site.</p>", unsafe_allow_html=True)
    st.markdown("---")

    for param in selected_parameters:
        site_data = analysis_df[['Site Name', param]].dropna()
        if site_data.empty:
            st.warning(f"No valid data for {param}")
            continue
        st.markdown(f"### 🔹 {param}")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=site_data, x='Site Name', y=param, ax=ax)
        ax.set_title(f"{param} – Distribution by Site")
        ax.set_ylabel(param)
        ax.set_xlabel("Site")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        st.markdown("---")
# Part 6-3
# --- Radar Plot ---
if selected_adv_analysis == "Radar Plot":
    st.markdown("## 🧭 Radar Plot for Site Comparison")
    st.markdown("<p style='color: gray;'>Normalized radar chart to compare sites based on parameter means</p>", unsafe_allow_html=True)
    st.markdown("---")

    if len(selected_parameters) < 3:
        st.info("Please select at least three parameters.")
    else:
        radar_df = (
            analysis_df
            .groupby("Site Name")[selected_parameters]
            .mean()
            .dropna()
        )
        radar_df = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

        categories = selected_parameters
        num_vars = len(categories)
        angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        for site in radar_df.index:
            values = radar_df.loc[site].tolist()
            values += values[:1]
            ax.plot(angles, values, label=site)
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_yticklabels([])
        ax.set_title("Radar Plot of Normalized Site Parameters", size=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)
        st.markdown("---")


# --- PCA Analysis ---
if selected_adv_analysis == "PCA Analysis":
    st.markdown("## 🔬 Principal Component Analysis (PCA)")
    st.markdown("<p style='color: gray;'>Dimensionality reduction and feature interpretation using 2 components</p>", unsafe_allow_html=True)
    st.markdown("---")

    if len(selected_parameters) < 2:
        st.info("Please select at least two parameters.")
    else:
        pca_df = analysis_df[['Site Name'] + selected_parameters].dropna()
        if len(pca_df) >= 10:
            X = pca_df[selected_parameters]
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            pca_result = pd.DataFrame(components, columns=['PC1', 'PC2'])
            pca_result['Site Name'] = pca_df['Site Name'].values

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=pca_result, x='PC1', y='PC2', hue='Site Name', ax=ax)
            ax.set_title("PCA: Principal Component Scatter")
            st.pyplot(fig)

            st.markdown("### 📈 Explained Variance Ratio")
            st.write(pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                'Explained Variance': pca.explained_variance_ratio_.round(3)
            }))
        else:
            st.warning("Not enough data for PCA.")


# --- KMeans Clustering ---
if selected_adv_analysis == "KMeans Clustering":
    st.markdown("## 📦 KMeans Clustering")
    st.markdown("<p style='color: gray;'>Clusters based on normalized parameter space</p>", unsafe_allow_html=True)
    st.markdown("---")

    if len(selected_parameters) >= 2:
        cluster_df = analysis_df[selected_parameters].dropna()
        if len(cluster_df) > 10:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(cluster_df)
            kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(scaled)
            analysis_df.loc[cluster_df.index, 'Cluster'] = kmeans.labels_
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=analysis_df, x=selected_parameters[0], y=selected_parameters[1], hue='Cluster', palette='Set2', ax=ax)
            ax.set_title("KMeans Clustering")
            st.pyplot(fig)
        else:
            st.info("Not enough valid data for clustering.")
    else:
        st.info("Please select at least two parameters.")


# --- Hierarchical Clustering ---
if selected_adv_analysis == "Hierarchical Clustering":
    st.markdown("## 🌲 Hierarchical Clustering – Dendrogram")
    st.markdown("<p style='color: gray;'>Hierarchical grouping of sites based on parameter averages</p>", unsafe_allow_html=True)
    st.markdown("---")

    if len(selected_parameters) < 2:
        st.info("Please select at least two parameters.")
    else:
        hc_df = analysis_df.groupby('Site Name')[selected_parameters].mean().dropna()
        if len(hc_df) >= 2:
            linked = linkage(hc_df, method='ward')
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(linked, labels=hc_df.index.tolist(), ax=ax)
            ax.set_title("Hierarchical Clustering Dendrogram")
            st.pyplot(fig)
        else:
            st.warning("Not enough data for clustering.")


# --- Time-Spatial Heatmap ---
if selected_adv_analysis == "Time-Spatial Heatmap":
    st.markdown("## 🗺️ Time-Spatial Heatmap")
    st.markdown("<p style='color: gray;'>Monthly average heatmap per site</p>", unsafe_allow_html=True)
    st.markdown("---")

    for param in selected_parameters:
        st.markdown(f"### 🔹 {param}")
        heat_df = analysis_df.copy()
        heat_df['MonthYear'] = heat_df['Date'].dt.to_period('M').dt.to_timestamp()
        pivot = heat_df.pivot_table(index='MonthYear', columns='Site Name', values=param, aggfunc='mean')
        pivot.index = pivot.index.strftime('%b %Y')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot.T, cmap='YlGnBu', cbar_kws={'label': param})
        ax.set_title(f"Heatmap of {param} by Site and Month")
        ax.set_xlabel("Month-Year")
        st.pyplot(fig)
        st.markdown("---")
# Part 6-3
# --- Radar Plot ---
if selected_adv_analysis == "Radar Plot":
    st.markdown("## 🧭 Radar Plot for Site Comparison")
    st.markdown("<p style='color: gray;'>Normalized radar chart to compare sites based on parameter means</p>", unsafe_allow_html=True)
    st.markdown("---")

    if len(selected_parameters) < 3:
        st.info("Please select at least three parameters.")
    else:
        radar_df = (
            analysis_df
            .groupby("Site Name")[selected_parameters]
            .mean()
            .dropna()
        )
        radar_df = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

        categories = selected_parameters
        num_vars = len(categories)
        angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        for site in radar_df.index:
            values = radar_df.loc[site].tolist()
            values += values[:1]
            ax.plot(angles, values, label=site)
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_yticklabels([])
        ax.set_title("Radar Plot of Normalized Site Parameters", size=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)
        st.markdown("---")


# --- PCA Analysis ---
if selected_adv_analysis == "PCA Analysis":
    st.markdown("## 🔬 Principal Component Analysis (PCA)")
    st.markdown("<p style='color: gray;'>Dimensionality reduction and feature interpretation using 2 components</p>", unsafe_allow_html=True)
    st.markdown("---")

    if len(selected_parameters) < 2:
        st.info("Please select at least two parameters.")
    else:
        pca_df = analysis_df[['Site Name'] + selected_parameters].dropna()
        if len(pca_df) >= 10:
            X = pca_df[selected_parameters]
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            pca_result = pd.DataFrame(components, columns=['PC1', 'PC2'])
            pca_result['Site Name'] = pca_df['Site Name'].values

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=pca_result, x='PC1', y='PC2', hue='Site Name', ax=ax)
            ax.set_title("PCA: Principal Component Scatter")
            st.pyplot(fig)

            st.markdown("### 📈 Explained Variance Ratio")
            st.write(pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                'Explained Variance': pca.explained_variance_ratio_.round(3)
            }))
        else:
            st.warning("Not enough data for PCA.")


# --- KMeans Clustering ---
if selected_adv_analysis == "KMeans Clustering":
    st.markdown("## 📦 KMeans Clustering")
    st.markdown("<p style='color: gray;'>Clusters based on normalized parameter space</p>", unsafe_allow_html=True)
    st.markdown("---")

    if len(selected_parameters) >= 2:
        cluster_df = analysis_df[selected_parameters].dropna()
        if len(cluster_df) > 10:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(cluster_df)
            kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(scaled)
            analysis_df.loc[cluster_df.index, 'Cluster'] = kmeans.labels_
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=analysis_df, x=selected_parameters[0], y=selected_parameters[1], hue='Cluster', palette='Set2', ax=ax)
            ax.set_title("KMeans Clustering")
            st.pyplot(fig)
        else:
            st.info("Not enough valid data for clustering.")
    else:
        st.info("Please select at least two parameters.")


# --- Hierarchical Clustering ---
if selected_adv_analysis == "Hierarchical Clustering":
    st.markdown("## 🌲 Hierarchical Clustering – Dendrogram")
    st.markdown("<p style='color: gray;'>Hierarchical grouping of sites based on parameter averages</p>", unsafe_allow_html=True)
    st.markdown("---")

    if len(selected_parameters) < 2:
        st.info("Please select at least two parameters.")
    else:
        hc_df = analysis_df.groupby('Site Name')[selected_parameters].mean().dropna()
        if len(hc_df) >= 2:
            linked = linkage(hc_df, method='ward')
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(linked, labels=hc_df.index.tolist(), ax=ax)
            ax.set_title("Hierarchical Clustering Dendrogram")
            st.pyplot(fig)
        else:
            st.warning("Not enough data for clustering.")


# --- Time-Spatial Heatmap ---
if selected_adv_analysis == "Time-Spatial Heatmap":
    st.markdown("## 🗺️ Time-Spatial Heatmap")
    st.markdown("<p style='color: gray;'>Monthly average heatmap per site</p>", unsafe_allow_html=True)
    st.markdown("---")

    for param in selected_parameters:
        st.markdown(f"### 🔹 {param}")
        heat_df = analysis_df.copy()
        heat_df['MonthYear'] = heat_df['Date'].dt.to_period('M').dt.to_timestamp()
        pivot = heat_df.pivot_table(index='MonthYear', columns='Site Name', values=param, aggfunc='mean')
        pivot.index = pivot.index.strftime('%b %Y')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot.T, cmap='YlGnBu', cbar_kws={'label': param})
        ax.set_title(f"Heatmap of {param} by Site and Month")
        ax.set_xlabel("Month-Year")
        st.pyplot(fig)
        st.markdown("---")
# Part 6-4
# --- Rolling Mean & Variance ---
if selected_adv_analysis == "Rolling Mean & Variance":
    st.markdown("## 🔁 Rolling Mean and Variance")
    st.markdown("<p style='color: gray;'>Detect changes in stability and smoothness of parameter over time</p>", unsafe_allow_html=True)
    st.markdown("---")

    window_size = st.slider("Select Rolling Window Size (months):", min_value=3, max_value=24, value=6)

    for param in selected_parameters:
        st.markdown(f"### 🔹 {param}")
        for site_id in selected_sites:
            site_data = analysis_df[analysis_df['Site ID'] == site_id].copy()
            site_data = site_data.sort_values('Date')
            site_data = site_data[['Date', param]].dropna()
            site_data = site_data.set_index('Date').resample('M').mean().interpolate()

            rolling_mean = site_data[param].rolling(window=window_size).mean()
            rolling_std = site_data[param].rolling(window=window_size).std()

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(site_data.index, site_data[param], label="Original", alpha=0.4)
            ax.plot(rolling_mean, label="Rolling Mean", color='blue')
            ax.plot(rolling_std, label="Rolling Std Dev", color='red')
            ax.set_title(f"{param} - Site {site_id}")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
        st.markdown("---")


# --- Trendline Regression ---
if selected_adv_analysis == "Trendline Regression":
    st.markdown("## 📉 Trendline Regression")
    st.markdown("<p style='color: gray;'>Linear regression for trend estimation across time</p>", unsafe_allow_html=True)
    st.markdown("---")

    for param in selected_parameters:
        st.markdown(f"### 🔹 {param}")
        for site_id in selected_sites:
            site_df = analysis_df[analysis_df['Site ID'] == site_id][['Date', param]].dropna()
            site_df = site_df.sort_values('Date')
            site_df = site_df.set_index('Date').resample('M').mean().interpolate()
            site_df = site_df.reset_index()

            if len(site_df) >= 6:
                site_df['Ordinal'] = site_df['Date'].map(pd.Timestamp.toordinal)
                X = site_df[['Ordinal']]
                y = site_df[param]
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(site_df['Date'], y, label="Observed", alpha=0.6)
                ax.plot(site_df['Date'], y_pred, label=f"Trendline (slope: {model.coef_[0]:.2f})", color='red')
                ax.set_title(f"{param} Trend at Site {site_id}")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
        st.markdown("---")


# --- Normality Test (Shapiro-Wilk) ---
if selected_adv_analysis == "Normality Test":
    st.markdown("## 🧮 Normality Test (Shapiro-Wilk)")
    st.markdown("<p style='color: gray;'>Tests if data is normally distributed per site</p>", unsafe_allow_html=True)
    st.markdown("---")

    for param in selected_parameters:
        st.markdown(f"### 🔹 {param}")
        normality_results = []
        for site in analysis_df['Site Name'].unique():
            values = analysis_df[analysis_df['Site Name'] == site][param].dropna()
            if len(values) >= 3:
                stat, p_value = shapiro(values)
                result = "Normal" if p_value > 0.05 else "Not Normal"
                normality_results.append((site, round(stat, 3), round(p_value, 4), result))

        if normality_results:
            result_df = pd.DataFrame(normality_results, columns=["Site", "W Statistic", "P-value", "Interpretation"])
            styled_df = result_df.style.applymap(
                lambda val: "color: red;" if val == "Not Normal" else "color: green;", subset=["Interpretation"]
            )
            st.dataframe(styled_df)
        st.markdown("---")
# Part 6-5
# --- Seasonal Decomposition ---
if selected_adv_analysis == "Seasonal Decomposition":
    st.markdown("## 📤 Seasonal Decomposition")
    st.markdown("<p style='color: gray;'>Breaks time series into trend, seasonality, and residuals</p>", unsafe_allow_html=True)
    st.markdown("---")

    for param in selected_parameters:
        st.markdown(f"### 🔹 {param}")
        for site in analysis_df['Site Name'].unique():
            site_data = analysis_df[analysis_df['Site Name'] == site][['MonthYear', param]].dropna()
            if len(site_data) >= 24:
                ts = site_data.set_index('MonthYear').resample('M').mean()[param].interpolate()
                try:
                    decomposition = seasonal_decompose(ts, model='additive', period=12)
                    fig, axs = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
                    decomposition.observed.plot(ax=axs[0], title='Observed')
                    decomposition.trend.plot(ax=axs[1], title='Trend')
                    decomposition.seasonal.plot(ax=axs[2], title='Seasonal')
                    decomposition.resid.plot(ax=axs[3], title='Residual')
                    axs[3].set_xlabel("Date")
                    fig.suptitle(f"{param} – Decomposition for {site}", fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not decompose {param} at {site}: {e}")
            else:
                st.info(f"Not enough data for {param} at {site} (min 24 monthly points).")


# --- Autocorrelation (ACF) ---
from statsmodels.graphics.tsaplots import plot_acf
if selected_adv_analysis == "Autocorrelation (ACF)":
    st.markdown("## 🔁 Autocorrelation (ACF)")
    st.markdown("<p style='color: gray;'>Shows lag correlation with previous values</p>", unsafe_allow_html=True)
    st.markdown("---")

    for param in selected_parameters:
        st.markdown(f"### 🔹 {param}")
        for site_id in selected_sites:
            site_data = analysis_df[analysis_df['Site ID'] == site_id][['Date', param]].dropna().sort_values('Date')
            if len(site_data) > 20:
                fig, ax = plt.subplots(figsize=(8, 4))
                plot_acf(site_data[param], lags=20, ax=ax, title=f"{param} - ACF (Site ID: {site_id})")
                st.pyplot(fig)
            else:
                st.info(f"Not enough data for ACF at Site ID {site_id}")
        st.markdown("---")


# --- Partial Autocorrelation (PACF) ---
from statsmodels.graphics.tsaplots import plot_pacf
if selected_adv_analysis == "Partial Autocorrelation (PACF)":
    st.markdown("## 🔁 Partial Autocorrelation (PACF)")
    st.markdown("<p style='color: gray;'>Highlights direct lag influence after removing previous lags</p>", unsafe_allow_html=True)
    st.markdown("---")

    for param in selected_parameters:
        st.markdown(f"### 🔹 {param}")
        for site_id in selected_sites:
            site_df = analysis_df[analysis_df['Site ID'] == site_id]
            series = site_df[['Date', param]].dropna().sort_values('Date')
            values = series[param].values

            if len(values) > 20 and np.std(values) > 0:
                try:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    plot_pacf(values, ax=ax, lags=20, method='ywm')
                    site_name = site_df['Site Name'].iloc[0]
                    ax.set_title(f"{param} – PACF at {site_name}")
                    st.pyplot(fig)
                except ValueError as e:
                    st.warning(f"PACF error for {param} at Site {site_id}: {e}")
            else:
                st.info(f"Not enough data for PACF at Site ID {site_id}")
        st.markdown("---")


# --- Forecasting with Prophet ---
if selected_adv_analysis == "Forecasting":
    st.markdown("## 🔮 Time Series Forecasting (Prophet)")
    st.markdown("<p style='color: gray;'>Forecasts the next 12 months using Prophet model</p>", unsafe_allow_html=True)
    st.markdown("---")

    for param in selected_parameters:
        st.markdown(f"### 🔹 Forecast for {param}")
        for site in selected_sites:
            site_df = analysis_df[(analysis_df["Site ID"] == site)][['Date', param]].dropna()
            if len(site_df) < 20:
                st.info(f"Not enough data to forecast for Site ID {site}")
                continue
            df_prophet = site_df.rename(columns={"Date": "ds", param: "y"})
            model = Prophet()
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=12, freq='M')
            forecast = model.predict(future)
            fig = model.plot(forecast)
            st.pyplot(fig)
        st.markdown("---")


# --- AI + XAI (SHAP + Predictive Modeling) ---
if selected_adv_analysis == "AI (XAI + Predictive Modeling)":
    st.markdown("## 🤖 AI: XAI + Predictive Modeling")
    st.markdown("<p style='color: gray;'>Train ML model and explain predictions using SHAP</p>", unsafe_allow_html=True)
    st.markdown("---")

    selected_inputs = st.multiselect("Select Input Parameters:", df.select_dtypes(include='number').columns)
    target_parameter = st.selectbox("Select Target Parameter:", df.select_dtypes(include='number').columns)
    model_type = st.selectbox("Select Model Type:", ["Linear Regression", "RandomForest", "XGBoost"])

    if st.button("Run Model Training"):
        if not selected_inputs or not target_parameter:
            st.warning("Please select inputs and target.")
        else:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            X = df[['Date'] + selected_inputs].dropna(subset=selected_inputs)
            y = df[['Date', target_parameter]].dropna(subset=[target_parameter])
            common_dates = set(X['Date']).intersection(set(y['Date']))
            X = X[X['Date'].isin(common_dates)].set_index('Date').sort_index()
            y = y[y['Date'].isin(common_dates)].set_index('Date').sort_index()
            if len(X) == len(y) and len(X) > 0:
                y = y.loc[X.index]
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "RandomForest":
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor()
                else:
                    model = xgb.XGBRegressor(objective='reg:squarederror', use_label_encoder=False, eval_metric='rmse')

                model.fit(X, y[target_parameter])
                st.success(f"{model_type} trained successfully.")
                st.session_state['trained_model'] = model
                st.session_state['X_train'] = X
                st.session_state['y_train'] = y[target_parameter]
                st.session_state['selected_inputs'] = selected_inputs
                st.session_state['target_parameter'] = target_parameter
            else:
                st.warning("No valid data after filtering for common dates.")

    # SHAP Analysis
    if st.button("Run SHAP Analysis"):
        if 'trained_model' in st.session_state:
            model = st.session_state['trained_model']
            X = st.session_state['X_train']
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

            st.markdown("### SHAP Summary Plot")
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot()

            st.markdown("### SHAP Decision Plot")
            shap.decision_plot(explainer.expected_value, shap_values.values, X.columns)
        else:
            st.warning("Please train the model first.")

    # Predictive Evaluation
    if st.button("Run Predictive Modeling"):
        if 'trained_model' in st.session_state:
            model = st.session_state['trained_model']
            X = st.session_state['X_train']
            y = st.session_state['y_train']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)

            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"R² Score: {r2:.4f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)
        else:
            st.warning("Please train the model first.")

