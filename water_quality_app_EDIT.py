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
    with tabs[0]:
        st.subheader("Summary Statistics")
        for param in selected_parameters:
            st.markdown(f"### {param}")
            summary = (
                analysis_df
                .groupby('Site Name')[param]
                .agg(['mean', 'median', 'std', 'min', 'max', 'count'])
                .round(2)
                .rename(columns={
                    'mean': 'Mean', 'median': 'Median', 'std': 'Std Dev',
                    'min': 'Min', 'max': 'Max', 'count': 'Count'
                })
            )
            st.dataframe(summary)
    with tabs[1]:
        st.subheader("Monthly Averages (Across Years)")
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
            5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }

        for param in selected_parameters:
            st.markdown(f"### {param}")
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

    with tabs[2]:
        st.subheader("Annual Averages")
        for param in selected_parameters:
            st.markdown(f"### {param}")
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
    "Seasonal Means", "Mann-Kendall Trend", "Flow vs Parameter",
    "Water Quality Index", "KMeans Clustering", "Time-Spatial Heatmap",
    "Boxplot by Site", "Normality Test", "Seasonal Decomposition",
    "Non-linear Correlation", "Rolling Mean & Variance", "Trendline Regression",
    "PCA Analysis", "Hierarchical Clustering", "Radar Plot",
    "Autocorrelation (ACF)", "Partial Autocorrelation (PACF)", "Forecasting"
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
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=analysis_df, x="Flow (CFS)", y=param, hue="Site Name", ax=ax)
                ax.set_title(f"Flow vs {param}")
                ax.set_xlabel("Flow (CFS)")
                ax.set_ylabel(param)
                from matplotlib.ticker import MaxNLocator
                ax.tick_params(axis='x', labelrotation=45)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))  # تعداد برچسب‌ها رو محدود کن
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))  # فرمت عددی
                ax.grid(True)
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
        pivot.index = pivot.index.strftime('%b %Y')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot.T, cmap='YlGnBu', cbar_kws={'label': param})
        ax.set_title(f"Heatmap of {param} by Site and Month")
        ax.set_xlabel("Month-Year")
        st.pyplot(fig)
with adv_tabs[6]:
    st.subheader("Boxplot of Parameters by Site")
    for param in selected_parameters:
        site_data = analysis_df[['Site Name', param]].dropna()
        if site_data.empty:
            st.warning(f"No valid data for {param}")
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=site_data, x='Site Name', y=param, ax=ax)
        ax.set_title(f"{param} – Distribution by Site")
        ax.set_ylabel(param)
        ax.set_xlabel("Site")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)        
st.markdown("---")
st.caption("Data Source: CRP Monitoring at Cypress Creek")
from scipy.stats import shapiro

with adv_tabs[7]:
    st.subheader("Shapiro-Wilk Normality Test (per Site)")
    for param in selected_parameters:
        st.markdown(f"### {param}")
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
        else:
            st.info(f"No valid data for {param}.")
from statsmodels.tsa.seasonal import seasonal_decompose

with adv_tabs[8]:
    st.subheader("Seasonal-Trend Decomposition")
    for param in selected_parameters:
        st.markdown(f"### {param}")
        for site in analysis_df['Site Name'].unique():
            site_data = analysis_df[analysis_df['Site Name'] == site][['MonthYear', param]].dropna()
            if len(site_data) >= 24:  # نیاز به حداقل داده برای تجزیه فصلی
                ts = site_data.set_index('MonthYear').resample('M').mean()[param]
                ts = ts.interpolate()  # پر کردن مقادیر گمشده
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

with adv_tabs[9]:
    st.subheader("Spearman & Kendall Correlation Matrix")
    if len(selected_parameters) < 2:
        st.info("Please select at least two parameters.")
    else:
        for method in ['spearman', 'kendall']:
            st.markdown(f"### {method.title()} Correlation")
            try:
                corr_df = analysis_df[selected_parameters].corr(method=method).round(2)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                            linewidths=0.5, square=True, cbar_kws={"shrink": .8})
                ax.set_title(f"{method.title()} Correlation Matrix")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not calculate {method} correlation: {e}")
with adv_tabs[10]:
    st.subheader("Rolling Mean and Variance")
    window_size = st.slider("Select Rolling Window Size (months):", min_value=3, max_value=24, value=6)
    
    for param in selected_parameters:
        st.markdown(f"### {param}")
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
            ax.set_xlabel("Date")
            ax.set_ylabel(param)
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
from sklearn.linear_model import LinearRegression

with adv_tabs[11]:
    st.subheader("Trendline Regression Analysis")
    for param in selected_parameters:
        st.markdown(f"### {param}")
        for site_id in selected_sites:
            site_df = analysis_df[analysis_df['Site ID'] == site_id][['Date', param]].dropna()
            site_df = site_df.sort_values('Date')
            site_df = site_df.set_index('Date').resample('M').mean().interpolate()
            site_df = site_df.reset_index().dropna()

            if len(site_df) >= 6:
                # تبدیل تاریخ به عدد
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
                ax.set_xlabel("Date")
                ax.set_ylabel(param)
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

with adv_tabs[12]:
    st.subheader("Principal Component Analysis (PCA)")
    if len(selected_parameters) < 2:
        st.info("Please select at least two parameters for PCA.")
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
            ax.set_title("PCA Scatter Plot")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax.grid(True)
            st.pyplot(fig)

            st.markdown("**Explained Variance Ratio:**")
            st.write(pd.DataFrame({
                'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                'Variance Explained': pca.explained_variance_ratio_.round(3)
            }))
        else:
            st.warning("Not enough data for PCA analysis.")       
from scipy.cluster.hierarchy import linkage, dendrogram

with adv_tabs[13]:
    st.subheader("Hierarchical Clustering – Dendrogram")
    if len(selected_parameters) < 2:
        st.info("Please select at least two parameters.")
    else:
        hc_df = analysis_df.groupby('Site Name')[selected_parameters].mean().dropna()
        if len(hc_df) >= 2:
            linked = linkage(hc_df, method='ward')
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(linked, labels=hc_df.index.tolist(), ax=ax)
            ax.set_title("Hierarchical Clustering of Sites")
            ax.set_ylabel("Distance")
            st.pyplot(fig)
        else:
            st.warning("Not enough data or sites for clustering.")

with adv_tabs[14]:
    st.subheader("Radar Plot for Site Comparison")
    if len(selected_parameters) < 3:
        st.info("Please select at least three parameters for meaningful radar plot.")
    else:
        radar_df = (
            analysis_df
            .groupby("Site Name")[selected_parameters]
            .mean()
            .dropna()
        )

        # Normalize to [0, 1]
        radar_df = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

        categories = selected_parameters
        num_vars = len(categories)

        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        for site in radar_df.index:
            values = radar_df.loc[site].tolist()
            values += values[:1]
            ax.plot(angles, values, label=site)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_yticklabels([])
        ax.set_title("Radar Plot of Site-Averaged Parameters", size=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)
from statsmodels.graphics.tsaplots import plot_acf

with adv_tabs[15]:
    st.subheader("Autocorrelation Function (ACF) Plot")
    if selected_parameters:
        for param in selected_parameters:
            st.markdown(f"**{param}**")
            for site_id in selected_sites:
                site_data = analysis_df[analysis_df['Site ID'] == site_id][['Date', param]].dropna().sort_values('Date')
                if len(site_data) > 20:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    plot_acf(site_data[param], lags=20, ax=ax, title=f"{param} - ACF (Site ID: {site_id})")
                    st.pyplot(fig)
                else:
                    st.info(f"Not enough data for ACF plot at Site ID {site_id}")
    else:
        st.warning("Please select at least one parameter.")  
# --- Partial Autocorrelation (PACF) ---
from statsmodels.graphics.tsaplots import plot_pacf

with adv_tabs[16]:
    st.subheader("Partial Autocorrelation (PACF)")

    for param in selected_parameters:
        st.markdown(f"**PACF for {param}**")
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
                st.info(f"Not enough variability or data points for {param} at Site ID {site_id}")
from prophet import Prophet

with adv_tabs[17]:
    st.subheader("Time Series Forecasting (Prophet)")

    if selected_parameters:
        for param in selected_parameters:
            st.markdown(f"### Forecasting: {param}")
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
    else:
        st.info("Please select at least one parameter.")
        
        
