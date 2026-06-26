import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Startup Funding Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS / STYLING
# ----------------------------

st.markdown("""
<style>
    /* Custom Main Font & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* KPI Card Styling */
    .kpi-container {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 20px;
    }
    .kpi-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .kpi-subtitle {
        color: #10b981;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 8px;
    }
    .kpi-subtitle.negative {
        color: #ef4444;
    }
    
    /* Insight Card Styling */
    .insight-card {
        background-color: rgba(30, 41, 59, 0.45);
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.03);
        border-right: 1px solid rgba(255, 255, 255, 0.03);
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }
    .insight-title {
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 6px;
        font-size: 1.05rem;
    }
    .insight-text {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# PATHS
# ----------------------------

BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "data" / "startup_funding.csv"
FORECAST_PATH = BASE_DIR / "outputs" / "funding_forecast.csv"
METRICS_PATH = BASE_DIR / "outputs" / "model_metrics.csv"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "outputs" / "feature_importance.csv"

# ----------------------------
# LOAD DATA (WITH CACHING)
# ----------------------------

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(DATA_PATH)
        df.rename(columns={
            "Date dd/mm/yyyy": "Date",
            "Amount in USD": "AmountInUSD",
            "Industry Vertical": "IndustryVertical",
            "City  Location": "CityLocation",
            "Startup Name": "StartupName",
            "Investors Name": "InvestorsName"
        }, inplace=True)

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(
                df["Date"],
                dayfirst=True,
                errors="coerce"
            )

        if "AmountInUSD" in df.columns:
            def clean_amount(val):
                if pd.isna(val):
                    return 0.0
                val_str = str(val).lower().strip()
                if any(x in val_str for x in ["undisclosed", "unknown", "nan"]):
                    return 0.0
                import re
                val_str = re.sub(r"[,\$\+ ]", "", val_str)
                try:
                    return float(val_str)
                except ValueError:
                    return 0.0
            df["AmountInUSD"] = df["AmountInUSD"].apply(clean_amount)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_forecast():
    if FORECAST_PATH.exists():
        try:
            return pd.read_csv(FORECAST_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        try:
            metrics = pd.read_csv(METRICS_PATH)
            metrics.rename(
                columns={
                    "Unnamed: 0": "Model"
                },
                inplace=True
            )
            return metrics
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data
def load_feature_importance():
    if FEATURE_IMPORTANCE_PATH.exists():
        try:
            return pd.read_csv(FEATURE_IMPORTANCE_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


# Load all datasets
df = load_data()
forecast = load_forecast()
metrics = load_metrics()
feature_importance = load_feature_importance()

# Helper function for rendering consistent KPI cards
def kpi_card(title, value, subtitle=None, is_negative=False):
    sub_class = "kpi-subtitle negative" if is_negative else "kpi-subtitle"
    sub_html = f'<div class="{sub_class}">{subtitle}</div>' if subtitle else ''
    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)

# ----------------------------
# SIDEBAR NAVIGATION
# ----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Executive Overview",
        "Dataset Explorer",
        "Funding Trends",
        "Industry Analysis",
        "City Analysis",
        "Investor Analytics",
        "Model Comparison",
        "Model Explainability",
        "Future Forecasts",
        "AI Generated Insights",
        "System Diagnostics",
        "About Project"
    ]
)

# ============================
# EXECUTIVE OVERVIEW
# ============================

if page == "Executive Overview":
    st.title("🚀 Executive Overview")

    if not df.empty:
        # Calculations
        total_records = len(df)
        total_funding = df["AmountInUSD"].sum() if "AmountInUSD" in df.columns else 0.0
        unique_startups = df["StartupName"].nunique() if "StartupName" in df.columns else 0
        unique_cities = df["CityLocation"].nunique() if "CityLocation" in df.columns else 0
        unique_industries = df["IndustryVertical"].nunique() if "IndustryVertical" in df.columns else 0
        avg_funding = df["AmountInUSD"].mean() if "AmountInUSD" in df.columns else 0.0
        max_funding = df["AmountInUSD"].max() if "AmountInUSD" in df.columns else 0.0

        best_model = "Unknown"
        if not metrics.empty and "RMSE" in metrics.columns and "Model" in metrics.columns:
            try:
                best_model = metrics.loc[metrics["RMSE"].idxmin()]["Model"]
            except Exception:
                pass

        # Calculate MoM or YoY Growth rate if Date and Amount exist
        growth_subtitle = "Insufficient chronological data"
        growth_val = "N/A"
        if "Date" in df.columns and "AmountInUSD" in df.columns:
            try:
                yearly = df.groupby(df["Date"].dt.year)["AmountInUSD"].sum()
                if len(yearly) > 1:
                    first_yr, last_yr = yearly.index.min(), yearly.index.max()
                    first_val, last_val = yearly.loc[first_yr], yearly.loc[last_yr]
                    yoy = ((last_val - first_val) / (first_val + 1e-5)) * 100
                    growth_val = f"{yoy:+.1f}%"
                    growth_subtitle = f"Overall YoY Growth ({first_yr} vs {last_yr})"
            except Exception:
                pass

        # KPI Layout
        col1, col2, col3 = st.columns(3)
        with col1:
            kpi_card("Total Funding Raised", f"${total_funding:,.0f}", growth_subtitle)
        with col2:
            kpi_card("Total Startups Funded", f"{unique_startups:,}", "Unique companies")
        with col3:
            kpi_card("Best Performing Model", best_model, "Selected dynamically by RMSE")

        col4, col5, col6 = st.columns(3)
        with col4:
            kpi_card("Average Deal Size", f"${avg_funding:,.0f}", "Mean funding amount")
        with col5:
            kpi_card("Highest Funding Round", f"${max_funding:,.0f}", "Maximum single transaction")
        with col6:
            kpi_card("Market Diversity", f"{unique_cities} Cities | {unique_industries} Sectors", "Geographic & vertical footprint")

        st.markdown("---")
        st.subheader("Executive Summary Insights")
        
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">Platform Summary & Core Findings</div>
            <div class="insight-text">
                This dashboard tracks historical funding patterns to produce a forward-looking forecast of startup investments. 
                Our platform aggregates <b>{total_records:,}</b> transactions across <b>{unique_startups:,}</b> distinct startup companies in India. 
                Bengaluru, Mumbai, and Delhi-NCR continue to act as key operational centers, while technology, e-commerce, and fintech lead the industry sectors. 
                The champion forecasting model identified by the system is <b>{best_model}</b>, optimized to predict investment volumes 12 months in advance.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("The dataset is empty or could not be loaded.")

# ============================
# DATASET EXPLORER
# ============================

elif page == "Dataset Explorer":
    st.title("🔍 Dataset Explorer")

    if not df.empty:
        st.subheader("Advanced Search & Filter Panel")
        
        # Filtering fields
        col1, col2, col3 = st.columns(3)
        with col1:
            search_query = st.text_input("Search (Startup Name / Investors Name)", "")
        with col2:
            all_cities = sorted(df["CityLocation"].dropna().unique().tolist()) if "CityLocation" in df.columns else []
            selected_cities = st.multiselect("Filter by City Location", all_cities)
        with col3:
            all_industries = sorted(df["IndustryVertical"].dropna().unique().tolist()) if "IndustryVertical" in df.columns else []
            selected_industries = st.multiselect("Filter by Industry Vertical", all_industries)

        col4, col5 = st.columns(2)
        with col4:
            if "Date" in df.columns and not df["Date"].isna().all():
                min_date = df["Date"].min().to_pydatetime()
                max_date = df["Date"].max().to_pydatetime()
                selected_dates = st.slider("Filter by Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
            else:
                selected_dates = None
        with col5:
            all_cols = df.columns.tolist()
            selected_cols = st.multiselect("Select Columns to Display", all_cols, default=all_cols)

        # Apply filters
        filtered_df = df.copy()

        if search_query:
            q = search_query.lower()
            startup_match = filtered_df["StartupName"].str.lower().str.contains(q, na=False) if "StartupName" in filtered_df.columns else False
            investor_match = filtered_df["InvestorsName"].str.lower().str.contains(q, na=False) if "InvestorsName" in filtered_df.columns else False
            filtered_df = filtered_df[startup_match | investor_match]

        if selected_cities:
            filtered_df = filtered_df[filtered_df["CityLocation"].isin(selected_cities)]

        if selected_industries:
            filtered_df = filtered_df[filtered_df["IndustryVertical"].isin(selected_industries)]

        if selected_dates and "Date" in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df["Date"] >= selected_dates[0]) & (filtered_df["Date"] <= selected_dates[1])]

        if selected_cols:
            display_df = filtered_df[selected_cols]
        else:
            display_df = filtered_df

        # Diagnostics / Summary
        st.write(f"Displaying **{len(display_df):,}** of **{len(df):,}** records.")

        # Display Dataframe
        st.dataframe(
            display_df,
            width='stretch',
            height=700
        )

        # Download option
        try:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Filtered Dataset (CSV)", csv_data, "filtered_startup_funding.csv", "text/csv")
        except Exception as e:
            st.error(f"Error preparing download: {e}")
    else:
        st.warning("The dataset is empty or could not be loaded.")

# ============================
# FUNDING TREND ANALYTICS
# ============================

elif page == "Funding Trends":
    st.title("📈 Funding Trend Analytics")

    if not df.empty:
        if "Date" in df.columns and "AmountInUSD" in df.columns:
            try:
                df_trends = df.dropna(subset=["Date", "AmountInUSD"])
                df_trends = df_trends.sort_values(by="Date")

                if not df_trends.empty:
                    # A. Monthly Funding Trend
                    monthly = df_trends.groupby(pd.Grouper(key="Date", freq="ME"))["AmountInUSD"].sum().reset_index()
                    fig_monthly = px.line(monthly, x="Date", y="AmountInUSD", title="Monthly Funding Trend", labels={"AmountInUSD": "Total Funding ($)"})
                    st.plotly_chart(fig_monthly, width='stretch')

                    # B. Quarterly Funding Trend
                    quarterly = df_trends.groupby(pd.Grouper(key="Date", freq="QE"))["AmountInUSD"].sum().reset_index()
                    fig_quarterly = px.bar(quarterly, x="Date", y="AmountInUSD", title="Quarterly Funding Trend", labels={"AmountInUSD": "Total Funding ($)"})
                    st.plotly_chart(fig_quarterly, width='stretch')

                    # C. Yearly Funding Trend
                    yearly = df_trends.groupby(pd.Grouper(key="Date", freq="YE"))["AmountInUSD"].sum().reset_index()
                    yearly["Year"] = yearly["Date"].dt.year.astype(str)
                    fig_yearly = px.bar(yearly, x="Year", y="AmountInUSD", title="Yearly Funding Trend", labels={"AmountInUSD": "Total Funding ($)"})
                    st.plotly_chart(fig_yearly, width='stretch')

                    # D. Cumulative Funding Growth
                    df_cum = df_trends.copy()
                    df_cum["CumulativeFunding"] = df_cum["AmountInUSD"].cumsum()
                    fig_cum = px.line(df_cum, x="Date", y="CumulativeFunding", title="Cumulative Funding Growth", labels={"CumulativeFunding": "Cumulative Funding ($)"})
                    st.plotly_chart(fig_cum, width='stretch')

                    # E & F. Moving Averages
                    monthly["MA_3"] = monthly["AmountInUSD"].rolling(3).mean()
                    monthly["MA_6"] = monthly["AmountInUSD"].rolling(6).mean()
                    fig_ma = go.Figure()
                    fig_ma.add_trace(go.Scatter(x=monthly["Date"], y=monthly["AmountInUSD"], name="Actual Monthly"))
                    fig_ma.add_trace(go.Scatter(x=monthly["Date"], y=monthly["MA_3"], name="3-Month Moving Average", line=dict(dash='dash')))
                    fig_ma.add_trace(go.Scatter(x=monthly["Date"], y=monthly["MA_6"], name="6-Month Moving Average", line=dict(dash='dot')))
                    fig_ma.update_layout(title="Monthly Funding with 3-Month and 6-Month Moving Averages", xaxis_title="Date", yaxis_title="Funding ($)")
                    st.plotly_chart(fig_ma, width='stretch')

                    # G. Funding Distribution Histogram
                    fig_hist = px.histogram(df_trends, x="AmountInUSD", nbins=50, title="Funding Transaction Size Distribution", labels={"AmountInUSD": "Deal Value ($)"})
                    fig_hist.update_xaxes(type="log", title="Deal Value (Log Scale)")
                    st.plotly_chart(fig_hist, width='stretch')

                    # H. Box Plot for Outliers
                    fig_box = px.box(df_trends, y="AmountInUSD", title="Outlier Detection Box Plot", labels={"AmountInUSD": "Deal Value ($)"})
                    fig_box.update_yaxes(type="log", title="Deal Value (Log Scale)")
                    st.plotly_chart(fig_box, width='stretch')

                    # I. Heatmap: Month vs Year Funding Amount
                    try:
                        df_heatmap = df_trends.copy()
                        df_heatmap["Year"] = df_heatmap["Date"].dt.year
                        df_heatmap["Month"] = df_heatmap["Date"].dt.strftime("%m-%B")
                        pivot = df_heatmap.pivot_table(index="Year", columns="Month", values="AmountInUSD", aggfunc="sum").fillna(0)
                        fig_heat = px.imshow(pivot, title="Funding Heatmap: Year vs Month", labels=dict(x="Month", y="Year", color="Total Funding ($)"))
                        st.plotly_chart(fig_heat, width='stretch')
                    except Exception as he:
                        st.error(f"Error plotting Heatmap: {he}")

                    # J. Seasonal Trend Analysis
                    df_seasonal = df_trends.copy()
                    df_seasonal["MonthName"] = df_seasonal["Date"].dt.strftime("%B")
                    months_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
                    seasonal_avg = df_seasonal.groupby("MonthName")["AmountInUSD"].mean().reindex(months_order).reset_index()
                    fig_seasonal = px.line(seasonal_avg, x="MonthName", y="AmountInUSD", title="Seasonal Trends (Average Transaction Size by Calendar Month)", markers=True)
                    st.plotly_chart(fig_seasonal, width='stretch')

                else:
                    st.warning("No records remain for trend plotting after cleaning null values.")
            except Exception as e:
                st.error(f"Error rendering trends: {e}")
        else:
            st.warning("Required trend columns ('Date', 'AmountInUSD') are missing from the dataset.")
    else:
        st.warning("The dataset is empty or could not be loaded.")

# ============================
# INDUSTRY ANALYTICS
# ============================

elif page == "Industry Analysis":
    st.title("🏭 Industry Analytics")

    if not df.empty:
        if "IndustryVertical" in df.columns and "AmountInUSD" in df.columns:
            try:
                # Aggregate industry data
                industry_grp = df.groupby("IndustryVertical")["AmountInUSD"].agg(["sum", "count", "mean"]).reset_index()
                industry_grp.columns = ["IndustryVertical", "TotalFunding", "DealsCount", "AverageFunding"]
                
                # Sort by funding
                top_industries = industry_grp.sort_values(by="TotalFunding", ascending=False)
                
                # Top 20 Industries Bar
                top_20 = top_industries.head(20)
                fig_ind_bar = px.bar(top_20, x="IndustryVertical", y="TotalFunding", title="Top 20 Industries by Total Funding", labels={"TotalFunding": "Total Funding ($)"})
                st.plotly_chart(fig_ind_bar, width='stretch')

                # Industry Share Pie (Top 10 + Other)
                top_10 = top_industries.head(10).copy()
                other_val = top_industries.iloc[10:]["TotalFunding"].sum() if len(top_industries) > 10 else 0.0
                if other_val > 0:
                    other_row = pd.DataFrame([{"IndustryVertical": "Other Industries", "TotalFunding": other_val, "DealsCount": 0, "AverageFunding": 0.0}])
                    top_10 = pd.concat([top_10, other_row], ignore_index=True)
                fig_pie = px.pie(top_10, values="TotalFunding", names="IndustryVertical", title="Industry Funding Share (Top 10 & Other)")
                st.plotly_chart(fig_pie, width='stretch')

                # Industry Growth Trends (Top 5 Industries YoY)
                if "Date" in df.columns:
                    top_5_names = top_industries.head(5)["IndustryVertical"].tolist()
                    df_top_5 = df[df["IndustryVertical"].isin(top_5_names)].copy()
                    df_top_5["Year"] = df_top_5["Date"].dt.year
                    # Pivot table of Year vs Top 5 Industries
                    pivot_ind_growth = df_top_5.groupby(["Year", "IndustryVertical"])["AmountInUSD"].sum().reset_index()
                    fig_ind_growth = px.line(pivot_ind_growth, x="Year", y="AmountInUSD", color="IndustryVertical", title="Yearly Funding Growth for Top 5 Sectors", markers=True)
                    st.plotly_chart(fig_ind_growth, width='stretch')

                # Industry Funding Distribution Box plot (Top 10 Industries)
                top_10_names = top_industries.head(10)["IndustryVertical"].tolist()
                df_top_10 = df[df["IndustryVertical"].isin(top_10_names)].copy()
                fig_dist = px.box(df_top_10, x="IndustryVertical", y="AmountInUSD", color="IndustryVertical", title="Transaction Size Distribution across Top 10 Industries")
                fig_dist.update_yaxes(type="log", title="Deal Size ($) - Log Scale")
                st.plotly_chart(fig_dist, width='stretch')

            except Exception as e:
                st.error(f"Error rendering industry analytics: {e}")
        else:
            st.warning("Required columns ('IndustryVertical', 'AmountInUSD') are missing.")
    else:
        st.warning("The dataset is empty or could not be loaded.")

# ============================
# CITY ANALYTICS
# ============================

elif page == "City Analysis":
    st.title("🏙️ City Analytics")

    if not df.empty:
        if "CityLocation" in df.columns and "AmountInUSD" in df.columns:
            try:
                # Aggregate City Data
                city_grp = df.groupby("CityLocation")["AmountInUSD"].agg(["sum", "count", "mean"]).reset_index()
                city_grp.columns = ["CityLocation", "TotalFunding", "DealsCount", "AverageFunding"]
                
                # Sort
                top_cities = city_grp.sort_values(by="TotalFunding", ascending=False)
                
                # Top Cities Bar Chart
                fig_city_bar = px.bar(top_cities.head(15), x="CityLocation", y="TotalFunding", title="Top 15 Startup Cities by Total Funding", labels={"TotalFunding": "Total Funding ($)"})
                st.plotly_chart(fig_city_bar, width='stretch')

                # Startup count by city
                top_deals_city = city_grp.sort_values(by="DealsCount", ascending=False).head(15)
                fig_deals_bar = px.bar(top_deals_city, x="CityLocation", y="DealsCount", title="Top 15 Most Active Startup Hubs (Deal Count)", labels={"DealsCount": "Number of Deals"})
                st.plotly_chart(fig_deals_bar, width='stretch')

                # City Ranking Table
                st.subheader("City Ranking Matrix")
                city_ranking = top_cities.copy().reset_index(drop=True)
                city_ranking.index = city_ranking.index + 1
                st.dataframe(city_ranking, width='stretch')

                # City Growth Trends (Top 5 Cities YoY)
                if "Date" in df.columns:
                    top_5_cities = top_cities.head(5)["CityLocation"].tolist()
                    df_top_5_city = df[df["CityLocation"].isin(top_5_cities)].copy()
                    df_top_5_city["Year"] = df_top_5_city["Date"].dt.year
                    pivot_city_growth = df_top_5_city.groupby(["Year", "CityLocation"])["AmountInUSD"].sum().reset_index()
                    fig_city_growth = px.line(pivot_city_growth, x="Year", y="AmountInUSD", color="CityLocation", title="Yearly Funding Growth for Top 5 Startup Cities", markers=True)
                    st.plotly_chart(fig_city_growth, width='stretch')

            except Exception as e:
                st.error(f"Error rendering city analytics: {e}")
        else:
            st.warning("Required columns ('CityLocation', 'AmountInUSD') are missing.")
    else:
        st.warning("The dataset is empty or could not be loaded.")

# ============================
# INVESTOR ANALYTICS
# ============================

elif page == "Investor Analytics":
    st.title("🤝 Investor Analytics")

    if not df.empty:
        if "InvestorsName" in df.columns and "AmountInUSD" in df.columns:
            try:
                # Separate list of individual investors since cell values can contain comma separation
                investors_series = df["InvestorsName"].dropna().str.split(",")
                investors_flat = [inv.strip() for sublist in investors_series for inv in sublist if inv.strip()]
                investors_df = pd.DataFrame(investors_flat, columns=["Investor"])
                
                # Top Investors by Deals Count
                top_investors = investors_df["Investor"].value_counts().reset_index()
                top_investors.columns = ["Investor", "DealsCount"]
                top_15_investors = top_investors.head(15)

                fig_deals = px.bar(top_15_investors, x="Investor", y="DealsCount", title="Top 15 Active Investors (Deal Count)", labels={"DealsCount": "Number of Deals"})
                st.plotly_chart(fig_deals, width='stretch')

                # Investor Funding Allocation
                # Since splitting amounts associated with multiple investors dynamically is mathematically arbitrary, 
                # we allocate the total deal size to each participating investor as standard practice.
                investor_deal_pairs = []
                for _, row in df.dropna(subset=["InvestorsName", "AmountInUSD"]).iterrows():
                    invs = str(row["InvestorsName"]).split(",")
                    for i in invs:
                        i_clean = i.strip()
                        if i_clean:
                            investor_deal_pairs.append((i_clean, row["AmountInUSD"]))
                
                pair_df = pd.DataFrame(investor_deal_pairs, columns=["Investor", "AmountInUSD"])
                inv_funding = pair_df.groupby("Investor")["AmountInUSD"].agg(["sum", "mean"]).reset_index()
                inv_funding.columns = ["Investor", "TotalFundingAssociated", "AverageDealSize"]
                
                top_funding_inv = inv_funding.sort_values(by="TotalFundingAssociated", ascending=False).head(15)
                fig_inv_fund = px.bar(top_funding_inv, x="Investor", y="TotalFundingAssociated", title="Top 15 Investors by Total Capital Associated", labels={"TotalFundingAssociated": "Associated Funding ($)"})
                st.plotly_chart(fig_inv_fund, width='stretch')

                # Distribution of Deal Sizes for Top 5 Investors
                top_5_funding_names = top_funding_inv.head(5)["Investor"].tolist()
                pair_top_5 = pair_df[pair_df["Investor"].isin(top_5_funding_names)]
                fig_inv_dist = px.box(pair_top_5, x="Investor", y="AmountInUSD", color="Investor", title="Deal Value Distribution for Top 5 Capital Investors")
                fig_inv_dist.update_yaxes(type="log", title="Deal Size ($) - Log Scale")
                st.plotly_chart(fig_inv_dist, width='stretch')

            except Exception as e:
                st.error(f"Error rendering investor analytics: {e}")
        else:
            st.warning("Required columns ('InvestorsName', 'AmountInUSD') are missing.")
    else:
        st.warning("The dataset is empty or could not be loaded.")

# ============================
# MODEL COMPARISON
# ============================

elif page == "Model Comparison":
    st.title("🤖 Model Comparison")

    if not metrics.empty:
        req_cols = ["Model", "RMSE", "MAE", "R2", "MAPE"]
        existing_cols = [c for c in req_cols if c in metrics.columns]

        if "Model" in metrics.columns:
            st.subheader("Model Performance Metrics")
            
            # Sort metrics to highlight model performance ranking
            rank_df = metrics.copy()
            if "RMSE" in rank_df.columns:
                rank_df = rank_df.sort_values(by="RMSE")
            st.dataframe(rank_df[existing_cols], width='stretch')

            # Dynamically determine and display the best model
            if "RMSE" in metrics.columns:
                try:
                    best_row = metrics.loc[metrics["RMSE"].idxmin()]
                    best_m_name = best_row["Model"]
                    best_m_rmse = best_row["RMSE"]
                    st.success(f"🏆 **Best Model Selected:** **{best_m_name}** (Lowest RMSE: **{best_m_rmse:,.2f}**)")
                except Exception:
                    pass

            # Visual charts for comparison
            col1, col2 = st.columns(2)
            
            with col1:
                if "RMSE" in metrics.columns:
                    fig_rmse = px.bar(metrics, x="Model", y="RMSE", title="RMSE Comparison (Lower is Better)", color="Model")
                    st.plotly_chart(fig_rmse, width='stretch')
                else:
                    st.warning("RMSE column is missing in metrics.")
            
            with col2:
                if "MAE" in metrics.columns:
                    fig_mae = px.bar(metrics, x="Model", y="MAE", title="MAE Comparison (Lower is Better)", color="Model")
                    st.plotly_chart(fig_mae, width='stretch')
                else:
                    st.warning("MAE column is missing in metrics.")

            col3, col4 = st.columns(2)
            
            with col3:
                if "R2" in metrics.columns:
                    fig_r2 = px.bar(metrics, x="Model", y="R2", title="R2 Score Comparison (Higher is Better)", color="Model")
                    st.plotly_chart(fig_r2, width='stretch')
                else:
                    st.warning("R2 column is missing in metrics.")
            
            with col4:
                if "MAPE" in metrics.columns:
                    fig_mape = px.bar(metrics, x="Model", y="MAPE", title="MAPE % Comparison (Lower is Better)", color="Model")
                    st.plotly_chart(fig_mape, width='stretch')
                else:
                    st.warning("MAPE column is missing in metrics.")
        else:
            st.warning("Model column is missing in model_metrics.csv.")
    else:
        st.warning("Model metrics data is not available.")

# ============================
# MODEL EXPLAINABILITY
# ============================

elif page == "Model Explainability":
    st.title("🔍 Model Explainability")

    if not feature_importance.empty:
        if "Feature" in feature_importance.columns and "Importance" in feature_importance.columns:
            try:
                st.subheader("Feature Importance Analysis")
                
                # Plotly Horizontal Bar
                fig_feat = px.bar(
                    feature_importance.head(10),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 10 Most Influential Features",
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                fig_feat.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_feat, width='stretch')

                # Feature Table
                st.subheader("Feature Rankings")
                st.dataframe(feature_importance, width='stretch')

                # Feature explanatory cards
                st.markdown("---")
                st.subheader("Feature Definitions & Impact Description")
                
                descriptions = {
                    "Lag_1": "The total funding volume in the immediately preceding month. Captures near-term momentum.",
                    "Lag_3": "The funding volume from three months ago. Standard time-lapsed baseline check.",
                    "Lag_6": "The funding volume from six months ago. Captures half-year cyclical momentum.",
                    "Rolling_Mean_3": "The average monthly funding over the past 3 months. Smooths out extreme monthly peaks and troughs.",
                    "Rolling_Mean_6": "The average monthly funding over the past 6 months. Captures medium-term trend direction.",
                    "Rolling_Std_3": "The standard deviation of monthly funding over the past 3 months. Reflects short-term market volatility.",
                    "Growth_Rate": "The percentage change in funding from month t-2 to month t-1. Captures acceleration/deceleration.",
                    "Year": "Calendar year. Models long-term macroeconomic shifts and secular market expansion/contraction.",
                    "Month": "Calendar month (1-12). Allows the model to capture seasonality effects (e.g., end of fiscal quarters).",
                    "Quarter": "Calendar quarter (1-4). Standard reporting periodicity capture."
                }

                for _, row in feature_importance.head(10).iterrows():
                    feat_name = row["Feature"]
                    feat_imp = row["Importance"]
                    feat_desc = descriptions.get(feat_name, "Custom time-series or macro feature engineered for optimal predictive utility.")
                    
                    st.markdown(f"""
                    <div class="insight-card">
                        <div class="insight-title">{feat_name} &nbsp;|&nbsp; Importance Score: {feat_imp:.4f}</div>
                        <div class="insight-text">{feat_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error rendering feature importance: {e}")
        else:
            st.warning("Required columns ('Feature', 'Importance') are missing in feature_importance.csv.")
    else:
        st.warning("feature_importance.csv was not found. Please run the model training pipeline to generate it.")

# ============================
# FUTURE FORECASTS
# ============================

elif page == "Future Forecasts":
    st.title("🚀 Future Forecasts")

    if not forecast.empty:
        if "Date" in forecast.columns and "Predicted_Funding_Amount" in forecast.columns:
            try:
                # Extrapolate Metrics
                first_pred = forecast["Predicted_Funding_Amount"].iloc[0]
                sixth_pred = forecast["Predicted_Funding_Amount"].iloc[5] if len(forecast) >= 6 else 0.0
                twelfth_pred = forecast["Predicted_Funding_Amount"].iloc[11] if len(forecast) >= 12 else 0.0

                col1, col2, col3 = st.columns(3)
                with col1:
                    kpi_card("Next Month Prediction", f"${first_pred:,.0f}", "Immediate outlook")
                with col2:
                    kpi_card("6-Month Prediction", f"${sixth_pred:,.0f}" if sixth_pred > 0 else "N/A", "Medium-term outlook")
                with col3:
                    kpi_card("12-Month Prediction", f"${twelfth_pred:,.0f}" if twelfth_pred > 0 else "N/A", "Long-term outlook")

                st.markdown("---")

                # Historical vs Forecast chart with confidence bounds
                fig_forecast = go.Figure()

                # Add historical if available
                if not df.empty and "Date" in df.columns and "AmountInUSD" in df.columns:
                    try:
                        hist_monthly = df.dropna(subset=["Date", "AmountInUSD"]).groupby(pd.Grouper(key="Date", freq="ME"))["AmountInUSD"].sum().reset_index()
                        # Limit historical view to last 24 months for visual clarity
                        hist_monthly_view = hist_monthly.tail(24)
                        fig_forecast.add_trace(go.Scatter(x=hist_monthly_view["Date"], y=hist_monthly_view["AmountInUSD"], name="Historical Monthly Funding", line=dict(color="#3b82f6")))
                    except Exception:
                        pass

                # Add predicted line
                fig_forecast.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Predicted_Funding_Amount"], name="Predicted Future Funding", line=dict(color="#ef4444", width=3)))

                # Add confidence intervals if they exist
                if "Predicted_Funding_Amount_Lower" in forecast.columns and "Predicted_Funding_Amount_Upper" in forecast.columns:
                    fig_forecast.add_trace(go.Scatter(
                        x=pd.concat([forecast["Date"], forecast["Date"].iloc[::-1]]),
                        y=pd.concat([forecast["Predicted_Funding_Amount_Upper"], forecast["Predicted_Funding_Amount_Lower"].iloc[::-1]]),
                        fill='toself',
                        fillcolor='rgba(239, 68, 68, 0.15)',
                        line=dict(color='rgba(255, 255, 255, 0)'),
                        hoverinfo="skip",
                        showlegend=True,
                        name="95% Confidence Interval"
                    ))

                fig_forecast.update_layout(title="Historical vs Future Predicted Funding Amount", xaxis_title="Date", yaxis_title="Funding Amount ($)")
                st.plotly_chart(fig_forecast, width='stretch')

                # Forecast growth %
                st.subheader("Forecast Calculations")
                
                # Pct change between historical average and forecast average
                if not df.empty and "AmountInUSD" in df.columns:
                    try:
                        hist_avg = df.dropna(subset=["Date", "AmountInUSD"]).groupby(pd.Grouper(key="Date", freq="ME"))["AmountInUSD"].sum().mean()
                        fore_avg = forecast["Predicted_Funding_Amount"].mean()
                        diff_avg = ((fore_avg - hist_avg) / (hist_avg + 1e-5)) * 100
                        st.write(f"The average forecasted monthly funding (**${fore_avg:,.0f}**) is expected to change by **{diff_avg:+.1f}%** compared to the historical monthly average (**${hist_avg:,.0f}**).")
                    except Exception:
                        pass

                # Forecast Table
                st.subheader("Future Forecast Table")
                st.dataframe(forecast, width='stretch')

                # Download option
                csv_fore = forecast.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Forecast Data (CSV)", csv_fore, "funding_forecast.csv", "text/csv")

            except Exception as e:
                st.error(f"Error executing forecasting visuals: {e}")
        else:
            st.warning("Required columns ('Date', 'Predicted_Funding_Amount') are missing from the forecast file.")
    else:
        st.warning("Forecast data is not available. Please run the predict pipeline to generate forecasts.")

# ============================
# AI GENERATED INSIGHTS
# ============================

elif page == "AI Generated Insights":
    st.title("🤖 AI Generated Insights")

    if not df.empty:
        st.subheader("Automated Business & Market Intelligence")
        
        # Rule-based calculations for AI text
        try:
            total_funding = df["AmountInUSD"].sum()
            avg_funding = df["AmountInUSD"].mean()
            
            top_industries = df.groupby("IndustryVertical")["AmountInUSD"].sum().sort_values(ascending=False).head(3).index.tolist()
            top_cities = df.groupby("CityLocation")["AmountInUSD"].sum().sort_values(ascending=False).head(3).index.tolist()
            
            # Forecast calculations
            growth_forecast = "Neutral"
            forecast_pct = 0.0
            if not forecast.empty and "Predicted_Funding_Amount" in forecast.columns:
                hist_avg = df.groupby(pd.Grouper(key="Date", freq="ME"))["AmountInUSD"].sum().mean()
                fore_avg = forecast["Predicted_Funding_Amount"].mean()
                forecast_pct = ((fore_avg - hist_avg) / (hist_avg + 1e-5)) * 100
                if forecast_pct > 5:
                    growth_forecast = "Expansionary"
                elif forecast_pct < -5:
                    growth_forecast = "Contractionary"
            
            # 1. Trend Insight
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">📈 Market Momentum Analysis</div>
                <div class="insight-text">
                    Historical funding totals reached <b>${total_funding:,.0f}</b> with an average deal size of <b>${avg_funding:,.0f}</b>. 
                    The historical run-rate indicates a healthy startup ecosystem. 
                    Based on time-series analysis, the overall funding velocity is currently calculated to be in a <b>{growth_forecast.lower()}</b> phase, with a projected shift of <b>{forecast_pct:+.1f}%</b> over the next 12 months.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 2. Sectoral Insight
            st.markdown(f"""
            <div class="insight-card" style="border-left-color: #10b981;">
                <div class="insight-title">🏭 Sectoral Allocation Insights</div>
                <div class="insight-text">
                    The top three industries attracting the highest concentration of venture capital are <b>{', '.join(top_industries)}</b>. 
                    Capital allocation is highly skewed towards technology and scale-ready business models. 
                    We recommend investors monitor early-stage developments in these high-velocity sectors, as they capture the majority of the market's capital share.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 3. Geographic Hubs
            st.markdown(f"""
            <div class="insight-card" style="border-left-color: #8b5cf6;">
                <div class="insight-title">🏙️ Geographic Hub Concentration</div>
                <div class="insight-text">
                    Venture capital deals remain highly concentrated in major metropolitan clusters. 
                    The leading cities by investment volume are <b>{', '.join(top_cities)}</b>. 
                    Bengaluru and Mumbai continue to dominate the ecosystem due to mature infrastructure, talent pooling, and proximity to active VC funds.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 4. Investor Activity
            investor_count = "N/A"
            if "InvestorsName" in df.columns:
                investor_count = f"{df['InvestorsName'].nunique():,}"
            st.markdown(f"""
            <div class="insight-card" style="border-left-color: #f59e0b;">
                <div class="insight-title">🤝 Investor Activity Profile</div>
                <div class="insight-text">
                    Our platform tracks activity from <b>{investor_count}</b> unique investor combinations. 
                    Co-investments are increasingly common, with institutional venture capitalists frequently partnering with corporate venture arms and angel networks in larger rounds. 
                    Angel investing remains highly active at the seed and early stages, providing critical initial traction before larger venture deals close.
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error compiling AI insights: {e}")
    else:
        st.warning("The dataset is empty or could not be loaded.")

# ============================
# SYSTEM DIAGNOSTICS
# ============================

elif page == "System Diagnostics":
    st.title("⚙️ System Diagnostics")

    st.subheader("Diagnostics Dashboard")

    # Paths & File verification
    paths_dict = {
        "Dataset Path": DATA_PATH,
        "Forecast Output Path": FORECAST_PATH,
        "Metrics Output Path": METRICS_PATH,
        "Feature Importance Path": FEATURE_IMPORTANCE_PATH
    }

    diag_data = []
    for name, path in paths_dict.items():
        exists = path.exists()
        size = f"{path.stat().st_size / 1024:.2f} KB" if exists else "N/A"
        diag_data.append({
            "File Name / Purpose": name,
            "Target Path": str(path),
            "Status": "✅ Found" if exists else "❌ Missing",
            "File Size": size
        })
    
    st.subheader("File Connectivity Matrix")
    st.table(pd.DataFrame(diag_data))

    # Columns & Row counts
    st.subheader("Data Structures Verification")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Raw Dataset Columns:**")
        st.write(df.columns.tolist() if not df.empty else "No Columns")
        st.write(f"Row count: **{len(df):,}**" if not df.empty else "Row count: **0**")
        
        st.markdown("**Model Evaluation Metrics Columns:**")
        st.write(metrics.columns.tolist() if not metrics.empty else "No Columns")
        st.write(f"Row count: **{len(metrics):,}**" if not metrics.empty else "Row count: **0**")

    with col2:
        st.markdown("**Forecast Output Columns:**")
        st.write(forecast.columns.tolist() if not forecast.empty else "No Columns")
        st.write(f"Row count: **{len(forecast):,}**" if not forecast.empty else "Row count: **0**")

        st.markdown("**Feature Importance Columns:**")
        st.write(feature_importance.columns.tolist() if not feature_importance.empty else "No Columns")
        st.write(f"Row count: **{len(feature_importance):,}**" if not feature_importance.empty else "Row count: **0**")

    # Data Quality Summary
    st.subheader("Data Quality Summary")
    if not df.empty:
        quality_info = []
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            dtype = str(df[col].dtype)
            quality_info.append({
                "Column Name": col,
                "Data Type": dtype,
                "Null Values Count": f"{null_count:,}",
                "Null Percentage (%)": f"{null_pct:.1f}%"
            })
        st.dataframe(pd.DataFrame(quality_info), width='stretch')
    else:
        st.warning("No data quality overview available (dataset empty).")

# ============================
# ABOUT
# ============================

elif page == "About Project":
    st.title("ℹ️ About Project")

    st.markdown(
        """
        ### Startup Funding Forecasting Platform

        This project builds a pipeline to forecast future startup funding volumes.
        We apply advanced statistical modeling and machine learning algorithms on historical deals data.

        #### Forecasting Models Implemented

        - **XGBoost**: Extreme Gradient Boosting regression model parameterized to capture lagging and rolling momentum inputs.
        - **Random Forest**: Ensemble decision-tree regressor robust to outliers and noisy historical features.
        - **Prophet**: Additive regression model developed by Facebook specifically designed for forecasting time-series data with strong seasonal effects.
        - **Linear Regression**: Baseline statistical model representing linear projections.

        #### Technologies Utilized

        - **Python**: Core scripting and pipeline management.
        - **Pandas & NumPy**: High-performance data cleaning, aggregation, and feature scaling.
        - **Scikit-Learn & XGBoost**: Machine learning training, validation splitting, and metric evaluation.
        - **Prophet**: Seasonality and confidence bound time-series fitting.
        - **Plotly**: Dynamic, interactive front-end charts.
        - **Streamlit**: Production-grade dashboard layout and navigation state management.
        """
    )