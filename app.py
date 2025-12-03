import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats

st.set_page_config(page_title="TrendLab", layout="wide")

# -------------------------- BEAUTIFUL UI CSS ------------------------------
st.markdown("""
<style>

body {
    background-color: #f6f8fa;
}

/* Gradient Title */
.big-title {
    font-size: 60px;
    font-weight: 900;
    text-align: center;
    background: -webkit-linear-gradient(45deg, #0078ff, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #666;
    margin-bottom: 40px;
}

/* Centered Buttons */
.center-buttons {
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-top: 20px;
}

/* Button Styling */
.stButton>button {
    background: linear-gradient(45deg, #0078ff, #00d4ff);
    color: white !important;
    border-radius: 10px;
    padding: 15px 25px;
    border: none;
    font-size: 18px !important;
    font-weight: bold;
    box-shadow: 0px 4px 12px rgba(0, 120, 255, 0.3);
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(45deg, #006be6, #00bde6);
}

/* Card Container */
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------- SESSION STATE ------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page

# -------------------------- HOME PAGE ------------------------------
if st.session_state.page == "home":

    st.markdown("<h1 class='big-title'>TrendLab</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Explore Data. Discover Insights. Forecast the Future.</p>", unsafe_allow_html=True)

    st.markdown("<div class='center-buttons'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📈 Time Series Analysis", use_container_width=True):
            go_to("time_series")

    with col2:
        if st.button("📊 Hypothesis Testing", use_container_width=True):
            go_to("hypothesis")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------- TIME SERIES PAGE ------------------------------
elif st.session_state.page == "time_series":

    st.button("⬅ Back", on_click=lambda: go_to("home"))
    st.markdown("<h1 class='big-title'>📈 Time Series Analysis</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            df = pd.read_csv(uploaded_file)

            date_cols = [c for c in df.columns if "date" in c.lower()]
            if len(date_cols) == 0:
                st.error("No date column found!")
                st.stop()

            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df.set_index(date_cols[0], inplace=True)

            st.subheader("📄 Data Preview")
            st.write(df.head())
            st.markdown("</div>", unsafe_allow_html=True)

        # Select column
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        column = st.selectbox("Select Numeric Column", num_cols)

        # Time Series Plot
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📉 Time Series Plot")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df[column])
        ax.set_title(f"Trend Over Time — {column}")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        # Decomposition
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔍 Trend, Seasonality & Residual Decomposition")
        try:
            result = seasonal_decompose(df[column], model='additive', period=12)
            fig = result.plot()
            fig.set_size_inches(12, 8)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Decomposition error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

        # ADF
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📌 ADF Stationarity Test")
        adf = adfuller(df[column])
        st.write(f"**ADF Statistic:** {adf[0]}")
        st.write(f"**p-value:** {adf[1]}")
        if adf[1] < 0.05:
            st.success("Data is Stationary")
        else:
            st.warning("Data is Not Stationary")
        st.markdown("</div>", unsafe_allow_html=True)

        # Forecast
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔮 Forecast (ARIMA)")
        steps = st.slider("Forecast Steps:", 5, 60, 12)

        try:
            model = ARIMA(df[column], order=(1, 1, 1))
            fit = model.fit()
            future = fit.forecast(steps=steps)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df[column], label="History")
            ax.plot(future, label="Forecast")
            ax.legend()
            st.pyplot(fig)
        except:
            st.warning("ARIMA failed. Try more data.")
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------- HYPOTHESIS PAGE ------------------------------
elif st.session_state.page == "hypothesis":

    st.button("⬅ Back", on_click=lambda: go_to("home"))
    st.markdown("<h1 class='big-title'>📊 Hypothesis Testing</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Data Preview")
        st.write(df.head())
        st.markdown("</div>", unsafe_allow_html=True)

        test = st.selectbox("Select Test", ["T-Test", "Chi-Square Test"])

        if test == "T-Test":
            num_cols = df.select_dtypes(include=['float64', 'int64']).columns
            c1 = st.selectbox("Sample 1", num_cols)
            c2 = st.selectbox("Sample 2", num_cols)

            if st.button("Run T-Test"):
                t, p = stats.ttest_ind(df[c1], df[c2])
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("📌 T-Test Result")
                st.write("T-Statistic:", t)
                st.write("p-value:", p)
                if p < 0.05:
                    st.success("Reject H₀ — Significant Difference")
                else:
                    st.warning("Fail to Reject H₀")
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) < 2:
                st.error("Need 2 categorical columns!")
            else:
                c1 = st.selectbox("Category 1", cat_cols)
                c2 = st.selectbox("Category 2", cat_cols)

                if st.button("Run Chi-Square"):
                    contingency = pd.crosstab(df[c1], df[c2])
                    chi, p, dof, exp = stats.chi2_contingency(contingency)

                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("📌 Chi-Square Result")
                    st.write("Chi-Square Value:", chi)
                    st.write("p-value:", p)
                    if p < 0.05:
                        st.success("Reject H₀ — Dependent Variables")
                    else:
                        st.warning("Fail to Reject H₀ — Independent")
                    st.markdown("</div>", unsafe_allow_html=True)
