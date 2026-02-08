import streamlit as st
import pandas as pd
import numpy as np

from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go

from config import LAST_HISTORICAL_DATE
from database import load_all_data, get_hospitals, get_hospital_wards
from ml_pipeline.predict import generate_forecast
from ml_pipeline.train import train_model_for_ward, should_retrain_model
from utils.plots import create_forecast_plot
from utils.helpers import get_current_datetime


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Hospital Bed Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Clean, professional CSS
# =========================
st.markdown(
    """
<style>
.main-header {
    font-size: 2.8rem;
    color: #1a237e;
    text-align: center;
    margin-bottom: 1.5rem;
    padding: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.ward-card {
    background: #f8fafc;
    border-radius: 12px;
    padding: 1.8rem;
    margin-bottom: 2.5rem;
    border-left: 6px solid #2563eb;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05),
                0 2px 4px -1px rgba(0, 0, 0, 0.03);
}

.historical-period {
    background: linear-gradient(135deg, #64748b 0%, #475569 100%);
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9em;
    display: inline-block;
}

.today-highlight {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
    color: white;
    padding: 8px 20px;
    border-radius: 25px;
    font-weight: 600;
    font-size: 1.1em;
    display: inline-block;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(220, 38, 38, 0.3);
}

.metric-card {
    background: white;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #e2e8f0;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
    margin: 5px 0;
}

.metric-label {
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.forecast-period {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9em;
    display: inline-block;
}

.section-divider {
    height: 2px;
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
    margin: 1.5rem 0;
    border: none;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Initialize
# =========================
current_datetime = get_current_datetime()
current_date = current_datetime.date()

# =========================
# Session state
# =========================
if "show_future_forecast" not in st.session_state:
    st.session_state.show_future_forecast = False

# =========================
# Header
# =========================
st.markdown(
    '<h1 class="main-header">Hospital Bed Occupancy Forecast</h1>',
    unsafe_allow_html=True,
)

# =========================
# Timeline display
# =========================
col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.markdown('<div class="historical-period">Historical Data</div>', unsafe_allow_html=True)
    st.markdown(f"Ends: {LAST_HISTORICAL_DATE.strftime('%b %d, %Y')}")

with col_b:
    st.markdown('<div class="forecast-period">Forecast Period</div>', unsafe_allow_html=True)
    st.markdown(f"Up to: {current_date.strftime('%b %d, %Y')}")

with col_c:
    st.markdown('<div class="today-highlight">TODAY</div>', unsafe_allow_html=True)
    st.markdown(f"{current_date.strftime('%b %d, %Y')}")

with col_d:
    st.markdown("Toggle in sidebar")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# =========================
# Helper functions
# =========================
def prepare_time_series(ward_data: pd.DataFrame) -> pd.Series:
    ts = ward_data.set_index("datetime")["occupied_beds"]
    ts = ts.asfreq("D").ffill().bfill()
    return ts


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, y_true)
    return np.nanmean(np.abs((y_true - y_pred) / denom)) * 100


def make_daily_y_X(ward_data, exog_cols=("staffed_beds", "closed_beds")):
    ward_data = ward_data.copy()
    ward_data["datetime"] = pd.to_datetime(ward_data["datetime"], errors="coerce")
    ward_data = ward_data.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")

    y = ward_data["occupied_beds"].resample("D").mean().ffill().bfill()
    X = ward_data[list(exog_cols)].resample("D").mean().ffill().bfill()

    X = X.loc[y.index]
    return y, X


def fit_sarimax_with_exog(y, X, test_days=28, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7)):
    cutoff = y.index.max() - pd.Timedelta(days=test_days)

    y_train = y[y.index <= cutoff]
    y_test = y[y.index > cutoff]

    X_train = X.loc[y_train.index]
    X_test = X.loc[y_test.index]

    model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=200)

    test_pred = res.get_forecast(steps=len(y_test), exog=X_test).predicted_mean
    mae = mean_absolute_error(y_test, test_pred)
    mape = safe_mape(y_test.values, test_pred.values)

    return res, mape, mae, y


def run_sarimax_what_if(
    ward_data,
    forecast_days=14,
    test_days=28,
    order=(2, 1, 2),
    seasonal_order=(1, 1, 1, 7),
    staffed_delta=10,
    closed_delta=5,
):
    y, X = make_daily_y_X(ward_data, exog_cols=("staffed_beds", "closed_beds"))
    res, mape, mae, y_daily = fit_sarimax_with_exog(
        y, X, test_days=test_days, order=order, seasonal_order=seasonal_order
    )

    future_index = pd.date_range(start=y.index.max() + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    last_exog = X.iloc[-1].copy()

    scenarios = {
        "Baseline (hold last)": last_exog,
        f"Staffed +{staffed_delta}": pd.Series(
            {"staffed_beds": last_exog["staffed_beds"] + staffed_delta, "closed_beds": last_exog["closed_beds"]}
        ),
        f"Closed +{closed_delta}": pd.Series(
            {"staffed_beds": last_exog["staffed_beds"], "closed_beds": last_exog["closed_beds"] + closed_delta}
        ),
        f"Staffed +{staffed_delta}, Closed +{closed_delta}": pd.Series(
            {
                "staffed_beds": last_exog["staffed_beds"] + staffed_delta,
                "closed_beds": last_exog["closed_beds"] + closed_delta,
            }
        ),
    }

    rows = []
    for name, exog_vals in scenarios.items():
        X_future = pd.DataFrame(
            [exog_vals.values] * forecast_days,
            index=future_index,
            columns=["staffed_beds", "closed_beds"],
        )
        fc = res.get_forecast(steps=forecast_days, exog=X_future).summary_frame()
        fc.index = future_index
        fc["scenario"] = name
        rows.append(fc)

    sim_df = pd.concat(rows).reset_index().rename(columns={"index": "date"})
    return sim_df, mape, mae, y_daily


# =========================
# Load data (cached)
# =========================
@st.cache_data(ttl=3600)
def load_cached_data():
    df = load_all_data()
    if df is None:
        return pd.DataFrame()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    return df


df = load_cached_data()
if df.empty:
    st.error("No data loaded from the database.")
    st.stop()

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## Dashboard Settings")

    hospital_options = get_hospitals()
    hospital = st.selectbox("Select Hospital", hospital_options)

    ward_options = get_hospital_wards(hospital)
    max_wards = st.slider("Max wards to display", 1, 6, 3)

    selected_wards = st.multiselect(
        "Select Wards to View",
        ward_options,
        default=ward_options[: min(2, len(ward_options))],
        max_selections=max_wards,
    )

    st.markdown("---")
    st.markdown("### Forecast Settings")

    forecast_ahead_days = st.slider(
        "Days of future forecast to show",
        min_value=7,
        max_value=365,
        value=14,
        step=7,
    )

    show_future = st.checkbox(
        "Show future forecast lines",
        value=st.session_state.show_future_forecast,
        help="Display forecast lines extending beyond today",
    )
    st.session_state.show_future_forecast = show_future

    if st.button("Refresh All Forecasts", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### About This Dashboard")
    st.markdown(
        """
This dashboard provides daily bed occupancy forecasts for hospital wards.

**Key Features:**
- Daily updated forecasts
- 95% confidence intervals
- Historical trend visualization
- Automatic model retraining
"""
    )

# =========================
# Main guards + filtering
# =========================
if not selected_wards:
    st.info("Please select a hospital and at least one ward from the sidebar to begin.")
    st.stop()

df_filtered = df[df["hospital"] == hospital].copy()
if df_filtered.empty:
    st.warning("No data found for the selected hospital.")
    st.stop()

# =========================
# Tabs
# =========================
tab_forecast, tab_sim = st.tabs(["Forecast", "What-If Simulation (SARIMAX)"])


# =========================
# Forecast tab (your deployed forecasting)
# =========================
with tab_forecast:
    for i in range(0, len(selected_wards), 2):
        ward_pair = selected_wards[i : i + 2]
        cols = st.columns(2)

        for j, ward in enumerate(ward_pair):
            with cols[j]:
                st.markdown('<div class="ward-card">', unsafe_allow_html=True)
                st.markdown(f"### 🏥 {ward}")

                ward_data = df_filtered[df_filtered["ward"] == ward].copy().sort_values("datetime")
                if ward_data.empty:
                    st.warning("No data available for this ward.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                ts = prepare_time_series(ward_data)

                if should_retrain_model(hospital, ward):
                    with st.spinner(f"Training model for {ward}..."):
                        success, _ = train_model_for_ward(hospital, ward, ts)
                        if not success:
                            st.error(f"Failed to train model for {ward}")
                            st.markdown("</div>", unsafe_allow_html=True)
                            continue

                with st.spinner(f"Generating forecast for {ward}..."):
                    forecast_df, mape, mae = generate_forecast(hospital, ward, ts, forecast_ahead_days)

                if forecast_df is None or forecast_df.empty:
                    st.error(f"Failed to create forecast for {ward}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                col1, col2, col3 = st.columns(3)

                with col1:
                    val = f"{mape:.1f}%" if pd.notna(mape) else "N/A"
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">MAPE</div>
                            <div class="metric-value">{val}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    current_max = float(ts.iloc[-30:].max()) if len(ts) else np.nan
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">30-Day Peak</div>
                            <div class="metric-value">{current_max:.0f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Data Days</div>
                            <div class="metric-value">{len(ts)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                fig = create_forecast_plot(
                    ward_name=ward,
                    historical_data=ward_data,
                    forecast_df=forecast_df,
                    mape=mape,
                    mae=mae,
                    current_date=current_date,
                    show_future=st.session_state.show_future_forecast,
                    forecast_ahead_days=forecast_ahead_days,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Simulation tab (SARIMAX What-If)
# =========================
with tab_sim:
    st.markdown("## What-If Simulation (SARIMAX with staffing inputs)")
    st.caption(
        "This tab fits a SARIMAX model with exogenous inputs (staffed_beds, closed_beds) and simulates operational scenarios."
    )

    sim_ward = st.selectbox("Select a ward to simulate", selected_wards)

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_forecast_days = st.slider("Forecast days", 7, 60, 14, 1)
    with sim_col2:
        staffed_delta = st.slider("Staffed beds change (+)", 0, 50, 10, 1)
    with sim_col3:
        closed_delta = st.slider("Closed beds change (+)", 0, 50, 5, 1)

    ward_data = df_filtered[df_filtered["ward"] == sim_ward].copy().sort_values("datetime")

    required_cols = {"occupied_beds", "staffed_beds", "closed_beds", "datetime"}
    missing = required_cols - set(ward_data.columns)

    if missing:
        st.error(f"Missing required columns for simulation: {sorted(list(missing))}")
        st.stop()

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Fitting SARIMAX and running scenarios..."):
            sim_df, sim_mape, sim_mae, y_daily = run_sarimax_what_if(
                ward_data=ward_data,
                forecast_days=sim_forecast_days,
                test_days=28,
                order=(2, 1, 2),
                seasonal_order=(1, 1, 1, 7),
                staffed_delta=staffed_delta,
                closed_delta=closed_delta,
            )

        st.success(f"Simulation complete — Backtest MAPE: {sim_mape:.1f}% | MAE: {sim_mae:.2f}")

        hist = ward_data.copy()
        hist["datetime"] = pd.to_datetime(hist["datetime"], errors="coerce")
        hist = hist.dropna(subset=["datetime"]).sort_values("datetime")
        hist = hist[hist["datetime"] >= hist["datetime"].max() - pd.Timedelta(days=60)]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist["datetime"],
                y=hist["occupied_beds"],
                mode="lines",
                name="Historical",
                line=dict(width=3),
            )
        )

        for scenario in sim_df["scenario"].unique():
            sub = sim_df[sim_df["scenario"] == scenario].copy()
            fig.add_trace(
                go.Scatter(
                    x=sub["date"],
                    y=sub["mean"],
                    mode="lines",
                    name=scenario,
                    line=dict(width=3),
                )
            )

        fig.add_vline(x=sim_df["date"].min(), line_dash="dash", opacity=0.6)

        fig.update_layout(
            title=f"SARIMAX What-If Scenarios — {sim_ward}",
            xaxis_title="Date",
            yaxis_title="Occupied Beds",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Scenario Forecast Table (first rows)")
        st.dataframe(sim_df[["date", "scenario", "mean", "mean_ci_lower", "mean_ci_upper"]].head(50))
