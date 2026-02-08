import plotly.graph_objects as go
import pandas as pd
from datetime import timedelta
import numpy as np
from config import LAST_HISTORICAL_DATE


def create_forecast_plot(
    ward_name,
    historical_data,
    forecast_df,
    mape,
    mae,
    current_date,
    show_future,
    forecast_ahead_days=14,
):
    """Create forecast plot"""

    # Filter historical data (last 60 days)
    history_start = pd.Timestamp(LAST_HISTORICAL_DATE) - timedelta(days=60)
    hist_filtered = historical_data[historical_data["datetime"] >= history_start].copy()

    # Ensure datetime types
    hist_filtered["datetime"] = pd.to_datetime(hist_filtered["datetime"], errors="coerce")
    hist_filtered = hist_filtered.dropna(subset=["datetime"]).sort_values("datetime")

    today_ts = pd.Timestamp(current_date)

    fig = go.Figure()

    # Historical line
    fig.add_trace(
        go.Scatter(
            x=hist_filtered["datetime"],
            y=hist_filtered["occupied_beds"],
            mode="lines",
            name="Historical Data",
            line=dict(width=3),
        )
    )

    if forecast_df is None or forecast_df.empty or forecast_df["mean"].isna().all():
        fig.update_layout(
            title=f"{ward_name} — No forecast available",
            xaxis_title="Date",
            yaxis_title="Occupied Beds",
        )
        return fig

    # Split forecast into past/today/future relative to current_date
    past_forecast = forecast_df[forecast_df.index.date <= current_date].copy()
    future_forecast = forecast_df[forecast_df.index.date > current_date].copy()

    # If show_future is off, only keep up to today
    if not show_future:
        future_forecast = future_forecast.iloc[0:0]

    # Confidence interval (if present)
    if "mean_ci_lower" in forecast_df.columns and "mean_ci_upper" in forecast_df.columns:
        ci_df = pd.concat([past_forecast, future_forecast]).copy()

        fig.add_trace(
            go.Scatter(
                x=ci_df.index,
                y=ci_df["mean_ci_upper"],
                mode="lines",
                name="Upper CI",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ci_df.index,
                y=ci_df["mean_ci_lower"],
                mode="lines",
                name="95% CI",
                fill="tonexty",
                line=dict(width=0),
                opacity=0.2,
            )
        )

    # Forecast mean (past/today)
    if not past_forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=past_forecast.index,
                y=past_forecast["mean"],
                mode="lines",
                name="Forecast (to today)",
                line=dict(width=3, dash="solid"),
            )
        )

    # Forecast mean (future)
    if not future_forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=future_forecast.index,
                y=future_forecast["mean"],
                mode="lines",
                name=f"Future Forecast (+{forecast_ahead_days}d)",
                line=dict(width=3, dash="dash"),
            )
        )

    # Today vertical marker
    fig.add_vline(x=today_ts, line_dash="dash", opacity=0.6)

    # Title + metrics
    mape_text = f"{mape:.1f}%" if pd.notna(mape) else "N/A"
    mae_text = f"{mae:.2f}" if pd.notna(mae) else "N/A"

    fig.update_layout(
        title=f"{ward_name} — MAPE: {mape_text} | MAE: {mae_text}",
        xaxis_title="Date",
        yaxis_title="Occupied Beds",
        hovermode="x unified",
        legend_title="Series",
        margin=dict(l=10, r=10, t=60, b=10),
    )

    return fig
