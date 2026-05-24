"""
Generates realistic synthetic supplier KPI data for demo purposes.
Replace this with your real data source (CSV, Excel, database, API).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


SUPPLIERS = [
    "AutoParts GmbH",
    "MetallWerk AG",
    "TechSupply KG",
    "Precision Parts Ltd",
    "EuroComponents SE",
]

KPI_DEFINITIONS = {
    "Liefertreue (%)":        {"target": 95, "unit": "%",   "higher_is_better": True},
    "Qualitätsrate (%)":      {"target": 98, "unit": "%",   "higher_is_better": True},
    "Durchlaufzeit (Tage)":   {"target": 5,  "unit": "d",   "higher_is_better": False},
    "Reklamationsquote (%)":  {"target": 2,  "unit": "%",   "higher_is_better": False},
    "Preisabweichung (%)":    {"target": 0,  "unit": "%",   "higher_is_better": False},
    "Reaktionszeit (Std.)":   {"target": 24, "unit": "h",   "higher_is_better": False},
}


def generate_supplier_data(
    months: int = 24,
    seed: int = 42,
    inject_anomalies: bool = True,
) -> pd.DataFrame:
    """Return a tidy DataFrame with monthly KPI values per supplier."""
    rng = np.random.default_rng(seed)
    today = datetime.today().replace(day=1)
    dates = [today - timedelta(days=30 * i) for i in range(months - 1, -1, -1)]

    rows = []
    for supplier in SUPPLIERS:
        # Each supplier gets slightly different baseline behaviour
        quality_bias = rng.uniform(-1, 2)
        delivery_bias = rng.uniform(-2, 3)

        for date in dates:
            row = {
                "Datum": date,
                "Lieferant": supplier,
                "Liefertreue (%)":      round(np.clip(rng.normal(93 + delivery_bias, 3), 60, 100), 1),
                "Qualitätsrate (%)":    round(np.clip(rng.normal(97 + quality_bias, 1.5), 80, 100), 1),
                "Durchlaufzeit (Tage)": round(np.clip(rng.normal(5, 1), 1, 20), 1),
                "Reklamationsquote (%)":round(np.clip(rng.normal(2, 0.8), 0, 15), 2),
                "Preisabweichung (%)":  round(rng.normal(0, 2), 2),
                "Reaktionszeit (Std.)": round(np.clip(rng.normal(20, 6), 1, 72), 1),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    if inject_anomalies:
        df = _inject_anomalies(df, rng)

    return df


def _inject_anomalies(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject realistic anomalies into ~5 % of records."""
    anomaly_mask = rng.random(len(df)) < 0.05
    idx = df[anomaly_mask].index

    for i in idx:
        kpi = rng.choice(list(KPI_DEFINITIONS.keys()))
        if KPI_DEFINITIONS[kpi]["higher_is_better"]:
            df.at[i, kpi] = round(float(df.at[i, kpi]) * rng.uniform(0.5, 0.75), 2)
        else:
            df.at[i, kpi] = round(float(df.at[i, kpi]) * rng.uniform(2.5, 4.0), 2)

    return df
