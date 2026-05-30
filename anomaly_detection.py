"""
Anomaly detection for supplier KPIs.
Uses three complementary methods and combines their signals.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats


KPI_COLS = [
    "Liefertreue (%)",
    "Qualitätsrate (%)",
    "Durchlaufzeit (Tage)",
    "Reklamationsquote (%)",
    "Preisabweichung (%)",
    "Reaktionszeit (Std.)",
]


# ── Z-Score method ──────────────────────────────────────────────────────────

def detect_zscore(series: pd.Series, threshold: float = 2.5) -> pd.Series:
    """Return boolean mask: True = anomaly."""
    z = np.abs(stats.zscore(series.dropna()))
    flags = pd.Series(False, index=series.index)
    flags[series.dropna().index] = z > threshold
    return flags


# ── IQR method ───────────────────────────────────────────────────────────────

def detect_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Return boolean mask: True = anomaly (classic box-plot fence)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    return (series < lo) | (series > hi)


# ── Isolation Forest ─────────────────────────────────────────────────────────

def detect_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.Series:
    """
    Multivariate anomaly detection across all KPI columns.
    Returns boolean Series; True = anomaly.
    """
    data = df[KPI_COLS].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.fillna(data.mean()))

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    preds = model.fit_predict(scaled)   # -1 = anomaly, 1 = normal
    scores = model.decision_function(scaled)  # lower = more anomalous

    flags = pd.Series(preds == -1, index=df.index)
    anomaly_scores = pd.Series(-scores, index=df.index)   # flip: higher = worse
    return flags, anomaly_scores


# ── Master function ───────────────────────────────────────────────────────────

def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds anomaly columns to *df* and returns the enriched DataFrame.

    New columns:
        anomaly_zscore        bool  – any KPI flagged by Z-Score
        anomaly_iqr           bool  – any KPI flagged by IQR
        anomaly_iforest       bool  – flagged by Isolation Forest
        anomaly_score         float – continuous severity score (0–1)
        anomaly_consensus     bool  – flagged by ≥2 of the three methods
        anomaly_kpis          str   – comma-separated list of offending KPIs
    """
    result = df.copy()

    # — per-KPI univariate flags —
    z_any = pd.Series(False, index=result.index)
    iq_any = pd.Series(False, index=result.index)
    offenders: dict[int, list[str]] = {i: [] for i in result.index}

    for col in KPI_COLS:
        z_flag = detect_zscore(result[col])
        iq_flag = detect_iqr(result[col])
        z_any |= z_flag
        iq_any |= iq_flag
        for i in result[z_flag | iq_flag].index:
            offenders[i].append(col)

    # — multivariate —
    if_flag, if_score = detect_isolation_forest(result)

    # — normalise score to 0-1 —
    s_min, s_max = if_score.min(), if_score.max()
    norm_score = (if_score - s_min) / (s_max - s_min + 1e-9)

    result["anomaly_zscore"]    = z_any
    result["anomaly_iqr"]       = iq_any
    result["anomaly_iforest"]   = if_flag
    result["anomaly_score"]     = norm_score.round(3)
    result["anomaly_consensus"] = (
        result["anomaly_zscore"].astype(int)
        + result["anomaly_iqr"].astype(int)
        + result["anomaly_iforest"].astype(int)
    ) >= 2
    result["anomaly_kpis"] = [", ".join(v) if v else "—" for v in offenders.values()]

    return result


# ── Scoring helper ────────────────────────────────────────────────────────────

def score_supplier(df_supplier: pd.DataFrame) -> dict:
    """
    Compute an aggregated performance score (0–100) for one supplier.
    Uses recent 3 months weighted more heavily.
    """
    from data_generator import KPI_DEFINITIONS

    df_s = df_supplier.sort_values("Datum")
    n = len(df_s)
    weights = np.linspace(0.5, 1.0, n)  # more recent → higher weight

    scores = []
    for kpi, meta in KPI_DEFINITIONS.items():
        target = meta["target"]
        vals = df_s[kpi].fillna(df_s[kpi].median())
        if meta["higher_is_better"]:
            kpi_score = np.clip(vals / target * 100, 0, 100)
        else:
            kpi_score = np.clip((1 - (vals - target) / (target + 1e-9)) * 100, 0, 100)
        scores.append(np.average(kpi_score, weights=weights))

    overall = round(float(np.mean(scores)), 1)
    grade = (
        "A" if overall >= 90 else
        "B" if overall >= 75 else
        "C" if overall >= 60 else
        "D"
    )
    return {"score": overall, "grade": grade}
