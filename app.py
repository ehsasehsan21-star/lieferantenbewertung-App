"""
Lieferantenbewertung – KPI Dashboard mit Anomalieerkennung
==========================================================
Run:  streamlit run app.py
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
 
from data_generator import generate_supplier_data, KPI_DEFINITIONS, SUPPLIERS
from anomaly_detection import run_anomaly_detection, score_supplier
 
# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lieferantenbewertung",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
 
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
 
    .main { background: #0f1117; }
 
    /* Grade badge */
    .grade-badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        padding: 0.1em 0.5em;
        border-radius: 6px;
        line-height: 1;
    }
    .grade-A { background: #0d3d2e; color: #22c55e; border: 1px solid #22c55e44; }
    .grade-B { background: #1a3a00; color: #84cc16; border: 1px solid #84cc1644; }
    .grade-C { background: #3a2800; color: #f59e0b; border: 1px solid #f59e0b44; }
    .grade-D { background: #3a0000; color: #ef4444; border: 1px solid #ef444444; }
 
    /* Anomaly row highlight */
    .anomaly-high  { color: #ef4444; font-weight: 600; }
    .anomaly-med   { color: #f59e0b; }
 
    /* KPI card */
    .kpi-card {
        background: #1a1d27;
        border: 1px solid #2d3148;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .kpi-value { font-size: 1.8rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
    .kpi-label { font-size: 0.78rem; color: #8b92b0; text-transform: uppercase; letter-spacing: 0.06em; }
    .kpi-delta { font-size: 0.82rem; margin-top: 2px; }
 
    /* Section title */
    .section-title {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #5b6080;
        margin: 1.4rem 0 0.6rem;
    }
 
    /* Sidebar */
    [data-testid="stSidebar"] { background: #0a0c14; border-right: 1px solid #1e2235; }
 
    /* Divider */
    hr { border-color: #1e2235; }
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Daten werden geladen …")
def load_data(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        uploaded_file.seek(0)  # reset file pointer before reading
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
 
        # Remove accidental whitespace from column names
        df.columns = df.columns.str.strip()
 
        # Safely parse date column
        if "Datum" in df.columns:
            df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
        else:
            st.error(f"Spalte 'Datum' nicht gefunden. Vorhandene Spalten: {list(df.columns)}")
            st.stop()
    else:
        df = generate_supplier_data(months=24, inject_anomalies=True)
 
    return run_anomaly_detection(df)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – filters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/delivery.png", width=40)
    st.markdown("## 📦 Lieferantenbewertung")
    st.markdown("---")
 
    uploaded = st.file_uploader(
        "📂 Eigene Daten hochladen",
        type=["csv", "xlsx"],
        help="CSV oder Excel mit Spalten: Datum, Lieferant, KPI-Spalten",
    )
 
    if uploaded is not None:
        uploaded.seek(0)  # reset before passing to cached function
    df_all = load_data(uploaded)
 
    st.markdown('<p class="section-title">Filter</p>', unsafe_allow_html=True)
 
    all_suppliers = sorted(df_all["Lieferant"].unique().tolist())
    selected_suppliers = st.multiselect(
        "Lieferanten",
        options=all_suppliers,
        default=all_suppliers,
    )
 
    min_date = df_all["Datum"].min().date()
    max_date = df_all["Datum"].max().date()
    date_range = st.date_input(
        "Zeitraum",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
 
    st.markdown('<p class="section-title">Anomalie-Methode</p>', unsafe_allow_html=True)
    method = st.radio(
        "Erkennungsmethode",
        ["Konsens (≥2 Methoden)", "Z-Score", "IQR", "Isolation Forest"],
        label_visibility="collapsed",
    )
 
    contamination_pct = st.slider(
        "Anomalie-Anteil (%)", min_value=1, max_value=15, value=5
    )
 
    st.markdown("---")
    st.markdown(
        "<small style='color:#5b6080'>Daten: synthetisch generiert<br>"
        "Modell: IsolationForest + Z-Score + IQR</small>",
        unsafe_allow_html=True,
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────────────────────────────────────
if len(date_range) == 2:
    start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_d, end_d = df_all["Datum"].min(), df_all["Datum"].max()
 
df = df_all[
    df_all["Lieferant"].isin(selected_suppliers)
    & (df_all["Datum"] >= start_d)
    & (df_all["Datum"] <= end_d)
].copy()
 
method_col = {
    "Konsens (≥2 Methoden)": "anomaly_consensus",
    "Z-Score": "anomaly_zscore",
    "IQR": "anomaly_iqr",
    "Isolation Forest": "anomaly_iforest",
}[method]
 
kpi_cols = list(KPI_DEFINITIONS.keys())
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ── TAB LAYOUT ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_supplier, tab_anomaly, tab_compare, tab_data = st.tabs([
    "🏠 Übersicht",
    "🏭 Lieferantendetail",
    "🚨 Anomalien",
    "📊 Vergleich",
    "🗂️ Rohdaten",
])
 
 
# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("## Gesamtübersicht Lieferantenperformance")
 
    # ── Score cards per supplier ─────────────────────────────────────────────
    cols = st.columns(len(selected_suppliers))
    for col, supplier in zip(cols, selected_suppliers):
        df_s = df[df["Lieferant"] == supplier]
        info = score_supplier(df_s)
        n_anomalies = int(df_s[method_col].sum())
        with col:
            grade = info["grade"]
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{supplier}</div>'
                f'<div style="display:flex;align-items:center;gap:10px;margin-top:6px">'
                f'<span class="grade-badge grade-{grade}">{grade}</span>'
                f'<span class="kpi-value" style="font-size:1.4rem">{info["score"]}</span>'
                f'</div>'
                f'<div class="kpi-delta" style="margin-top:6px;color:#ef4444">🚨 {n_anomalies} Anomalien</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
 
    st.markdown("---")
 
    # ── KPI trend (multi-supplier) ───────────────────────────────────────────
    st.markdown("### KPI-Trends über Zeit")
    kpi_choice = st.selectbox("KPI auswählen", kpi_cols)
 
    fig = px.line(
        df,
        x="Datum",
        y=kpi_choice,
        color="Lieferant",
        markers=True,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    target = KPI_DEFINITIONS[kpi_choice]["target"]
    fig.add_hline(
        y=target,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text=f"Ziel: {target}",
        annotation_position="bottom right",
    )
    # Highlight anomalies
    df_anom = df[df[method_col]]
    fig.add_trace(go.Scatter(
        x=df_anom["Datum"],
        y=df_anom[kpi_choice],
        mode="markers",
        marker=dict(symbol="x", size=10, color="#ef4444", line=dict(width=2)),
        name="⚠️ Anomalie",
        hovertemplate="<b>Anomalie</b><br>%{x}<br>Wert: %{y}",
    ))
    fig.update_layout(
        height=380,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,22,35,0.6)",
    )
    st.plotly_chart(fig, use_container_width=True)
 
    # ── Heatmap ──────────────────────────────────────────────────────────────
    st.markdown("### KPI-Heatmap (Durchschnittswerte)")
    pivot = df.groupby("Lieferant")[kpi_cols].mean().round(2)
 
    # Normalise each KPI to 0-1 for colour (respecting direction)
    norm = pd.DataFrame(index=pivot.index, columns=pivot.columns)
    for col in kpi_cols:
        mn, mx = pivot[col].min(), pivot[col].max()
        if mx == mn:
            norm[col] = 0.5
        elif KPI_DEFINITIONS[col]["higher_is_better"]:
            norm[col] = (pivot[col] - mn) / (mx - mn)
        else:
            norm[col] = 1 - (pivot[col] - mn) / (mx - mn)
 
    fig_heat = go.Figure(go.Heatmap(
        z=norm.values.astype(float),
        x=[c.split(" (")[0] for c in kpi_cols],
        y=pivot.index.tolist(),
        colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]],
        text=pivot.values.round(2),
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        showscale=True,
        zmin=0, zmax=1,
    ))
    fig_heat.update_layout(
        height=260,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,22,35,0.6)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(side="top"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
 
 
# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – SUPPLIER DETAIL
# ════════════════════════════════════════════════════════════════════════════
with tab_supplier:
    st.markdown("## Lieferantendetail-Analyse")
    supplier = st.selectbox("Lieferant wählen", selected_suppliers, key="detail_supplier")
    df_s = df[df["Lieferant"] == supplier].sort_values("Datum")
    info = score_supplier(df_s)
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gesamtscore", f"{info['score']} / 100", delta=None)
    c2.metric("Note", info["grade"])
    c3.metric("Datenpunkte", len(df_s))
    c4.metric("Anomalien", int(df_s[method_col].sum()), delta=None)
 
    st.markdown("---")
 
    # KPI subplots
    fig_sub = make_subplots(
        rows=2, cols=3,
        subplot_titles=[c.split(" (")[0] for c in kpi_cols],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )
 
    colours = ["#60a5fa", "#34d399", "#f472b6", "#fbbf24", "#a78bfa", "#fb923c"]
    for idx, (col, clr) in enumerate(zip(kpi_cols, colours)):
        r, c = divmod(idx, 3)
        target = KPI_DEFINITIONS[col]["target"]
 
        fig_sub.add_trace(
            go.Scatter(x=df_s["Datum"], y=df_s[col],
                       mode="lines+markers", name=col,
                       line=dict(color=clr, width=2),
                       marker=dict(size=5), showlegend=False),
            row=r + 1, col=c + 1,
        )
        # Anomaly markers
        df_a = df_s[df_s[method_col]]
        fig_sub.add_trace(
            go.Scatter(x=df_a["Datum"], y=df_a[col],
                       mode="markers",
                       marker=dict(symbol="x", size=10, color="#ef4444", line=dict(width=2)),
                       showlegend=False, name="Anomalie"),
            row=r + 1, col=c + 1,
        )
        fig_sub.add_hline(
            y=target, line_dash="dot", line_color="rgba(245,158,11,0.33)",
            row=r + 1, col=c + 1,
        )
 
    fig_sub.update_layout(
        height=540,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,22,35,0.6)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_sub, use_container_width=True)
 
    # Radar chart
    st.markdown("### Radar-Profil")
    avg = df_s[kpi_cols].mean()
    targets = [KPI_DEFINITIONS[k]["target"] for k in kpi_cols]
 
    # Normalise to 0-100 relative to target
    norm_vals = []
    for col in kpi_cols:
        t = KPI_DEFINITIONS[col]["target"]
        if KPI_DEFINITIONS[col]["higher_is_better"]:
            norm_vals.append(min(avg[col] / t * 100, 100))
        else:
            norm_vals.append(min(t / max(avg[col], 0.01) * 100, 100))
 
    labels = [c.split(" (")[0] for c in kpi_cols]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=norm_vals + [norm_vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(96,165,250,0.2)",
        line=dict(color="#60a5fa", width=2),
        name=supplier,
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[100] * (len(labels) + 1),
        theta=labels + [labels[0]],
        line=dict(color="rgba(245,158,11,0.27)", dash="dot", width=1),
        name="Ziel 100%",
        showlegend=True,
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 110], tickfont=dict(size=9)),
            bgcolor="rgba(20,22,35,0.8)",
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=60, r=60, t=20, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
 
 
# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – ANOMALY DETAIL
# ════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    st.markdown("## 🚨 Anomalie-Analyse")
 
    df_anom = df[df[method_col]].copy()
 
    c1, c2, c3 = st.columns(3)
    c1.metric("Anomalien gesamt", len(df_anom))
    c2.metric("Anomalie-Rate", f"{len(df_anom)/len(df)*100:.1f} %")
    c3.metric("Betroffene Lieferanten", df_anom["Lieferant"].nunique())
 
    st.markdown("---")
 
    # Anomaly timeline
    st.markdown("### Anomalien im Zeitverlauf")
    timeline = df_anom.groupby(["Datum", "Lieferant"]).size().reset_index(name="Anzahl")
    fig_tl = px.bar(
        timeline, x="Datum", y="Anzahl", color="Lieferant",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_tl.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,22,35,0.6)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_tl, use_container_width=True)
 
    # Anomaly score scatter
    st.markdown("### Anomalie-Score Verteilung")
    fig_sc = px.scatter(
        df,
        x="Datum",
        y="anomaly_score",
        color="Lieferant",
        size="anomaly_score",
        hover_data=["Lieferant", "anomaly_kpis"],
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.75,
    )
    fig_sc.add_hline(y=0.5, line_dash="dot", line_color="rgba(239,68,68,0.67)",
                     annotation_text="Schwelle")
    fig_sc.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,22,35,0.6)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_sc, use_container_width=True)
 
    # KPI breakdown
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Anomalien nach Lieferant")
        by_sup = df_anom["Lieferant"].value_counts().reset_index()
        by_sup.columns = ["Lieferant", "Anzahl"]
        fig_sup = px.bar(by_sup, x="Anzahl", y="Lieferant", orientation="h",
                         color="Anzahl", color_continuous_scale="Reds",
                         template="plotly_dark")
        fig_sup.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)",
                              coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_sup, use_container_width=True)
 
    with col_r:
        st.markdown("### Betroffene KPIs")
        kpi_count: dict[str, int] = {}
        for row in df_anom["anomaly_kpis"]:
            for k in str(row).split(", "):
                if k and k != "—":
                    kpi_count[k] = kpi_count.get(k, 0) + 1
        if kpi_count:
            kpi_df = pd.DataFrame(kpi_count.items(), columns=["KPI", "Anzahl"])
            fig_kpi = px.pie(kpi_df, names="KPI", values="Anzahl",
                             template="plotly_dark",
                             color_discrete_sequence=px.colors.qualitative.Bold)
            fig_kpi.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)",
                                  margin=dict(l=0,r=0,t=10,b=10))
            st.plotly_chart(fig_kpi, use_container_width=True)
 
    # Table
    st.markdown("### Anomalie-Tabelle")
    cols_show = ["Datum", "Lieferant", "anomaly_score", "anomaly_kpis"] + kpi_cols
    st.dataframe(
        df_anom[cols_show]
        .sort_values("anomaly_score", ascending=False)
        .rename(columns={"anomaly_score": "Score", "anomaly_kpis": "Betroffene KPIs"})
        .style.bar(subset=["Score"], color="#ef444466"),
        use_container_width=True,
        height=340,
    )
 
 
# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("## 📊 Lieferantenvergleich")
 
    # Score bar chart
    scores = []
    for s in selected_suppliers:
        info = score_supplier(df[df["Lieferant"] == s])
        scores.append({"Lieferant": s, "Score": info["score"], "Note": info["grade"]})
    df_scores = pd.DataFrame(scores).sort_values("Score", ascending=True)
 
    fig_sc = px.bar(
        df_scores, x="Score", y="Lieferant", orientation="h",
        color="Score",
        color_continuous_scale=[[0, "#ef4444"], [0.6, "#f59e0b"], [1, "#22c55e"]],
        text="Score",
        template="plotly_dark",
        range_color=[50, 100],
    )
    fig_sc.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_sc.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,22,35,0.6)",
        coloraxis_showscale=False,
        margin=dict(l=0, r=40, t=10, b=0),
    )
    st.plotly_chart(fig_sc, use_container_width=True)
 
    # Box plots per KPI
    st.markdown("### KPI-Verteilungen im Vergleich")
    kpi_box = st.selectbox("KPI", kpi_cols, key="box_kpi")
    fig_box = px.box(
        df, x="Lieferant", y=kpi_box, color="Lieferant",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
        points="outliers",
    )
    target_val = KPI_DEFINITIONS[kpi_box]["target"]
    fig_box.add_hline(y=target_val, line_dash="dot", line_color="#f59e0b",
                      annotation_text=f"Ziel {target_val}")
    fig_box.update_layout(
        height=360,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,22,35,0.6)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_box, use_container_width=True)
 
    # Anomaly rate table
    st.markdown("### Anomalie-Rate je Lieferant")
    anom_table = (
        df.groupby("Lieferant")[method_col]
        .agg(["sum", "count"])
        .rename(columns={"sum": "Anomalien", "count": "Gesamt"})
    )
    anom_table["Rate (%)"] = (anom_table["Anomalien"] / anom_table["Gesamt"] * 100).round(1)
    anom_table["Score"] = [score_supplier(df[df["Lieferant"] == s])["score"] for s in anom_table.index]
    anom_table["Note"]  = [score_supplier(df[df["Lieferant"] == s])["grade"] for s in anom_table.index]
    st.dataframe(
        anom_table.style.bar(subset=["Rate (%)"], color="#ef444466")
                        .bar(subset=["Score"], color="#22c55e66"),
        use_container_width=True,
    )
 
 
# ════════════════════════════════════════════════════════════════════════════
# TAB 5 – RAW DATA
# ════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("## 🗂️ Rohdaten")
 
    col_filter = st.multiselect(
        "Spalten anzeigen",
        options=df.columns.tolist(),
        default=["Datum", "Lieferant"] + kpi_cols + ["anomaly_consensus", "anomaly_score", "anomaly_kpis"],
    )
 
    only_anomalies = st.checkbox("Nur Anomalien anzeigen", value=False)
    df_view = df[col_filter].copy()
    if only_anomalies:
        df_view = df_view[df[method_col]]
 
    st.dataframe(df_view.reset_index(drop=True), use_container_width=True, height=480)
 
    # Download buttons
    c1, c2 = st.columns(2)
    with c1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSV herunterladen", csv_bytes, "lieferanten_kpis.csv", "text/csv")
    with c2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="KPIs")
            df[df[method_col]].to_excel(writer, index=False, sheet_name="Anomalien")
        st.download_button(
            "⬇️ Excel herunterladen",
            buf.getvalue(),
            "lieferanten_kpis.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
 
            "⬇️ Excel herunterladen",
            buf.getvalue(),
            "lieferanten_kpis.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
