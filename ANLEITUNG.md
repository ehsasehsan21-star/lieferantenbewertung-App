# 📦 Lieferantenbewertungs-App — Schritt-für-Schritt-Anleitung

## Was macht die App?
Diese App bewertet Lieferantenleistungen anhand von KPIs (Key Performance Indicators) und
erkennt automatisch Anomalien (ungewöhnliche Abweichungen) mit drei statistischen Methoden:
- **Z-Score** – wie weit weicht ein Wert vom Durchschnitt ab?
- **IQR** – liegt der Wert außerhalb des normalen Wertebereichs (Box-Plot)?
- **Isolation Forest** – KI-basierte multivariate Anomalieerkennung (scikit-learn)

---

## 🗂️ Projektstruktur

```
lieferanten_app/
├── app.py                ← Hauptdatei: Streamlit-UI
├── data_generator.py     ← Synthetische Testdaten (ersetzbar durch echte Daten)
├── anomaly_detection.py  ← Alle Anomalie-Algorithmen + Scoring
└── requirements.txt      ← Python-Pakete
```

---

## 🚀 Schritt-für-Schritt: App lokal starten

### Schritt 1 — Python prüfen
Öffne das Terminal (Finder → Programme → Dienstprogramme → Terminal):
```bash
python3 --version
```
✅ Du brauchst Python 3.9 oder neuer. Falls nicht installiert:
→ https://www.python.org/downloads/

---

### Schritt 2 — Projektordner anlegen & hineinwechseln
```bash
mkdir ~/lieferanten_app
cd ~/lieferanten_app
```

---

### Schritt 3 — Virtuelle Umgebung erstellen
Eine virtuelle Umgebung isoliert deine Pakete vom System-Python.
```bash
python3 -m venv venv
source venv/bin/activate
```
Du siehst jetzt `(venv)` am Anfang jeder Zeile — das ist korrekt!

---

### Schritt 4 — Pakete installieren
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Das dauert 1–2 Minuten. Pakete werden heruntergeladen:
- `streamlit` → Web-Framework
- `pandas` / `numpy` → Datentabellen und Zahlen
- `plotly` → Interaktive Diagramme
- `scikit-learn` → Isolation Forest Algorithmus
- `scipy` → Z-Score Berechnung
- `openpyxl` → Excel-Dateien lesen/schreiben

---

### Schritt 5 — App starten
```bash
streamlit run app.py
```
Der Browser öffnet sich automatisch unter **http://localhost:8501** 🎉

---

## 📐 Wie funktioniert der Code?

### `data_generator.py`
```
generate_supplier_data()
  → erstellt 24 Monate Monatsdaten für 5 Lieferanten
  → jeder Lieferant hat leicht unterschiedliche Basiswerte
  → ~5% der Datenpunkte werden als Anomalien "injiziert"
```
**Echte Daten einbinden:** Ersetze `generate_supplier_data()` in `app.py` durch:
```python
df = pd.read_csv("deine_daten.csv", parse_dates=["Datum"])
```
Deine CSV-Datei braucht diese Spalten:
```
Datum, Lieferant, Liefertreue (%), Qualitätsrate (%), Durchlaufzeit (Tage),
Reklamationsquote (%), Preisabweichung (%), Reaktionszeit (Std.)
```

---

### `anomaly_detection.py`
Drei Methoden kombiniert:

| Methode | Typ | Funktionsprinzip |
|---|---|---|
| Z-Score | univariat | Abstand vom Mittelwert in Standardabweichungen |
| IQR | univariat | Werte außerhalb Quartil-Grenzen (×1.5) |
| Isolation Forest | multivariat | KI isoliert Ausreißer durch zufällige Splits |

Konsens-Regel: Ein Punkt gilt als Anomalie, wenn **≥ 2 Methoden** ihn markieren.

---

### `app.py`
Die App ist in 5 Tabs aufgeteilt:

| Tab | Inhalt |
|---|---|
| 🏠 Übersicht | Scorecards, Trendlinien, KPI-Heatmap |
| 🏭 Lieferantendetail | KPI-Subplots, Radar-Chart pro Lieferant |
| 🚨 Anomalien | Zeitreihe, Scatter, Tabelle aller Anomalien |
| 📊 Vergleich | Ranking, Box-Plots, Anomalierate |
| 🗂️ Rohdaten | Alle Daten, CSV/Excel-Export |

---

## ❓ Häufige Fehler & Lösungen

| Fehler | Lösung |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements.txt` erneut ausführen |
| `command not found: streamlit` | Sicherstellen, dass venv aktiv ist: `source venv/bin/activate` |
| Browser öffnet sich nicht | Manuell öffnen: http://localhost:8501 |
| Port 8501 belegt | `streamlit run app.py --server.port 8502` |

---

## 🔄 App stoppen
Im Terminal: **Strg + C**

## 🔄 App erneut starten
```bash
cd ~/lieferanten_app
source venv/bin/activate
streamlit run app.py
```
