# 🪐 Exoplanet Explorer

An interactive Streamlit dashboard for exploring confirmed exoplanets from the
[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).

## Features

| Tab | Description |
|-----|-------------|
| 📊 Overview | Discoveries per year, discovery-method breakdown, planet-radius histogram |
| 🔭 Scatter Plots | Fully configurable mass–radius diagram and orbital-period vs. stellar temperature chart |
| 🗺️ Sky Map | Sky distribution of host stars plotted by RA / Dec |
| 📋 Raw Data | Filterable table with CSV download |

**Sidebar filters** let you narrow results by discovery method, year range, and maximum distance.

## Getting Started

### Prerequisites

* Python ≥ 3.9

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (default: http://localhost:8501).

> **Offline / no-network mode** – if the NASA Exoplanet Archive cannot be reached a synthetic
> demo dataset is shown automatically, so the app always starts.

## Data Source

Live data is fetched on first load via the NASA Exoplanet Archive TAP service and cached for
the session:

```
https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=…&format=csv
```

## Tech Stack

* [Streamlit](https://streamlit.io) – web app framework
* [pandas](https://pandas.pydata.org/) – data manipulation
* [Plotly](https://plotly.com/python/) – interactive charts
* [NumPy](https://numpy.org/) – numerical helpers
* [Requests](https://requests.readthedocs.io/) – HTTP data fetching