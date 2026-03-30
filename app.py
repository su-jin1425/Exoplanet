import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Exoplanet Explorer",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading ──────────────────────────────────────────────────────────────
NASA_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    "select+pl_name,hostname,disc_year,discoverymethod,"
    "pl_orbper,pl_rade,pl_bmasse,pl_eqt,st_teff,st_rad,st_mass,"
    "sy_dist,pl_orbsmax,rastr,decstr"
    "+from+ps"
    "+where+default_flag=1"
    "&format=csv"
)


@st.cache_data(show_spinner="Loading exoplanet data from NASA Exoplanet Archive…")
def load_data() -> pd.DataFrame:
    """Fetch confirmed planets table from NASA Exoplanet Archive (TAP)."""
    try:
        response = requests.get(NASA_TAP_URL, timeout=30)
        response.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
    except Exception:
        # Fallback: generate a small synthetic dataset so the app still runs
        st.warning(
            "⚠️ Could not reach NASA Exoplanet Archive. "
            "Showing a synthetic demo dataset instead."
        )
        df = _synthetic_data()

    df = df.rename(
        columns={
            "pl_name": "Planet Name",
            "hostname": "Host Star",
            "disc_year": "Discovery Year",
            "discoverymethod": "Discovery Method",
            "pl_orbper": "Orbital Period (days)",
            "pl_rade": "Planet Radius (Earth radii)",
            "pl_bmasse": "Planet Mass (Earth masses)",
            "pl_eqt": "Equilibrium Temp (K)",
            "st_teff": "Stellar Temp (K)",
            "st_rad": "Stellar Radius (Solar)",
            "st_mass": "Stellar Mass (Solar)",
            "sy_dist": "Distance (pc)",
            "pl_orbsmax": "Semi-major Axis (AU)",
        }
    )
    return df


def _synthetic_data() -> pd.DataFrame:
    """Return a small synthetic exoplanet dataset for offline / fallback use."""
    rng = np.random.default_rng(42)
    n = 300
    methods = ["Transit", "Radial Velocity", "Imaging", "Microlensing", "Astrometry"]
    method_weights = [0.75, 0.18, 0.03, 0.03, 0.01]

    df = pd.DataFrame(
        {
            "pl_name": [f"Demo-{i}b" for i in range(n)],
            "hostname": [f"Star-{i}" for i in range(n)],
            "disc_year": rng.integers(1995, 2025, n),
            "discoverymethod": rng.choice(methods, n, p=method_weights),
            "pl_orbper": np.abs(rng.lognormal(3.0, 2.0, n)),
            "pl_rade": np.abs(rng.lognormal(0.5, 0.8, n)),
            "pl_bmasse": np.abs(rng.lognormal(1.5, 2.0, n)),
            "pl_eqt": rng.integers(200, 3000, n).astype(float),
            "st_teff": rng.integers(3000, 9000, n).astype(float),
            "st_rad": np.abs(rng.lognormal(0.1, 0.4, n)),
            "st_mass": np.abs(rng.lognormal(0.05, 0.35, n)),
            "sy_dist": np.abs(rng.lognormal(4.0, 1.2, n)),
            "pl_orbsmax": np.abs(rng.lognormal(-0.5, 1.5, n)),
        }
    )
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
df_raw = load_data()

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/"
    "NASA_logo.svg/200px-NASA_logo.svg.png",
    width=80,
)
st.sidebar.title("🪐 Exoplanet Explorer")
st.sidebar.markdown("Data from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).")

# Discovery method filter
all_methods = sorted(df_raw["Discovery Method"].dropna().unique())
selected_methods = st.sidebar.multiselect(
    "Discovery Method",
    options=all_methods,
    default=all_methods,
)

# Year range filter
year_min = int(df_raw["Discovery Year"].dropna().min())
year_max = int(df_raw["Discovery Year"].dropna().max())
year_range = st.sidebar.slider(
    "Discovery Year",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
)

# Distance filter
dist_max_raw = float(df_raw["Distance (pc)"].dropna().max())
dist_cap = st.sidebar.number_input(
    "Max Distance (parsecs)",
    min_value=1.0,
    max_value=float(dist_max_raw),
    value=float(dist_max_raw),
    step=100.0,
)

# Apply filters
mask = (
    df_raw["Discovery Method"].isin(selected_methods)
    & (df_raw["Discovery Year"] >= year_range[0])
    & (df_raw["Discovery Year"] <= year_range[1])
    & (df_raw["Distance (pc)"].fillna(dist_cap + 1) <= dist_cap)
)
df = df_raw[mask].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🪐 Exoplanet Explorer")
st.markdown(
    "An interactive dashboard for exploring confirmed exoplanets from the "
    "**NASA Exoplanet Archive**. Use the sidebar filters to narrow your selection."
)

# ── KPI row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Planets selected", f"{len(df):,}")
col2.metric("Host stars", f"{df['Host Star'].nunique():,}")
col3.metric(
    "Median radius (R⊕)",
    f"{df['Planet Radius (Earth radii)'].median():.2f}"
    if df["Planet Radius (Earth radii)"].notna().any()
    else "—",
)
col4.metric(
    "Median distance (pc)",
    f"{df['Distance (pc)'].median():.0f}"
    if df["Distance (pc)"].notna().any()
    else "—",
)

st.divider()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_overview, tab_scatter, tab_sky, tab_data = st.tabs(
    ["📊 Overview", "🔭 Scatter Plots", "🗺️ Sky Map", "📋 Raw Data"]
)

# ── Tab 1 – Overview ─────────────────────────────────────────────────────────
with tab_overview:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Discoveries per Year")
        yearly = (
            df.groupby("Discovery Year")
            .size()
            .reset_index(name="Count")
            .sort_values("Discovery Year")
        )
        fig_year = px.bar(
            yearly,
            x="Discovery Year",
            y="Count",
            color="Count",
            color_continuous_scale="Viridis",
            labels={"Count": "Planets discovered"},
        )
        fig_year.update_layout(coloraxis_showscale=False, margin=dict(t=20))
        st.plotly_chart(fig_year, use_container_width=True)

    with col_right:
        st.subheader("Discovery Method Breakdown")
        method_counts = df["Discovery Method"].value_counts().reset_index()
        method_counts.columns = ["Method", "Count"]
        fig_pie = px.pie(
            method_counts,
            names="Method",
            values="Count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_layout(margin=dict(t=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Planet Radius Distribution")
    radius_col = "Planet Radius (Earth radii)"
    valid_r = df[radius_col].dropna()
    if valid_r.empty:
        st.info("No radius data available for current selection.")
    else:
        fig_hist = px.histogram(
            df,
            x=radius_col,
            nbins=60,
            color="Discovery Method",
            barmode="overlay",
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Set2,
            range_x=[0, min(valid_r.quantile(0.99), 30)],
        )
        fig_hist.add_vline(
            x=1.0, line_dash="dash", line_color="white",
            annotation_text="Earth", annotation_position="top right",
        )
        fig_hist.add_vline(
            x=11.2, line_dash="dash", line_color="orange",
            annotation_text="Jupiter", annotation_position="top right",
        )
        fig_hist.update_layout(margin=dict(t=20))
        st.plotly_chart(fig_hist, use_container_width=True)

# ── Tab 2 – Scatter Plots ────────────────────────────────────────────────────
with tab_scatter:
    st.subheader("Mass–Radius Diagram")
    numeric_cols = [
        "Planet Radius (Earth radii)",
        "Planet Mass (Earth masses)",
        "Orbital Period (days)",
        "Equilibrium Temp (K)",
        "Stellar Temp (K)",
        "Distance (pc)",
        "Semi-major Axis (AU)",
    ]
    present_cols = [c for c in numeric_cols if c in df.columns]

    sc_col1, sc_col2, sc_col3 = st.columns(3)
    x_axis = sc_col1.selectbox("X axis", present_cols, index=1)
    y_axis = sc_col2.selectbox(
        "Y axis",
        present_cols,
        index=0 if present_cols[0] != x_axis else 1,
    )
    color_by = sc_col3.selectbox(
        "Color by",
        ["Discovery Method", "Discovery Year", "Equilibrium Temp (K)"],
    )

    plot_df = df[[x_axis, y_axis, color_by, "Planet Name", "Host Star"]].dropna(
        subset=[x_axis, y_axis]
    )

    if plot_df.empty:
        st.info("No data with both axes present for the current selection.")
    else:
        use_log_x = st.checkbox("Log scale X", value=True)
        use_log_y = st.checkbox("Log scale Y", value=True)

        fig_sc = px.scatter(
            plot_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            hover_name="Planet Name",
            hover_data={"Host Star": True},
            log_x=use_log_x,
            log_y=use_log_y,
            opacity=0.6,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_sc.update_traces(marker=dict(size=5))
        fig_sc.update_layout(margin=dict(t=20))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()
    st.subheader("Orbital Period vs. Stellar Effective Temperature")
    op_df = df[
        ["Orbital Period (days)", "Stellar Temp (K)", "Planet Radius (Earth radii)",
         "Discovery Method", "Planet Name"]
    ].dropna(subset=["Orbital Period (days)", "Stellar Temp (K)"])

    if op_df.empty:
        st.info("No sufficient data for this chart with current filters.")
    else:
        fig_op = px.scatter(
            op_df,
            x="Stellar Temp (K)",
            y="Orbital Period (days)",
            color="Discovery Method",
            size="Planet Radius (Earth radii)",
            size_max=18,
            hover_name="Planet Name",
            log_y=True,
            opacity=0.65,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_op.update_layout(margin=dict(t=20))
        st.plotly_chart(fig_op, use_container_width=True)

# ── Tab 3 – Sky Map ───────────────────────────────────────────────────────────
with tab_sky:
    st.subheader("Sky Distribution of Exoplanet Host Stars")
    sky_df = df[["rastr", "decstr", "Planet Name", "Host Star", "Discovery Method"]].copy()

    def _dms_to_deg(val: str, is_ra: bool) -> float | None:
        """Convert sexagesimal RA/Dec string to decimal degrees."""
        try:
            s = str(val).strip().replace("−", "-")
            sign = -1 if s.startswith("-") else 1
            s = s.lstrip("+-")
            parts = [float(p) for p in s.split()]
            deg = sign * (parts[0] + parts[1] / 60 + parts[2] / 3600)
            if is_ra:
                deg *= 15  # hours → degrees
            return deg
        except Exception:
            return None

    if "rastr" in df.columns and "decstr" in df.columns:
        sky_df["RA (deg)"] = sky_df["rastr"].apply(lambda v: _dms_to_deg(v, is_ra=True))
        sky_df["Dec (deg)"] = sky_df["decstr"].apply(lambda v: _dms_to_deg(v, is_ra=False))
        sky_df = sky_df.dropna(subset=["RA (deg)", "Dec (deg)"])

        if sky_df.empty:
            st.info("No sky-coordinate data available for the current selection.")
        else:
            fig_sky = px.scatter(
                sky_df,
                x="RA (deg)",
                y="Dec (deg)",
                color="Discovery Method",
                hover_name="Planet Name",
                hover_data={"Host Star": True},
                opacity=0.5,
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"RA (deg)": "Right Ascension (°)", "Dec (deg)": "Declination (°)"},
            )
            fig_sky.update_traces(marker=dict(size=4))
            fig_sky.update_layout(
                margin=dict(t=20),
                xaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_sky, use_container_width=True)
    else:
        st.info("Sky coordinate columns not present in the loaded dataset.")

# ── Tab 4 – Raw Data ─────────────────────────────────────────────────────────
with tab_data:
    st.subheader(f"Filtered Dataset — {len(df):,} planets")
    display_cols = [
        "Planet Name", "Host Star", "Discovery Year", "Discovery Method",
        "Planet Radius (Earth radii)", "Planet Mass (Earth masses)",
        "Orbital Period (days)", "Semi-major Axis (AU)", "Equilibrium Temp (K)",
        "Stellar Temp (K)", "Distance (pc)",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=500,
    )
    csv = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="exoplanets_filtered.csv",
        mime="text/csv",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data sourced from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) "
    "via its Table Access Protocol (TAP) service. "
    "Built with [Streamlit](https://streamlit.io) · "
    "[Plotly](https://plotly.com/python/) · "
    "[pandas](https://pandas.pydata.org/)."
)
