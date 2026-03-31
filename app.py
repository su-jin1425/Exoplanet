import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import requests
import io
matplotlib.use('Agg')

st.set_page_config(
    page_title="Exoplanet Habitability Predictor",
    layout="wide"
)

BG      = "#0d1117"
CARD    = "#161b22"
GRID    = "#21262d"
ACCENT  = "#58a6ff"
GREEN   = "#3fb950"
RED     = "#f85149"
YELLOW  = "#d29922"
PURPLE  = "#bc8cff"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"

SPACE_CMAP = LinearSegmentedColormap.from_list(
    "space", ["#1a237e", "#4a148c", "#6a1b9a", "#c2185b", "#e65100", "#f9a825"]
)


@st.cache_resource
def load_bundle():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("model.pkl not found. Run a.ipynb first.")
    st.stop()

model        = bundle['model']
base_cols    = bundle['base_feature_cols']
derived_cols = bundle['derived_feature_cols']
all_cols     = bundle['all_feature_cols']

FEATURE_LABELS = {
    'pl_orbper':  'Orbital Period (days)',
    'pl_rade':    'Planet Radius (Earth Radii)',
    'pl_masse':   'Planet Mass (Earth Masses)',
    'pl_orbsmax': 'Semi-major Axis (AU)',
    'pl_orbeccen':'Orbital Eccentricity',
    'st_teff':    'Star Temperature (K)',
    'st_rad':     'Star Radius (Solar Radii)',
    'st_mass':    'Star Mass (Solar Masses)',
    'sy_dist':    'Distance from Earth (pc)'
}
DERIVED_LABELS = {
    't_eq':         'Equilibrium Temperature (K)',
    'stellar_flux': 'Stellar Flux (x Earth)',
    'hz_ratio':     'HZ Ratio (1.0 = center)'
}
FEATURE_LABELS_SHORT = {
    'pl_orbper':   'Orbital Period',
    'pl_rade':     'Planet Radius',
    'pl_masse':    'Planet Mass',
    'pl_orbsmax':  'Semi-major Axis',
    'pl_orbeccen': 'Eccentricity',
    'st_teff':     'Star Temp',
    'st_rad':      'Star Radius',
    'st_mass':     'Star Mass',
    'sy_dist':     'Distance',
    't_eq':        'Equil. Temp *',
    'stellar_flux':'Stellar Flux *',
    'hz_ratio':    'HZ Ratio *',
}
FEATURE_DEFAULTS = {
    'pl_orbper': 365.25, 'pl_rade': 1.0,   'pl_masse': 1.0,
    'pl_orbsmax': 1.0,   'pl_orbeccen': 0.02, 'st_teff': 5778.0,
    'st_rad': 1.0,       'st_mass': 1.0,   'sy_dist': 10.0
}
FEATURE_RANGES = {
    'pl_orbper':  (0.1,   10000.0),
    'pl_rade':    (0.1,   30.0),
    'pl_masse':   (0.01,  5000.0),
    'pl_orbsmax': (0.001, 100.0),
    'pl_orbeccen':(0.0,   1.0),
    'st_teff':    (2000.0,50000.0),
    'st_rad':     (0.01,  100.0),
    'st_mass':    (0.01,  100.0),
    'sy_dist':    (0.1,   10000.0)
}
PRESETS = {
    "Earth Analog": {
        'pl_orbper': 365.25, 'pl_rade': 1.0,  'pl_masse': 1.0,
        'pl_orbsmax': 1.0,   'pl_orbeccen': 0.02, 'st_teff': 5778.0,
        'st_rad': 1.0,       'st_mass': 1.0,  'sy_dist': 10.0
    },
    "Hot Jupiter": {
        'pl_orbper': 3.5,  'pl_rade': 13.0, 'pl_masse': 950.0,
        'pl_orbsmax': 0.04,'pl_orbeccen': 0.01, 'st_teff': 5800.0,
        'st_rad': 1.1,     'st_mass': 1.05, 'sy_dist': 150.0
    },
    "Super-Earth HZ": {
        'pl_orbper': 210.0,'pl_rade': 1.7,  'pl_masse': 5.5,
        'pl_orbsmax': 0.9, 'pl_orbeccen': 0.06, 'st_teff': 4950.0,
        'st_rad': 0.78,    'st_mass': 0.79, 'sy_dist': 38.0
    },
    "Lava World": {
        'pl_orbper': 0.85, 'pl_rade': 1.6,  'pl_masse': 6.0,
        'pl_orbsmax': 0.012,'pl_orbeccen': 0.0, 'st_teff': 5500.0,
        'st_rad': 0.9,     'st_mass': 0.88, 'sy_dist': 22.0
    },
    "Ice Giant": {
        'pl_orbper': 4380.0,'pl_rade': 4.0, 'pl_masse': 17.0,
        'pl_orbsmax': 19.2, 'pl_orbeccen': 0.04, 'st_teff': 5800.0,
        'st_rad': 1.0,      'st_mass': 1.0, 'sy_dist': 10.0
    }
}
PRESET_DESCRIPTIONS = {
    "Earth Analog":   "Sun-like star, 1 AU orbit. Should predict habitable.",
    "Hot Jupiter":    "Gas giant in 3.5-day orbit. Clearly not habitable.",
    "Super-Earth HZ": "Rocky planet around a K-dwarf in the habitable zone.",
    "Lava World":     "Tidally locked, extremely close orbit. Surface is molten.",
    "Ice Giant":      "Uranus-like: large, cold, far from star."
}


def compute_derived(vals):
    st_rad  = max(vals.get('st_rad', 1.0), 0.01)
    st_teff = max(vals.get('st_teff', 5778.0), 500.0)
    a_au    = max(vals.get('pl_orbsmax', 1.0), 1e-4)
    st_rad_au  = st_rad * 0.00465
    luminosity = ((st_teff / 5778.0) ** 4) * (st_rad ** 2)
    t_eq        = st_teff * np.sqrt(st_rad_au / (2.0 * a_au)) * (0.7 ** 0.25)
    stellar_flux= luminosity / (a_au ** 2)
    hz_center   = max(np.sqrt(luminosity), 1e-6)
    hz_ratio    = a_au / hz_center
    return {'t_eq': t_eq, 'stellar_flux': stellar_flux, 'hz_ratio': hz_ratio}


def habitability_factors(vals, derived):
    t_eq         = derived['t_eq']
    stellar_flux = derived['stellar_flux']
    hz_ratio     = derived['hz_ratio']
    pl_rade      = vals.get('pl_rade', 1.0)
    pl_masse     = vals.get('pl_masse', 1.0)
    pl_orbeccen  = vals.get('pl_orbeccen', 0.0)
    st_teff      = vals.get('st_teff', 5778.0)
    return [
        ('Equilibrium Temp',  f'{t_eq:.1f} K',
         175 <= t_eq <= 340,          '175-340 K'),
        ('Planet Radius',     f'{pl_rade:.2f} Re',
         0.5 <= pl_rade <= 2.5,       '0.5-2.5 Re'),
        ('Planet Mass',       f'{pl_masse:.2f} Me',
         pl_masse <= 13.0,            '<=13 Me'),
        ('Stellar Flux',      f'{stellar_flux:.3f} Se',
         0.2 <= stellar_flux <= 2.5,  '0.2-2.5 Se'),
        ('Eccentricity',      f'{pl_orbeccen:.3f}',
         pl_orbeccen <= 0.4,          '<=0.4'),
        ('Star Temp',         f'{st_teff:.0f} K',
         3500 <= st_teff <= 7500,     '3500-7500 K'),
        ('HZ Ratio',          f'{hz_ratio:.3f}',
         0.5 <= hz_ratio <= 2.0,      '0.5-2.0'),
    ]


@st.cache_data(ttl=3600)
def load_sample_data():
    cols = "pl_name,hostname,pl_orbper,pl_rade,pl_masse,pl_orbsmax,pl_orbeccen,st_teff,st_rad,st_mass,sy_dist"
    url  = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        f"?query=select+top+50+{cols}+from+ps"
        "&format=csv"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), comment='#')
        base = ['pl_orbper', 'pl_rade', 'pl_masse', 'pl_orbsmax', 'pl_orbeccen',
                'st_teff', 'st_rad', 'st_mass', 'sy_dist']
        df = df.dropna(thresh=int(len(base) * 0.55)).reset_index(drop=True)

        st_rad_s  = df['st_rad'].fillna(1.0).clip(lower=0.01)
        st_teff_s = df['st_teff'].fillna(5778.0).clip(lower=500)
        a_au_s    = df['pl_orbsmax'].clip(lower=1e-4)
        st_rad_au = st_rad_s * 0.00465
        luminosity = ((st_teff_s / 5778.0) ** 4) * (st_rad_s ** 2)
        df['t_eq']         = st_teff_s * np.sqrt(st_rad_au / (2.0 * a_au_s)) * (0.7 ** 0.25)
        df['stellar_flux'] = luminosity / (a_au_s ** 2)
        hz_center          = np.sqrt(luminosity).clip(lower=1e-6)
        df['hz_ratio']     = a_au_s / hz_center

        t_eq_ok   = df['t_eq'].between(175, 340)
        radius_ok = df['pl_rade'].between(0.5, 2.5)
        flux_ok   = df['stellar_flux'].between(0.2, 2.5)
        ecc_ok    = df['pl_orbeccen'].fillna(0.0) <= 0.4
        star_ok   = df['st_teff'].between(3500, 7500)
        mass_ok   = df['pl_masse'].fillna(df['pl_rade'] ** 2.5) <= 13.0
        df['Habitable'] = (t_eq_ok & radius_ok & flux_ok & mass_ok & ecc_ok & star_ok).map(
            {True: 'Yes', False: 'No'}
        )
        return df, None
    except Exception as e:
        return None, str(e)


def make_probability_donut(hab_prob):
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    fill_color = GREEN if hab_prob >= 0.5 else RED
    not_prob   = 1 - hab_prob
    ax.pie(
        [hab_prob, not_prob],
        colors=[fill_color, GRID],
        startangle=90,
        wedgeprops=dict(width=0.38, edgecolor=BG, linewidth=4),
        counterclock=False,
    )
    ax.text(0, 0.12, f"{hab_prob*100:.1f}%",
            ha='center', va='center', fontsize=28, fontweight='bold', color=fill_color)
    ax.text(0, -0.18,
            "Potentially Habitable" if hab_prob >= 0.5 else "Not Habitable",
            ha='center', va='center', fontsize=10, color=TEXT)
    ax.set_title("Habitability Score", color=TEXT, fontsize=12, fontweight='bold', pad=14)
    ax.axis('off')
    plt.tight_layout()
    return fig


def make_factor_radar(factors):
    labels = [f[0] for f in factors]
    passes = [1.0 if f[2] else 0.0 for f in factors]
    N      = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    passes_plot = passes + [passes[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)

    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(np.linspace(0, 2*np.pi, 200), [r]*200,
                color=GRID, linewidth=0.8, alpha=0.7, zorder=1)
    for angle in angles:
        ax.plot([angle, angle], [0, 1.05], color=GRID, linewidth=0.6, alpha=0.5, zorder=1)

    n_pass = sum(f[2] for f in factors)
    fill_color = GREEN if n_pass >= 6 else (YELLOW if n_pass >= 4 else RED)
    ax.fill(angles_plot, passes_plot, alpha=0.20, color=fill_color, zorder=2)
    ax.plot(angles_plot, passes_plot, color=fill_color, linewidth=2.5, zorder=3)
    for angle, val, passed in zip(angles, passes, [f[2] for f in factors]):
        ax.scatter([angle], [val], color=GREEN if passed else RED,
                   s=80, zorder=5, edgecolors=BG, linewidth=1.5)

    ax.set_xticks(angles)
    ax.set_xticklabels([l.replace(' ', '\n') for l in labels],
                       color=TEXT, fontsize=8.5)
    ax.set_yticks([])
    ax.set_ylim(0, 1.18)
    ax.spines['polar'].set_color(GRID)
    ax.set_title(f"Factors  ({n_pass}/{N} pass)", color=TEXT,
                 fontsize=12, fontweight='bold', pad=18)
    plt.tight_layout()
    return fig


def make_importance_chart(fi):
    NEGLIGIBLE = 1e-3
    fi_df = pd.DataFrame(list(fi.items()), columns=['Feature', 'Importance'])
    fi_df['label']   = fi_df['Feature'].map(lambda f: FEATURE_LABELS_SHORT.get(f, f))
    fi_df['clipped'] = np.where(np.abs(fi_df['Importance']) < NEGLIGIBLE, 0.0, fi_df['Importance'])
    fi_df = fi_df.sort_values('clipped', ascending=True)

    sig = fi_df[fi_df['clipped'] >= NEGLIGIBLE]
    ngl = fi_df[fi_df['clipped'] <  NEGLIGIBLE]

    fig = plt.figure(figsize=(12, 5.5), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.8, 1], wspace=0.38,
                             left=0.03, right=0.97, top=0.86, bottom=0.13)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(CARD)
    x_max = sig['clipped'].max() if len(sig) else 0.1
    norm  = sig['clipped'] / x_max
    colors = [SPACE_CMAP(float(v)) for v in norm]

    bars = ax1.barh(sig['label'], sig['clipped'], color=colors,
                    height=0.58, edgecolor=BG, linewidth=0.8)
    for bar, col in zip(bars, colors):
        w = bar.get_width()
        ax1.barh(bar.get_y() + bar.get_height()/2, w * 0.88,
                 height=bar.get_height() * 0.35, color='white',
                 alpha=0.08, left=0, zorder=4)

    for bar, val in zip(bars, sig['clipped']):
        ax1.text(val + x_max * 0.028,
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', fontsize=10,
                 fontweight='bold', color=TEXT)

    ax1.set_xlim(0, x_max * 1.32)
    ax1.set_xlabel('Permutation Importance (Balanced Accuracy drop)', color=SUBTEXT, fontsize=10)
    ax1.set_title('Key Predictors', color=TEXT, fontsize=13, fontweight='bold', pad=10)
    ax1.tick_params(colors=TEXT, labelsize=10)
    for sp in ax1.spines.values(): sp.set_edgecolor(GRID)
    ax1.grid(axis='x', color=GRID, linewidth=0.5, alpha=0.7)
    ax1.set_axisbelow(True)
    ax1.xaxis.set_tick_params(labelcolor=SUBTEXT)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(CARD)
    y_pos = list(range(len(ngl)))
    raw   = ngl['Importance'].values
    for y, x in zip(y_pos, raw):
        ax2.plot([0, x], [y, y], color=GRID, linewidth=1.5, zorder=1)
    dot_colors = [GREEN if v >= 0 else RED for v in raw]
    ax2.scatter(raw, y_pos, color=dot_colors, s=65, zorder=3,
                edgecolors=BG, linewidth=1.2)
    ax2.axvline(0, color=SUBTEXT, linewidth=1, linestyle='--', alpha=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(ngl['label'], fontsize=9, color=TEXT)
    ax2.set_xlabel('Raw value', color=SUBTEXT, fontsize=9)
    ax2.set_title('Negligible\n(< 0.001)', color=SUBTEXT, fontsize=11, fontweight='bold', pad=8)
    ax2.tick_params(colors=SUBTEXT, labelsize=8)
    for sp in ax2.spines.values(): sp.set_edgecolor(GRID)
    ax2.grid(axis='x', color=GRID, linewidth=0.4, alpha=0.6)
    ax2.set_axisbelow(True)

    fig.suptitle('Feature Importance — Permutation Method  (* = derived feature)',
                 color=TEXT, fontsize=12, fontweight='bold', y=0.97)
    return fig


def make_roc_pr_chart(bundle):
    roc_data = bundle.get('roc_curve', {})
    fpr      = roc_data.get('fpr', [])
    tpr      = roc_data.get('tpr', [])
    roc_auc  = bundle.get('roc_auc',  float('nan'))
    report   = bundle.get('report', {})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=BG)
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.97, top=0.86, bottom=0.15)

    ax = axes[0]
    ax.set_facecolor(CARD)
    if fpr and tpr:
        ax.plot([0,1], [0,1], '--', color=SUBTEXT, linewidth=1.2, alpha=0.5, label='Random')
        ax.fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
        ax.plot(fpr, tpr, color=ACCENT, linewidth=2.5, label=f'ROC-AUC = {roc_auc:.3f}')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.07)
    ax.set_xlabel('False Positive Rate', color=SUBTEXT, fontsize=10)
    ax.set_ylabel('True Positive Rate', color=SUBTEXT, fontsize=10)
    ax.set_title('ROC Curve', color=TEXT, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, facecolor=GRID, edgecolor=GRID, labelcolor=TEXT, loc='lower right')
    ax.tick_params(colors=SUBTEXT)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.7)

    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    classes = ['Not Habitable', 'Habitable']
    metrics = ['precision', 'recall', 'f1-score']
    x       = np.arange(len(metrics))
    width   = 0.28
    bar_cols = [RED, GREEN]
    for i, (cls, col) in enumerate(zip(classes, bar_cols)):
        if cls in report:
            vals   = [report[cls][m] for m in metrics]
            offset = (i - 0.5) * width
            b = ax2.bar(x + offset, vals, width, color=col,
                        alpha=0.85, edgecolor=BG, linewidth=0.8, label=cls)
            for bar, v in zip(b, vals):
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.018,
                         f'{v:.2f}', ha='center', va='bottom',
                         fontsize=8.5, color=TEXT, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Precision', 'Recall', 'F1-Score'], color=TEXT, fontsize=10)
    ax2.set_ylim(0, 1.22)
    ax2.set_title('Per-Class Metrics', color=TEXT, fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, facecolor=GRID, edgecolor=GRID, labelcolor=TEXT)
    ax2.tick_params(colors=SUBTEXT)
    for sp in ax2.spines.values(): sp.set_edgecolor(GRID)
    ax2.grid(axis='y', color=GRID, linewidth=0.5, alpha=0.7)
    ax2.set_axisbelow(True)

    fig.suptitle('Model Performance', color=TEXT, fontsize=13, fontweight='bold', y=0.97)
    return fig


def make_cv_chart(bundle):
    keys = [
        ('Accuracy',      'accuracy',         None,              None),
        ('Balanced Acc',  'balanced_accuracy', None,              None),
        ('ROC-AUC',       'roc_auc',          None,              None),
        ('Avg Precision', 'avg_precision',     None,              None),
        ('CV Bal Acc',    'cv_bal_acc_mean',   'cv_bal_acc_std',  True),
        ('CV ROC-AUC',    'cv_roc_mean',       'cv_roc_std',      True),
    ]
    labels = [k[0] for k in keys]
    vals   = [bundle.get(k[1], float('nan')) for k in keys]
    errs   = [bundle.get(k[2], 0) if k[2] else 0 for k in keys]
    is_cv  = [bool(k[3]) for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.22)
    ax.set_facecolor(CARD)

    x      = np.arange(len(labels))
    norm_v = np.array([v if not np.isnan(v) else 0 for v in vals])
    colors = [SPACE_CMAP(float(v)) for v in norm_v]
    hatches = ['///' if cv else '' for cv in is_cv]

    for xi, val, err, col, hatch in zip(x, vals, errs, colors, hatches):
        bar = ax.bar(xi, val, color=col, width=0.55, edgecolor=BG,
                     linewidth=0.8, hatch=hatch,
                     yerr=err if err else None, capsize=5,
                     error_kw=dict(ecolor=SUBTEXT, capthick=1.5, elinewidth=1.5))
        if not np.isnan(val):
            ax.text(xi, val + max(err, 0) + 0.016, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=9,
                    color=TEXT, fontweight='bold')

    ax.axhline(1.0, color=SUBTEXT, linewidth=0.8, linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha='right', color=TEXT, fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel('Score', color=SUBTEXT, fontsize=10)
    ax.set_title('Metrics Overview  (/// = cross-validated)',
                 color=TEXT, fontsize=12, fontweight='bold')
    ax.tick_params(colors=SUBTEXT)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.grid(axis='y', color=GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    return fig


st.title("Exoplanet Habitability Predictor")
st.markdown(
    "<span style='color:#8b949e'>Model: Histogram Gradient Boosting"
    " &nbsp;&middot;&nbsp; Trained on NASA Exoplanet Archive</span>",
    unsafe_allow_html=True
)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Feature Analysis", "Model Info", "Sample Data"])

with tab1:
    st.subheader("Quick Presets")
    preset_cols = st.columns(len(PRESETS))
    for idx, (name, vals) in enumerate(PRESETS.items()):
        with preset_cols[idx]:
            if st.button(name, use_container_width=True, key=f"btn_{name}"):
                for feat, val in vals.items():
                    if feat in base_cols:
                        st.session_state[f"inp_{feat}"] = float(val)
                st.rerun()
            st.caption(PRESET_DESCRIPTIONS[name])

    st.markdown("---")
    st.subheader("Planet Parameters")
    input_vals = {}
    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]
    for i, feat in enumerate(base_cols):
        label   = FEATURE_LABELS.get(feat, feat)
        default = float(FEATURE_DEFAULTS.get(feat, 1.0))
        lo, hi  = FEATURE_RANGES.get(feat, (0.0, 1000.0))
        current = float(st.session_state.get(f"inp_{feat}", default))
        with cols[i % 3]:
            input_vals[feat] = st.number_input(
                label, min_value=lo, max_value=hi,
                value=current, step=(hi - lo) / 200,
                format="%.4f", key=f"inp_{feat}"
            )

    st.markdown("---")
    derived = compute_derived(input_vals)
    dcol1, dcol2, dcol3 = st.columns(3)
    dcol1.metric("Equilibrium Temperature", f"{derived['t_eq']:.1f} K",
                 help="Blackbody surface temperature assuming albedo 0.3")
    dcol2.metric("Stellar Flux", f"{derived['stellar_flux']:.3f} x Earth",
                 help="Stellar energy flux relative to Earth")
    dcol3.metric("HZ Ratio", f"{derived['hz_ratio']:.3f}",
                 help="1.0 = centre of habitable zone.")

    if st.button("Predict Habitability", type="primary"):
        row = {feat: input_vals.get(feat, FEATURE_DEFAULTS.get(feat, 0.0)) for feat in base_cols}
        row.update(derived)
        x = np.array([[row[c] for c in all_cols]])

        proba      = model.predict_proba(x)[0]
        mc         = list(model.classes_)
        pos_idx    = mc.index(1) if 1 in mc else None
        hab_prob   = proba[pos_idx] if pos_idx is not None else 0.0
        pred_label = 1 if hab_prob >= 0.5 else 0

        st.markdown("### Prediction Result")
        donut_col, radar_col = st.columns(2)
        factors = habitability_factors(input_vals, derived)

        with donut_col:
            st.pyplot(make_probability_donut(hab_prob), use_container_width=True)

        with radar_col:
            st.pyplot(make_factor_radar(factors), use_container_width=True)

        st.markdown("#### Factor Breakdown")
        n_pass = sum(1 for f in factors if f[2])
        st.markdown(
            f"<div style='color:{SUBTEXT};margin-bottom:8px'>"
            f"{n_pass} of {len(factors)} habitability criteria met</div>",
            unsafe_allow_html=True
        )
        for name_f, val_f, passes, threshold in factors:
            icon  = "Pass" if passes else "Fail"
            bg    = "#0d2318" if passes else "#2b0f0f"
            color = GREEN    if passes else RED
            st.markdown(
                f"<div style='background:{bg};border-left:3px solid {color};"
                f"padding:7px 14px;border-radius:4px;margin:4px 0;"
                f"display:flex;justify-content:space-between;align-items:center'>"
                f"<span><b style='color:{TEXT}'>{name_f}</b></span>"
                f"<span style='color:{TEXT};font-family:monospace;font-size:0.95em'>{val_f}</span>"
                f"<span style='color:{SUBTEXT};font-size:0.82em'>{threshold}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

with tab2:
    st.subheader("Feature Importance")
    st.caption(
        "Permutation importance: how much balanced accuracy drops when a feature is randomly shuffled "
        "on the held-out test set. * marks derived features. Values below 0.001 are noise-level."
    )
    fi = bundle.get('feature_importance', {})
    if fi:
        st.pyplot(make_importance_chart(fi), use_container_width=True)
    else:
        st.info("No feature importance found in bundle. Re-run a.ipynb.")

    st.markdown("---")
    st.subheader("Habitability Criteria Reference")
    crit_df = pd.DataFrame([
        ('Equilibrium Temp', '175 - 340 K',       'Physics-derived from star and orbit'),
        ('Planet Radius',    '0.5 - 2.5 Re',      'Rocky planet range'),
        ('Planet Mass',      '<= 13 Me',           'Below gas giant threshold'),
        ('Stellar Flux',     '0.2 - 2.5 Se',      'Energy flux for liquid water'),
        ('Eccentricity',     '<= 0.4',            'Stable, not too elliptical'),
        ('Star Temperature', '3500 - 7500 K',     'M to F-type stars'),
        ('HZ Ratio',         '0.5 - 2.0',         'Within the habitable zone'),
    ], columns=['Criterion', 'Threshold', 'Rationale'])
    st.dataframe(crit_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Preset Comparison")
    preset_rows = []
    for pname, pvals in PRESETS.items():
        d = compute_derived(pvals)
        preset_rows.append({
            'Preset': pname,
            'Radius (Re)': pvals['pl_rade'],
            'Mass (Me)':   pvals['pl_masse'],
            'SMA (AU)':    pvals['pl_orbsmax'],
            'T_star (K)':  pvals['st_teff'],
            'T_eq (K)':    round(d['t_eq'], 1),
            'Flux (Se)':   round(d['stellar_flux'], 3)
        })
    st.dataframe(pd.DataFrame(preset_rows), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Sample Data from NASA Exoplanet Archive")
    st.caption(
        "Live fetch of 50 planets from the NASA TAP API. "
        "Derived columns (Equil. Temp, Stellar Flux, HZ Ratio) and habitability label "
        "are computed on the fly using the same logic as the model."
    )

    sample_df, err = load_sample_data()

    if err:
        st.error(f"Could not fetch data from NASA archive: {err}")
    else:
        total   = len(sample_df)
        n_hab   = (sample_df['Habitable'] == 'Yes').sum()
        n_not   = total - n_hab

        m1, m2, m3 = st.columns(3)
        m1.metric("Planets Shown", total)
        m2.metric("Potentially Habitable", int(n_hab))
        m3.metric("Not Habitable", int(n_not))

        st.markdown("---")

        DISPLAY_COLS = {
            'pl_name':     'Planet Name',
            'hostname':    'Host Star',
            'pl_orbper':   'Orbital Period (days)',
            'pl_rade':     'Radius (Re)',
            'pl_masse':    'Mass (Me)',
            'pl_orbsmax':  'Semi-major Axis (AU)',
            'pl_orbeccen': 'Eccentricity',
            'st_teff':     'Star Temp (K)',
            'st_mass':     'Star Mass (Mo)',
            'sy_dist':     'Distance (pc)',
            't_eq':        'Equil. Temp (K)',
            'stellar_flux':'Stellar Flux (Se)',
            'hz_ratio':    'HZ Ratio',
            'Habitable':   'Habitable',
        }

        display_df = sample_df[[c for c in DISPLAY_COLS if c in sample_df.columns]].copy()
        display_df = display_df.rename(columns=DISPLAY_COLS)

        for col in ['Orbital Period (days)', 'Radius (Re)', 'Mass (Me)',
                    'Semi-major Axis (AU)', 'Eccentricity', 'Star Temp (K)',
                    'Star Mass (Mo)', 'Distance (pc)']:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")

        for col in ['Equil. Temp (K)', 'Stellar Flux (Se)', 'HZ Ratio']:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")

        def highlight_habitable(row):
            color = "#0d2318" if row.get('Habitable') == 'Yes' else ""
            return [f"background-color: {color}" for _ in row]

        styled = display_df.style.apply(highlight_habitable, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Column Descriptions")
        col_desc = pd.DataFrame([
            ('Planet Name',          'Canonical planet identifier from NASA archive'),
            ('Host Star',            'Name of the host star'),
            ('Orbital Period (days)','Time for one full orbit around the host star'),
            ('Radius (Re)',          'Planet radius relative to Earth'),
            ('Mass (Me)',            'Planet mass relative to Earth'),
            ('Semi-major Axis (AU)', 'Average orbital distance from the star in AU'),
            ('Eccentricity',         'Orbital shape: 0 = circular, 1 = parabolic'),
            ('Star Temp (K)',         'Effective surface temperature of the host star'),
            ('Star Mass (Mo)',        'Host star mass relative to the Sun'),
            ('Distance (pc)',         'Distance from Earth in parsecs'),
            ('Equil. Temp (K)',       'Derived: estimated blackbody surface temperature'),
            ('Stellar Flux (Se)',     'Derived: stellar energy flux relative to Earth'),
            ('HZ Ratio',             'Derived: orbital distance / habitable zone center'),
            ('Habitable',            'Derived: Yes if all 6 habitability criteria are met'),
        ], columns=['Column', 'Description'])
        st.dataframe(col_desc, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Performance Charts")
    st.pyplot(make_roc_pr_chart(bundle), use_container_width=True)

    st.markdown("---")
    st.pyplot(make_cv_chart(bundle), use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Configuration**")
        st.markdown("**Algorithm:** HistGradientBoostingClassifier")
        st.markdown("**Feature importance:** Permutation (balanced accuracy)")
        st.markdown("**Data source:** NASA Exoplanet Archive TAP API")
        st.markdown("**Target:** Physics-derived habitability label")
        st.markdown("**Train / Test split:** 80% / 20% — Stratified")
        st.markdown("**Missing values:** Native NaN support (no imputation)")
        st.markdown("**Class balancing:** class_weight = balanced")
        st.markdown("**Early stopping:** Enabled (validation_fraction = 0.1)")

    with c2:
        st.markdown("**Feature Summary**")
        feat_rows = []
        for f in all_cols:
            ftype = '* Derived' if f in derived_cols else 'Input'
            label = DERIVED_LABELS.get(f, FEATURE_LABELS.get(f, f))
            feat_rows.append({'Feature': f, 'Description': label, 'Type': ftype})
        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

        if 'report' in bundle:
            st.markdown("**Classification Report**")
            report = bundle['report']
            report_rows = []
            for cls in ['Not Habitable', 'Habitable']:
                if cls in report:
                    r = report[cls]
                    report_rows.append({
                        'Class': cls,
                        'Precision': f'{r["precision"]:.3f}',
                        'Recall':    f'{r["recall"]:.3f}',
                        'F1':        f'{r["f1-score"]:.3f}',
                        'Support':   int(r['support'])
                    })
            if report_rows:
                st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)