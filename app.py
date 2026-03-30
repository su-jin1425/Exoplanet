import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib.patches import Patch

st.set_page_config(
    page_title="Exoplanet Classification Dashboard",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .prediction-box-conf {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #52b788;
        margin-top: 1rem;
    }
    .prediction-box-flag {
        background: linear-gradient(135deg, #7b2d00, #c1440e);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #ff7043;
        margin-top: 1rem;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bundle():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("model.pkl not found. Please run the Jupyter notebook first to generate it.")
    st.stop()

model = bundle['model']
model_name = bundle['model_name']
imputer = bundle['imputer']
scaler = bundle['scaler']
feature_cols = bundle['feature_cols']
feature_stats = bundle['feature_stats']
accuracy = bundle['accuracy']
roc_auc = bundle['roc_auc']
report = bundle['report']
confusion_mat = np.array(bundle['confusion_matrix'])
roc_data = bundle['roc_curve']
feature_importance = bundle['feature_importance']
target_distribution = bundle['target_distribution']
corr_matrix = bundle['correlation_matrix']
df_sample = pd.DataFrame(bundle['df_sample'])

st.markdown("""
<div class="main-header">
    <h1>🪐 NASA Exoplanet Classification Dashboard</h1>
    <p style="font-size:1.1rem; opacity:0.85;">
        Random Forest model trained on the NASA Exoplanet Archive
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🤖 Model Info")
    st.info(f"**{model_name}**")
    st.metric("Accuracy", f"{accuracy:.4f}")
    st.metric("ROC-AUC", f"{roc_auc:.4f}")
    macro_f1 = report.get('macro avg', {}).get('f1-score', 0)
    st.metric("Macro F1", f"{macro_f1:.4f}")

    st.markdown("---")
    st.markdown("## 📊 Display Options")
    dark_bg = st.checkbox("Dark plot theme", value=True)
    show_sample_data = st.checkbox("Show sample data table", value=False)

    st.markdown("---")
    st.markdown(f"**Features:** {len(feature_cols)}")
    total_samples = sum(target_distribution.values())
    st.markdown(f"**Dataset rows:** {total_samples:,}")

plot_style = 'dark_background' if dark_bg else 'seaborn-v0_8-whitegrid'

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Predict", "📈 Model Performance", "🔬 Feature Analysis",
    "🗺️ Data Explorer", "📋 Report"
])

with tab1:
    st.subheader("Enter Exoplanet Parameters")
    st.markdown("Fill in the planet and star properties below, then click **Predict**.")

    LABELS = {
        'pl_orbper': ('Orbital Period (days)', '⏱️'),
        'pl_rade': ('Planet Radius (Earth Radii)', '🌍'),
        'pl_masse': ('Planet Mass (Earth Masses)', '⚖️'),
        'pl_orbsmax': ('Semi-major Axis (AU)', '📏'),
        'pl_orbeccen': ('Orbital Eccentricity', '🔄'),
        'pl_orbincl': ('Orbital Inclination (deg)', '📐'),
        'st_teff': ('Star Effective Temp (K)', '🌡️'),
        'st_rad': ('Star Radius (Solar Radii)', '☀️'),
        'st_mass': ('Star Mass (Solar Masses)', '🏋️'),
        'st_logg': ('Star Surface Gravity (log g)', '🌊'),
        'st_met': ('Star Metallicity [Fe/H]', '⚗️'),
        'sy_dist': ('Distance from Earth (pc)', '🔭'),
        'sy_vmag': ('Visual Magnitude (V mag)', '💫')
    }

    input_values = {}
    cols_input = st.columns(3)
    for idx, col in enumerate(feature_cols):
        stats = feature_stats[col]
        label, icon = LABELS.get(col, (col, ''))
        with cols_input[idx % 3]:
            val = st.number_input(
                f"{icon} {label}",
                min_value=float(stats['min']),
                max_value=float(stats['max']),
                value=float(stats['median']),
                step=float(max((stats['max'] - stats['min']) / 100, 1e-6)),
                key=f"input_{col}",
                help=f"Range: [{stats['min']:.3g}, {stats['max']:.3g}]  |  Median: {stats['median']:.3g}"
            )
            input_values[col] = val

    st.markdown("---")
    predict_btn = st.button("🚀 Run Prediction", use_container_width=False, type="primary")

    if predict_btn:
        input_array = np.array([[input_values[c] for c in feature_cols]])
        input_imp = imputer.transform(input_array)
        input_scaled = scaler.transform(input_imp)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]
        confidence = prob[pred]

        res_col, prob_col = st.columns([1, 1])

        with res_col:
            if pred == 1:
                st.markdown(f"""
                <div class="prediction-box-flag">
                    <h2 style="color:#ff7043; margin:0;">⚠️ CONTROVERSIAL</h2>
                    <p style="color:white; font-size:1rem; margin:0.5rem 0;">The planet is likely controversial</p>
                    <h1 style="color:#ffccbc; margin:0;">{confidence*100:.1f}%</h1>
                    <p style="color:rgba(255,255,255,0.7); font-size:0.9rem;">confidence</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box-conf">
                    <h2 style="color:#52b788; margin:0;">✅ CONFIRMED</h2>
                    <p style="color:white; font-size:1rem; margin:0.5rem 0;">The planet is likely confirmed</p>
                    <h1 style="color:#b7e4c7; margin:0;">{confidence*100:.1f}%</h1>
                    <p style="color:rgba(255,255,255,0.7); font-size:0.9rem;">confidence</p>
                </div>
                """, unsafe_allow_html=True)

        with prob_col:
            st.markdown("**Probability Breakdown**")
            with plt.style.context(plot_style):
                fig, ax = plt.subplots(figsize=(5, 3.5))
                bar_colors = ['#2ecc71', '#e74c3c']
                bars = ax.bar(['Confirmed (0)', 'Controversial (1)'],
                              prob, color=bar_colors, alpha=0.88, edgecolor='white', linewidth=1.2)
                ax.set_ylim(0, 1.15)
                ax.set_ylabel('Probability')
                ax.set_title(f'{model_name} — Output', fontsize=10, fontweight='bold')
                for bar, p in zip(bars, prob):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{p:.3f}', ha='center', fontsize=11, fontweight='bold')
                plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("**Input Summary**")
        input_df = pd.DataFrame([{
            'Feature': LABELS.get(c, (c, ''))[0],
            'Value': round(input_values[c], 4),
            'Dataset Median': round(feature_stats[c]['median'], 4)
        } for c in feature_cols])
        st.dataframe(input_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Model Performance")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Accuracy", f"{accuracy:.4f}", f"{(accuracy-0.5)*100:.1f}% above chance")
    with m2:
        st.metric("ROC-AUC", f"{roc_auc:.4f}")
    with m3:
        weighted_f1 = report.get('weighted avg', {}).get('f1-score', 0)
        st.metric("Weighted F1", f"{weighted_f1:.4f}")
    with m4:
        st.metric("Macro F1", f"{macro_f1:.4f}")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("#### Confusion Matrix")
        with plt.style.context(plot_style):
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Confirmed', 'Controversial'],
                        yticklabels=['Confirmed', 'Controversial'],
                        linewidths=0.5, annot_kws={'size': 14})
            ax.set_title(f'Confusion Matrix — {model_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Actual', fontsize=11)
            plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_roc:
        st.markdown("#### ROC Curve")
        with plt.style.context(plot_style):
            fig, ax = plt.subplots(figsize=(6, 5))
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            ax.plot(fpr, tpr, color='#3498db', linewidth=2.5,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
            ax.fill_between(fpr, tpr, alpha=0.12, color='#3498db')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
            ax.set_xlabel('False Positive Rate', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontsize=11)
            ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
            ax.legend(loc='lower right', fontsize=10)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("#### Classification Report")
    rows = []
    for key in ['0', '1', 'macro avg', 'weighted avg']:
        if key in report:
            d = report[key]
            rows.append({
                'Class': 'Confirmed (0)' if key == '0' else
                         'Controversial (1)' if key == '1' else key,
                'Precision': round(d.get('precision', 0), 4),
                'Recall': round(d.get('recall', 0), 4),
                'F1-Score': round(d.get('f1-score', 0), 4),
                'Support': int(d.get('support', 0))
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Feature Analysis")

    feat_tab1, feat_tab2, feat_tab3 = st.tabs(["Distributions", "Correlations", "Importance"])

    with feat_tab1:
        st.markdown("#### Feature Distributions")
        selected_feat = st.multiselect(
            "Choose features to display:",
            options=feature_cols,
            default=feature_cols[:6]
        )
        if selected_feat:
            n_cols = 3
            n_rows = (len(selected_feat) + n_cols - 1) // n_cols
            with plt.style.context(plot_style):
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([[axes]])
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                axes_flat = axes.flatten()
                for i, feat in enumerate(selected_feat):
                    col_data = df_sample[feat].dropna() if feat in df_sample.columns else pd.Series([])
                    if len(col_data) > 0:
                        axes_flat[i].hist(col_data, bins=40, color='#4a9eff', edgecolor='none', alpha=0.8)
                        axes_flat[i].axvline(feature_stats[feat]['mean'], color='#e74c3c',
                                             linestyle='--', linewidth=1.5,
                                             label=f"Mean: {feature_stats[feat]['mean']:.3g}")
                        axes_flat[i].axvline(feature_stats[feat]['median'], color='#f39c12',
                                             linestyle='-.', linewidth=1.5,
                                             label=f"Median: {feature_stats[feat]['median']:.3g}")
                        axes_flat[i].set_title(feat, fontsize=10, fontweight='bold')
                        axes_flat[i].legend(fontsize=7)
                for j in range(len(selected_feat), len(axes_flat)):
                    axes_flat[j].set_visible(False)
                plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with feat_tab2:
        st.markdown("#### Feature Correlation Heatmap")
        corr_df = pd.DataFrame(corr_matrix).reindex(index=feature_cols, columns=feature_cols)
        with plt.style.context(plot_style):
            fig, ax = plt.subplots(figsize=(12, 9))
            mask = np.triu(np.ones_like(corr_df, dtype=bool))
            sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f',
                        cmap='RdYlBu_r', center=0, ax=ax,
                        linewidths=0.5, cbar_kws={'shrink': 0.8},
                        annot_kws={'size': 8})
            ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
            plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        corr_pairs = []
        for i, f1 in enumerate(feature_cols):
            for j, f2 in enumerate(feature_cols):
                if j > i and f1 in corr_df.index and f2 in corr_df.columns:
                    corr_pairs.append({'Feature 1': f1, 'Feature 2': f2,
                                       'Correlation': round(corr_df.loc[f1, f2], 4)})
        if corr_pairs:
            st.markdown("##### Top Correlated Pairs")
            pairs_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(pairs_df.head(10), use_container_width=True, hide_index=True)

    with feat_tab3:
        st.markdown("#### Feature Importance")
        fi_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
        fi_df = fi_df.sort_values('Importance', ascending=True)

        with plt.style.context(plot_style):
            fig, ax = plt.subplots(figsize=(10, 7))
            bar_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi_df)))
            bars = ax.barh(fi_df['Feature'], fi_df['Importance'], color=bar_colors)
            ax.set_xlabel('Importance Score', fontsize=11)
            ax.set_title(f'Feature Importance — {model_name}', fontsize=13, fontweight='bold')
            for bar, val in zip(bars, fi_df['Importance']):
                ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', va='center', fontsize=8)
            plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        fi_sorted = fi_df.sort_values('Importance', ascending=False).copy()
        fi_sorted['Importance %'] = (fi_sorted['Importance'] / fi_sorted['Importance'].sum() * 100).round(2)
        st.dataframe(fi_sorted, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Data Explorer")

    col_x, col_y, col_opt = st.columns(3)
    with col_x:
        x_feat = st.selectbox("X-axis Feature", feature_cols, index=0)
    with col_y:
        y_feat = st.selectbox("Y-axis Feature", feature_cols, index=1 if len(feature_cols) > 1 else 0)
    with col_opt:
        use_log = st.checkbox("Log scale X", value=True)

    scatter_df = df_sample[[x_feat, y_feat, 'pl_controv_flag']].dropna()
    scatter_df = scatter_df[scatter_df[x_feat] > 0]

    with plt.style.context(plot_style):
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter_colors = scatter_df['pl_controv_flag'].map({0: '#2ecc71', 1: '#e74c3c'})
        xvals = np.log10(scatter_df[x_feat]) if use_log else scatter_df[x_feat]
        ax.scatter(xvals, scatter_df[y_feat], c=scatter_colors, alpha=0.4, s=12)
        ax.set_xlabel(f"log10({x_feat})" if use_log else x_feat, fontsize=11)
        ax.set_ylabel(y_feat, fontsize=11)
        ax.set_title(f"{x_feat} vs {y_feat}", fontsize=13, fontweight='bold')
        legend_elements = [Patch(facecolor='#2ecc71', label='Confirmed (0)'),
                           Patch(facecolor='#e74c3c', label='Controversial (1)')]
        ax.legend(handles=legend_elements)
        plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Class Distribution")
    tc1, tc2 = st.columns(2)
    vals = [target_distribution.get(0, 0), target_distribution.get(1, 0)]

    with tc1:
        with plt.style.context(plot_style):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(vals, labels=['Confirmed (0)', 'Controversial (1)'],
                   colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=90)
            ax.set_title('Class Balance', fontweight='bold')
            plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tc2:
        with plt.style.context(plot_style):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(['Confirmed (0)', 'Controversial (1)'], vals,
                   color=['#2ecc71', '#e74c3c'], alpha=0.85, edgecolor='white')
            for i, v in enumerate(vals):
                ax.text(i, v + max(vals) * 0.01, f'{v:,}', ha='center', fontweight='bold')
            ax.set_title('Class Counts', fontweight='bold')
            ax.set_ylabel('Count')
            plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    if show_sample_data:
        st.markdown("#### Sample Data (first 100 rows)")
        st.dataframe(df_sample.head(100), use_container_width=True)

with tab5:
    st.subheader(f"Full Classification Report — {model_name}")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Accuracy", f"{accuracy:.4f}")
    with r2:
        st.metric("ROC-AUC", f"{roc_auc:.4f}")
    with r3:
        st.metric("Macro F1", f"{macro_f1:.4f}")
    with r4:
        st.metric("Weighted F1", f"{report.get('weighted avg', {}).get('f1-score', 0):.4f}")

    rows = []
    for key in ['0', '1', 'macro avg', 'weighted avg']:
        if key in report:
            d = report[key]
            rows.append({
                'Class': 'Confirmed (0)' if key == '0' else
                         'Controversial (1)' if key == '1' else key,
                'Precision': round(d.get('precision', 0), 4),
                'Recall': round(d.get('recall', 0), 4),
                'F1-Score': round(d.get('f1-score', 0), 4),
                'Support': int(d.get('support', 0))
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Model Parameters")
    params = model.get_params()
    param_df = pd.DataFrame([{'Parameter': k, 'Value': str(v)} for k, v in params.items()])
    st.dataframe(param_df, use_container_width=True, hide_index=True)
