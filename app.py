import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Exoplanet Habitability Explorer",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme styling
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    .stMetric label {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700;
    }
    h1, h2, h3 {
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .highlight-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #6366f1;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.title("🪐 Advanced Exoplanet Habitability Ranking System")
st.markdown("""
An interactive multi-stage ranking framework combining **NASA Exoplanet Archive** parameters with **NASA ExoMiner++** planet validation probabilities and **Kopparapu et al. Habitable Zone models**.
""")
st.markdown("---")

# Load Data
DATA_FILE = "exoplanet_habitability_rankings.csv"

@st.cache_data
def load_rankings():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        return df
    else:
        st.error(f"Data file '{DATA_FILE}' not found. Please run 'exoplanet_pipeline.py' first.")
        return pd.DataFrame()

df = load_rankings()

if not df.empty:
    # Sidebar Filters
    st.sidebar.header("🔍 Filter Parameters")

    # Habitability Composite Score Slider
    min_composite = st.sidebar.slider(
        "Minimum Composite Habitability Score",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05
    )

    # Minimum ExoMiner Probability
    min_exominer = st.sidebar.slider(
        "Minimum Real Planet Probability (P_real)",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05
    )

    # Stellar Type Filter
    stellar_types = sorted(df['stellar_type'].dropna().unique().tolist())
    selected_stellar = st.sidebar.multiselect(
        "Select Host Star Spectral Types",
        options=stellar_types,
        default=stellar_types
    )

    # Boolean Checkboxes
    rocky_only = st.sidebar.checkbox("Show Rocky Planets Only (R ≤ 1.6 R⊕)", value=False)
    hz_only = st.sidebar.checkbox("Show Conservative Habitable Zone Planets Only", value=False)

    # Apply Filters
    filtered_df = df[
        (df['composite_habitability_score'] >= min_composite) &
        (df['P_real_planet'] >= min_exominer) &
        (df['stellar_type'].isin(selected_stellar))
    ].copy()

    if rocky_only:
        filtered_df = filtered_df[filtered_df['is_rocky'] == 1]

    if hz_only:
        filtered_df = filtered_df[filtered_df['P_HZ'] == 1]

    # Display Metrics Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Catalog Candidates", f"{len(df):,}")
    with col2:
        st.metric("Filtered Candidates", f"{len(filtered_df):,}")
    with col3:
        high_cand = (df['composite_habitability_score'] >= 0.85).sum()
        st.metric("Top Tier Candidates (Score ≥ 0.85)", f"{high_cand}")
    with col4:
        rocky_count = (filtered_df['is_rocky'] == 1).sum()
        st.metric("Filtered Rocky Planets", f"{rocky_count}")

    st.markdown("### 🏆 Ranked Potentially Habitable Candidates")
    st.write(f"Showing **{len(filtered_df)}** exoplanet candidates meeting active filter criteria.")

    # Data Table Formatting
    display_cols = [
        'pl_name', 'hostname', 'stellar_type', 'pl_rade', 'eq_temp_k',
        'pl_insol', 'earth_similarity_index', 'P_real_planet', 'P_HZ',
        'physics_habitability_score', 'ml_habitability_score', 'composite_habitability_score'
    ]

    # Filter available display columns
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    renamed_cols = {
        'pl_name': 'Planet Name',
        'hostname': 'Host Star',
        'stellar_type': 'Star Type',
        'pl_rade': 'Radius (R⊕)',
        'eq_temp_k': 'T_eq (K)',
        'pl_insol': 'Insolation (S⊕)',
        'earth_similarity_index': 'Proxy ESI',
        'P_real_planet': 'P(Real Planet)',
        'P_HZ': 'In HZ',
        'physics_habitability_score': 'Physics Score',
        'ml_habitability_score': 'ML Score',
        'composite_habitability_score': 'Composite Score'
    }

    display_df = filtered_df[display_cols].rename(columns=renamed_cols)

    # Format numeric columns
    format_dict = {
        'Radius (R⊕)': '{:.2f}',
        'T_eq (K)': '{:.1f}',
        'Insolation (S⊕)': '{:.2f}',
        'Proxy ESI': '{:.3f}',
        'P(Real Planet)': '{:.3f}',
        'Physics Score': '{:.3f}',
        'ML Score': '{:.3f}',
        'Composite Score': '{:.3f}'
    }

    for col, fmt in format_dict.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else 'N/A')

    st.dataframe(display_df, use_container_width=True)

    # Visualization Section
    st.markdown("---")
    st.subheader("📊 Habitability Distribution & Scientific Insights")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Score Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.set_style("darkgrid")
        plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white', 'figure.facecolor': '#0b0f19', 'axes.facecolor': '#1e293b'})
        sns.histplot(filtered_df['composite_habitability_score'], kde=True, color='#38bdf8', ax=ax, bins=20)
        ax.set_title("Distribution of Composite Habitability Scores", color='white')
        ax.set_xlabel("Composite Habitability Score", color='white')
        ax.set_ylabel("Count", color='white')
        st.pyplot(fig)

    with chart_col2:
        st.markdown("#### Radius vs Insolation (Habitable Zone)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white', 'figure.facecolor': '#0b0f19', 'axes.facecolor': '#1e293b'})
        scatter = ax2.scatter(
            filtered_df['pl_insol'],
            filtered_df['pl_rade'],
            c=filtered_df['composite_habitability_score'],
            cmap='viridis',
            alpha=0.8,
            edgecolors='w',
            linewidth=0.5
        )
        ax2.set_xscale('log')
        ax2.set_title("Planet Radius vs. Insolation Flux", color='white')
        ax2.set_xlabel("Insolation (Earth Flux Units, Log Scale)", color='white')
        ax2.set_ylabel("Radius (Earth Radii)", color='white')
        ax2.axhline(1.6, color='#f43f5e', linestyle='--', label='Rocky Threshold (1.6 R⊕)')
        ax2.legend()
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("Composite Score", color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        st.pyplot(fig2)

else:
    st.warning("No data loaded.")
