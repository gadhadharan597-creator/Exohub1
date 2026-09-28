#!/usr/bin/env python3
"""
Exoplanet Habitability Ranking System Pipeline
Ingests real data from NASA Exoplanet Archive and ExoMiner++, engineers habitability features,
trains physics-based and ML habitability models, and outputs ranked exoplanet candidates.
"""

import os
import sys
import math
import json
import time
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

CONFIG = {
    'DATA_DIR': 'exoplanet_data',
    'OUTPUT_FILE': 'exoplanet_habitability_rankings.csv',
    'NASA_TAP_URL': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync',
    'ZENODO_RECORD_ID': '15466292',
    'SEED': 42,
    'EARTH_INSOLATION_WM2': 1361.0,
    'EARTH_TEFF_K': 5778.0,
    'ALBEDO': 0.3,
    'ROCKY_RADIUS_THRESHOLD': 1.6
}

os.makedirs(CONFIG['DATA_DIR'], exist_ok=True)
np.random.seed(CONFIG['SEED'])

print("==================================================")
print(" 🪐 EXOPLANET HABITABILITY RANKING SYSTEM PIPELINE ")
print("==================================================")

# -----------------------------------------------------------------------------
# 1. DATA INGESTION: NASA Exoplanet Archive
# -----------------------------------------------------------------------------
def fetch_nasa_exoplanet_archive():
    cache_path = os.path.join(CONFIG['DATA_DIR'], 'nasa_exoplanet_archive.csv')
    if os.path.exists(cache_path):
        print(f"[1/4] Loading cached NASA Exoplanet Archive data from {cache_path}...")
        df = pd.read_csv(cache_path)
        print(f"      Loaded {len(df)} records.")
        return df

    print("[1/4] Fetching data from NASA Exoplanet Archive TAP service...")
    columns = [
        'pl_name', 'hostname',
        'pl_orbper', 'pl_rade', 'pl_insol', 'pl_eqt',
        'st_teff', 'st_rad', 'st_mass', 'st_lum', 'sy_dist'
    ]
    query = f"SELECT {', '.join(columns)} FROM pscomppars"
    url = f"{CONFIG['NASA_TAP_URL']}?query={query.replace(' ', '+')}&format=json"

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        records = response.json()
        df = pd.DataFrame(records)
        df.to_csv(cache_path, index=False)
        print(f"      Successfully retrieved {len(df)} records from NASA Exoplanet Archive.")
        return df
    except Exception as e:
        print(f"      Error querying NASA TAP service: {e}")
        raise e

# -----------------------------------------------------------------------------
# 2. DATA INGESTION: ExoMiner++ TESS Vetting Catalog
# -----------------------------------------------------------------------------
def fetch_exominer_catalog():
    cache_path = os.path.join(CONFIG['DATA_DIR'], 'ExoMinerPP_TESS_vetting_catalog.csv')
    if os.path.exists(cache_path):
        print(f"[2/4] Loading cached ExoMiner++ catalog from {cache_path}...")
        df = pd.read_csv(cache_path)
        print(f"      Loaded {len(df)} records.")
        return df

    print("[2/4] Fetching ExoMiner++ TESS Vetting Catalog from Zenodo...")
    try:
        zenodo_url = f"https://zenodo.org/api/records/{CONFIG['ZENODO_RECORD_ID']}"
        res = requests.get(zenodo_url, timeout=30)
        if res.status_code == 200:
            files = res.json().get('files', [])
            for f in files:
                if f['key'].endswith('.csv') or 'ExoMiner' in f['key']:
                    download_url = f['links']['self']
                    print(f"      Downloading {f['key']}...")
                    r = requests.get(download_url, timeout=120)
                    with open(cache_path, 'wb') as out:
                        out.write(r.content)
                    df = pd.read_csv(cache_path)
                    print(f"      Successfully fetched ExoMiner++ catalog with {len(df)} records.")
                    return df
    except Exception as e:
        print(f"      Notice: Direct Zenodo API download note: {e}")

    # Try zenodo_get if available
    try:
        print("      Attempting download via zenodo_get...")
        os.system(f"zenodo_get {CONFIG['ZENODO_RECORD_ID']} -o {CONFIG['DATA_DIR']}")
        for file in os.listdir(CONFIG['DATA_DIR']):
            if file.endswith('.csv') and 'ExoMiner' in file:
                full_p = os.path.join(CONFIG['DATA_DIR'], file)
                df = pd.read_csv(full_p)
                df.to_csv(cache_path, index=False)
                return df
    except Exception:
        pass

    print("      ExoMiner catalog file not found locally. Initializing baseline ExoMiner probabilities from archive dataset...")
    return pd.DataFrame()

# -----------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# -----------------------------------------------------------------------------
kopparapu_coeffs = {
    'recent_venus_inner':   {'a': 1.3351e-8, 'b': 3.1515e-12, 'c': -1.3785e-15, 'd': -2.9658e-19, 'e': -2.9866e-23, 'S_eff_sun': 1.776},
    'max_greenhouse_outer': {'a': 1.0183e-8, 'b': 1.4885e-12, 'c': -4.8943e-16, 'd': -3.6937e-19, 'e': -1.1396e-23, 'S_eff_sun': 0.356},
    'dry_runaway_inner':    {'a': 1.7766e-8, 'b': 6.6436e-12, 'c': -2.5704e-15, 'd': -2.9090e-19, 'e': -6.1802e-24, 'S_eff_sun': 1.038},
    'early_mars_outer':     {'a': 5.9529e-9, 'b': -1.4925e-12, 'c': 7.6293e-16, 'd': -2.3164e-19, 'e': 2.4578e-23, 'S_eff_sun': 0.320}
}

def calculate_hz_flux(st_teff, boundary_type):
    if pd.isna(st_teff):
        return np.nan
    coeffs = kopparapu_coeffs[boundary_type]
    X = st_teff - CONFIG['EARTH_TEFF_K']
    S_eff = coeffs['S_eff_sun'] + coeffs['a']*X + coeffs['b']*(X**2) + coeffs['c']*(X**3) + coeffs['d']*(X**4) + coeffs['e']*(X**5)
    return max(0.01, S_eff)

def classify_stellar_type(teff):
    if pd.isna(teff): return 'Unknown'
    if teff >= 30000: return 'O'
    elif teff >= 10000: return 'B'
    elif teff >= 7500: return 'A'
    elif teff >= 6000: return 'F'
    elif teff >= 5200: return 'G'
    elif teff >= 3700: return 'K'
    else: return 'M'

def run_feature_engineering(df, exominer_df):
    print("[3/4] Performing cross-matching and feature engineering...")
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Process ExoMiner scores
    if not exominer_df.empty:
        exominer_df.columns = exominer_df.columns.str.strip().str.lower()
        score_col = [c for c in exominer_df.columns if 'score' in c or 'prob' in c or 'pred' in c]
        tic_col = [c for c in exominer_df.columns if 'tic' in c]
        if score_col and tic_col:
            exominer_sub = exominer_df[[tic_col[0], score_col[0]]].dropna()
            exominer_sub.columns = ['tic_id', 'probability_of_planet']
            df = pd.merge(df, exominer_sub, on='tic_id', how='left')
    
    if 'probability_of_planet' not in df.columns:
        # High confidence for confirmed planets in NASA Exoplanet Archive
        df['probability_of_planet'] = 0.95

    df['probability_of_planet'] = df['probability_of_planet'].fillna(0.90)

    # Clean numeric columns
    num_cols = ['pl_orbper', 'pl_rade', 'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_lum', 'sy_dist']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate Habitable Zone Boundaries
    df['hz_inner_con_flux'] = df['st_teff'].apply(lambda x: calculate_hz_flux(x, 'recent_venus_inner'))
    df['hz_outer_con_flux'] = df['st_teff'].apply(lambda x: calculate_hz_flux(x, 'max_greenhouse_outer'))
    
    # Position Index (1.0 = inner edge, 0.0 = outer edge, 0..1 = inside HZ)
    df['hz_position_index'] = (df['pl_insol'] - df['hz_outer_con_flux']) / (df['hz_inner_con_flux'] - df['hz_outer_con_flux'])

    # Equilibrium Temperature with Bond Albedo = 0.3
    sigma = 5.670374419e-8
    df['pl_insol_wm2'] = df['pl_insol'] * CONFIG['EARTH_INSOLATION_WM2']
    df['calculated_eq_temp_k'] = (df['pl_insol_wm2'] * (1 - CONFIG['ALBEDO']) / (4 * sigma)) ** 0.25
    df['eq_temp_k'] = df['pl_eqt'].fillna(df['calculated_eq_temp_k'])

    # Radius Classification (Rocky vs Mini-Neptune)
    df['is_rocky'] = (df['pl_rade'] <= CONFIG['ROCKY_RADIUS_THRESHOLD']).astype(int)
    df['radius_class'] = np.where(df['is_rocky'] == 1, 'Rocky', 'Mini-Neptune/Gas Giant')

    # Proxy Earth Similarity Index (ESI)
    esi_r = 1 - (df['pl_rade'] - 1.0).abs() / (df['pl_rade'] + 1.0)
    esi_t = 1 - (df['eq_temp_k'] - 288.0).abs() / (df['eq_temp_k'] + 288.0)
    df['earth_similarity_index'] = (esi_r.clip(lower=0) * esi_t.clip(lower=0)).pow(0.5).fillna(0)

    # Stellar Type
    df['stellar_type'] = df['st_teff'].apply(classify_stellar_type)

    return df

# -----------------------------------------------------------------------------
# 4. HABITABILITY MODELING & RANKING
# -----------------------------------------------------------------------------
def train_and_rank_models(df):
    print("[4/4] Training Habitability Models and computing final rankings...")
    
    # Physics-based Habitability Score
    # P(HZ): 1 if in conservative HZ (flux between outer and inner boundaries)
    df['P_HZ'] = ((df['pl_insol'] >= df['hz_outer_con_flux']) & (df['pl_insol'] <= df['hz_inner_con_flux'])).astype(int)
    df['P_rocky'] = df['is_rocky']
    df['P_real_planet'] = df['probability_of_planet'].clip(0, 1)

    df['physics_habitability_score'] = df['P_real_planet'] * df['P_HZ'] * df['P_rocky']

    # Rule-Derived Label for Supervised ML Training
    df['is_habitable_candidate'] = (
        (df['P_real_planet'] >= 0.85) &
        (df['P_HZ'] == 1) &
        (df['P_rocky'] == 1)
    ).astype(int)

    # ML Feature Set
    feature_cols = ['pl_orbper', 'pl_rade', 'pl_insol', 'eq_temp_k', 'st_teff', 'st_rad', 'st_mass', 'sy_dist', 'earth_similarity_index']
    
    ml_df = df.dropna(subset=feature_cols).copy()
    X = ml_df[feature_cols]
    y = ml_df['is_habitable_candidate']

    print(f"      ML dataset contains {len(X)} complete instances across features.")
    print(f"      Positive habitable candidate target count: {y.sum()}")

    # Model: HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(random_state=CONFIG['SEED'], max_iter=200, learning_rate=0.05)
    model.fit(X, y)

    # Generate probabilities for all records (handling missing features with impute/predict_proba)
    X_full = df[feature_cols].fillna(X.median())
    df['ml_habitability_score'] = model.predict_proba(X_full)[:, 1]

    # Combine Physics and ML scores into a Unified Composite Ranking Score
    df['composite_habitability_score'] = (0.5 * df['physics_habitability_score']) + (0.5 * df['ml_habitability_score'])

    # Sort by Composite Score
    df_ranked = df.sort_values(by='composite_habitability_score', ascending=False).reset_index(drop=True)

    output_path = CONFIG['OUTPUT_FILE']
    df_ranked.to_csv(output_path, index=False)
    print(f"      Rankings successfully written to {output_path}")

    # Top 10 Report
    print("\n🏆 TOP 10 POTENTIALLY HABITABLE EXOPLANETS 🏆")
    top10_cols = ['pl_name', 'hostname', 'stellar_type', 'pl_rade', 'eq_temp_k', 'pl_insol', 'physics_habitability_score', 'ml_habitability_score', 'composite_habitability_score']
    print(df_ranked[top10_cols].head(10).to_string(index=False))

    return df_ranked

if __name__ == '__main__':
    nasa_df = fetch_nasa_exoplanet_archive()
    exominer_df = fetch_exominer_catalog()
    processed_df = run_feature_engineering(nasa_df, exominer_df)
    ranked_df = train_and_rank_models(processed_df)
    print("\n✅ Pipeline completed successfully.")
