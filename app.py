"""
Smart Crop & Agriculture Analyzer - Interactive Streamlit Web Application
=========================================================================
Standalone Streamlit application connecting directly to serialized ML models:
1. Crop Recommendation (Classification: Random Forest, Decision Tree, KNN)
2. Crop Yield Prediction (Regression: Linear Regression, Random Forest Regressor)
3. Soil Pattern Analysis (Clustering: K-Means with PCA 2D visualization)
4. Comprehensive Model Performance & College Viva Guide
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Suppress loky CPU detection warnings on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Set matplotlib style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Crop & Agriculture Analyzer",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Agriculture Green-and-White Theme
st.markdown(
    """
    <style>
    .main {
        background-color: #FBFDFB;
    }
    h1, h2, h3 {
        color: #1B5E20 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f4fbf4 100%);
        border: 1px solid #C8E6C9;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(46, 125, 50, 0.08);
        margin-bottom: 15px;
    }
    .metric-card-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #43A047;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1B5E20;
    }
    .metric-card-sub {
        font-size: 0.82rem;
        color: #558B2F;
        margin-top: 4px;
    }
    .result-banner {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 6px solid #2E7D32;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
    }
    .result-title {
        font-size: 1.1rem;
        color: #1B5E20;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .result-highlight {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2E7D32;
        text-transform: capitalize;
    }
    section[data-testid="stSidebar"] {
        background-color: #F1F8E9;
        border-right: 1px solid #DCEDC8;
    }
    div.stButton > button:first-child {
        background-color: #2E7D32;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.55rem 1.4rem;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #1B5E20;
        box-shadow: 0 4px 10px rgba(46, 125, 50, 0.25);
    }
    .viva-badge {
        display: inline-block;
        background-color: #E0F2F1;
        color: #00695C;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 8px;
        border: 1px solid #80CBC4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# STANDALONE INFERENCE HELPERS
# -----------------------------------------------------------------------------
def predict_crop(model, input_dict):
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    input_df = pd.DataFrame([input_dict])[feature_cols]
    predicted_crop = model.predict(input_df)[0]
    probabilities = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        classes = model.classes_
        probabilities = dict(zip(classes, probs.tolist()))
    return predicted_crop, probabilities

def explain_crop_inputs(input_dict):
    n, p, k = input_dict["N"], input_dict["P"], input_dict["K"]
    temp, hum, ph, rain = input_dict["temperature"], input_dict["humidity"], input_dict["ph"], input_dict["rainfall"]
    n_status = "High (Optimal for nitrogen-demanding crops like rice, cotton)" if n > 80 else ("Low (Legumes adapt well)" if n < 40 else "Medium")
    p_status = "High (Ideal for phosphorus fruits like apples, grapes)" if p > 70 else ("Low" if p < 30 else "Optimal")
    k_status = "Very High (Supports heavy fruit formation)" if k > 60 else ("Adequate" if k >= 30 else "Low")
    ph_status = "Acidic (Favors tea, coffee, potatoes)" if ph < 5.5 else ("Alkaline (Favors cotton, barley)" if ph > 7.5 else "Neutral")
    rain_status = "High Rainfall (Ideal for wetland rice, jute)" if rain > 150 else ("Low Rainfall (Millets, pulses favored)" if rain < 60 else "Moderate Rainfall")
    return {
        "Nitrogen (N)": f"{n} kg/ha → {n_status}",
        "Phosphorus (P)": f"{p} kg/ha → {p_status}",
        "Potassium (K)": f"{k} kg/ha → {k_status}",
        "Soil pH": f"{ph:.1f} → {ph_status}",
        "Temperature": f"{temp:.1f} °C → {'Warm/Tropical' if temp >= 20 else 'Cool/Temperate'}",
        "Humidity": f"{hum:.1f} % → {'High Humidity' if hum > 70 else 'Moderate Humidity'}",
        "Rainfall": f"{rain:.1f} mm → {rain_status}",
    }

def predict_yield(pipeline, input_dict):
    feature_cols = ["Crop", "Season", "State_Name", "Area"]
    input_df = pd.DataFrame([input_dict])[feature_cols]
    predicted_yield = float(pipeline.predict(input_df)[0])
    predicted_yield = max(0.01, round(predicted_yield, 2))
    area = float(input_dict["Area"])
    estimated_production = round(predicted_yield * area, 2)
    return predicted_yield, estimated_production

def predict_soil_cluster(cluster_model_data, input_dict):
    scaler = cluster_model_data["scaler"]
    kmeans = cluster_model_data["kmeans"]
    pca = cluster_model_data["pca"]
    profiles = cluster_model_data.get("profiles", {
        0: {"name": "Cluster 0: Legume & Neutral Soil Pattern", "description": "Moderate N, P, K, pH ~ 7.2", "characteristics": "Supports nitrogen-fixing pulses (chickpea, lentil).", "recommended_action": "Maintain organic matter.", "badge_color": "#2E7D32"},
        1: {"name": "Cluster 1: High Nitrogen Soil Pattern", "description": "High N (~96), pH ~ 6.5", "characteristics": "Optimal for heavy feeders (rice, maize, banana, cotton).", "recommended_action": "Apply split urea doses.", "badge_color": "#1565C0"},
        2: {"name": "Cluster 2: Acidic Plantation Soil Pattern", "description": "Low N (~22), Acidic pH (~5.8)", "characteristics": "Suits tropical tree crops (coconut, mango).", "recommended_action": "Apply lime if growing neutral crops.", "badge_color": "#E65100"},
        3: {"name": "Cluster 3: Orchard High P & K Soil Pattern", "description": "Extremely High P (~133) & K (~200)", "characteristics": "Favors heavy fruit crops (grapes, apple).", "recommended_action": "Monitor potash levels.", "badge_color": "#6A1B9A"}
    })
    feature_cols = ["N", "P", "K", "ph"]
    sample_df = pd.DataFrame([input_dict])[feature_cols]
    sample_scaled = scaler.transform(sample_df)
    assigned_cluster = int(kmeans.predict(sample_scaled)[0])
    sample_pca = pca.transform(sample_scaled)[0]
    return assigned_cluster, (float(sample_pca[0]), float(sample_pca[1])), profiles.get(assigned_cluster, {})

# -----------------------------------------------------------------------------
# RESOURCE CACHING (LOAD MODELS ONCE)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_all_models_and_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    if not os.path.exists(models_dir):
        return None
    try:
        return {
            "crop_model": joblib.load(os.path.join(models_dir, "crop_classifier.pkl")),
            "crop_all": joblib.load(os.path.join(models_dir, "crop_classifier_all.pkl")),
            "yield_model": joblib.load(os.path.join(models_dir, "yield_regressor.pkl")),
            "yield_all": joblib.load(os.path.join(models_dir, "yield_regressor_all.pkl")),
            "soil_model": joblib.load(os.path.join(models_dir, "soil_cluster_model.pkl")),
            "preprocessing": joblib.load(os.path.join(models_dir, "preprocessing_objects.pkl")),
            "evaluation": joblib.load(os.path.join(models_dir, "evaluation_results.pkl")),
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

@st.cache_data
def load_raw_dataset_samples():
    crop_df = pd.read_csv("data/crop_recommendation.csv")
    yield_df = pd.read_csv("data/crop_yield.csv", nrows=1000)
    return crop_df, yield_df

artifacts = load_all_models_and_artifacts()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style='text-align: center; padding: 10px 0;'>
        <h2 style='color: #2E7D32; margin-bottom: 2px;'>🌱 AGRO ANALYZER</h2>
        <span style='font-size: 0.85rem; color: #558B2F;'>Smart Agriculture Decision Support</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation Menu:",
    [
        "🏠 1. Home Dashboard",
        "🌾 2. Crop Recommendation (Classification)",
        "🚜 3. Crop Yield Prediction (Regression)",
        "🧪 4. Soil Pattern Clustering (K-Means)",
        "📊 5. Model Performance & Viva Guide",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='font-size: 0.82rem; color: #37474F;'>
        <b>College Project Info:</b><br>
        • <b>Architecture:</b> Scikit-Learn + Streamlit<br>
        • <b>ML Tasks:</b> Classification, Regression, Clustering<br>
        • <b>No Deep Learning / No LLMs</b><br>
        • <b>Datasets:</b> Real Indian Agricultural Data
    </div>
    """,
    unsafe_allow_html=True,
)

if artifacts is None:
    st.error("🚨 Models not found in `models/` directory!")
    st.stop()

# =============================================================================
# PAGE 1: HOME DASHBOARD
# =============================================================================
if menu == "🏠 1. Home Dashboard":
    st.markdown(
        """
        <div style='background: linear-gradient(120deg, #E8F5E9 0%, #C8E6C9 100%); padding: 25px; border-radius: 12px; margin-bottom: 25px;'>
            <h1 style='margin: 0; color: #1B5E20; font-size: 2.3rem;'>🌾 Smart Crop & Agriculture Analyzer</h1>
            <p style='margin-top: 8px; font-size: 1.05rem; color: #2E7D32;'>
                An end-to-end Machine Learning decision support system that helps farmers and agronomists analyze soil conditions, 
                recommend optimal crops, predict harvest yields, and detect natural soil nutrient patterns using traditional Machine Learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class='metric-card'>
                <div class='metric-card-title'>Module A • Classification</div>
                <div class='metric-card-value'>Crop Recommendation</div>
                <div class='metric-card-sub'>
                    <b>Algorithm:</b> Random Forest (99.55% Acc)<br>
                    <b>Inputs:</b> N, P, K, Temp, Humidity, pH, Rain<br>
                    <b>Target:</b> 22 Crop Classes
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class='metric-card'>
                <div class='metric-card-title'>Module B • Regression</div>
                <div class='metric-card-value'>Yield Prediction</div>
                <div class='metric-card-sub'>
                    <b>Algorithm:</b> Random Forest Regressor (R² 0.91)<br>
                    <b>Inputs:</b> Crop, Season, State, Area<br>
                    <b>Target:</b> Yield (Tonnes / Hectare)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class='metric-card'>
                <div class='metric-card-title'>Module C • Clustering</div>
                <div class='metric-card-value'>Soil Pattern Analysis</div>
                <div class='metric-card-sub'>
                    <b>Algorithm:</b> K-Means (k=4) + 2D PCA<br>
                    <b>Inputs:</b> Nitrogen, Phosphorus, Potassium, pH<br>
                    <b>Target:</b> 4 Soil Chemistry Profiles
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📚 Real-World Datasets Overview")
    crop_df, yield_df_sample = load_raw_dataset_samples()
    prep_obj = artifacts["preprocessing"]

    tab_rec, tab_yld = st.tabs(["🌾 Crop Recommendation Dataset", "🚜 Crop Yield / Production Dataset"])

    with tab_rec:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(crop_df):,}")
        c2.metric("Total Features", "7 Soil & Weather")
        c3.metric("Unique Crop Classes", f"{crop_df['label'].nunique()} Crops")
        c4.metric("Class Balance", "100 Samples / Crop (Balanced)")
        st.dataframe(crop_df.head(), use_container_width=True)

    with tab_yld:
        total_yld_records = prep_obj["yield_summary"]["num_records"]
        num_crops = len(prep_obj["yield_summary"]["crops"])
        num_states = len(prep_obj["yield_summary"]["states"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cleaned Records", f"{total_yld_records:,}")
        c2.metric("Covered Crops", f"{num_crops} Major Crops")
        c3.metric("Indian States", f"{num_states} States & UTs")
        c4.metric("Seasons", "6 Agricultural Seasons")
        st.dataframe(yield_df_sample.head(6), use_container_width=True)

# =============================================================================
# PAGE 2: CROP RECOMMENDATION (CLASSIFICATION)
# =============================================================================
elif menu == "🌾 2. Crop Recommendation (Classification)":
    st.markdown("## 🌾 Multiclass Crop Recommendation")
    st.write("Enter soil nutrients and ambient weather conditions to receive an instant recommended crop.")

    st.markdown("##### ⚡ Quick-Fill Presets for Testing:")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    preset_vals = None
    if p_col1.button("🌾 Preset: Rice (Wetland)"):
        preset_vals = {"N": 90, "P": 42, "K": 43, "temp": 21.0, "hum": 82.0, "ph": 6.5, "rain": 203.0}
    elif p_col2.button("🌽 Preset: Maize (Moderate)"):
        preset_vals = {"N": 71, "P": 54, "K": 20, "temp": 22.6, "hum": 65.0, "ph": 5.7, "rain": 110.0}
    elif p_col3.button("🍎 Preset: Apple (High P & K)"):
        preset_vals = {"N": 20, "P": 134, "K": 199, "temp": 22.0, "hum": 92.0, "ph": 5.9, "rain": 112.0}
    elif p_col4.button("🌱 Preset: Chickpea (Legume)"):
        preset_vals = {"N": 40, "P": 68, "K": 79, "temp": 17.0, "hum": 17.0, "ph": 7.4, "rain": 80.0}

    with st.form("crop_rec_form"):
        col_n, col_p, col_k, col_ph = st.columns(4)
        default_n = preset_vals["N"] if preset_vals else 90
        default_p = preset_vals["P"] if preset_vals else 42
        default_k = preset_vals["K"] if preset_vals else 43
        default_ph = preset_vals["ph"] if preset_vals else 6.5

        val_n = col_n.number_input("Nitrogen (N) [kg/ha]", min_value=0, max_value=150, value=default_n, step=1)
        val_p = col_p.number_input("Phosphorus (P) [kg/ha]", min_value=5, max_value=150, value=default_p, step=1)
        val_k = col_k.number_input("Potassium (K) [kg/ha]", min_value=5, max_value=210, value=default_k, step=1)
        val_ph = col_ph.number_input("Soil pH (0 - 14)", min_value=3.5, max_value=10.0, value=default_ph, step=0.1)

        col_temp, col_hum, col_rain = st.columns(3)
        default_temp = preset_vals["temp"] if preset_vals else 24.5
        default_hum = preset_vals["hum"] if preset_vals else 75.0
        default_rain = preset_vals["rain"] if preset_vals else 180.0

        val_temp = col_temp.slider("Temperature (°C)", min_value=8.0, max_value=45.0, value=default_temp, step=0.5)
        val_hum = col_hum.slider("Humidity (%)", min_value=10.0, max_value=100.0, value=default_hum, step=1.0)
        val_rain = col_rain.slider("Rainfall (mm)", min_value=20.0, max_value=300.0, value=default_rain, step=5.0)

        submit_crop = st.form_submit_button("🌱 Recommend Suitable Crop")

    if submit_crop:
        input_data = {"N": val_n, "P": val_p, "K": val_k, "temperature": val_temp, "humidity": val_hum, "ph": val_ph, "rainfall": val_rain}
        best_model = artifacts["crop_model"]
        crop_all = artifacts["crop_all"]

        pred_crop, proba_dict = predict_crop(best_model, input_data)
        explanation = explain_crop_inputs(input_data)

        st.markdown(
            f"""
            <div class='result-banner'>
                <div class='result-title'>🎯 Optimal Recommended Crop:</div>
                <div class='result-highlight'>🌿 {pred_crop.upper()}</div>
                <div style='margin-top: 8px; color: #1B5E20; font-size: 0.95rem;'>
                    <b>Selected Algorithm:</b> {crop_all['best_name']} (Test Accuracy: <b>{crop_all['results'][crop_all['best_name']]['accuracy']*100:.2f}%</b>)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.markdown("##### 📊 Top 5 Model Confidence Predictions")
            if proba_dict:
                top5 = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                top5_df = pd.DataFrame(top5, columns=["Crop", "Probability"])
                top5_df["Confidence %"] = (top5_df["Probability"] * 100).round(1)

                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.barplot(data=top5_df, y="Crop", x="Confidence %", palette="Greens_r", ax=ax)
                ax.set_xlim(0, 100)
                st.pyplot(fig)

        with res_col2:
            st.markdown("##### 🧪 Agronomic Explanation of Input Conditions")
            for feature, expl in explanation.items():
                st.markdown(f"• **{feature}:** {expl}")

# =============================================================================
# PAGE 3: CROP YIELD PREDICTION (REGRESSION)
# =============================================================================
elif menu == "🚜 3. Crop Yield Prediction (Regression)":
    st.markdown("## 🚜 Crop Yield Prediction")
    st.write("Predict expected harvest yield in **Tonnes per Hectare (t/ha)** and total output based on farm area.")

    yield_meta = artifacts["preprocessing"]["yield_summary"]
    crops_list = yield_meta["crops"]
    seasons_list = yield_meta["seasons"]
    states_list = yield_meta["states"]

    st.markdown("##### ⚡ Quick-Fill Presets for Testing:")
    yp_col1, yp_col2, yp_col3, yp_col4 = st.columns(4)
    preset_yield = None
    if yp_col1.button("🌾 Punjab Wheat"):
        preset_yield = {"crop": "Wheat", "season": "Rabi", "state": "Punjab", "area": 10.0}
    elif yp_col2.button("🍬 UP Sugarcane"):
        preset_yield = {"crop": "Sugarcane", "season": "Whole Year", "state": "Uttar Pradesh", "area": 5.0}
    elif yp_col3.button("🌽 Karnataka Maize"):
        preset_yield = {"crop": "Maize", "season": "Kharif", "state": "Karnataka", "area": 8.0}
    elif yp_col4.button("☁️ Maharashtra Cotton"):
        preset_yield = {"crop": "Cotton(lint)", "season": "Kharif", "state": "Maharashtra", "area": 15.0}

    with st.form("yield_form"):
        c_crop, c_season, c_state = st.columns(3)
        default_crop = preset_yield["crop"] if (preset_yield and preset_yield["crop"] in crops_list) else "Rice"
        default_season = preset_yield["season"] if (preset_yield and preset_yield["season"] in seasons_list) else seasons_list[0]
        default_state = preset_yield["state"] if (preset_yield and preset_yield["state"] in states_list) else "Uttar Pradesh"
        default_area = preset_yield["area"] if preset_yield else 5.0

        val_crop = c_crop.selectbox("Crop Type:", crops_list, index=crops_list.index(default_crop))
        val_season = c_season.selectbox("Season:", seasons_list, index=seasons_list.index(default_season))
        val_state = c_state.selectbox("State / UT:", states_list, index=states_list.index(default_state))
        val_area = st.number_input("Cultivated Farm Area (Hectares):", min_value=0.1, max_value=10000.0, value=float(default_area), step=0.5)

        submit_yield = st.form_submit_button("🚜 Predict Expected Yield")

    if submit_yield:
        input_yield_data = {"Crop": val_crop, "Season": val_season, "State_Name": val_state, "Area": float(val_area)}
        yield_pipeline = artifacts["yield_model"]
        pred_yield, est_prod = predict_yield(yield_pipeline, input_yield_data)

        st.markdown(
            f"""
            <div class='result-banner'>
                <div class='result-title'>🎯 Forecasted Crop Productivity:</div>
                <div class='result-highlight'>🌾 {pred_yield:.2f} Tonnes / Hectare</div>
                <div style='margin-top: 10px; color: #1B5E20; font-size: 1.05rem;'>
                    <b>Total Estimated Harvest:</b> <span style='font-size: 1.3rem; font-weight: 700; color: #2E7D32;'>{est_prod:,.2f} Tonnes</span> across {val_area:.1f} hectares
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# PAGE 4: SOIL PATTERN CLUSTERING (K-MEANS)
# =============================================================================
elif menu == "🧪 4. Soil Pattern Clustering (K-Means)":
    st.markdown("## 🧪 Soil Pattern Analysis using K-Means")
    st.write("Group agricultural fields into similar soil nutrient patterns using **Unsupervised K-Means Clustering (k=4)**.")

    soil_model_data = artifacts["soil_model"]

    with st.form("soil_cluster_form"):
        sc1, sc2, sc3, sc4 = st.columns(4)
        in_n = sc1.number_input("Nitrogen (N) [kg/ha]:", min_value=0, max_value=150, value=95, step=1)
        in_p = sc2.number_input("Phosphorus (P) [kg/ha]:", min_value=5, max_value=150, value=45, step=1)
        in_k = sc3.number_input("Potassium (K) [kg/ha]:", min_value=5, max_value=210, value=38, step=1)
        in_ph = sc4.number_input("Soil pH:", min_value=3.5, max_value=10.0, value=6.5, step=0.1)

        submit_soil = st.form_submit_button("🧪 Analyze Soil Pattern")

    if submit_soil:
        input_soil = {"N": in_n, "P": in_p, "K": in_k, "ph": in_ph}
        c_id, pca_pt, profile = predict_soil_cluster(soil_model_data, input_soil)

        st.markdown(
            f"""
            <div class='result-banner'>
                <div class='result-title'>🎯 Assigned Soil Cluster:</div>
                <div class='result-highlight' style='color: {profile.get("badge_color", "#2E7D32")};'>{profile.get("name", f"Cluster {c_id}")}</div>
                <div style='margin-top: 8px; font-size: 1rem; color: #2E7D32;'>
                    <b>Overview:</b> {profile.get("description", "")}<br>
                    <b>Agronomic Trait:</b> {profile.get("characteristics", "")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### 🗺️ 2D PCA Scatter Visualization of Soil Patterns")
        pca_df = soil_model_data["pca_df"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#2E7D32", "#1565C0", "#E65100", "#6A1B9A"]
        for i in range(4):
            subset = pca_df[pca_df["Cluster"] == i]
            ax.scatter(subset["PCA1"], subset["PCA2"], c=colors[i], label=f"Cluster {i} (n={len(subset)})", alpha=0.45, s=35)
        ax.scatter(pca_pt[0], pca_pt[1], c="#D50000", marker="*", s=280, label="Your Soil Sample", edgecolors="black", linewidth=1.5, zorder=10)
        ax.set_title("Soil Nutrient Clusters in 2D PCA Space", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right")
        st.pyplot(fig)

# =============================================================================
# PAGE 5: MODEL PERFORMANCE & VIVA GUIDE
# =============================================================================
elif menu == "📊 5. Model Performance & Viva Guide":
    st.markdown("## 📊 Model Evaluation, Comparisons & Viva Guide")
    eval_data = artifacts["evaluation"]

    tab_eval_clf, tab_eval_reg, tab_eval_clus, tab_viva = st.tabs([
        "🌾 Classification Performance", "🚜 Regression Performance", "🧪 Clustering Performance", "🎓 College Viva Voce Guide"
    ])

    with tab_eval_clf:
        clf_res = eval_data["classification"]["results"]
        metrics_list = [{"Model": name, "Accuracy": d["accuracy"], "Precision": d["precision"], "Recall": d["recall"], "F1-Score": d["f1_score"]} for name, d in clf_res.items()]
        st.dataframe(pd.DataFrame(metrics_list), use_container_width=True)

        rf_cm = clf_res["Random Forest"]["confusion_matrix"]
        rf_classes = clf_res["Random Forest"]["classes"]
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Greens", xticklabels=rf_classes, yticklabels=rf_classes, ax=ax_cm, cbar=False, annot_kws={"size": 7})
        ax_cm.set_title("Confusion Matrix: Random Forest Classifier", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        st.pyplot(fig_cm)

    with tab_eval_reg:
        reg_res = eval_data["regression"]["results"]
        r_list = [{"Algorithm": name, "R² Score": round(d["R2_Score"], 4), "MAE (t/ha)": round(d["MAE"], 4), "RMSE (t/ha)": round(d["RMSE"], 4)} for name, d in reg_res.items()]
        st.dataframe(pd.DataFrame(r_list), use_container_width=True)

    with tab_eval_clus:
        st.metric("Silhouette Score (k=4)", f"{eval_data['clustering']['silhouette']:.4f}")
        st.dataframe(eval_data["clustering"]["summary_df"], use_container_width=True)

    with tab_viva:
        st.markdown("### 🎓 Viva Voce Cheat Sheet")
        with st.expander("❓ Q1: Why choose Random Forest over single Decision Trees?"):
            st.write("A single Decision Tree has high variance and easily overfits. Random Forest uses bootstrap aggregation (bagging) over randomized tree ensembles to cancel out variance and maximize generalization.")
        with st.expander("❓ Q2: Why is Feature Scaling mandatory for KNN and K-Means?"):
            st.write(r"KNN and K-Means rely on Euclidean distances ($d = \sqrt{\sum(x_i - y_i)^2}$). High magnitude features (like Rainfall: 200 mm) would overwhelm smaller features (like pH: 6.5) without standardization.")
        with st.expander("❓ Q3: How is Crop Yield calculated?"):
            st.write(r"Yield is mathematically derived as $\text{Yield} = \frac{\text{Production (tonnes)}}{\text{Area (hectares)}}$. Outliers (> 100 t/ha) and non-tonne crops like Coconut were removed.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #78909C; font-size: 0.85rem;'>🌾 <b>Smart Crop & Agriculture Analyzer</b> • Academic AI/ML Project</div>", unsafe_allow_html=True)
