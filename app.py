import streamlit as st
import numpy as np
import joblib
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor | Decision Tree",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #7d3c98 0%, #a569bd 60%, #d2b4de 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(125,60,152,0.25);
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p  { color: rgba(255,255,255,0.85); margin: 0.4rem 0 0; font-size: 1rem; }

    .result-positive {
        background: linear-gradient(135deg, #fdf2f8, #f5cba7);
        border-left: 6px solid #e67e22;
        padding: 1.8rem 2rem;
        border-radius: 14px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(230,126,34,0.15);
    }
    .result-negative {
        background: linear-gradient(135deg, #f0f8ff, #aed6f1);
        border-left: 6px solid #2980b9;
        padding: 1.8rem 2rem;
        border-radius: 14px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(41,128,185,0.15);
    }
    .result-positive h3 { color: #784212; margin: 0 0 0.5rem; font-size: 1.3rem; }
    .result-negative h3 { color: #1a5276; margin: 0 0 0.5rem; font-size: 1.3rem; }
    .result-positive p, .result-negative p { margin: 0; color: #555; font-size: 0.95rem; }

    .prob-bar-wrap {
        background: #f0f0f0;
        border-radius: 8px;
        height: 14px;
        margin: 0.6rem 0;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
    }

    .metric-card {
        background: white;
        border: 1px solid #e8daef;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-card .val { font-size: 1.7rem; font-weight: 700; color: #7d3c98; }
    .metric-card .lbl { font-size: 0.75rem; color: #888; margin-top: 0.2rem; }

    .stButton > button {
        background: linear-gradient(135deg, #7d3c98, #a569bd) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 2.5rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(125,60,152,0.3) !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(125,60,152,0.45) !important;
    }

    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e8daef;
        border-radius: 10px;
        padding: 0.8rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model  = joblib.load(os.path.join(BASE_DIR, "models", "dt_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load model: {e}")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌳 Heart Disease Predictor</h1>
    <p>Decision Tree Classifier · Enter patient vitals to assess heart disease risk</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Patient Information")

    st.markdown("### 🧬 Demographics")
    age = st.slider("Age (years)", 20, 80, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

    st.markdown("### 🫀 Cardiac Symptoms")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                      format_func=lambda x: {0:"Typical Angina", 1:"Atypical Angina",
                                              2:"Non-Anginal Pain", 3:"Asymptomatic"}[x])
    thalach = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, 150)
    exang   = st.selectbox("Exercise-Induced Angina", [0, 1],
                            format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 7.0, 1.0, step=0.1)
    slope   = st.selectbox("Slope of ST Segment", [0, 1, 2],
                            format_func=lambda x: {0:"Upsloping", 1:"Flat", 2:"Downsloping"}[x])

    st.markdown("### 🩸 Lab Results")
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
    chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                             format_func=lambda x: "No" if x == 0 else "Yes")
    restecg  = st.selectbox("Resting ECG", [0, 1, 2],
                             format_func=lambda x: {0:"Normal", 1:"ST-T Abnormality",
                                                    2:"LV Hypertrophy"}[x])

    st.markdown("### 🔬 Other Tests")
    ca   = st.selectbox("Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3],
                         format_func=lambda x: {1:"Normal", 2:"Fixed Defect",
                                                3:"Reversible Defect"}[x])

# ── Summary Cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="metric-card"><div class="val">{age}</div><div class="lbl">Age</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><div class="val">{thalach}</div><div class="lbl">Max Heart Rate</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><div class="val">{chol}</div><div class="lbl">Cholesterol</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="metric-card"><div class="val">{trestbps}</div><div class="lbl">Blood Pressure</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Full Input Summary ─────────────────────────────────────────────────────────
with st.expander("📊 Full Input Summary", expanded=False):
    import pandas as pd
    feat_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
                  'thalach','exang','oldpeak','slope','ca','thal']
    feat_vals  = [age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]
    summary_df = pd.DataFrame({'Feature': feat_names, 'Value': feat_vals})
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── Predict ────────────────────────────────────────────────────────────────────
st.markdown("### 🔮 Prediction")
predict_btn = st.button("Analyse Risk →")

if predict_btn:
    if not model_loaded:
        st.error("Model not loaded. Check the models/ folder.")
    else:
        input_arr    = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                   thalach, exang, oldpeak, slope, ca, thal]])
        input_scaled = scaler.transform(input_arr)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]

        prob_pos = probability[1] * 100
        prob_neg = probability[0] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <h3>⚠️ High Risk of Heart Disease Detected</h3>
                <p>The Decision Tree predicts <strong>presence of heart disease</strong> with
                <strong>{prob_pos:.1f}% confidence</strong>. Please consult a cardiologist promptly.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <h3>✅ Low Risk — No Heart Disease Detected</h3>
                <p>The Decision Tree predicts <strong>absence of heart disease</strong> with
                <strong>{prob_neg:.1f}% confidence</strong>. Continue healthy lifestyle habits.</p>
            </div>
            """, unsafe_allow_html=True)

        # Probability bars
        st.markdown("#### Prediction Confidence")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**No Disease — {prob_neg:.1f}%**")
            st.markdown(f"""
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill" style="width:{prob_neg:.1f}%;background:linear-gradient(90deg,#2980b9,#7fb3d3);"></div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"**Heart Disease — {prob_pos:.1f}%**")
            st.markdown(f"""
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill" style="width:{prob_pos:.1f}%;background:linear-gradient(90deg,#e67e22,#f0b27a);"></div>
            </div>
            """, unsafe_allow_html=True)

        # Key metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction",       "Disease" if prediction == 1 else "No Disease")
        m2.metric("Disease Probability", f"{prob_pos:.1f}%")
        m3.metric("Confidence Level",
                  "High" if max(prob_pos, prob_neg) > 70 else
                  "Moderate" if max(prob_pos, prob_neg) > 55 else "Low")

        # Decision path info
        tree_depth = model.get_depth()
        tree_leaves = model.get_n_leaves()
        st.markdown(f"""
        <div style="background:#f8f0fc;border-radius:10px;padding:1rem;margin-top:1rem;
                    border:1px solid #d7bde2;font-size:0.88rem;color:#555;">
            🌳 <strong>Tree Info</strong> &nbsp;|&nbsp;
            Max Depth: <strong>{tree_depth}</strong> &nbsp;|&nbsp;
            Number of Leaves: <strong>{tree_leaves}</strong> &nbsp;|&nbsp;
            Features Used: <strong>13</strong>
        </div>
        """, unsafe_allow_html=True)

        st.info("⚠️ **Disclaimer**: This tool is for educational purposes only and does not replace professional medical diagnosis.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:0.8rem;'>"
    "Built with Streamlit · Decision Tree Classifier · Heart Disease Dataset"
    "</p>",
    unsafe_allow_html=True,
)
