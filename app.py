import streamlit as st
import numpy as np
import joblib
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor | Decision Tree",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a6b3a 0%, #27ae60 60%, #2ecc71 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(39,174,96,0.25);
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p  { color: rgba(255,255,255,0.85); margin: 0.4rem 0 0; font-size: 1rem; }

    .result-box {
        background: linear-gradient(135deg, #eafaf1, #d5f5e3);
        border-left: 6px solid #27ae60;
        padding: 1.8rem 2rem;
        border-radius: 14px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(39,174,96,0.12);
    }
    .result-box .price {
        font-size: 3rem;
        font-weight: 700;
        color: #1a6b3a;
        line-height: 1;
    }
    .result-box .label {
        font-size: 0.9rem;
        color: #555;
        margin-top: 0.4rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e8f5e9;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-card .val { font-size: 1.7rem; font-weight: 700; color: #27ae60; }
    .metric-card .lbl { font-size: 0.75rem; color: #888; margin-top: 0.2rem; }

    .stButton > button {
        background: linear-gradient(135deg, #1a6b3a, #27ae60) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 2.5rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(39,174,96,0.3) !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(39,174,96,0.45) !important;
    }

    .range-bar {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #27ae60, #f1c40f, #e74c3c);
        margin: 0.5rem 0;
        position: relative;
    }
    .range-indicator {
        width: 14px; height: 14px;
        background: #1a6b3a;
        border: 2px solid white;
        border-radius: 50%;
        position: absolute;
        top: -2px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }

    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e8f5e9;
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
    <h1>🏠 House Price Predictor</h1>
    <p>Decision Tree Regression · Estimate median home value from neighbourhood features</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏘️ Property & Neighbourhood")

    st.markdown("### 🏗️ Structure")
    rm      = st.slider("Avg Rooms per Dwelling",  3.0, 9.0, 6.3, step=0.1)
    age     = st.slider("% Owner-Occupied Units Built Before 1940", 0.0, 100.0, 68.0, step=0.5)
    chas    = st.selectbox("Charles River Adjacent?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    st.markdown("### 📍 Location")
    dis     = st.slider("Weighted Dist. to Employment Centres", 1.0, 13.0, 3.8, step=0.1)
    rad     = st.slider("Accessibility to Radial Highways (index)", 1, 24, 5)
    crim    = st.slider("Per Capita Crime Rate", 0.0, 90.0, 3.6, step=0.1)

    st.markdown("### 🌿 Environment")
    nox     = st.slider("Nitric Oxides Concentration (ppm × 10)", 0.38, 0.87, 0.55, step=0.01)
    zn      = st.slider("% Residential Land (>25k sq ft)", 0.0, 100.0, 11.0, step=1.0)
    indus   = st.slider("% Non-Retail Business Acres", 0.5, 28.0, 11.1, step=0.1)

    st.markdown("### 💰 Tax & Social")
    tax     = st.slider("Full-Value Property Tax Rate (per $10k)", 187, 711, 408, step=5)
    ptratio = st.slider("Pupil-Teacher Ratio", 12.6, 22.0, 18.5, step=0.1)
    lstat   = st.slider("% Lower-Status Population", 1.7, 38.0, 12.6, step=0.1)
    b       = st.slider("B — 1000(Bk - 0.63)²  (racial diversity proxy)", 0.3, 396.9, 356.0, step=0.5)

# ── Summary cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="metric-card"><div class="val">{rm:.1f}</div><div class="lbl">Avg Rooms</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><div class="val">{lstat:.1f}%</div><div class="lbl">Lower-Status Pop.</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><div class="val">{crim:.2f}</div><div class="lbl">Crime Rate</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="metric-card"><div class="val">{ptratio:.1f}</div><div class="lbl">Pupil-Teacher Ratio</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Input summary ──────────────────────────────────────────────────────────────
with st.expander("📊 Full Input Summary", expanded=False):
    import pandas as pd
    feat_names = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat']
    feat_vals  = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]
    feat_desc  = [
        'Per Capita Crime Rate', '% Residential Land (large)', '% Non-Retail Business',
        'Charles River Adjacent', 'Nitric Oxides Conc.', 'Avg Rooms per Dwelling',
        '% Old Owner-Occupied Units', 'Dist. to Employment', 'Highway Access Index',
        'Property Tax Rate', 'Pupil-Teacher Ratio', 'Racial Diversity (B)', '% Lower-Status Pop.'
    ]
    summary_df = pd.DataFrame({'Feature': feat_names, 'Description': feat_desc, 'Value': feat_vals})
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── Predict ────────────────────────────────────────────────────────────────────
st.markdown("### 🔮 Price Prediction")
predict_btn = st.button("Predict House Price →")

if predict_btn:
    if not model_loaded:
        st.error("Model not loaded. Check the models/ folder.")
    else:
        input_arr    = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
        input_scaled = scaler.transform(input_arr)
        prediction   = model.predict(input_scaled)[0]
        prediction   = max(5.0, min(50.0, prediction))

        price_usd = prediction * 1000

        st.markdown(f"""
        <div class="result-box">
            <div class="price">${price_usd:,.0f}</div>
            <div class="label">Estimated Median House Value · {prediction:.1f} (×$1,000 index units)</div>
        </div>
        """, unsafe_allow_html=True)

        # Bracket indicator
        pct = (prediction - 5) / (50 - 5)
        bracket = "Budget" if prediction < 15 else "Mid-Range" if prediction < 30 else "Premium"
        bracket_color = "#e74c3c" if prediction < 15 else "#f1c40f" if prediction < 30 else "#27ae60"

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Value (index)", f"{prediction:.1f}")
        col2.metric("Market Bracket", bracket)
        col3.metric("Price Range Position", f"Top {100 - int(pct*100)}%")

        # Visual price bar
        left_pct = int(pct * 100)
        st.markdown(f"""
        <div style="margin: 1rem 0 0.3rem;">
            <span style="font-size:0.85rem;color:#666;">Budget ($5k)</span>
            <span style="float:right;font-size:0.85rem;color:#666;">Premium ($50k)</span>
        </div>
        <div class="range-bar">
            <div class="range-indicator" style="left: calc({left_pct}% - 7px);"></div>
        </div>
        <div style="text-align:center;font-size:0.8rem;color:#888;margin-top:0.3rem;">
            Your property sits at <strong style="color:{bracket_color}">{bracket}</strong> level
        </div>
        """, unsafe_allow_html=True)

        # Feature influence note
        st.info(
            f"📌 Key drivers: **rm** ({rm} rooms), **lstat** ({lstat}% lower-status), "
            f"**crim** ({crim:.2f} crime rate), **nox** ({nox:.3f} pollution)."
        )

        st.caption("⚠️ This model is for educational purposes. Predictions are based on a synthetic dataset.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:0.8rem;'>"
    "Built with Streamlit · Decision Tree Regression · Boston Housing Dataset"
    "</p>",
    unsafe_allow_html=True,
)
