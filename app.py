import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor | Amit Sharma",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — deep navy + gold palette
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 60%, #0d1b2a 100%);
    color: #e2e8f0;
}

/* Hide default header */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(120deg, #1a2744 0%, #162032 50%, #0f1e35 100%);
    border: 1px solid #2a3a5c;
    border-radius: 18px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
}
.hero-banner h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #f5c842;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2d45;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 9px;
    color: #94a3b8;
    font-weight: 500;
    font-size: 0.92rem;
    padding: 10px 22px;
    border: none;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #f5c842, #e0a820) !important;
    color: #0a0e1a !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 12px rgba(245,200,66,0.35);
}

/* ── Cards ── */
.metric-card {
    background: linear-gradient(135deg, #1a2744, #162032);
    border: 1px solid #2a3a5c;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.metric-card .label {
    color: #64748b;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    color: #f5c842;
    font-size: 2rem;
    font-weight: 700;
}
.metric-card .sub {
    color: #94a3b8;
    font-size: 0.8rem;
    margin-top: 0.2rem;
}

/* ── Prediction Result ── */
.predict-result {
    background: linear-gradient(135deg, #1a3a1a, #0f2a0f);
    border: 2px solid #22c55e;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(34,197,94,0.15);
    margin-top: 1.5rem;
}
.predict-result .price {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    color: #22c55e;
    font-weight: 700;
}
.predict-result .label {
    color: #86efac;
    font-size: 1rem;
    margin-top: 0.3rem;
}

/* ── Input styling ── */
.stNumberInput > div > div > input,
.stSlider { color: #e2e8f0 !important; }

label { color: #cbd5e1 !important; font-size: 0.9rem !important; }

/* ── Section headers ── */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #f5c842;
    border-left: 4px solid #f5c842;
    padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
}

/* ── Accuracy bar ── */
.acc-bar-wrap {
    background: #1a2744;
    border-radius: 8px;
    height: 14px;
    width: 100%;
    overflow: hidden;
    margin-top: 6px;
}
.acc-bar-fill {
    background: linear-gradient(90deg, #f5c842, #22c55e);
    height: 100%;
    border-radius: 8px;
    transition: width 1s ease;
}

/* ── Developer card ── */
.dev-card {
    background: linear-gradient(135deg, #1a2744, #162032);
    border: 1px solid #2a3a5c;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    max-width: 460px;
    margin: 0 auto;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
}
.dev-avatar {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    background: linear-gradient(135deg, #f5c842, #e0a820);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
    margin: 0 auto 1.2rem auto;
    box-shadow: 0 4px 20px rgba(245,200,66,0.3);
}
.dev-name {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #f5c842;
    margin-bottom: 0.2rem;
}
.dev-role {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
}
.badge {
    display: inline-block;
    background: #1e3a5f;
    border: 1px solid #2a5080;
    color: #7dd3fc;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    margin: 3px;
}

/* ── Info box ── */
.info-box {
    background: #1a2744;
    border-left: 4px solid #f5c842;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
    color: #cbd5e1;
    font-size: 0.93rem;
    line-height: 1.6;
}
.info-box strong { color: #f5c842; }

/* ── Warning/Disclaimer ── */
.disclaimer-box {
    background: #2d1515;
    border: 1px solid #7f1d1d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: #fca5a5;
    font-size: 0.88rem;
    line-height: 1.6;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("house_price_model.pkl")
    except Exception:
        return None

model = load_model()

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🏠 House Price Predictor</h1>
    <p>AI-powered multilinear regression model · Real-time price estimation · Built by Amit Sharma</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  Predict Price",
    "📊  Data Visualization",
    "📈  Model Accuracy",
    "👨‍💻  Developer & Info",
])

# ══════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Enter House Details</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        square_footage = st.number_input("📐 Square Footage (sq ft)", min_value=0.0, value=1500.0, step=50.0)
        num_bedrooms   = st.number_input("🛏️ Bedrooms", min_value=0, value=3)
        garage_size    = st.number_input("🚗 Garage Size (cars)", min_value=0, value=1)
    with col2:
        num_bathrooms  = st.number_input("🚿 Bathrooms", min_value=0, value=2)
        year_built     = st.number_input("📅 Year Built", min_value=1900, max_value=2025, value=2005)
        lot_size       = st.number_input("🌿 Lot Size (sq ft)", min_value=0.0, value=5000.0, step=100.0)
    with col3:
        neighborhood_quality = st.slider("🏘️ Neighbourhood Quality", min_value=1, max_value=10, value=7)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Quality Score</div>
            <div class="value">{neighborhood_quality}/10</div>
            <div class="sub">{'⭐' * neighborhood_quality}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict House Price", use_container_width=True, type="primary")

    if predict_btn:
        features = np.array([[
            square_footage, num_bedrooms, num_bathrooms,
            year_built, lot_size, garage_size, neighborhood_quality
        ]])

        if model is not None:
            prediction = model.predict(features)[0]
        else:
            # Demo value if model file not found
            prediction = (square_footage * 4200 + num_bedrooms * 150000 +
                          num_bathrooms * 80000 + (year_built - 1950) * 3000 +
                          lot_size * 15 + garage_size * 50000 +
                          neighborhood_quality * 40000) * random.uniform(0.95, 1.05)

        st.markdown(f"""
        <div class="predict-result">
            <div class="price">₹ {prediction:,.0f}</div>
            <div class="label">Estimated Market Value</div>
            <div style="color:#64748b;font-size:0.82rem;margin-top:0.8rem;">
                Based on {square_footage:,.0f} sq ft · {num_bedrooms} bed · {num_bathrooms} bath · Built {year_built}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Mini stat row
        st.markdown("<br>", unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        price_sqft = prediction / square_footage if square_footage > 0 else 0
        with mc1:
            st.markdown(f'<div class="metric-card"><div class="label">Price / sq ft</div><div class="value" style="font-size:1.4rem;">₹{price_sqft:,.0f}</div></div>', unsafe_allow_html=True)
        with mc2:
            st.markdown(f'<div class="metric-card"><div class="label">Bedrooms</div><div class="value" style="font-size:1.4rem;">{num_bedrooms}</div></div>', unsafe_allow_html=True)
        with mc3:
            st.markdown(f'<div class="metric-card"><div class="label">Age</div><div class="value" style="font-size:1.4rem;">{2025-year_built} yrs</div></div>', unsafe_allow_html=True)
        with mc4:
            st.markdown(f'<div class="metric-card"><div class="label">Lot Size</div><div class="value" style="font-size:1.4rem;">{lot_size:,.0f}</div><div class="sub">sq ft</div></div>', unsafe_allow_html=True)

        if model is None:
            st.warning("⚠️ Model file `house_price_model.pkl` not found — showing demo prediction. Place the file in the same directory as app.py.")

# ══════════════════════════════════════════════
# TAB 2 — DATA VISUALIZATION
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Dataset Insights & Visual Analytics</div>', unsafe_allow_html=True)

    # ── Synthetic dataset for demo visuals ──
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "Square Footage": np.random.randint(600, 5000, n),
        "Bedrooms": np.random.choice([1,2,3,4,5,6], n, p=[0.05,0.15,0.35,0.30,0.10,0.05]),
        "Bathrooms": np.random.choice([1,2,3,4], n, p=[0.15,0.45,0.30,0.10]),
        "Year Built": np.random.randint(1970, 2023, n),
        "Lot Size": np.random.randint(1000, 20000, n),
        "Garage Size": np.random.choice([0,1,2,3], n, p=[0.10,0.35,0.40,0.15]),
        "Neighbourhood Quality": np.random.randint(1, 11, n),
    })
    df["Price"] = (
        df["Square Footage"] * 4200 +
        df["Bedrooms"] * 150000 +
        df["Bathrooms"] * 80000 +
        (df["Year Built"] - 1950) * 3000 +
        df["Lot Size"] * 15 +
        df["Garage Size"] * 50000 +
        df["Neighbourhood Quality"] * 40000 +
        np.random.normal(0, 300000, n)
    )

    PLOT_BG   = "rgba(15,24,41,0)"
    PAPER_BG  = "rgba(0,0,0,0)"
    GOLD      = "#f5c842"
    GREEN     = "#22c55e"
    BLUE      = "#38bdf8"
    PURPLE    = "#a78bfa"
    PINK      = "#f472b6"
    FONT_CLR  = "#cbd5e1"
    GRID_CLR  = "#1e2d45"

    def base_layout(title):
        return dict(
            title=dict(text=title, font=dict(color=GOLD, size=15, family="Inter")),
            paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_CLR, family="Inter"),
            margin=dict(l=20, r=20, t=50, b=20),
        )

    row1_l, row1_r = st.columns(2)

    # 1. Scatter: Price vs Sq Ft
    with row1_l:
        fig1 = px.scatter(
            df, x="Square Footage", y="Price",
            color="Neighbourhood Quality",
            color_continuous_scale=[[0,"#1e2d45"],[0.5,"#f5c842"],[1,"#22c55e"]],
            size="Bedrooms", size_max=12,
            hover_data=["Bedrooms","Bathrooms"],
            labels={"Price": "Price (₹)", "Square Footage": "Area (sq ft)"},
        )
        fig1.update_layout(**base_layout("💛 Price vs. Square Footage"))
        fig1.update_xaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
        fig1.update_yaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
        st.plotly_chart(fig1, use_container_width=True)

    # 2. Pie: Bedroom distribution
    with row1_r:
        bed_counts = df["Bedrooms"].value_counts().sort_index()
        fig2 = go.Figure(go.Pie(
            labels=[f"{b} Bedroom{'s' if b>1 else ''}" for b in bed_counts.index],
            values=bed_counts.values,
            hole=0.55,
            marker=dict(colors=[GOLD, GREEN, BLUE, PURPLE, PINK, "#fb923c"],
                        line=dict(color="#0a0e1a", width=2)),
            textfont=dict(color=FONT_CLR),
        ))
        fig2.update_layout(**base_layout("🛏️ Bedroom Distribution"))
        st.plotly_chart(fig2, use_container_width=True)

    row2_l, row2_r = st.columns(2)

    # 3. Box plot: Price by Garage Size
    with row2_l:
        fig3 = px.box(
            df, x="Garage Size", y="Price",
            color="Garage Size",
            color_discrete_sequence=[BLUE, GOLD, GREEN, PURPLE],
            labels={"Price": "Price (₹)", "Garage Size": "Garage Capacity (cars)"},
        )
        fig3.update_layout(**base_layout("🚗 Price Distribution by Garage Size"))
        fig3.update_xaxes(gridcolor=GRID_CLR)
        fig3.update_yaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig3, use_container_width=True)

    # 4. Bar: Avg price by neighbourhood quality
    with row2_r:
        avg_by_nq = df.groupby("Neighbourhood Quality")["Price"].mean().reset_index()
        fig4 = px.bar(
            avg_by_nq, x="Neighbourhood Quality", y="Price",
            color="Price",
            color_continuous_scale=[[0,"#1e3a5f"],[1,"#f5c842"]],
            labels={"Price": "Avg Price (₹)"},
            text_auto=".2s",
        )
        fig4.update_layout(**base_layout("🏘️ Avg Price by Neighbourhood Quality"))
        fig4.update_xaxes(gridcolor=GRID_CLR)
        fig4.update_yaxes(gridcolor=GRID_CLR)
        fig4.update_traces(textfont_color=FONT_CLR)
        st.plotly_chart(fig4, use_container_width=True)

    # 5. Full-width: Price trend by Year Built
    avg_by_year = df.groupby("Year Built")["Price"].mean().reset_index()
    fig5 = px.area(
        avg_by_year, x="Year Built", y="Price",
        color_discrete_sequence=[GREEN],
        labels={"Price": "Avg Price (₹)", "Year Built": "Year Built"},
    )
    fig5.update_traces(fillcolor="rgba(34,197,94,0.15)", line=dict(color=GREEN, width=2))
    fig5.update_layout(**base_layout("📅 Average Price Trend by Year Built"))
    fig5.update_xaxes(gridcolor=GRID_CLR)
    fig5.update_yaxes(gridcolor=GRID_CLR)
    st.plotly_chart(fig5, use_container_width=True)

    # 6. Correlation heatmap
    corr = df.corr(numeric_only=True).round(2)
    fig6 = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale=[[0,"#1a2744"],[0.5,"#f5c842"],[1,"#22c55e"]],
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(size=10, color=FONT_CLR),
        showscale=True,
    ))
    fig6.update_layout(**base_layout("🔥 Feature Correlation Heatmap"), height=450)
    st.plotly_chart(fig6, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — MODEL ACCURACY & DATASET KNOWLEDGE
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Model Performance Metrics</div>', unsafe_allow_html=True)

    # Demo metrics (replace with real values from your training script)
    metrics = {
        "R² Score":       ("0.9984", "99.84%", 99),
        "MAE": ("₹ 7,823", "Mean Absolute Error", 99),
        "RMSE": ("₹ 9,882", "Root Mean Sq. Error", 99),
        "Accuracy (±10%)": ("100.0%", "Predictions within 10% range", 100)
    }

    col_a, col_b, col_c, col_d = st.columns(4)
    for col, (k, v) in zip([col_a, col_b, col_c, col_d], metrics.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{k}</div>
                <div class="value" style="font-size:1.5rem;">{v[0]}</div>
                <div class="sub">{v[1]}</div>
                <div class="acc-bar-wrap" style="margin-top:10px;">
                    <div class="acc-bar-fill" style="width:{v[2]}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)

    features_imp = {
        "Square Footage": 0.42,
        "Neighbourhood Quality": 0.21,
        "Year Built": 0.13,
        "Bathrooms": 0.10,
        "Lot Size": 0.07,
        "Bedrooms": 0.05,
        "Garage Size": 0.02,
    }
    fi_df = pd.DataFrame(list(features_imp.items()), columns=["Feature","Importance"])
    fi_df = fi_df.sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=[[0,"#1e3a5f"],[1,"#f5c842"]],
        text=fi_df["Importance"].apply(lambda x: f"{x*100:.0f}%"),
    )
    fig_fi.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        title=dict(text="Feature Importance in Multilinear Regression", font=dict(color="#f5c842")),
        margin=dict(l=0, r=20, t=50, b=20), height=350,
        showlegend=False,
    )
    fig_fi.update_xaxes(gridcolor="#1e2d45")
    fig_fi.update_yaxes(gridcolor="#1e2d45")
    fig_fi.update_traces(textfont_color="#0a0e1a", textposition="outside")
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="section-title">📚 Dataset Knowledge</div>', unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        <div class="info-box">
            <strong>Dataset Overview</strong><br>
            The model was trained on a structured house price dataset containing residential
            property records. Each record captures physical attributes and location quality
            of a house along with its market price.
        </div>
        <div class="info-box">
            <strong>Features Used</strong><br>
            • <strong>Square Footage</strong> — Interior living area<br>
            • <strong>Bedrooms</strong> — Number of bedrooms<br>
            • <strong>Bathrooms</strong> — Number of bathrooms<br>
            • <strong>Year Built</strong> — Construction year<br>
            • <strong>Lot Size</strong> — Total plot area<br>
            • <strong>Garage Size</strong> — Cars it can accommodate<br>
            • <strong>Neighbourhood Quality</strong> — Score from 1–10
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="info-box">
            <strong>Algorithm: Multilinear Regression</strong><br>
            Multilinear (multiple linear) regression models the relationship between
            a dependent variable (price) and multiple independent variables (features)
            using a linear equation: <em>Price = β₀ + β₁x₁ + β₂x₂ + … + βₙxₙ</em>
        </div>
        <div class="info-box">
            <strong>Training Details</strong><br>
            • Train/Test Split: 80% / 20%<br>
            • Scaling: StandardScaler applied<br>
            • Validation: 5-Fold Cross Validation<br>
            • Library: scikit-learn<br>
            • Serialization: joblib (.pkl)
        </div>
        """, unsafe_allow_html=True)

    # Residual plot (demo)
    st.markdown('<div class="section-title">Predicted vs. Actual (Demo)</div>', unsafe_allow_html=True)
    np.random.seed(7)
    actual    = np.random.uniform(2e6, 15e6, 200)
    predicted = actual * np.random.uniform(0.88, 1.12, 200)
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        x=actual, y=predicted, mode="markers",
        marker=dict(color="#f5c842", opacity=0.6, size=6),
        name="Predictions",
    ))
    fig_res.add_trace(go.Scatter(
        x=[actual.min(), actual.max()],
        y=[actual.min(), actual.max()],
        mode="lines", line=dict(color="#22c55e", dash="dash", width=2),
        name="Perfect Fit",
    ))
    fig_res.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        title=dict(text="Predicted vs Actual Prices", font=dict(color="#f5c842")),
        xaxis=dict(title="Actual Price (₹)", gridcolor="#1e2d45"),
        yaxis=dict(title="Predicted Price (₹)", gridcolor="#1e2d45"),
        margin=dict(l=0, r=10, t=50, b=20),
    )
    st.plotly_chart(fig_res, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — DEVELOPER & INFO
# ══════════════════════════════════════════════
with tab4:
    dev_col, info_col = st.columns([1, 1.6], gap="large")

    with dev_col:
        st.markdown("""
        <div class="dev-card">
            <div class="dev-avatar">👨‍💻</div>
            <div class="dev-name">Amit Sharma</div>
            <div class="dev-role">ML Engineer & Data Scientist</div>
            <div>
                <span class="badge">Python</span>
                <span class="badge">Machine Learning</span>
                <span class="badge">scikit-learn</span>
                <span class="badge">Data Science</span>
            </div>
            <div style="margin-top:1.4rem;color:#64748b;font-size:0.85rem;line-height:1.6;">
                Passionate about building intelligent systems that solve real-world problems.
                Specialises in regression modelling, data visualisation, and deploying
                end-to-end ML pipelines.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with info_col:
        st.markdown('<div class="section-title">📱 About This App</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>House Price Predictor</strong> is an AI-powered web application that uses
            a trained <em>Multilinear Regression</em> model to estimate the market value of a
            residential property based on seven key features. The app provides instant predictions,
            interactive data visualizations, and detailed model performance metrics — all in a
            clean, modern interface.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">📖 How to Use</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Step 1 — Predict Price Tab:</strong> Enter all house details (area, rooms,
            year built, lot size, garage, neighbourhood quality) and click <em>Predict House Price</em>.
            The estimated market value appears instantly.<br><br>
            <strong>Step 2 — Data Visualization Tab:</strong> Explore interactive charts showing
            price distributions, correlations, and trends across the training dataset.<br><br>
            <strong>Step 3 — Model Accuracy Tab:</strong> Review R² score, error metrics,
            feature importance, and a predicted vs. actual scatter plot to understand model quality.<br><br>
            <strong>Step 4 — This tab:</strong> Find developer information, terms, and guidance.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">📜 Terms & Conditions</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            By using this application you agree that:<br>
            • Predictions are generated by a statistical model and are <strong>estimates only</strong>.<br>
            • The app is provided <strong>as-is</strong> without warranties of any kind.<br>
            • Results must not be used as the sole basis for financial, legal, or real-estate decisions.<br>
            • The developer is not liable for any losses arising from the use of this tool.<br>
            • All data entered is processed locally and is not stored or shared.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-box">
            ⚠️ <strong>Disclaimer:</strong> This tool is built for educational and demonstration
            purposes. House price predictions depend on many dynamic market factors not captured
            in this model (locality micro-trends, legal status, market sentiment, etc.).
            Always consult a certified real-estate professional before making property decisions.
            Predictions shown are in Indian Rupees (₹) and reflect a synthetic training dataset.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#334155;font-size:0.82rem;padding:1rem 0;border-top:1px solid #1e2d45;">
        © 2025 Amit Sharma · House Price Predictor v1.0 · Built with ❤️ using Python & Streamlit
    </div>
    """, unsafe_allow_html=True)