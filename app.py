
import streamlit as st
import pandas as pd
import joblib

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: #f5f5f3;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem 3rem 3rem !important;
    max-width: 1300px !important;
    margin: 0 auto !important;
}

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 24px;
    background: white;
    border: 1px solid #e8e8e6;
    border-radius: 14px;
    margin-bottom: 28px;
}
.topbar-icon {
    width: 34px; height: 34px;
    background: #E24B4A;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
}
.topbar-title {
    font-size: 15px;
    font-weight: 600;
    color: #1a1a1a;
    letter-spacing: -0.3px;
}
.topbar-sub {
    font-size: 12px;
    color: #888;
    margin-top: 1px;
}
.topbar-badge {
    margin-left: auto;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    padding: 3px 10px;
    border-radius: 20px;
    background: #f0efec;
    border: 1px solid #e0dfd9;
    color: #888;
    letter-spacing: 0.5px;
}

/* ── Section label ── */
.section-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.4px;
    color: #aaa;
    text-transform: uppercase;
    margin-bottom: 16px;
    margin-top: 4px;
}

/* ── Result card ── */
.result-card {
    background: white;
    border-radius: 16px;
    border: 1px solid #e8e8e6;
    padding: 28px;
    margin-bottom: 16px;
}

/* ── Score number ── */
.score-number {
    font-family: 'DM Mono', monospace;
    font-size: 64px;
    font-weight: 500;
    line-height: 1;
    margin-bottom: 4px;
}
.score-sub {
    font-size: 12px;
    color: #888;
}

/* ── Risk colors ── */
.risk-high { color: #E24B4A; }
.risk-mid  { color: #BA7517; }
.risk-low  { color: #3B6D11; }

/* ── Gauge ── */
.gauge-bg {
    height: 6px;
    border-radius: 3px;
    background: #f0efec;
    overflow: hidden;
    margin: 20px 0 4px;
}
.gauge-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}
.gauge-fill-high { background: #E24B4A; }
.gauge-fill-mid  { background: #EF9F27; }
.gauge-fill-low  { background: #639922; }

.gauge-ticks {
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    color: #bbb;
    margin-bottom: 20px;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.4px;
    margin-bottom: 24px;
}
.badge-high { background: #FCEBEB; color: #A32D2D; }
.badge-mid  { background: #FAEEDA; color: #854F0B; }
.badge-low  { background: #EAF3DE; color: #3B6D11; }
.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.dot-high { background: #E24B4A; }
.dot-mid  { background: #BA7517; }
.dot-low  { background: #3B6D11; }

/* ── Insights ── */
.insight {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 13px;
    border-radius: 9px;
    margin-bottom: 7px;
    font-size: 12.5px;
    line-height: 1.5;
    border: 1px solid;
}
.insight-warn { background: #FAEEDA; border-color: #FAC775; color: #854F0B; }
.insight-ok   { background: #EAF3DE; border-color: #C0DD97; color: #3B6D11; }

/* ── Meta rows ── */
.meta-row {
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #aaa;
    padding: 5px 0;
    border-bottom: 1px solid #f0efec;
}
.meta-val { color: #555; }

/* ── Divider ── */
.divider {
    height: 1px;
    background: #f0efec;
    margin: 20px 0;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #bbb;
    font-size: 13px;
}
.empty-icon {
    font-size: 36px;
    margin-bottom: 12px;
    display: block;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    font-size: 12px !important;
    color: #666 !important;
    font-family: 'Sora', sans-serif !important;
    margin-bottom: 4px !important;
}

div[data-testid="stButton"] button {
    width: 100%;
    background: #E24B4A !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px !important;
    letter-spacing: 0.3px !important;
    transition: opacity 0.15s !important;
}
div[data-testid="stButton"] button:hover {
    opacity: 0.85 !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TOP BAR
# ──────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-icon">📉</div>
  <div>
    <div class="topbar-title">Churn Predictor</div>
    <div class="topbar-sub">Modelo de retención de clientes</div>
  </div>
  <span class="topbar-badge">v2.1 · TOP-10 VARS</span>
</div>
""", unsafe_allow_html=True)



# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("modelo_churn.pkl")

model = load_model()

# ──────────────────────────────────────────────
# LAYOUT
# ──────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT: INPUTS ──────────────────────────────
with col_left:
    st.markdown("<div style='padding: 0 8px'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Datos del cliente</div>', unsafe_allow_html=True)

    contract = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])

    c1, c2 = st.columns(2)
    with c1:
        security = st.selectbox("Seguridad online", ["Yes", "No"])
    with c2:
        tech = st.selectbox("Soporte técnico", ["Yes", "No"])

    tenure = st.slider("Antigüedad (meses)", 0, 72, 12)
    monthly = st.slider("Cargo mensual ($)", 20, 200, 80)

    c3, c4 = st.columns(2)
    with c3:
        internet = st.selectbox("Internet", ["Fiber optic", "DSL", "No"])
    with c4:
        backup = st.selectbox("Online Backup", ["Yes", "No"])

    payment = st.selectbox("Método de pago", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    predict = st.button("Calcular riesgo de fuga")
    st.markdown("</div>", unsafe_allow_html=True)

# ── RIGHT: RESULT ─────────────────────────────
with col_right:
    st.markdown("<div style='padding: 0 8px'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Resultado</div>', unsafe_allow_html=True)

    if not predict:
        st.markdown("""
        <div class="result-card empty-state">
          <span class="empty-icon">📊</span>
          Completa los datos y calcula el riesgo
        </div>
        """, unsafe_allow_html=True)

    else:
        total_charges = tenure * monthly

        input_df = pd.DataFrame([{
            "Contract":         contract,
            "OnlineSecurity":   security,
            "tenure":           tenure,
            "PaymentMethod":    payment,
            "TechSupport":      tech,
            "TotalCharges":     total_charges,
            "InternetService":  internet,
            "MonthlyCharges":   monthly,
            "OnlineBackup":     backup,
            # defaults for remaining features
            "gender":           "Female",
            "SeniorCitizen":    0,
            "Partner":          "Yes",
            "Dependents":       "No",
            "PhoneService":     "Yes",
            "MultipleLines":    "No",
            "DeviceProtection": "No",
            "StreamingTV":      "Yes",
            "StreamingMovies":  "Yes",
            "PaperlessBilling": "Yes",
        }])

        prob     = model.predict_proba(input_df)[0][1]
        prob_pct = round(prob * 100, 1)

        if prob_pct >= 70:
            level, label = "high", "Alto riesgo"
        elif prob_pct >= 40:
            level, label = "mid",  "Riesgo medio"
        else:
            level, label = "low",  "Bajo riesgo"

        # ── Score + gauge ──
        st.markdown(f"""
        <div class="result-card">
          <div class="score-number risk-{level}">{prob_pct}%</div>
          <div class="score-sub">Cargo total estimado: ${total_charges:,.0f}</div>

          <div class="gauge-bg">
            <div class="gauge-fill gauge-fill-{level}" style="width:{prob_pct}%"></div>
          </div>
          <div class="gauge-ticks">
            <span>0%</span><span>40%</span><span>70%</span><span>100%</span>
          </div>

          <div>
            <span class="risk-badge badge-{level}">
              <span class="dot dot-{level}"></span>
              {label}
            </span>
          </div>

          <div class="divider"></div>

          <div class="meta-row"><span>contrato</span>   <span class="meta-val">{contract}</span></div>
          <div class="meta-row"><span>antigüedad</span> <span class="meta-val">{tenure} meses</span></div>
          <div class="meta-row"><span>cargo mensual</span><span class="meta-val">${monthly}</span></div>
          <div class="meta-row"><span>internet</span>   <span class="meta-val">{internet}</span></div>
          <div class="meta-row"><span>pago</span>       <span class="meta-val">{payment}</span></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Insights ──
        warnings, positives = [], []

        if contract == "Month-to-month":
            warnings.append("Contrato mensual — alta flexibilidad para cancelar en cualquier momento.")
        if security == "No":
            warnings.append("Sin seguridad online activa — factor de riesgo significativo.")
        if tech == "No":
            warnings.append("Sin soporte técnico — reduce la percepción de valor del servicio.")
        if tenure < 6:
            warnings.append("Antigüedad menor a 6 meses — clientes nuevos son más volátiles.")
        if monthly > 90:
            warnings.append(f"Cargo mensual de ${monthly} — por encima del promedio del segmento.")
        if payment == "Electronic check":
            warnings.append("Pago por cheque electrónico — correlacionado con mayor churn.")
        if internet == "Fiber optic":
            warnings.append("Fibra óptica — usuarios con alta expectativa de servicio.")

        if contract != "Month-to-month":
            positives.append("Contrato de largo plazo — ancla al cliente.")
        if security == "Yes":
            positives.append("Seguridad online activa — buena adopción de servicios adicionales.")
        if tech == "Yes":
            positives.append("Soporte técnico activo — mayor engagement con la plataforma.")
        if tenure > 24:
            positives.append(f"Antigüedad de {tenure} meses — cliente consolidado y leal.")

        shown = warnings[:3] + positives[:2]

        if shown:
            st.markdown('<div class="section-label" style="margin-top:4px">Factores detectados</div>',
                        unsafe_allow_html=True)
            for item in shown:
                is_warn = item in warnings
                icon    = "▲" if is_warn else "✓"
                cls     = "insight-warn" if is_warn else "insight-ok"
                st.markdown(
                    f'<div class="insight {cls}"><span>{icon}</span><span>{item}</span></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)
