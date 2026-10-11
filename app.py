#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAC vs NAI — Clinical Decision Support System (10-Variable ElasticNet Model)
Veterinary Endocrinology · Differentiation of Hyperadrenocorticism from Non-Adrenal Illness
Modern High-Clarity UI/UX · Web Streamlit Interface
"""
import os, json, io, base64, textwrap
import unicodedata
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

def render_html(html_str):
    """Safely renders HTML in Streamlit without triggering Markdown indented code blocks."""
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(BASE, "HAC.png")

# Modèle 10 vars prioritaire, fallback ancien F3 si absent
CANDIDATES = [("modele_M10_logregEN.joblib", "metadata_M10.json"),
              ("modele_F3_logregEN.joblib", "metadata.json")]
MODEL_PATH, META_PATH = None, None
for mp, me in CANDIDATES:
    if os.path.exists(os.path.join(BASE, mp)):
        MODEL_PATH, META_PATH = os.path.join(BASE, mp), os.path.join(BASE, me)
        break

@st.cache_resource
def load_model():
    pipe = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return pipe, meta

pipe, META = load_model()
SEUIL_T8 = float(META.get("seuil_T8_histo", 27.6))
SEUIL_DEFAUT = float(META.get("seuil_proba_defaut", 0.63))
PERF = META.get("performances_nestedCV", {})
PERF_HISTO = META.get("performances_histo_T8", {})

# ---------- i18n Dictionary ----------
TR = {
    "page_title": {"EN": "HAC vs NAI — Clinical Decision Support", "FR": "HAC vs NAI — Aide à la Décision Clinique"},
    "app_title": {"EN": "HAC vs NAI", "FR": "HAC vs NAI"},
    "app_tagline": {"EN": "Differentiating Hyperadrenocorticism from Non-Adrenal Illness without ACTH stimulation",
                    "FR": "Différenciation de l'hyperadrénocorticisme et des maladies non surrénaliennes sans stimulation à l'ACTH"},
    "model_badge": {"EN": "M10 ElasticNet Logistic Regression · Nested-CV AUC {auc}±{sd} · n=123",
                    "FR": "Modèle M10 ElasticNet · Nested-CV AUC {auc}±{sd} · n=123 cas"},
    "lang_btn": {"EN": "🇫🇷 Français", "FR": "🇬🇧 English"},
    "tab1": {"EN": "🔬 Single Patient Analysis", "FR": "🔬 Analyse Patient Unique"},
    "tab2": {"EN": "📁 Cohort / Batch File (CSV & Excel)", "FR": "📁 Fichier & Cohorte (CSV / Excel)"},
    
    # Unit System
    "unit_sys_title": {"EN": "1. Select Laboratory Unit System", "FR": "1. Sélectionnez le système d'unités du laboratoire"},
    "u_uk": {"EN": "🇬🇧 UK / SI Units (nmol/L, µmol/L, mmol/L, ×10⁹/L)", "FR": "🇬🇧 Unités UK / SI (nmol/L, µmol/L, mmol/L, ×10⁹/L)"},
    "u_us": {"EN": "🇺🇸 US Units (µg/dL, mg/dL, mg/dL, ×10³/µL)", "FR": "🇺🇸 Unités US (µg/dL, mg/dL, mg/dL, ×10³/µL)"},
    "u_adv": {"EN": "⚙️ Advanced unit customization (per variable)", "FR": "⚙️ Personnalisation avancée par variable"},
    "u_cort_l": {"EN": "Cortisol (T8, T3)", "FR": "Cortisol (T8, T3)"},
    "u_crea_l": {"EN": "Creatinine", "FR": "Créatinine"},
    "u_ca_l": {"EN": "Calcium", "FR": "Calcium"},
    "u_counts_l": {"EN": "Cell Counts (CBC)", "FR": "Numérations (NFS)"},
    "u_applied_notice": {"EN": "Values are automatically converted to standard training units before inference.",
                         "FR": "Les valeurs sont automatiquement converties vers les unités standard d'entraînement avant prédiction."},

    # Input Cards
    "sec_ldst": {"EN": "LDDST Dynamic Protocol (Cortisol)", "FR": "Protocole LDDST (Freinage faible à la dexaméthasone)"},
    "sec_ldst_sub": {"EN": "Gold standard dynamic screening test. 8h non-suppression is required for historical diagnosis.",
                     "FR": "Test dynamique de référence. La non-suppression à 8h définit la règle classique."},
    "sec_cbc": {"EN": "Complete Blood Count (CBC / NFS)", "FR": "Hémogramme (NFS / Leucogramme)"},
    "sec_cbc_sub": {"EN": "Stress leukogram markers typical of hypercortisolemia.",
                    "FR": "Marqueurs de leucogramme de stress caractéristiques de l'excès de cortisol."},
    "sec_bio": {"EN": "Biochemistry, Urinalysis & Age", "FR": "Biochimie, Urologie & Âge"},
    "sec_bio_sub": {"EN": "Metabolic hallmarks, urinary concentration & patient maturity.",
                    "FR": "Empreinte métabolique, concentration urinaire et âge du chien."},
    
    # Fields
    "f_T8": {"EN": "Cortisol T8 (8 hours)", "FR": "Cortisol T8 (8 heures)"},
    "f_T3": {"EN": "Cortisol T3 (3–4 hours)", "FR": "Cortisol T3 (3–4 heures)"},
    "c_cens": {"EN": "Below detection limit (< threshold)", "FR": "Inférieur au seuil de détection (< seuil)"},
    "f_lymph": {"EN": "Lymphocytes", "FR": "Lymphocytes"},
    "f_thr": {"EN": "Platelets (Thrombocytes)", "FR": "Plaquettes (Thrombocytes)"},
    "f_eos": {"EN": "Eosinophils", "FR": "Éosinophiles"},
    "f_usg": {"EN": "Urine Specific Gravity (USG)", "FR": "Densité urinaire (USG)"},
    "c_usg": {"EN": "Highly concentrated (> 1.050)", "FR": "Très concentrée (> 1.050)"},
    "f_crea": {"EN": "Creatinine", "FR": "Créatinine"},
    "f_ca": {"EN": "Total Calcium", "FR": "Calcium total"},
    "f_alkp": {"EN": "ALKP Ratio (× ULN)", "FR": "Ratio PAL / ALKP (× LSN)"},
    "f_alkp_help": {"EN": "Measured ALKP divided by the lab's Upper Limit of Normal (unitless).",
                    "FR": "PAL mesurée divisée par la Limite Supérieure de la Normale du labo (sans unité)."},
    "f_age": {"EN": "Age", "FR": "Âge"},
    "years": {"EN": "years", "FR": "ans"},

    # Threshold & Controls
    "threshold_card_title": {"EN": "2. Decision Threshold & Analysis", "FR": "2. Seuil décisionnel & Déclenchement"},
    "threshold_slider_label": {"EN": "HAC Probability Decision Threshold", "FR": "Seuil décisionnel de probabilité HAC"},
    "threshold_help": {"EN": "Default 0.63 optimizes Youden index (Sensitivity 86.4%, Specificity 95.3% on out-of-fold validation). Lower to 0.50 for screening rule-out, or raise to 0.75 for strict rule-in.",
                      "FR": "Défaut 0.63 optimise l'indice de Youden (Sensibilité 86.4%, Spécificité 95.3%). Abaisser vers 0.50 pour éliminer la maladie (rule-out), ou élever vers 0.75 pour confirmer strictement (rule-in)."},
    "btn_predict": {"EN": "Analyze Patient Profile", "FR": "Analyser le Profil Patient"},

    # Diagnostic Results
    "results_title": {"EN": "Diagnostic Interpretation & AI Analysis", "FR": "Interprétation Diagnostique & Analyse IA"},
    "verdict_hac": {"EN": "HYPERADRENOCORTICISM (HAC) SUSPECTED", "FR": "HYPERADRÉNOCORTICISME (HAC) SUSPECTÉ"},
    "verdict_nai": {"EN": "NON-ADRENAL ILLNESS (NAI) PROBABLE", "FR": "MALADIE NON SURRÉNALIENNE (NAI) PROBABLE"},
    "confidence_high_hac": {"EN": "Very high probability — Biological and hormonal profile strongly compatible with Cushing's syndrome.",
                           "FR": "Très forte probabilité — Profil biologique et hormonal hautement évocateur d'un Cushing."},
    "confidence_mod_hac": {"EN": "Positive above optimal threshold — Clinical correlation and adrenal imaging recommended.",
                          "FR": "Positif au-dessus du seuil — Corrélation clinique et imagerie surrénalienne recommandées."},
    "confidence_equivocal": {"EN": "Borderline / Equivocal zone — Repeat testing or close monitoring advised.",
                            "FR": "Zone grise / limite — Surveillance clinique ou confirmation à distance conseillées."},
    "confidence_mod_nai": {"EN": "Negative below threshold — Preserved suppression and/or quiet metabolic profile.",
                          "FR": "Négatif sous le seuil — Suppression préservée et/ou profil métabolique non concordant."},
    "confidence_high_nai": {"EN": "Very low probability — Strongly in favor of non-adrenal concurrent illness.",
                           "FR": "Très faible probabilité — Fortement en faveur d'une affection non surrénalienne."},
    "proba_metric_label": {"EN": "Calculated Model Probability P(HAC)", "FR": "Probabilité calculée par le modèle P(HAC)"},
    "risk_meter_label": {"EN": "Probability Risk Scale vs Decision Threshold", "FR": "Échelle de risque vs Seuil décisionnel"},
    "missing_warning": {"EN": "⚠️ Missing value(s) detected: Imputed with training cohort median. Interpretation should be cautious.",
                        "FR": "⚠️ Valeur(s) manquante(s) détectée(s) : Imputation par la médiane d'entraînement. Interpréter avec réserve."},

    # Comparison Table
    "comp_table_title": {"EN": "Model vs Historical LDDST Rule Comparison", "FR": "Comparaison Modèle IA vs Règle Historique LDDST"},
    "th_criterion": {"EN": "Diagnostic Criterion", "FR": "Critère Diagnostique"},
    "th_model": {"EN": "10-Variable AI Model", "FR": "Modèle IA 10 Variables"},
    "th_histo": {"EN": "Historical LDDST Alone (T8 > {t8})", "FR": "LDDST Seul Historique (T8 > {t8})"},
    "row_verdict": {"EN": "Result / Verdict", "FR": "Résultat / Verdict"},
    "row_spec": {"EN": "Diagnostic Specificity", "FR": "Spécificité Diagnostique"},
    "row_sens": {"EN": "Diagnostic Sensitivity", "FR": "Sensibilité Diagnostique"},
    "row_rationale": {"EN": "Clinical Rationale", "FR": "Apport Clinique"},
    "rat_model": {"EN": "Synthesizes endocrine suppression with systemic stress hematology and liver/renal markers, preventing false positives from concurrent disease.",
                  "FR": "Synthétise la suppression hormonale avec l'hématologie de stress et les marqueurs biochimiques, éliminant les faux positifs dus aux maladies intercurrentes."},
    "rat_histo": {"EN": "Relies solely on cortisol 8h. High sensitivity (93%), but 27% false positive rate when dogs are stressed or sick from non-adrenal diseases.",
                  "FR": "Repose uniquement sur le cortisol 8h. Forte sensibilité (93%), mais 27% de faux positifs en présence de maladies non surrénaliennes ou de stress."},

    # Biological Profiling
    "factors_title": {"EN": "Key Biomarker Contributions", "FR": "Contribution des Principaux Biomarqueurs"},
    "fact_lddst_hac": {"EN": "Failed Cortisol Suppression (T8 high)", "FR": "Défaut de freinage du cortisol (T8 élevé)"},
    "fact_lddst_nai": {"EN": "Preserved Cortisol Suppression", "FR": "Suppression adéquate du cortisol"},
    "fact_lymph_hac": {"EN": "Lymphopenia (Stress Leucogram)", "FR": "Lymphopénie (Leucogramme de stress)"},
    "fact_lymph_nai": {"EN": "Normal Lymphocyte Count", "FR": "Lymphocytes dans les normes"},
    "fact_thr_hac": {"EN": "Thrombocytosis (High Platelets)", "FR": "Thrombocytose (Plaquettes élevées)"},
    "fact_thr_nai": {"EN": "Normal Platelet Count", "FR": "Plaquettes dans les normes"},
    "fact_usg_hac": {"EN": "Poorly Concentrated Urine (Low USG)", "FR": "Défaut de concentration urinaire (USG basse)"},
    "fact_usg_nai": {"EN": "Preserved Urine Concentration", "FR": "Concentration urinaire préservée"},
    "fact_alkp_hac": {"EN": "Marked ALKP Induction", "FR": "Induction marquée des PAL / ALKP"},
    "fact_alkp_nai": {"EN": "Mild / Normal ALKP", "FR": "PAL modérées ou normales"},

    # Tab 2: Batch File
    "tab2_title": {"EN": "Batch Processing & Model Validation", "FR": "Traitement par Lot & Validation du Modèle"},
    "tab2_intro": {"EN": "Process a cohort file (CSV, XLS, XLSX) containing the 10 variables. If the 'Diagnosis' column is provided (HAC/1 vs NAI/0), the app calculates full ROC metrics, bootstrap 95% confidence intervals, and recalibration parameters.",
                   "FR": "Traitez un fichier de cohorte (CSV, XLS, XLSX) contenant les 10 variables. Si la colonne 'Diagnosis' est présente (HAC/1 vs NAI/0), l'application calcule automatiquement les courbes ROC, les intervalles de confiance bootstrap à 95% et la recalibration."},
    "tmpl_download_title": {"EN": "Ready-to-use Templates", "FR": "Templates prêts à l'emploi"},
    "tmpl_unit_label": {"EN": "Template units", "FR": "Unités du template"},
    "tmpl_edit_hint": {"EN": "Edit the example rows below (add your cases), then download.",
                       "FR": "Modifiez les lignes d'exemple ci-dessous (ajoutez vos cas), puis téléchargez."},
    "btn_tmpl_download": {"EN": "⬇️ Download template CSV", "FR": "⬇️ Télécharger le template CSV"},
    "file_drop_label": {"EN": "Drag and drop your CSV or Excel file here", "FR": "Glissez-déposez votre fichier CSV ou Excel ici"},
    "btn_download_results": {"EN": "⬇️ Download Predictions (CSV)", "FR": "⬇️ Télécharger les Prédictions (CSV)"},
    "btn_download_recal": {"EN": "⬇️ Download Recalibration JSON", "FR": "⬇️ Télécharger les Paramètres de Recalibration (JSON)"},
    
    # Footer
    "disclaimer": {"EN": "Clinical Decision Support System. Developed for veterinary practitioners. For research and clinical assistance — always correlate with complete physical examination, history, and imaging.",
                   "FR": "Système d'aide à la décision clinique destiné aux vétérinaires. Outil d'assistance diagnostique — à corréler systématiquement avec l'examen clinique, l'anamnèse et l'imagerie."},
}

if "lang" not in st.session_state:
    st.session_state.lang = "EN"

def T(k):
    v = TR.get(k, {}).get(st.session_state.lang, k)
    return v if isinstance(v, str) else str(k)

# ---------- Page Config & Modern Custom CSS ----------
st.set_page_config(
    page_title=T("page_title"),
    page_icon=ICON_PATH if os.path.exists(ICON_PATH) else "🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom High-Clarity CSS
MODERN_CSS = """
<style>
/* Modern Typography & Clean Palette */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* App Background & Container Spacing */
.stApp {
    background: #F8FAFC;
    color: #0F172A;
}

/* Header Container */
.app-header {
    background: linear-gradient(135deg, #063C44 0%, #0B525D 50%, #0E7490 100%);
    border-radius: 20px;
    padding: 24px 32px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 10px 25px -5px rgba(14, 116, 144, 0.25);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 24px;
}

.header-logo {
    width: 76px;
    height: 76px;
    border-radius: 18px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.25);
    border: 2px solid rgba(255, 255, 255, 0.2);
}

.header-title-text {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.2;
}

.header-tagline {
    font-size: 14px;
    color: #A5F3FC;
    margin: 4px 0 0 0;
    font-weight: 500;
}

.header-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    padding: 4px 12px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 8px;
    color: #E0F2FE;
}

/* Modern Card Components */
.clean-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04);
    transition: all 0.2s ease;
}

.clean-card:hover {
    border-color: #CBD5E1;
    box-shadow: 0 6px 20px -2px rgba(15, 23, 42, 0.07);
}

.card-title {
    font-size: 16px;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.card-subtitle {
    font-size: 12px;
    color: #64748B;
    margin-bottom: 16px;
    line-height: 1.4;
}

/* Diagnostic Verdict Banners */
.verdict-banner-hac {
    background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
    border: 2px solid #F43F5E;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px -4px rgba(244, 63, 94, 0.15);
}

.verdict-banner-nai {
    background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
    border: 2px solid #10B981;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px -4px rgba(16, 185, 129, 0.15);
}

.verdict-title {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.3px;
    margin-bottom: 6px;
}

.verdict-title-hac { color: #BE123C; }
.verdict-title-nai { color: #047857; }

.verdict-desc {
    font-size: 14px;
    font-weight: 500;
    color: #334155;
    margin-bottom: 16px;
}

/* Probability Gauge Component */
.gauge-container {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 16px 20px;
    border: 1px solid #E2E8F0;
    margin-top: 14px;
}

.gauge-track {
    position: relative;
    height: 16px;
    border-radius: 8px;
    background: linear-gradient(90deg, #10B981 0%, #34D399 40%, #FBBF24 63%, #F43F5E 80%, #BE123C 100%);
    margin: 18px 0 10px 0;
}

.gauge-pointer {
    position: absolute;
    top: -6px;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #0F172A;
    border: 3px solid #FFFFFF;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    transform: translateX(-50%);
    transition: left 0.4s ease;
}

.gauge-threshold-line {
    position: absolute;
    top: -8px;
    bottom: -8px;
    width: 3px;
    background: #000000;
    border-radius: 2px;
    transform: translateX(-50%);
    z-index: 2;
}

.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    font-weight: 600;
    color: #64748B;
}

/* Biological Factor Pills */
.factor-pill {
    background: #F1F5F9;
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}

.factor-pill.hac {
    background: #FFF1F2;
    border-color: #FECDD3;
    color: #9F1239;
}

.factor-pill.nai {
    background: #F0FDF4;
    border-color: #BBF7D0;
    color: #166534;
}

.factor-pill-icon {
    font-size: 18px;
}

.factor-pill-text {
    font-size: 13px;
    font-weight: 600;
}

/* Comparison Table Styling */
.comp-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    font-size: 13px;
}

.comp-table th {
    background: #F8FAFC;
    color: #475569;
    font-weight: 700;
    padding: 12px;
    border-bottom: 2px solid #E2E8F0;
    text-align: left;
}

.comp-table td {
    padding: 12px;
    border-bottom: 1px solid #E2E8F0;
    color: #1E293B;
}

.comp-table tr:hover {
    background: #F8FAFC;
}

/* Clean Custom Form Inputs */
div[data-baseweb="input"] {
    border-radius: 10px !important;
}

div.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}

/* Primary Action Button Glow */
button[kind="primary"] {
    background: linear-gradient(135deg, #0E7490 0%, #0891B2 100%) !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(14, 116, 144, 0.35) !important;
}

button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(14, 116, 144, 0.45) !important;
    transform: translateY(-1px);
}
</style>
"""
st.markdown(MODERN_CSS, unsafe_allow_html=True)

# Encode icon to base64 for header
logo_b64 = ""
if os.path.exists(ICON_PATH):
    with open(ICON_PATH, "rb") as img_f:
        logo_b64 = base64.b64encode(img_f.read()).decode("utf-8")

# ---------- Header Component ----------
_logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="header-logo" alt="HAC Icon"/>' if logo_b64 else '🩺'

st.markdown(f"""
<div class="app-header">
    <div class="header-left">
        {_logo_html}
        <div>
            <h1 class="header-title-text">{T("app_title")}</h1>
            <p class="header-tagline">{T("app_tagline")}</p>
            <div class="header-badge">{T("model_badge").format(auc=PERF.get('AUC_mean', 0.964), sd=PERF.get('AUC_sd', 0.032))}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top Bar Language Switcher
_c_spacer, _c_lang = st.columns([10, 2])
with _c_lang:
    if st.button(T("lang_btn"), use_container_width=True):
        st.session_state.lang = "FR" if st.session_state.lang == "EN" else "EN"
        st.rerun()

# ---------- Feature Engineering & Modeling Functions ----------
def build_features_row(raw):
    """Maps 10 raw entries into 12 engineered features expected by the model."""
    def num(x):
        try:
            if x is None or (isinstance(x, str) and x.strip() == ""):
                return np.nan
            return float(str(x).replace(",", "."))
        except:
            return np.nan

    if raw.get("T3_cens"):
        _t3x = num(raw.get("T3"))
        t3v, t3s = (_t3x, 1) if np.isfinite(_t3x) else (13.8, 1)
    else:
        t3 = num(raw.get("T3"))
        t3v, t3s = (t3, 0) if np.isfinite(t3) else (np.nan, np.nan)

    if raw.get("T8_cens"):
        _t8x = num(raw.get("T8"))
        t8v, t8s = (_t8x, 1) if np.isfinite(_t8x) else (13.8, 1)
    else:
        t8 = num(raw.get("T8"))
        t8v, t8s = (t8, 0) if np.isfinite(t8) else (np.nan, np.nan)

    logT3 = np.log1p(t3v) if np.isfinite(t3v) else np.nan
    logT8 = np.log1p(t8v) if np.isfinite(t8v) else np.nan

    if raw.get("USG_sup"):
        usgv = 1.051
    else:
        u = num(raw.get("USG"))
        usgv = u if np.isfinite(u) else np.nan

    def log1p_c(x, off=0.0):
        v = num(x)
        return float(np.log1p(max(v, 0) + off)) if np.isfinite(v) else np.nan

    row = {
        "logT8": logT8, "T8_suppr": t8s, "logT3": logT3, "T3_suppr": t3s,
        "log_Lymphocytes": log1p_c(raw.get("Lymphocytes")),
        "log_Thrombocytes": log1p_c(raw.get("Thrombocytes")),
        "USG_val": usgv,
        "log_Creatinine": log1p_c(raw.get("Creatinine")),
        "Totalcalcium": num(raw.get("Totalcalcium")),
        "log_Eosinophils": log1p_c(raw.get("Eosinophils"), 0.01),
        "log_RatioALKP": log1p_c(raw.get("RatioALKP")),
        "Age": num(raw.get("Age")),
    }
    cols = META["features"] if "features" in META else list(row.keys())
    return pd.DataFrame([row], columns=cols), {"T8_val": t8v, "T3_val": t3v}

def predict_proba(feat):
    return float(pipe.predict_proba(feat)[0, 1])

def bootstrap_ci(y_true, y_score, metric_fn, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            vals.append(metric_fn(y_true[idx], y_score[idx]))
        except:
            continue
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def se_sp(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp/(tp+fn) if (tp+fn) else np.nan, tn/(tn+fp) if (tn+fp) else np.nan

def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def seuil_youden_cv(yt, pm, seed=42):
    n = len(yt)
    k = max(2, min(5, int(np.min(np.bincount(yt)))))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.zeros(n, dtype=int); thrs = []
    for tr, te in skf.split(np.zeros(n), yt):
        fpr, tpr, thr = roc_curve(yt[tr], pm[tr])
        thrs.append(float(thr[int(np.argmax(tpr - fpr))]))
        oof[te] = (pm[te] >= thrs[-1]).astype(int)
    se, sp = se_sp(yt, oof)
    return float(np.median(thrs)), float(se), float(sp)

def platt_cv(yt, pm, seed=42):
    z = _logit(pm).reshape(-1, 1)
    k = max(2, min(5, int(np.min(np.bincount(yt)))))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = cross_val_predict(LogisticRegression(), z, yt, cv=skf, method="predict_proba")[:, 1]
    fin = LogisticRegression().fit(z, yt)
    return oof, float(fin.intercept_[0]), float(fin.coef_[0][0])

# ---------- Unit Conversion Standards ----------
UNIT_PRESETS = {
    "UK": {"cort": "nmol/L", "crea": "µmol/L", "ca": "mmol/L", "counts": "×10⁹/L"},
    "US": {"cort": "µg/dL", "crea": "mg/dL", "ca": "mg/dL", "counts": "×10³/µL"},
}
UNIT_FACTORS = {
    "cort": {"nmol/L": 1.0, "µg/dL": 27.59, "ug/dL": 27.59, "µg/L": 2.759, "ug/L": 2.759},
    "crea": {"µmol/L": 1.0, "umol/L": 1.0, "mg/dL": 88.4, "mg/L": 8.84},
    "ca": {"mmol/L": 1.0, "mg/dL": 0.2495, "mg/L": 0.02495},
    "counts": {"×10⁹/L": 1.0, "x10^9/L": 1.0, "×10³/µL": 1.0, "x10^3/uL": 1.0, "/µL": 0.001, "/uL": 0.001},
}
UNIT_CHOICES = {
    "cort": ["nmol/L", "µg/dL", "µg/L"],
    "crea": ["µmol/L", "mg/dL", "mg/L"],
    "ca": ["mmol/L", "mg/dL", "mg/L"],
    "counts": ["×10⁹/L", "×10³/µL", "/µL"],
}
CORT_CENS = {"nmol/L": 27.6, "µg/dL": 1.0, "ug/dL": 1.0, "µg/L": 10.0, "ug/L": 10.0}

# Défauts UK garantis dès le 1er run (jamais de preset vide -> jamais d'erreur KeyError),
# modifiables manuellement ensuite. Le preset UK/US est la source de vérité : quand il
# change, il répercute sur les 4 menus par-variable (qui gardent leur état sinon, car à `key`).
for _k, _v in [("upres1", "UK"), ("upres2", "UK"),
               ("ucort1", "nmol/L"), ("ucrea1", "µmol/L"),
               ("uca1", "mmol/L"), ("ucounts1", "×10⁹/L")]:
    st.session_state.setdefault(_k, _v)

def _sync_preset1():
    p = UNIT_PRESETS.get(st.session_state.get("upres1"), UNIT_PRESETS["UK"])
    for _k, _slot in [("ucort1", "cort"), ("ucrea1", "crea"),
                      ("uca1", "ca"), ("ucounts1", "counts")]:
        st.session_state[_k] = p[_slot]
CORT_SEUIL_DISP = {"nmol/L": 27.6, "µg/dL": 1.0, "ug/dL": 1.0, "µg/L": 10.0, "ug/L": 10.0}

def _norm_header(s):
    """Normalize a CSV header: lowercase, accents stripped, alphanumerics only.
    Lets 'Créatinine [µmol/L]' or 'T8 cortisol [nmol/L]' match canonical names."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())

def find_col(columns, *aliases):
    """Find a column by canonical aliases (exact match first, then substring).
    Accepts old bare headers ('T8', 'Totalcalcium') and new readable ones
    ('T8 cortisol [nmol/L]', 'Calcium total [mmol/L]')."""
    norm = [(_norm_header(c), c) for c in columns]
    for a in aliases:
        q = _norm_header(a)
        if not q:
            continue
        for n, orig in norm:
            if n == q:
                return orig
    for a in aliases:
        q = _norm_header(a)
        if not q:
            continue
        for n, orig in norm:
            if q in n:
                return orig
    return None

def _uk_num(x, kind, unit):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return np.nan
        return float(str(x).replace(",", ".")) * UNIT_FACTORS[kind].get(unit, 1.0)
    except:
        return np.nan

def _uk_censored_val(x, kind, unit):
    try:
        if x is None or "<" not in str(x):
            return np.nan, False
        xv = float(str(x).replace("<", "").replace(",", ".").strip())
        return (xv / 2.0) * UNIT_FACTORS[kind].get(unit, 1.0), True
    except:
        return np.nan, False

def warn_units(vals_uk, units, EN):
    w = []
    def arr(k):
        try: return np.asarray(vals_uk.get(k, []), dtype=float)
        except: return np.array([])
    t8 = arr("T8"); t8 = t8[np.isfinite(t8)]
    cr = arr("Crea"); cr = cr[np.isfinite(cr)]
    ca = arr("Ca"); ca = ca[np.isfinite(ca)]
    ly = arr("Lymph"); ly = ly[np.isfinite(ly)]
    th = arr("Thr"); th = th[np.isfinite(th)]
    eo = arr("Eos"); eo = eo[np.isfinite(eo)]
    uc = units.get("cort", "nmol/L")
    if uc == "nmol/L" and len(t8) and np.nanmax(t8) < 25:
        w.append("T8 all < 25 nmol/L — looks like µg/dL. Switch unit system to US?" if EN else "T8 tous < 25 nmol/L — on dirait des µg/dL. Passer en unités US ?")
    if uc in ("µg/dL", "ug/dL") and len(t8) and np.nanmax(t8) > 60:
        w.append("T8 > 60 µg/dL — looks like nmol/L. Switch unit system to UK?" if EN else "T8 > 60 µg/dL — on dirait des nmol/L. Passer en unités UK / SI ?")
    uk = units.get("crea", "µmol/L")
    if uk == "µmol/L" and len(cr) and np.nanmax(cr) < 10:
        w.append("Creatinine all < 10 µmol/L — mg/dL? Check unit system." if EN else "Créatinine < 10 µmol/L — valeur en mg/dL ? Vérifiez le système d'unités.")
    if uk == "mg/dL" and len(cr) and np.nanmax(cr) > 15:
        w.append("Creatinine > 15 mg/dL — looks like µmol/L." if EN else "Créatinine > 15 mg/dL — on dirait des µmol/L.")
    ua = units.get("ca", "mmol/L")
    if ua == "mmol/L" and len(ca) and np.nanmax(ca) > 5:
        w.append("Calcium > 5 mmol/L — mg/dL? Check unit system." if EN else "Calcium > 5 mmol/L — valeur en mg/dL ? Vérifiez le système d'unités.")
    for arrv, nm in [(ly, "Lymphocytes"), (th, "Thrombocytes"), (eo, "Eosinophils")]:
        if len(arrv) and np.nanmax(arrv) > (20000 if nm == "Thrombocytes" else 50):
            w.append(f"{nm} very high — raw /µL counts? Check unit system." if EN else f"{nm} très élevés — numération en /µL ? Vérifiez les unités.")
            break
    return w

def build_template_df(unit, lang):
    """Template headers: Cas/Case first, readable labels, unit in brackets.
    Order: Cas, T3, T8 (form order), CBC, USG, bio, ALKP, Age, Diagnosis."""
    us = (unit == "US")
    FR = (lang == "FR")
    cort = "µg/dL" if us else "nmol/L"
    cnt = "×10³/µL" if us else "×10⁹/L"
    cre = "mg/dL" if us else "µmol/L"
    ca = "mg/dL" if us else "mmol/L"
    pfx = "Cas" if FR else "Case"
    cols = [(pfx),
            f"T3 cortisol [{cort}]", f"T8 cortisol [{cort}]",
            f"Lymphocytes [{cnt}]", f"Thrombocytes [{cnt}]",
            f"{'Éosinophiles' if FR else 'Eosinophils'} [{cnt}]",
            "USG",
            f"{'Créatinine' if FR else 'Creatinine'} [{cre}]",
            f"{'Calcium total' if FR else 'Total calcium'} [{ca}]",
            f"Ratio ALKP [{'× LSN' if FR else '× ULN'}]",
            ("Âge [années]" if FR else "Age [years]"),
            ("Diagnostic" if FR else "Diagnosis")]
    if us:
        vals = [
            [1, 2.67, 6.45, 2.3, 426.0, 0.03, "1.025", 0.78, 12.18, 146.4, 6.0, "HAC"],
            [2, "< 1.0", "< 1.0", 2.5, 300.0, 0.3, "1.030", 1.02, 10.42, 2.0, 10.0, "NAI"],
            [3, 1.63, 1.27, 1.0, 500.0, 0.05, "> 1.050", 0.79, 11.22, 8.0, 11.0, "HAC"],
        ]
    else:
        vals = [
            [1, 73.7, 178.0, 2.3, 426.0, 0.03, "1.025", 69.0, 3.04, 146.4, 6.0, "HAC"],
            [2, "< 27.6", "< 27.6", 2.5, 300.0, 0.3, "1.030", 90.0, 2.6, 2.0, 10.0, "NAI"],
            [3, 45.0, 35.0, 1.0, 500.0, 0.05, "> 1.050", 70.0, 2.80, 8.0, 11.0, "HAC"],
        ]
    return pd.DataFrame(vals, columns=cols)

# ---------- Tabs Navigation ----------
tab1, tab2 = st.tabs([T("tab1"), T("tab2")])

# ==============================================================================
# TAB 1: SINGLE PATIENT ANALYSIS
# ==============================================================================
with tab1:
    # 1. Unit Selection Box
    with st.container(border=True):
        st.markdown(f"**{T('unit_sys_title')}**")
        upres1 = st.radio(
            T("unit_sys_title"),
            ["UK", "US"],
            index=0,
            horizontal=True,
            format_func=lambda x: T("u_uk") if x == "UK" else T("u_us"),
            label_visibility="collapsed",
            key="upres1",
            on_change=_sync_preset1
        )

        with st.expander(T("u_adv")):
            _a1, _a2, _a3, _a4 = st.columns(4)
            _p1 = UNIT_PRESETS.get(upres1, UNIT_PRESETS["UK"])
            with _a1:
                u_cort1 = st.selectbox(T("u_cort_l"), UNIT_CHOICES["cort"],
                                       index=UNIT_CHOICES["cort"].index(_p1["cort"]), key="ucort1")
            with _a2:
                u_crea1 = st.selectbox(T("u_crea_l"), UNIT_CHOICES["crea"],
                                       index=UNIT_CHOICES["crea"].index(_p1["crea"]), key="ucrea1")
            with _a3:
                u_ca1 = st.selectbox(T("u_ca_l"), UNIT_CHOICES["ca"],
                                     index=UNIT_CHOICES["ca"].index(_p1["ca"]), key="uca1")
            with _a4:
                u_counts1 = st.selectbox(T("u_counts_l"), UNIT_CHOICES["counts"],
                                         index=UNIT_CHOICES["counts"].index(_p1["counts"]), key="ucounts1")
        units1 = {"cort": u_cort1, "crea": u_crea1, "ca": u_ca1, "counts": u_counts1}
        st.caption(f"ℹ️ {T('u_applied_notice')}")

    # 2. Input Fields grouped into 3 aesthetic clinical cards
    col_lddst, col_cbc, col_bio = st.columns([1, 1, 1.1])
    thr1 = CORT_CENS.get(u_cort1, 27.6)

    # Card A: LDDST
    with col_lddst:
        with st.container(border=True):
            st.markdown(f"**🧪 {T('sec_ldst')}**")
            st.caption(T("sec_ldst_sub"))
            t8_default = "60" if upres1 != "US" else "2.17"
            t3_default = "80" if upres1 != "US" else "2.90"
            
            T3 = st.text_input(f"{T('f_T3')} [{u_cort1}]", value=t3_default, help=f"Intermediate 3-4h test point")
            T3c = st.checkbox(f"T3 {T('c_cens')} (< {thr1:g})", value=False)
            st.divider()
            T8 = st.text_input(f"{T('f_T8')} [{u_cort1}]", value=t8_default, help=f"Reference threshold: {thr1:g} {u_cort1}")
            T8c = st.checkbox(f"T8 {T('c_cens')} (< {thr1:g})", value=False)

    # Card B: CBC / Hematology
    with col_cbc:
        with st.container(border=True):
            st.markdown(f"**🩸 {T('sec_cbc')}**")
            st.caption(T("sec_cbc_sub"))
            Lymph = st.text_input(f"{T('f_lymph')} [{u_counts1}]", value="1.2")
            Thr = st.text_input(f"{T('f_thr')} [{u_counts1}]", value="350")
            Eos = st.text_input(f"{T('f_eos')} [{u_counts1}]", value="0.10")

    # Card C: Biochemistry & Urinalysis & Age
    with col_bio:
        with st.container(border=True):
            st.markdown(f"**📊 {T('sec_bio')}**")
            st.caption(T("sec_bio_sub"))
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                USG = st.text_input(T("f_usg"), value="1.012")
                USGsup = st.checkbox(T("c_usg"), value=False)
                crea_def = "80" if upres1 != "US" else "0.90"
                Crea = st.text_input(f"{T('f_crea')} [{u_crea1}]", value=crea_def)
            with c_b2:
                ca_def = "2.70" if upres1 != "US" else "10.8"
                Ca = st.text_input(f"{T('f_ca')} [{u_ca1}]", value=ca_def)
                rALKP = st.text_input(T("f_alkp"), value="5.0", help=T("f_alkp_help"))
                Age = st.text_input(f"{T('f_age')} ({T('years')})", value="9.5")

    # 3. Decision Threshold & Predict Button Bar
    with st.container(border=True):
        st.markdown(f"**⚙️ {T('threshold_card_title')}**")
        c_th1, c_th2 = st.columns([3, 1.2])
        with c_th1:
            seuil_proba = st.slider(
                T("threshold_slider_label"),
                min_value=0.10,
                max_value=0.90,
                value=SEUIL_DEFAUT,
                step=0.01,
                help=T("threshold_help")
            )
        with c_th2:
            st.write("")
            st.write("")
            do_predict = st.button(f"🔍 {T('btn_predict')}", type="primary", use_container_width=True)

    # 4. Prediction Execution & Diagnostic Rendering
    if do_predict:
        def _resolve_cort_st(txt, box, unit):
            val = _uk_num(txt, "cort", unit)
            if "<" in str(txt or "") or box:
                return 13.8, True, np.nan, (bool(box) and np.isfinite(val))
            if np.isfinite(val):
                return val, False, val, False
            return np.nan, False, np.nan, False
        _t8uk, T8c, _t8h, _t8_contra = _resolve_cort_st(T8, T8c, u_cort1)
        _t3uk, T3c, _t3h, _t3_contra = _resolve_cort_st(T3, T3c, u_cort1)
        if _t8_contra:
            st.info(f"ℹ️ T8 taken as censored (< {thr1:g} {u_cort1}) — typed value ignored." if st.session_state.lang == "EN"
                    else f"ℹ️ T8 pris comme censuré (< {thr1:g} {u_cort1}) — valeur saisie ignorée.")
        if _t3_contra:
            st.info(f"ℹ️ T3 taken as censored (< {thr1:g} {u_cort1}) — typed value ignored." if st.session_state.lang == "EN"
                    else f"ℹ️ T3 pris comme censuré (< {thr1:g} {u_cort1}) — valeur saisie ignorée.")
        _lyuk = _uk_num(Lymph, "counts", u_counts1)
        _thuk = _uk_num(Thr, "counts", u_counts1)
        _eouk = _uk_num(Eos, "counts", u_counts1)
        _cruk = _uk_num(Crea, "crea", u_crea1)
        _cauk = _uk_num(Ca, "ca", u_ca1)

        # Plausibility Check Warnings
        w_list = warn_units({
            "T8": [_t8h], "T3": [_t3uk], "Crea": [_cruk], "Ca": [_cauk],
            "Lymph": [_lyuk], "Thr": [_thuk], "Eos": [_eouk]
        }, units1, st.session_state.lang == "EN")
        for wm in w_list:
            st.warning(f"⚠️ {wm}")

        raw = {
            "T8": ("" if T8c else _t8uk), "T8_cens": T8c,
            "T3": ("" if T3c else _t3uk), "T3_cens": T3c,
            "Lymphocytes": _lyuk, "Thrombocytes": _thuk, "Eosinophils": _eouk,
            "USG": USG, "USG_sup": USGsup, "Creatinine": _cruk,
            "Totalcalcium": _cauk, "RatioALKP": rALKP, "Age": Age
        }
        feat, aux = build_features_row(raw)

        if bool(feat.isna().values.any()):
            st.warning(T("missing_warning"))

        p = predict_proba(feat)
        is_hac = p >= seuil_proba
        p_pct = p * 100
        thr_pct = seuil_proba * 100

        # Confidence Level
        if is_hac:
            conf_text = T("confidence_high_hac") if p >= 0.80 else T("confidence_mod_hac")
            banner_class = "verdict-banner-hac"
            verdict_class = "verdict-title-hac"
            verdict_title = T("verdict_hac")
            badge_icon = "🚨"
        else:
            if p >= 0.45:
                conf_text = T("confidence_equivocal")
            elif p <= 0.20:
                conf_text = T("confidence_high_nai")
            else:
                conf_text = T("confidence_mod_nai")
            banner_class = "verdict-banner-nai"
            verdict_class = "verdict-title-nai"
            verdict_title = T("verdict_nai")
            badge_icon = "🛡️"

        # Historical Rule Evaluation
        t8v = aux["T8_val"]
        hist_pos = np.isfinite(t8v) and t8v > SEUIL_T8
        _fdisp = UNIT_FACTORS["cort"].get(u_cort1, 1.0)
        _t8show = f"{t8v / _fdisp:.2f} {u_cort1}" if np.isfinite(t8v) else f"< {thr1:g} {u_cort1}"
        hist_text = f"Positif (T8 = {_t8show})" if hist_pos else f"Négatif (T8 = {_t8show})"
        if st.session_state.lang == "EN":
            hist_text = f"Positive (T8 = {_t8show})" if hist_pos else f"Negative (T8 = {_t8show})"

        # RENDER DIAGNOSTIC RESULTS
        render_html(f"""
        <div class="{banner_class}">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
                <div>
                    <div class="verdict-title {verdict_class}">{badge_icon} {verdict_title}</div>
                    <div class="verdict-desc">{conf_text}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:12px; font-weight:700; color:#64748B; text-transform:uppercase;">{T("proba_metric_label")}</div>
                    <div style="font-size:38px; font-weight:800; line-height:1.1; color:{'#BE123C' if is_hac else '#047857'};">{p_pct:.1f}%</div>
                    <div style="font-size:12px; font-weight:600; color:#64748B;">Seuil décisionnel : {thr_pct:.1f}%</div>
                </div>
            </div>
            <div class="gauge-container">
                <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:700; color:#334155;">
                    <span>🛡️ NAI (Maladie non surrénalienne)</span>
                    <span style="color:#0F172A;">📍 Patient: {p_pct:.1f}% | Seuil: {thr_pct:.0f}%</span>
                    <span>🚨 HAC (Hyperadrénocorticisme)</span>
                </div>
                <div class="gauge-track">
                    <div class="gauge-threshold-line" style="left: {thr_pct}%;" title="Seuil Youden = {thr_pct:.1f}%"></div>
                    <div class="gauge-pointer" style="left: {min(max(p_pct, 2.0), 98.0)}%;" title="Patient: {p_pct:.1f}%"></div>
                </div>
                <div class="gauge-labels">
                    <span>0% (Forte certitude NAI)</span>
                    <span>Zone de transition</span>
                    <span>100% (Forte certitude HAC)</span>
                </div>
            </div>
        </div>
        """)

        # Comparative Table & Biological Profiling Columns
        c_diag1, c_diag2 = st.columns([1.1, 1])

        with c_diag1:
            with st.container(border=True):
                st.markdown(f"**⚖️ {T('comp_table_title')}**")
                render_html(f"""
                <table class="comp-table">
                    <thead>
                        <tr>
                            <th>{T("th_criterion")}</th>
                            <th>{T("th_model")}</th>
                            <th>{T("th_histo").format(t8=f"{thr1:g} {u_cort1}")}</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><b>{T("row_verdict")}</b></td>
                            <td><span style="color:{'#BE123C' if is_hac else '#047857'}; font-weight:800;">{'HAC' if is_hac else 'NAI'} ({p_pct:.1f}%)</span></td>
                            <td><span style="font-weight:700;">{hist_text}</span></td>
                        </tr>
                        <tr>
                            <td><b>{T("row_spec")}</b></td>
                            <td><b style="color:#047857;">95.3%</b> (Très peu de faux positifs)</td>
                            <td><b style="color:#B45309;">73.4%</b> (27% de faux positifs)</td>
                        </tr>
                        <tr>
                            <td><b>{T("row_sens")}</b></td>
                            <td><b>86.4%</b></td>
                            <td><b>93.2%</b></td>
                        </tr>
                        <tr>
                            <td><b>{T("row_rationale")}</b></td>
                            <td style="font-size:12px; color:#475569;">{T("rat_model")}</td>
                            <td style="font-size:12px; color:#475569;">{T("rat_histo")}</td>
                        </tr>
                    </tbody>
                </table>
                """)

        with c_diag2:
            with st.container(border=True):
                st.markdown(f"**🧬 {T('factors_title')}**")

                # Cortisol Status Pill
                if np.isfinite(_t8uk) and _t8uk > 27.6:
                    render_html(f'<div class="factor-pill hac"><span class="factor-pill-icon">🔴</span><span class="factor-pill-text">{T("fact_lddst_hac")} — T8={_t8show}</span></div>')
                else:
                    render_html(f'<div class="factor-pill nai"><span class="factor-pill-icon">🟢</span><span class="factor-pill-text">{T("fact_lddst_nai")} — T8={_t8show}</span></div>')

                # Lymphocytes Status Pill
                if np.isfinite(_lyuk) and _lyuk < 1.0:
                    render_html(f'<div class="factor-pill hac"><span class="factor-pill-icon">🔴</span><span class="factor-pill-text">{T("fact_lymph_hac")} — {Lymph} {u_counts1}</span></div>')
                else:
                    render_html(f'<div class="factor-pill nai"><span class="factor-pill-icon">🟢</span><span class="factor-pill-text">{T("fact_lymph_nai")} — {Lymph} {u_counts1}</span></div>')

                # Platelets Status Pill
                if np.isfinite(_thuk) and _thuk > 400:
                    render_html(f'<div class="factor-pill hac"><span class="factor-pill-icon">🔴</span><span class="factor-pill-text">{T("fact_thr_hac")} — {Thr} {u_counts1}</span></div>')
                else:
                    render_html(f'<div class="factor-pill nai"><span class="factor-pill-icon">🟢</span><span class="factor-pill-text">{T("fact_thr_nai")} — {Thr} {u_counts1}</span></div>')

                # Urinalysis USG Pill
                try:
                    usg_float = float(str(USG).replace(",", "."))
                except:
                    usg_float = 1.012
                if not USGsup and usg_float < 1.020:
                    render_html(f'<div class="factor-pill hac"><span class="factor-pill-icon">🔴</span><span class="factor-pill-text">{T("fact_usg_hac")} — USG {USG}</span></div>')
                else:
                    render_html(f'<div class="factor-pill nai"><span class="factor-pill-icon">🟢</span><span class="factor-pill-text">{T("fact_usg_nai")} — USG {"> 1.050" if USGsup else USG}</span></div>')

                # ALKP Pill
                try:
                    alkp_float = float(str(rALKP).replace(",", "."))
                except:
                    alkp_float = 1.0
                if alkp_float >= 4.0:
                    render_html(f'<div class="factor-pill hac"><span class="factor-pill-icon">🔴</span><span class="factor-pill-text">{T("fact_alkp_hac")} — {rALKP}× LSN</span></div>')
                else:
                    render_html(f'<div class="factor-pill nai"><span class="factor-pill-icon">🟢</span><span class="factor-pill-text">{T("fact_alkp_nai")} — {rALKP}× LSN</span></div>')

# ==============================================================================
# TAB 2: BATCH / COHORT FILE PROCESSING
# ==============================================================================
with tab2:
    with st.container(border=True):
        st.markdown(f"**📁 {T('tab2_title')}**")
        st.caption(T("tab2_intro"))

        # Template generator: choose units, edit rows, download
        _tu = st.radio(T("tmpl_unit_label"), ["UK", "US"], index=0, horizontal=True,
                       format_func=lambda x: T("u_uk") if x == "UK" else T("u_us"),
                       key="tmpl_unit")
        _tdf = build_template_df(_tu, st.session_state.lang)
        _edited = st.data_editor(_tdf, num_rows="dynamic", use_container_width=True,
                                 key=f"tmpl_editor_{_tu}_{st.session_state.lang}")
        st.caption(T("tmpl_edit_hint"))
        _tbuf = io.BytesIO(); _edited.to_csv(_tbuf, index=False)
        st.download_button(T("btn_tmpl_download"), _tbuf.getvalue(),
                           f"template_HAC_NAI_{_tu}.csv", "text/csv", use_container_width=True)

    # Unit & Calibration Options for File
    with st.container(border=True):
        st.markdown(f"**⚙️ {T('u_file')}**")
        _uc1, _uc2, _uc3 = st.columns([1.2, 1.5, 1])
        with _uc1:
            upres2 = st.radio(T("u_file"), ["UK", "US"], index=0, horizontal=True,
                              format_func=lambda x: T("u_uk") if x == "UK" else T("u_us"), key="upres2", label_visibility="collapsed")
        with _uc2:
            seuil2 = st.slider(T("slider2"), 0.1, 0.9, SEUIL_DEFAUT, 0.01, key="s2")
        with _uc3:
            recal_opt = st.selectbox(T("recal_lab"), [T("r_auto"), T("r_seuil"), T("r_platt"), T("r_none")], index=0)

        units2 = dict(UNIT_PRESETS.get(upres2, UNIT_PRESETS["UK"]))
        thr2disp = CORT_SEUIL_DISP.get(units2["cort"], 27.6)

    # Upload Drag & Drop
    uploaded_file = st.file_uploader(T("file_drop_label"), type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                try:
                    df_in = pd.read_csv(uploaded_file, sep=None, engine="python")
                except:
                    uploaded_file.seek(0)
                    df_in = pd.read_csv(uploaded_file)
            else:
                df_in = pd.read_excel(uploaded_file)
        except Exception as err:
            st.error(f"❌ Error reading file: {err}")
            st.stop()

        df_in.columns = [str(c).strip() for c in df_in.columns]
        def get_col(*names):
            return find_col(list(df_in.columns), *names)

        cT8, cT3 = get_col("T8"), get_col("T3")
        if cT8 is None or cT3 is None:
            st.error("❌ 'T8' and 'T3' columns are mandatory for LDDST assessment.")
            st.stop()

        cDiag = get_col("Diagnosis", "diagnosis", "diagnostic", "label")
        def g(r, c):
            return r[c] if (c is not None and c in df_in.columns) else np.nan

        probs, t8pos, ok_list = [], [], []
        _w = {"T8": [], "T3": [], "Crea": [], "Ca": [], "Lymph": [], "Thr": [], "Eos": []}

        for _, r in df_in.iterrows():
            t3raw, t8raw = g(r, cT3), g(r, cT8)
            usgraw = g(r, get_col("USG", "usg", "densite"))
            t3c = isinstance(t3raw, str) and "<" in t3raw
            t8c = isinstance(t8raw, str) and "<" in t8raw
            usup = isinstance(usgraw, str) and ">" in usgraw

            if t8c:
                _v8, _ok8 = _uk_censored_val(t8raw, "cort", units2["cort"])
                t8uk = _v8 if _ok8 and np.isfinite(_v8) else 13.8
            else:
                t8uk = _uk_num(t8raw, "cort", units2["cort"])

            if t3c:
                _v3, _ok3 = _uk_censored_val(t3raw, "cort", units2["cort"])
                t3uk = _v3 if _ok3 and np.isfinite(_v3) else 13.8
            else:
                t3uk = _uk_num(t3raw, "cort", units2["cort"])

            _ly = _uk_num(g(r, get_col("Lymphocytes", "lymphocytes")), "counts", units2["counts"])
            _th = _uk_num(g(r, get_col("Thrombocytes", "thrombocytes")), "counts", units2["counts"])
            _eo = _uk_num(g(r, get_col("eosinophil", "Eosinophils", "eosinophils")), "counts", units2["counts"])
            _cr = _uk_num(g(r, get_col("Creatinine", "creatinine")), "crea", units2["crea"])
            _ca = _uk_num(g(r, get_col("calcium", "Totalcalcium", "totalcalcium")), "ca", units2["ca"])

            for _k, _v in [("T8", (np.nan if t8c else t8uk)), ("T3", t3uk), ("Crea", _cr), ("Ca", _ca),
                           ("Lymph", _ly), ("Thr", _th), ("Eos", _eo)]:
                _w[_k].append(_v)

            raw = {
                "T8": t8uk, "T8_cens": t8c,
                "T3": t3uk, "T3_cens": t3c,
                "Lymphocytes": _ly, "Thrombocytes": _th, "Eosinophils": _eo,
                "USG": ("" if usup else usgraw), "USG_sup": usup,
                "Creatinine": _cr, "Totalcalcium": _ca,
                "RatioALKP": g(r, get_col("RatioALKP", "ratioalkp")),
                "Age": g(r, get_col("Age", "age"))
            }
            feat, aux = build_features_row(raw)
            try:
                probs.append(predict_proba(feat))
                ok_list.append(True)
            except:
                probs.append(np.nan)
                ok_list.append(False)

            tv = aux["T8_val"]
            t8pos.append(1 if (np.isfinite(tv) and tv > SEUIL_T8) else 0)

        for _wm in warn_units(_w, units2, st.session_state.lang == "EN"):
            st.warning(f"⚠️ {_wm}")

        df_in["proba_HAC_modele"] = probs
        df_in["pred_modele"] = (df_in["proba_HAC_modele"] >= seuil2).astype("Int64").map({1: "HAC", 0: "NAI"})
        df_in[f"pred_LDDST_T8>{thr2disp:g}({units2['cort']})"] = pd.Series(t8pos).map({1: "HAC", 0: "NAI"})

        st.success(f"✅ {len(df_in)} {T('results_title').lower()} ({int(np.sum(ok_list))} valid).")

        # Visual preview
        st.dataframe(df_in.head(50), use_container_width=True)

        # Performance evaluation if Diagnosis is present
        if cDiag is not None:
            y = df_in[cDiag].astype(str).str.strip().str.lower().map({
                "hyperadrenocorticism": 1, "hac": 1, "1": 1, "1.0": 1, "true": 1,
                "non-adrenal illness": 0, "nai": 0, "0": 0, "0.0": 0, "false": 0
            })
            m = y.notna() & df_in["proba_HAC_modele"].notna()
            if m.sum() < 10:
                st.info("ℹ️ Fewer than 10 cases with verified diagnosis — displaying predictions only.")
            else:
                yt = y[m].astype(int).values
                pm = df_in.loc[m, "proba_HAC_modele"].values
                predm = (pm >= seuil2).astype(int)
                ph = np.array(t8pos)[m.values]

                auc_m = roc_auc_score(yt, pm)
                _, lo_a, hi_a = bootstrap_ci(yt, pm, roc_auc_score)
                se_m, sp_m = se_sp(yt, predm)
                _, lo_se, hi_se = bootstrap_ci(yt, predm, lambda a, b: (b[a == 1] == 1).mean() if (a == 1).sum() else np.nan)
                _, lo_sp, hi_sp = bootstrap_ci(yt, predm, lambda a, b: (b[a == 0] == 0).mean() if (a == 0).sum() else np.nan)

                se_h, sp_h = se_sp(yt, ph)
                try: auc_h = roc_auc_score(yt, ph)
                except: auc_h = (se_h + sp_h) / 2

                # Metric Cards
                with st.container(border=True):
                    st.markdown("**📊 Cohort Metrics & Statistical Performance**")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("AUC Modèle 10-vars", f"{auc_m:.3f}", f"95% CI [{lo_a:.3f}–{hi_a:.3f}]")
                    m2.metric("Sensibilité (Se)", f"{se_m:.1%}", f"95% CI [{lo_se:.2f}–{hi_se:.2f}]")
                    m3.metric("Spécificité (Sp)", f"{sp_m:.1%}", f"95% CI [{lo_sp:.2f}–{hi_sp:.2f}]")
                    m4.metric("LDDST Seul (T8)", f"AUC {auc_h:.3f}", f"Se {se_h:.1%} / Sp {sp_h:.1%}")

                # ROC Plot
                fpr_m, tpr_m, _ = roc_curve(yt, pm)
                fpr_h, tpr_h, _ = roc_curve(yt, ph)

                fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#F8FAFC")
                ax.set_facecolor("#FFFFFF")
                ax.plot(fpr_m, tpr_m, color="#0E7490", lw=2.5, label=f"Modèle 10-vars (AUC = {auc_m:.3f})")
                ax.plot(fpr_h, tpr_h, color="#BE123C", lw=2, linestyle="--", label=f"LDDST seul T8>{thr2disp:g} (AUC = {auc_h:.3f})")
                ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle=":", lw=1.5, label="Aléatoire (AUC = 0.500)")
                ax.set_xlabel("1 – Spécificité (Taux faux positifs)", fontsize=11, fontweight="600")
                ax.set_ylabel("Sensibilité (Taux vrais positifs)", fontsize=11, fontweight="600")
                ax.set_title("Courbe ROC : Modèle 10 Variables vs Test LDDST Seul", fontsize=12, fontweight="700", pad=12)
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", loc="lower right")
                st.pyplot(fig)

        # Download Buttons
        buf_out = io.BytesIO(); df_in.to_csv(buf_out, index=False)
        st.download_button(T("btn_download_results"), buf_out.getvalue(), "predictions_HAC_NAI.csv", "text/csv", type="primary")

# ---------- Footer ----------
st.markdown(f"""
<div style="text-align:center; padding:32px 16px 12px 16px; color:#94A3B8; font-size:12px;">
    {T("disclaimer")}<br/>
    <b>HAC vs NAI AI Diagnostic Tool</b> · ElasticNet M10 Engine · 10 Variables (No ACTHst)
</div>
""", unsafe_allow_html=True)
