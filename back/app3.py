### Le programme jumelé (`main.py`)
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuration
st.set_page_config(page_title="Dashboard Parentalité", layout="wide", page_icon="👶")

# --- DESIGN ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #fffff 0%, #fffff 50%, #4a148c 100%); }
    [data-testid="stSidebar"] { background-color: #4a148c; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p { color: white; }
    h1, h2 { color: #4a148c; font-family: 'Helvetica', sans-serif; }
    div.stButton > button:first-child {
        background-color: #ab47bc; color: white; border-radius: 20px; font-weight: bold; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("👶 Dashboard Prédictif : Parentalité & Famille")

# --- CHARGEMENT ---
@st.cache_resource
def load_resources():
    # Modèle 1 : Classification (Random Forest)
    model_proba = joblib.load("models/model_rf_parentalite.pkl")
    cols_proba = joblib.load("models/columns_parentalite.pkl")
    # Modèle 2 : Régression (ANN)
    model_nb = joblib.load("models/model_nb_enfant.pkl")
    cols_nb = joblib.load("models/columns_nb_enfant.pkl")
    scaler_nb = joblib.load("models/scaler_nb_enfant.pkl")
    return model_proba, cols_proba, model_nb, cols_nb, scaler_nb

m_proba, c_proba, m_nb, c_nb, s_nb = load_resources()

# --- SIDEBAR (Saisie unique) ---
st.sidebar.header("📋 Profil de l'individu")
age = st.sidebar.slider("Âge", 18, 100, 30)
nombre_fs = st.sidebar.number_input("Nombre de frères et soeurs", 0, 15, 2)
minutes_tv = st.sidebar.number_input("Minutes TV / jour", 0, 1440, 120)
minutes_l = st.sidebar.number_input("Minutes Lecture / jour", 0, 1440, 30)
sexe = st.sidebar.radio("Sexe", ["H", "F"])
statut = st.sidebar.selectbox("Statut Marital", ["Marie", "Seul", "Divorce", "Veuf"])
occupation = st.sidebar.selectbox("Occupation", ["Autre inactif", "Exerce profession", "Retraite", "Etudiant", "Chomeur", "Au foyer"])
qualification = st.sidebar.selectbox("Qualification", ["Cadre", "Non concerne", "Employe de bureau", "Ouvrier qualifie", "Profession intermediaire", "Autre", "Ouvrier specialise", "Technicien"])
taille = st.sidebar.selectbox("Espace logement", ["Comme il faut", "Pas assez", "Trop", "Refus"])
etudie = st.sidebar.selectbox("Étudie ?", ["Oui", "Non"])
jardinage = st.sidebar.selectbox("Jardinage ?", ["Oui", "Non"])
cuisine = st.sidebar.selectbox("Cuisine ?", ["Oui", "Non"])
sport = st.sidebar.selectbox("Sport ?", ["Oui", "Non"])
lecture_bd = st.sidebar.selectbox("Lit des BD ?", ["Oui", "Non"])
ecoute_rp = st.sidebar.selectbox("Écoute Radio ?", ["Oui", "Non"])
journal = st.sidebar.selectbox("Journal Intime ?", ["Oui", "Non"])

# --- PRÉPARATION DES DONNÉES ---
input_data = pd.DataFrame({
    'Sexe': [sexe], 'Age': [age], 'Statut': [statut], 'Occupation': [occupation],
    'Qualification': [qualification], 'Etudie': [etudie], 'NombreFS': [nombre_fs],
    'Jardinage': [jardinage], 'Cuisine': [cuisine], 'Sport': [sport],
    'MinutesTV': [minutes_tv], 'LectureBD': [lecture_bd], 'MinutesL': [minutes_l],
    'EcouteRP': [ecoute_rp], 'JournalIntime': [journal], 'Taille': [taille]
})

if st.button("Lancer l'analyse de l'individu"):
    # 1. Traitement pour Probabilité (Classification)
    input_encoded_p = pd.get_dummies(input_data)
    X_p = pd.DataFrame(0, index=[0], columns=c_proba)
    for col in input_encoded_p.columns:
        if col in X_p.columns: X_p[col] = input_encoded_p[col].values
    
    res_proba = m_proba.predict_proba(X_p)[0][1]
    
    # 2. Traitement pour Nombre (ANN + Scaling)
    input_encoded_n = pd.get_dummies(input_data)
    X_n = pd.DataFrame(0, index=[0], columns=c_nb)
    for col in input_encoded_n.columns:
        if col in X_n.columns: X_n[col] = input_encoded_n[col].values
    
    X_n_scaled = s_nb.transform(X_n)
    res_nb_brut = m_nb.predict(X_n_scaled)
    res_nb = max(0, int(round(float(res_nb_brut.item()))))

    # --- AFFICHAGE ---
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Probabilité de Parentalité")
        color = "green" if res_proba > 0.5 else "orange"
        st.markdown(f"<h1 style='color:{color};'>{res_proba:.1%}</h1>", unsafe_allow_html=True)
        # st.write("Modèle : Random Forest Classifier")

    with col2:
        st.subheader("👶 Nombre d'enfants estimé")
        st.markdown(f"<h1 style='color:#ab47bc;'>{res_nb}</h1>", unsafe_allow_html=True)
        # st.write("Modèle : Artificial Neural Network (Regression)")
        # st.caption(f"Valeur brute ANN : {res_nb_brut.item():.2f}")

    st.balloons()