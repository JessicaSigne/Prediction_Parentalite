import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(page_title="Prédicteur de Parentalité", layout="wide")

# --- AJOUT DU DESIGN ROSÉ VIOLET ---
st.markdown("""
    <style>
    /* Fond principal de l'application */
    .stApp {
        background: linear-gradient(135deg, #fffff 0%, #fffff 50%, #4a148c 100%);
    }
    
    /* Style de la barre latérale (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #4a148c; /* Rose plus soutenu f8bbd0*/
    }
    
    /* Personnalisation des titres */
    h1 {
        color: #4a148c; /* Violet foncé */
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Style du bouton Prédire */
    div.stButton > button:first-child {
        background-color: #ab47bc;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    div.stButton > button:first-child:hover {
        background-color: #4a148c;
        border: 1px solid white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Prédiction : L'individu aura-t-il des enfants ?")

# 1. Chargement du modèle et des colonnes
model = joblib.load("models/model_rf_parentalite.pkl")
model_columns = joblib.load("models/columns_parentalite.pkl")


# 3. Transformation des saisies en DataFrame

# --- 1. Variables Numériques ---
age = st.sidebar.slider("Âge", 18, 100, 30)
nombre_fs = st.sidebar.number_input("Nombre de frères et soeurs", 0, 15, 2)
minutes_tv = st.sidebar.number_input("Minutes TV par jour", 0, 1440, 120)
minutes_l = st.sidebar.number_input("Minutes Lecture par jour", 0, 1440, 30)

# --- 2. Variables Catégorielles Spécifiques ---
sexe = st.sidebar.radio("Sexe", ["H", "F"])

statut = st.sidebar.selectbox("Statut Marital", 
    ["Marie", "Seul", "Divorce", "Veuf"])

occupation = st.sidebar.selectbox("Occupation", 
    ["Autre inactif", "Exerce profession", "Retraite", "Etudiant", "Chomeur", "Au foyer"])

qualification = st.sidebar.selectbox("Qualification", 
    ["Cadre", "Non concerne", "Employe de bureau", "Ouvrier qualifie", 
     "Profession intermediaire", "Autre", "Ouvrier specialise", "Technicien"])

taille_ressentie = st.sidebar.selectbox("Taille de l'appartement ressentie", 
    ["Comme il faut", "Pas assez", "Trop", "Refus"])

# --- 3. Variables Binaires (Oui/Non) ---
etudie = st.sidebar.radio("Étudie", ["Oui", "Non"])
jardinage = st.sidebar.radio("Jardinage", ["Oui", "Non"])
cuisine = st.sidebar.radio("Cuisine", ["Oui", "Non"])
sport = st.sidebar.radio("Sport", ["Oui", "Non"])
lecture_bd = st.sidebar.radio("Lecture BD", ["Oui", "Non"])
ecoute_rp = st.sidebar.radio("Écoute Radio/Podcast", ["Oui", "Non"])
journal_intime = st.sidebar.radio("Journal Intime", ["Oui", "Non"])

# --- 4. Transformation en DataFrame ---
# Note : 'NombreE' est exclu ici car c'est ce qu'on cherche à comparer/prédire
input_data = pd.DataFrame({
    'Sexe': [sexe],
    'Age': [age],
    'Statut': [statut],
    'Occupation': [occupation],
    'Qualification': [qualification],
    'Etudie': [etudie],
    'NombreFS': [nombre_fs],
    'Jardinage': [jardinage],
    'Cuisine': [cuisine],
    'Sport': [sport],
    'MinutesTV': [minutes_tv],
    'LectureBD': [lecture_bd],
    'MinutesL': [minutes_l],
    'EcouteRP': [ecoute_rp],
    'JournalIntime': [journal_intime],
    'Taille': [taille_ressentie]
})

st.write("### Données saisies pour la prédiction :")
st.dataframe(input_data)

# 4. Application du Get Dummies identique au modèle
input_encoded = pd.get_dummies(input_data)

# 5. Alignement des colonnes (Crucial !)
# On crée un DataFrame vide avec les colonnes du modèle et on remplit
full_input = pd.DataFrame(columns=model_columns)
full_input = pd.concat([full_input, input_encoded]).fillna(0)
full_input = full_input[model_columns] # On remet les colonnes dans le bon ordre

# 6. Prédiction
if st.button("Prédire"):
    prediction = model.predict(full_input)
    proba = model.predict_proba(full_input)[0][1]
    
    if prediction[0] == 1:
        st.success(f"Résultat : **Probabilité élevée d'avoir des enfants** ({proba:.2%})")
    else:
        st.info(f"Résultat : **Probabilité faible d'avoir des enfants** ({proba:.2%})")