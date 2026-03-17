# version 3:
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Parentalité d'un individu", layout="wide")

st.title("Prédiction de la possibilité de parentalité d'un individu")

st.subheader("📂 Étape 1 : Prédiction sur un échantillon de données")

# --- CHARGEMENT DU MODÈLE ---
model = joblib.load("models/model_rf_parentalite.pkl")
model_columns = joblib.load("models/columns_parentalite.pkl")

# --- ÉTAPE 1 : CHARGER X_TEST ---
# st.subheader("Exercice Application famille1/Donnees_test_parentalite.csv")
test_file = st.file_uploader("Chargement du fichier à analyser", type="csv", key="test")

if test_file is not None:
    df_test = pd.read_csv(test_file, sep=';', decimal=',', engine='python')
    st.write("Aperçu du fichier de test :", df_test.head())

    # Préparation et Prédiction
    df_encoded = pd.get_dummies(df_test)
    X_final = pd.DataFrame(columns=model_columns)
    X_final = pd.concat([X_final, df_encoded]).fillna(0)
    X_final = X_final[model_columns]

    y_pred = model.predict(X_final)
    y_probs = model.predict_proba(X_final)[:, 1]
    df_test['Prediction'] = y_pred
    df_test['Confiance (%)'] = (y_probs * 100).round(2)

    
    st.success(f"✅ Prédictions générées pour {len(df_test)} lignes.")
    st.dataframe(df_test.head())

    st.divider()
    st.subheader("📊 Étape 2 : Comparaison avec la Réalité")
    
    # --- ÉTAPE 2 : CHARGER LA RÉALITÉ ---
    # st.subheader("2. Charger le fichier avec la colonne 'NombreE'")
    real_file = st.file_uploader("Chargement du fichier contenant les données réelles", type="csv", key="real")

    if real_file is not None:
        df_real = pd.read_csv(real_file, sep=';', decimal=',', engine='python')
        
        if 'NombreE' in df_real.columns:
            # On transforme NombreE en binaire (0 ou 1)
            y_true = (df_real['NombreE'] > 0).astype(int)
            
            # --- CALCUL DES STATS ---
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)

            col1, col2 = st.columns(2)

            with col1:
                st.write("### 🎯 Métriques de Performance")
                st.metric("Précision Globale (Accuracy)", f"{acc:.2%}")
                st.metric("Fiabilité (Précision)", f"{prec:.2%}")
                
                # Tableau de comparaison rapide
                df_compare = pd.DataFrame({
                    'Réalité (NombreE)': df_real['NombreE'],
                    'Réalité (Binaire)': y_true,
                    'Prédiction IA': y_pred
                })
                st.write("Détail des 10 premières lignes :")
                st.dataframe(df_compare.head(10))

            with col2:
                st.write("### 🧩 Matrice de Confusion")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                            xticklabels=['Prédit: 0', 'Prédit: 1'],
                            yticklabels=['Réel: 0', 'Réel: 1'], ax=ax)
                st.pyplot(fig)
                
            # Bouton de téléchargement final
            final_report = df_test.copy()
            final_report['NombreE_Reel'] = df_real['NombreE']
            csv = final_report.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger le rapport de comparaison", csv, "comparaison_finale.csv")
            
        else:
            st.error("❌ Le deuxième fichier doit contenir la colonne 'NombreE' pour la comparaison.")