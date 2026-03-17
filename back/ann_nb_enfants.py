import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# Chargement et nettoyage des données
donnees=pd.read_csv('data/dataLabs.csv', sep=';', decimal=',')
donnees.drop_duplicates(inplace=True)

donnees = donnees[(donnees['MinutesTV'] <= 1140) & (donnees['MinutesL'] <= 1140)]

data=donnees.copy()
print(data)

# --- ENCODAGE SÉCURISÉ ---
# 1. On sépare les variables numériques et catégorielles
var_num = ['Age', 'NombreFS', 'MinutesTV', 'MinutesL']
var_cat = ['Sexe', 'Statut', 'Occupation', 'Qualification', 'Etudie', 'Jardinage', 'Cuisine', 'Sport', 'LectureBD', 'EcouteRP', 'JournalIntime', 'Taille']

# 2. On s'assure que les numériques sont bien des nombres
for col in var_num:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 3. On fait le get_dummies SANS drop_first pour l'App
input_encoded = pd.get_dummies(data, columns=var_cat, drop_first=False, dtype=int)

model_columns = list(input_encoded.columns)
full_input = pd.DataFrame(columns=model_columns)
full_input = pd.concat([full_input, input_encoded]).fillna(0)

data_final = full_input.astype('float32')

X= data_final.drop(columns=["NombreE"], errors='ignore')
y = data_final["NombreE"]

# Séparation en données d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- AJOUT DU SCALING ---
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 1. On apprend et on transforme les données d'entraînement
X_train = scaler.fit_transform(X_train)

# 2. On transforme les données de test (SANS faire de fit !)
X_test = scaler.transform(X_test)

# 3. ON SAUVEGARDE LE SCALER (Indispensable pour ton App Streamlit)
joblib.dump(scaler, 'models/scaler_nb_enfant.pkl')
# ------------------------

# Modelisation
#  v1
# model_reg = Sequential([
#     (Dense(64, activation='relu', input_dim=X_train.shape[1])),
#     (BatchNormalization()),
#     (Dropout(0.2)),
#     (Dense(32, activation='relu')),
#     (Dropout(0.2)),
#     (Dense(16, activation='relu')),
#     (Dropout(0.1)),
#     (Dense(8, activation='relu')),
#     (Dropout(0.1)),
#     (Dense(1, activation='relu')) # Relu ici force le résultat à être >= 0 # Relu ici force le résultat à être >= 0
# ])


# # --- CHANGEMENT CRUCIAL POUR LA RÉGRESSION ---
# # 1 seul neurone en sortie, SANS activation (ou activation 'linear')
# # On ne veut plus une probabilité (0 à 1), mais un nombre (0, 1, 2...)
# model_reg.add(Dense(1, activation='relu')) 

# Tu utilises linear. C'est correct, mais pour un nombre d'enfants (toujours positif), 
# la fonction ReLU en sortie est souvent plus stable pour éviter les prédictions négatives.

#différence entre linear et sigmoid :
# Linear : La sortie peut être n'importe quelle valeur réelle. C'est idéal pour la régression où tu veux prédire des nombres (comme le nombre d'enfants). 
# Cependant, cela peut parfois conduire à des prédictions négatives ou très grandes si les données ne sont pas bien normalisées.

# # Compilation avec MSE (Mean Squared Error) au lieu de Crossentropy
# model_reg.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# # Entraînement
# history = model_reg.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32)


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# v2
# 1. Architecture plus "nerveuse"
model_reg = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    BatchNormalization(),
    Dropout(0.3), # Un peu plus de dropout pour éviter le surapprentissage
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    
    # Sortie : Toujours 1 neurone, ReLU est bien pour garantir du positif
    Dense(1, activation='relu') 
])

# 2. Utilisation d'un Learning Rate plus faible
optimizer = Adam(learning_rate=0.0005)

model_reg.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# 3. Ajout du EarlyStopping
# Cela arrête l'entraînement si le modèle ne progresse plus sur le jeu de validation
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# 4. Entraînement avec plus d'époques (car le EarlyStopping gère l'arrêt)
history = model_reg.fit(
    X_train, y_train, 
    validation_split=0.2, 
    epochs=200, 
    batch_size=16, # Batch plus petit pour une meilleure généralisation
    callbacks=[early_stop],
    verbose=1
)


#checking the performance of the model
score = model_reg.evaluate(X_train,y_train, verbose=0)
print('train Model Accuracy = ',score[1])
score = model_reg.evaluate(X_test, y_test, verbose=0)
print('test Model Accuracy = ',score[1])


# 1. Obtenir les prédictions numériques (ex: 1.2, 0.7...)
y_pred_reg = model_reg.predict(X_test)

# 2. Calcul des métriques de régression
mae = mean_absolute_error(y_test, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
r2 = r2_score(y_test, y_pred_reg)

print(f"--- Évaluation Régression ANN ---")
print(f"Erreur Moyenne (MAE) : {mae:.2f} enfants") 
print(f"Ecart-type erreur (RMSE) : {rmse:.2f}")
print(f"Score R² (Précision relative) : {r2:.4f}")


# 1. Calcul des prédictions sur le jeu de test
y_pred = model_reg.predict(X_test).flatten()

# 2. Création de la figure
plt.figure(figsize=(15, 6))

# --- GRAPHIQUE 1 : Valeurs Réelles vs Prédites ---
plt.subplot(1, 2, 1)
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Comparaison : Réel vs Prédit')
plt.xlabel('Nombre d\'enfants Réel')
plt.ylabel('Nombre d\'enfants Prédit')

# --- GRAPHIQUE 2 : Distribution des Erreurs (Résidus) ---
plt.subplot(1, 2, 2)
residus = y_test - y_pred
sns.histplot(residus, kde=True, color='purple')
plt.axvline(x=0, color='black', linestyle='--')
plt.title('Distribution des Résidus (Erreurs)')
plt.xlabel('Erreur (Réel - Prédit)')
plt.tight_layout()
plt.show()


# 1. Graphique de l'Erreur (MAE - Mean Absolute Error)
# C'est l'équivalent de ton graphique d'accuracy pour la régression
plt.figure(figsize=(10, 6))
# On retire plt.ylim(0, 1) car l'erreur peut être supérieure à 1 enfant
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Performance du Modèle (MAE)')
plt.ylabel('Erreur moyenne (nb enfants)')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Validation'], loc='upper right')
plt.show()

# 2. Graphique de la Perte (Loss / MSE)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Évolution de la Perte (MSE)')
plt.ylabel('Perte (Mean Squared Error)')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Validation'], loc='upper right')
plt.show()


# Faire les prédictions
y_pred = model_reg.predict(X_test)

# On crée un DataFrame pour comparer
# 2. On aplatit ET on arrondit à l'entier le plus proche
# .flatten() : passe de 2D à 1D
# .round() : arrondit (ex: 2.24 -> 2.0)
# .astype(int) : transforme en nombre entier (ex: 2.0 -> 2)
comparaison = pd.DataFrame({
    'Valeur Réelle (y_test)': y_test.values,
    'Prédiction (y_pred)': np.round(y_pred.flatten()).astype(int)  # On a besoin de "flatten" pour aplatir les prédictions en 1D
})

# On ajoute une colonne pour voir tout de suite si c'est juste ou faux
comparaison['Correction'] = comparaison['Valeur Réelle (y_test)'] == comparaison['Prédiction (y_pred)']

# Afficher les 15 premiers résultats
print(comparaison.head(50))

# Calculer le taux d'erreur spécifique sur ces 15 lignes
print(f"\nNombre de succès sur cet échantillon : {comparaison.head(50)['Correction'].sum()} / 50")


import joblib

# Sauvegarde le modèle Random Forest
joblib.dump(model_reg, 'models/model_nb_enfant.pkl')

# Sauvegarde la liste des colonnes après get_dummies

model_columns = list(X.columns)
joblib.dump(model_columns, 'models/columns_nb_enfant.pkl')