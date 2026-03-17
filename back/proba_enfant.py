import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns 


donnees=pd.read_csv('data/dataLabs.csv', sep=';', decimal=',')
donnees.head()

donnees.drop_duplicates(inplace=True)

#Création de Flag_enfant: si NombreE >= 1 Y=1 sinon Y=0
donnees['Flag_enfant'] = (donnees['NombreE'] >= 1).astype(int)

donnees = donnees[(donnees['MinutesTV'] <= 1140) & (donnees['MinutesL'] <= 1140)]

donnees2 =donnees.copy()

# Sauvegarder les caractéristiques (X_test) sans la réponse
from sklearn.model_selection import train_test_split
X_brut=donnees.drop(columns=["NombreE", "Flag_enfant"]) # variables explicatives ou features  (tout sauf NombreE et Flag_enfant)
y_brut=donnees["NombreE"] # variable cible ou target (Flag_enfant)

# Séparation en données d'entraînement (80%) et de test (20%)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_brut, y_brut, test_size=0.2, random_state=42)
X_test_b.to_csv('data/Donnees_test_parentalite.csv', index=False, sep=';', decimal=',')

print(f"Fichiers CSV créés avec succès ! avec {len(X_test_b)} lignes et {X_test_b.shape[1]} colonnes (sans la cible).")

df_test_complet = donnees.loc[X_test_b.index]

# 2. Sauvegarde en CSV avec toutes les colonnes
df_test_complet.to_csv('data/Donnees_parentalite_completes.csv', index=False, sep=';', decimal=',')

print(f"Fichier sauvegardé avec {df_test_complet.shape[1]} colonnes et {len(df_test_complet)} lignes.")

donnees['Flag_enfant'].value_counts().plot(kind="pie", 
                             autopct='%1.1f%%', 
                             labels=["1","0"])

def analyser_variables_predictrices(df, target_col):
    # On identifie les colonnes à analyser (on exclut la cible et le Nombre d'enfants)
    cols_a_exclure = [target_col, 'NombreE']
    features = [c for c in df.columns if c not in cols_a_exclure]
    
    for col in features:
        plt.figure(figsize=(10, 4))
        
        # CAS 1 : Variable Numérique -> Analyse des Ratios
        if df[col].dtype in ['int64', 'float64']:
            # Calcul des moyennes par groupe et ratio
            stats = df.groupby(target_col)[col].mean()
            global_mean = df[col].mean()
            
            # On crée un petit DataFrame pour le plot
            plot_data = pd.DataFrame({
                'Ratio': stats / global_mean
            })
            plot_data.plot(kind='barh', color=['skyblue', 'orange'], ax=plt.gca())
            plt.axvline(1, color='red', linestyle='--')
            plt.title(f"Impact de {col} (Ratio par rapport à la moyenne)")
            plt.xlabel("Ratio (1.0 = Moyenne globale)")

        # CAS 2 : Variable Catégorielle -> Barres empilées (Proportions)
        else:
            # Tableau croisé en pourcentage
            ct = pd.crosstab(df[col], df[target_col], normalize='index')
            ct.plot(kind='bar', stacked=True, ax=plt.gca(), color=['#3498db', '#e67e22'])
            plt.title(f"Répartition de {target_col} selon {col}")
            plt.ylabel("Proportion")
            plt.legend(["Sans Enfant", "Avec Enfant"], loc='upper right')
            plt.xticks(rotation=45)

        plt.tight_layout()
        # plt.show()

# Utilisation :
analyser_variables_predictrices(donnees, 'Flag_enfant')

#Encodage des variables catégorielles
# One-Hot Encoding : Transforme toutes les colonnes 'objet' en colonnes 0/1
donnees = pd.get_dummies(donnees, drop_first=True, dtype=int)

#  Ajout de labels pour la clarté
ax = donnees[['MinutesL', 'MinutesTV']].plot(kind="box", figsize=(8, 5))

# Personnalisation (recommandé)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# plt.show()

# MinutesL a des valeurs proches de 1800. C'est physiquement impossible sur une journée de 24h donc 1140 min.
# Suppression des valeurs aberrantes

data =donnees.copy()
data.shape

#Correlation variables
# 1. On identifie toutes les colonnes numériques SAUF 'Flag_enfant'
cols_etude = data.select_dtypes(include=['number']).columns.drop("Flag_enfant")

# 2. On calcule la moyenne par groupe (0 et 1) uniquement pour ces colonnes
# On transpose (.T) pour avoir les variables en lignes
comp = data.groupby("Flag_enfant")[cols_etude].mean().T

# 3. On divise par la moyenne globale pour obtenir le ratio
# On s'assure de diviser chaque ligne par la moyenne de la colonne correspondante
comp = comp.div(data[cols_etude].mean(), axis=0)

# 4. On renomme les colonnes pour que ce soit plus clair sur le graphique
# Attention : l'ordre des colonnes ici correspond aux valeurs 0 et 1 de ta colonne
comp.columns = ["Sans_Enfant", "Avec_Enfant"]

# 5. Calcul du ratio de différence
comp["difference"] = comp["Avec_Enfant"] / comp["Sans_Enfant"]

# 6. Tri et affichage
comp.sort_values(by="difference").plot(kind="bar", figsize=(10, 5))
plt.axhline(y=1, color='red', linestyle='--') # Ligne de référence
plt.title("Comparaison des profils (Ratio Avec vs Sans Enfant)")
# plt.show()

# Suppression de la colonne NombreE qui est redondante avec Flag_enfant
data.drop(columns=['NombreE'], inplace=True)

# Affichage de la corrélation absolue avec la variable cible, triée par ordre décroissant
data.corr().Flag_enfant.abs().sort_values()[:-1].plot(kind="bar", figsize=(15,8))

# Sélection des variables les plus corrélées avec la cible. utiliser la pvalue >0.05
corr_with_target = data.corr()['Flag_enfant'].abs().sort_values(ascending=False)
selected_vars = corr_with_target[corr_with_target > 0.05].index.tolist()
selected_vars

# 1. Calcul de la corrélation avec la cible
corr_with_target = data.corr()['Flag_enfant'].abs().sort_values(ascending=False)

# 2. Seuil de sélection
threshold = 0.05

print(f"--- Analyse des corrélations (Seuil : {threshold}) ---\n")

for col, val in corr_with_target.items():
    if col == 'Flag_enfant':
        continue  # On ne s'auto-analyse pas
    
    if val > threshold:
        print(f"✅ La colonne '{col}' est corrélée (score: {val:.4f})")
    else:
        print(f"❌ La colonne '{col}' n'est pas assez corrélée (score: {val:.4f})")

# 3. Création de la liste finale
selected_vars = corr_with_target[corr_with_target > threshold].index.tolist()


# Entrainement du modèle
# Train test split


#  Séparation des features et de la cible encodées
X = data.drop(columns=["Flag_enfant"])
y = data["Flag_enfant"]

#  LE NOUVEAU SPLIT (L'étape CRUCIALE)
# En utilisant exactement le même random_state=42, les index de X_test_final 
# seront IDENTIQUES à ceux de X_test_b utilisé plus haut.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vérification de sécurité
if all(X_test.index == X_test_b.index):
    print("✅ Succès : Les index du modèle correspondent parfaitement aux index des CSV Streamlit !")
else:
    print("❌ Erreur : Les index ne correspondent pas. Vérifiez le random_state.")


####### MODELE 1 : KNN


from sklearn.neighbors import KNeighborsClassifier  

#choisir le modele
model = KNeighborsClassifier(5) #dites moi vos voisins et je vous dis qui vous etes!!

#entrainer le modele 
model.fit(X_train, y_train)

#voir la performance du modele: precision
print('Precision train: {:.2f}'.format(model.score(X_train, y_train)))
print('Precision test : {:.2f}'.format(model.score(X_test, y_test)))

# Faire les prédictions
y_pred = model.predict(X_test)

# On crée un DataFrame pour comparer
comparaison = pd.DataFrame({
    'Valeur Réelle (y_test)': y_test.values,
    'Prédiction (y_pred)': y_pred
})

# On ajoute une colonne pour voir tout de suite si c'est juste ou faux
comparaison['Correction'] = comparaison['Valeur Réelle (y_test)'] == comparaison['Prédiction (y_pred)']

# Afficher les 15 premiers résultats
print(comparaison.head())

# Calculer le taux d'erreur spécifique sur ces 15 lignes
print(f"\nNombre de succès sur cet échantillon : {comparaison.head(5)['Correction'].sum()} / 15")

# Matrice de confusion
from sklearn.metrics import confusion_matrix

# 1. Calcul de la matrice
cm = confusion_matrix(y_test, y_pred)

# 2. Préparation des textes pour chaque carré
# On définit les noms des catégories (0: Sans Enfant, 1: Avec Enfant)
labels = ['Sans Enfant', 'Avec Enfant']

# Création des annotations personnalisées
group_names = ['Vrais Négatifs\n(Bien prédit : Sans)', 
               'Faux Positifs\n(Erreur : Dit "Avec" mais c\'est "Sans")', 
               'Faux Négatifs\n(Erreur : Dit "Sans" mais c\'est "Avec")', 
               'Vrais Positifs\n(Bien prédit : Avec)']

group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
annot_labels = [f"{n}\nNb: {c}" for n, c in zip(group_names, group_counts)]
annot_labels = np.asarray(annot_labels).reshape(2,2)

# 3. Affichage avec Matplotlib et Seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, cbar=False)

plt.xlabel('Prédictions du Modèle', fontsize=12, fontweight='bold')
plt.ylabel('Réalité (Terrain)', fontsize=12, fontweight='bold')
plt.title('Interprétation de la Matrice de Confusion', fontsize=15, pad=20)
# plt.show()


####### MODELE 2 : Random Forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# 1. Séparation en données d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Création et entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. voir la performance du modele: precision
print('Precision train: {:.2f}'.format(model.score(X_train, y_train)))
print('Precision test : {:.2f}'.format(model.score(X_test, y_test)))


# Faire les prédictions
y_pred = model.predict(X_test)

# On crée un DataFrame pour comparer
comparaison = pd.DataFrame({
    'Valeur Réelle (y_test)': y_test.values,
    'Prédiction (y_pred)': y_pred
})

# On ajoute une colonne pour voir tout de suite si c'est juste ou faux
comparaison['Correction'] = comparaison['Valeur Réelle (y_test)'] == comparaison['Prédiction (y_pred)']

# Afficher les 15 premiers résultats
print(comparaison.head(50))

# Calculer le taux d'erreur spécifique sur ces 15 lignes
print(f"\nNombre de succès sur cet échantillon : {comparaison.head(50)['Correction'].sum()} / 50")

# Matrice de confusion
from sklearn.metrics import confusion_matrix

# 1. Calcul de la matrice
cm = confusion_matrix(y_test, y_pred)

# 2. Préparation des textes pour chaque carré
# On définit les noms des catégories (0: Sans Enfant, 1: Avec Enfant)
labels = ['Sans Enfant', 'Avec Enfant']

# Création des annotations personnalisées
group_names = ['Vrais Négatifs\n(Bien prédit : Sans)', 
               'Faux Positifs\n(Erreur : Dit "Avec" mais c\'est "Sans")', 
               'Faux Négatifs\n(Erreur : Dit "Sans" mais c\'est "Avec")', 
               'Vrais Positifs\n(Bien prédit : Avec)']

group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
annot_labels = [f"{n}\nNb: {c}" for n, c in zip(group_names, group_counts)]
annot_labels = np.asarray(annot_labels).reshape(2,2)

# 3. Affichage avec Matplotlib et Seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, cbar=False)

plt.xlabel('Prédictions du Modèle', fontsize=12, fontweight='bold')
plt.ylabel('Réalité (Terrain)', fontsize=12, fontweight='bold')
plt.title('Interprétation de la Matrice de Confusion', fontsize=15, pad=20)
# plt.show()


# Sauvegarde du modèle RF car plus pertinent
import joblib
joblib.dump(model, 'models/model_rf_parentalite.pkl')


# Sauvegarde la liste des colonnes après get_dummies
model_columns = list(X_train.columns)
joblib.dump(model_columns, 'models/columns_parentalite.pkl')


###### MODELE 3 : ANN
X= donnees.drop(columns=["Flag_enfant", "NombreE"], errors='ignore')
y = donnees["Flag_enfant"]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping

classifier = Sequential()

# 1. Couche d'entrée : On commence avec un peu plus de neurones 
# pour capter la complexité, mais on normalise TOUT DE SUITE.
classifier.add(BatchNormalization()) 

# 2. Couches cachées avec Dropout :
# On utilise une structure en "entonnoir" (64 -> 32 -> 16).
# Le Dropout aide à réduire les Faux Positifs en forçant le réseau à ne pas
# se reposer sur une seule variable "miracle" (comme le statut marié).
classifier.add(Dense(64, activation='relu'))
classifier.add(Dropout(0.3)) # On désactive 30% des neurones aléatoirement

classifier.add(Dense(32, activation='relu'))
classifier.add(Dropout(0.2)) # On désactive 20% des neurones aléatoirement

classifier.add(Dense(16, activation='relu'))
classifier.add(Dropout(0.1)) # On désactive 10% des neurones aléatoirement

classifier.add(Dense(8, activation='relu'))
classifier.add(Dropout(0.1)) # On désactive 10% des neurones aléatoirement

classifier.add(Dense(4, activation='relu'))
classifier.add(Dropout(0.1)) # On désactive 10% des neurones aléatoirement

classifier.add(Dense(2, activation='relu'))
classifier.add(Dropout(0.1)) # On désactive 10% des neurones aléatoirement
# Le fait de changer drastiquement d'autant de neurones entre les couches peut aider à créer des frontières de décision plus complexes que de simples seuils.

# 3. Couche de sortie
classifier.add(Dense(1, activation='sigmoid'))

# différence entre relu et sigmoid : 
# ReLU est plus rapide à calculer et évite le problème de vanishing gradient. Plusieurs neurones qui se suivent s'il se rend compte que la donnée n'est pas pertinente, il peut les "éteindre" (mettre à 0) et se concentrer sur les autres. Cependant, il peut aussi "éteindre" des neurones utiles si la donnée est un peu bruyante.
# Sigmoid est utile pour les sorties binaires (comme ici, 0 ou 1), mais peut ralentir l'apprentissage.

# 4. Compilation avec un Learning Rate contrôlé
# 'adam' est excellent, mais on peut parfois le stabiliser.
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Early Stopping : L'arme secrète
# On arrête l'entraînement si l'accuracy de test ne progresse plus pendant 10 époques.
# Cela évite de s'entraîner "trop longtemps" pour rien.
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 6. Entraînement
history = classifier.fit(
    X_train, y_train, 
    validation_split=0.20, # Plus de données pour la validation (20% au lieu de 10)
    batch_size=32,         # Taille de lot un peu plus grande pour lisser les gradients
    epochs=100, 
    callbacks=[early_stop], 
    verbose=1
)


## Evaluation du modèle
score = classifier.evaluate(X_train,y_train, verbose=0)
print('train Model Accuracy = ',score[1])
score = classifier.evaluate(X_test, y_test, verbose=0)
print('test Model Accuracy = ',score[1])

import matplotlib.pyplot as plt

history=classifier.fit(X_train, y_train, validation_split=0.20, epochs=20, batch_size=10,verbose=0)


# 1. Graphique de l'Accuracy (Ton code)
plt.figure(figsize=(10, 6))
plt.ylim(0, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
# plt.show()

# 2. Graphique de la Perte (Loss) - Très important pour un ANN
plt.figure(figsize=(10, 6))
plt.ylim(0, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss (binary_crossentropy)')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
# plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
# 1. Obtenir les probabilités
y_probs = classifier.predict(X_test)

# 2. Transformer en classes (seuil 0.5)
y_pred = (y_probs > 0.5).astype(int)

# Calcul des métriques
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_val = roc_auc_score(y_test, y_probs) # On utilise les probas pour l'AUC

print(f"--- Évaluation de l'ANN ---")
print(f"Accuracy  (Justesse)     : {acc:.4f}")
print(f"Précision (Fiabilité)    : {prec:.4f}")
print(f"Recall    (Sensibilité)  : {rec:.4f}")
print(f"F1-Score  (Équilibre)    : {f1:.4f}")
print(f"AUC Score (Séparation)   : {auc_val:.4f}")

# Matrice de confusion :

# Matrice de confusion
from sklearn.metrics import confusion_matrix

# 1. Obtenir les probabilités de sortie de l'ANN
y_pred_probs = classifier.predict(X_test)

# 2. Convertir les probabilités en 0 ou 1 (seuil de 0.5)
y_pred = (y_pred_probs > 0.5).astype(int)

# 3. Calcul de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# 4. Préparation des textes pour chaque carré
# On définit les noms des catégories (0: Sans Enfant, 1: Avec Enfant)
labels = ['Sans Enfant', 'Avec Enfant']

# Création des annotations personnalisées
group_names = ['Vrais Négatifs\n(Bien prédit : Sans)', 
               'Faux Positifs\n(Erreur : Dit "Avec" mais c\'est "Sans")', 
               'Faux Négatifs\n(Erreur : Dit "Sans" mais c\'est "Avec")', 
               'Vrais Positifs\n(Bien prédit : Avec)']

group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
annot_labels = [f"{n}\nNb: {c}" for n, c in zip(group_names, group_counts)]
annot_labels = np.asarray(annot_labels).reshape(2,2)

# 5. Affichage avec Matplotlib et Seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, cbar=False)

plt.xlabel('Prédictions du Modèle', fontsize=12, fontweight='bold')
plt.ylabel('Réalité (Terrain)', fontsize=12, fontweight='bold')
plt.title('Interprétation de la Matrice de Confusion', fontsize=15, pad=20)
# plt.show()

