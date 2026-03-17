# 👶 Projet de Prédiction de Natalité : ML & Deep Learning

## 👤 Auteur : 
**Jessica SIGNE**  
📧 Email : jessicasigne44@gmail.com  
📅 Date : Mars 2026 

GitHub : https://github.com/JessicaSigne/Projet_RL_Puissance4.git

---

Ce projet a été développé dans le cadre d'un cours de **Deep Learning**. L'objectif est d'explorer deux problématiques liées à la parentalité en utilisant des approches de Machine Learning classique et de Deep Learning.

## 🎯 Objectifs du Projet
1. **Classification (Probabilité)** : Déterminer si un individu aura des enfants ou non.
2. **Régression (Nombre d'enfants)** : Prédire le nombre précis d'enfants potentiels d'un individu.

## 🧠 Méthodologie et Modèles
Le projet compare plusieurs architectures pour identifier la plus performante :
- **Classification** : Comparaison entre **KNN**, **Random Forest** et **ANN**. 
  * *Résultat* : Le **Random Forest** a été retenu pour l'interface finale car il offrait la meilleure précision et stabilité.
- **Régression** : Utilisation d'un **Réseau de Neurones Artificiels (ANN)** multi-couches développé avec TensorFlow/Keras.
  * *Architecture* : Utilisation de couches `Dense` avec activation `ReLU`, `BatchNormalization` pour la stabilité, et `Dropout` pour limiter le surapprentissage.

## 🚀 Interfaces Streamlit
Le projet propose trois modes d'utilisation :
- **Analyse par Lot (`app2.py`)** : Permet de charger un fichier CSV complet pour générer des prédictions massives et comparer les résultats avec les données réelles (Matrice de confusion).
- **Analyse Individuelle (`app.py`)** : Formulaire interactif pour tester la probabilité de parentalité d'un profil spécifique.
- **Interface Jumelée (`main.py`)** : Regroupe la prédiction de probabilité (Random Forest) et l'estimation du nombre d'enfants (ANN) sur un seul tableau de bord.

## 💻 Installation
1. Installez les dépendances :
   ```pip install -r requirements.txt```

## Lancez l'application principale:

```streamlit run main.py```

## 📂 Structure du Projet
```/back :``` Scripts d'entraînement des modèles (ANN et RF) et interface visuel.

```/data :``` Jeux de données CSV et exports de tests.

```/models :``` Fichiers binaires (.pkl) des modèles, scalers et listes de colonnes.

## 📈 Analyses


## 📄 Licence

Ce projet est réalisé dans un cadre pédagogique.  
Tous droits réservés © 2026 Jessica SIGNE


## 📞 Contacts

**Jessica SIGNE**  
📧 jessicasigne44@gmail.com

Pour toute question sur le projet, l'implémentation ou les résultats, n'hésitez pas à me contacter !