import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Gestionnaire CLI : Projet Parentalité")
    
    # 1. Action principale (train ou interface)
    parser.add_argument(
        'action', 
        choices=['train', 'interface'], 
        help="Action : 'train' (Entraînement) ou 'interface' (Visualisation)"
    )

    # 2. Option pour l'entraînement
    parser.add_argument(
        '--target', 
        choices=['proba', 'nombre', 'all'], 
        default='all',
        help="Cible : 'proba' (RF), 'nombre' (ANN) ou 'all'"
    )

    # 3. Option pour choisir l'application
    parser.add_argument(
        '--app', 
        choices=['solo', 'batch'], 
        default='solo',
        help="Type d'interface : 'solo' (Jumelée) ou 'batch' (CSV)"
    )

    args = parser.parse_args()

    # --- LOGIQUE ENTRAÎNEMENT ---
    if args.action == 'train':
        print("--- ⚙️ Lancement des entraînements ---")
        if args.target in ['proba', 'all']:
            print("Exécution : back/proba_enfant.py")
            subprocess.run(["python", "back/proba_enfant.py"])
        if args.target in ['nombre', 'all']:
            print("Exécution : back/ann_nb_enfants.py")
            subprocess.run(["python", "back/ann_nb_enfants.py"])

    # --- LOGIQUE INTERFACE (STREAMLIT) ---
    elif args.action == 'interface':
        if args.app == 'solo':
            print("--- 🖥️ Lancement Interface Individuelle Jumelée (app3.py) ---")
            subprocess.run(["streamlit", "run", "back/app3.py"])
        
        elif args.app == 'batch':
            print("--- 📂 Lancement Interface Chargement de fichier (app.py) ---")
            subprocess.run(["streamlit", "run", "back/app.py"])

if __name__ == "__main__":
    main()