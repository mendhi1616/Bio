# Live-Cell — App no-code (biologistes)

## Installation
1. conda env create -f environment.yml
2. conda activate livecell-app
3. Organisez vos données :
```
data/
  wild-type/
    wild-type_01.tif ...
  KO/
    KO_01.tif ...
```
4. Lancez : streamlit run app.py

## Utilisation
- Indiquez le chemin vers `data/`.
- Sélectionnez les conditions détectées (ex. `wild-type` et `KO`).
- Choisissez la segmentation (`auto` par défaut; `cellpose` si installé).
- Cliquez **Lancer l'analyse**.
- Téléchargez les CSV (`outputs/`).

## Statistiques
- T‑tests de Welch : (i) n_cells au dernier temps, (ii) AUC prolifération, (iii) AUC survie.
- AUC par intégration trapézoïdale.
