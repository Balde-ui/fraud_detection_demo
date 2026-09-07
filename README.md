# 🏦 Fraud Detection — Live Demo

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red)
![API](https://img.shields.io/badge/API-Flask%20%2F%20Render-green)

Dashboard interactif de détection de fraude bancaire par Machine Learning.

## 🌐 Demo live
👉 **[Accéder au dashboard](https://frauddetectiondemo-zriwmvzpi7wa3gy2vnxugu.streamlit.app/)**

## 🏗️ Architecture

Le modèle n'est pas embarqué dans ce dépôt : le dashboard est un pur client
d'une API de scoring déployée séparément.

```
┌──────────────────────┐      POST /predict       ┌────────────────────────┐
│  Streamlit dashboard │ ───────────────────────► │  API Flask (Render)    │
│  (ce dépôt)          │ ◄─────────────────────── │  modèle XGBoost        │
└──────────────────────┘   fraud_probability      └────────────────────────┘
                           is_fraud, risk_level
```

**API** : `https://fraud-detection-eh3v.onrender.com`

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | GET | État du service (utilisé pour le réveil) |
| `/predict` | POST | Score **une** transaction |

Requête — `Time`, `V1`…`V28`, `Amount` en JSON :

```json
{ "Time": 48877.0, "V1": 1.476, "V2": -0.750, "...": "...", "Amount": 16.53 }
```

Réponse :

```json
{ "fraud_probability": 0.0009, "is_fraud": 0, "risk_level": "LOW" }
```

L'API scorant une transaction par requête, le dashboard envoie les 200
transactions en parallèle (8 workers, ~15 s) et met le résultat en cache.

## ⏱️ Note sur le démarrage à froid
L'API est hébergée sur Render (plan gratuit) : elle se met en veille après
15 minutes d'inactivité et met environ **50 secondes** à redémarrer. Le
dashboard réveille l'API avant l'analyse et affiche un message explicite si
elle n'est pas encore disponible.

## 🔍 Ce que fait ce dashboard
- Analyse automatique de 200 transactions bancaires
- Score de risque par transaction (LOW / MEDIUM / HIGH)
- Statistiques globales : fraudes détectées, montant à risque
- Tableau filtrable par niveau de risque

## 🛠️ Stack
Python · XGBoost · Flask · Streamlit · Render

## 📊 Modèle
- Algorithme : XGBoost Classifier
- AUC-ROC : **0.976**
- Recall : **86%** des fraudes détectées

## 📁 Structure

```
streamlit_app.py      # dashboard, client de l'API
data/sample_demo.csv  # 200 transactions d'exemple
requirements.txt
README.md
```

## 🚀 Lancer en local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Pour pointer vers une autre API, définir la variable d'environnement
`FRAUD_API_URL`.

## 📁 Données
Échantillon du dataset public :
[Credit Card Fraud Detection — Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 👤 Auteur
**Alhassane Baldé**  
Étudiant en Master IA & Big Data  
[LinkedIn](https://www.linkedin.com/in/alhassane-bald%C3%A9-0007bb19a/) · 
[GitHub](https://github.com/Balde-ui)
