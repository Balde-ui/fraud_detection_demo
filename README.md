# 🏦 Fraud Detection : démo

![Python](https://img.shields.io/badge/Python-3.14-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red)
![API](https://img.shields.io/badge/API-Flask%20%2F%20Render-green)

Dashboard de démonstration d'un système de détection de fraude bancaire.

## 🌐 Démo en ligne
👉 **[Accéder au dashboard](https://frauddetectiondemo-zriwmvzpi7wa3gy2vnxugu.streamlit.app/)**

## 🏗️ Architecture

Ce dépôt ne contient **aucun modèle** : le dashboard est un client d'une API de scoring déployée séparément.

```
┌──────────────────────┐  POST /predict/batch (X-API-Key)  ┌────────────────────────┐
│  Dashboard Streamlit │ ────────────────────────────────► │  API Flask (Render)    │
│  (ce dépôt)          │ ◄──────────────────────────────── │  modèle XGBoost V3     │
└──────────────────────┘  probabilité, décision, niveau,   └────────────────────────┘
                          seuil utilisé, variables clés
```

**API** : `https://fraud-detection-eh3v.onrender.com` (documentation interactive sur `/apidocs/`)

| Route | Méthode | Accès | Rôle |
|---|---|---|---|
| `/health` | GET | public | État du service (sert aussi à le réveiller) |
| `/predict` | POST | clé d'API | Score une transaction |
| `/predict/batch` | POST | clé d'API | Score de 1 à 100 transactions |

Exemple de réponse pour une transaction :

```json
{
  "is_fraud": 1,
  "fraud_probability": 0.999467,
  "risk_level": "HIGH",
  "model_version": "3.0.0",
  "threshold_used": 0.174897,
  "top_features": [{"feature": "V14", "contribution": 3.2652}, {"feature": "V17", "contribution": 1.7438}, {"feature": "V10", "contribution": 1.0897}]
}
```

Le dashboard ne contient aucun seuil : la décision (`is_fraud`), le niveau de risque et le seuil utilisé viennent de l'API.

## 🔍 Ce que montre le dashboard
- **200 transactions tirées du jeu de test** (les dernières heures du dataset), jamais vues par le modèle : 180 légitimes et 20 fraudes. La proportion de fraudes (10 %) est volontairement gonflée par rapport à la réalité du dataset (0,12 %) pour avoir des cas à montrer ; le dashboard l'indique à l'écran.
- Les décisions de l'API et la **matrice de confusion** de l'échantillon.
- Un **curseur de seuil** pour voir en direct le compromis recall / precision / coût (fraudes manquées + 5 par fausse alerte).
- Pour chaque transaction, les trois variables qui pèsent le plus sur le score.

## 📊 Modèle
XGBoost V3, évalué sur un jeu de test temporel (les 6 dernières heures, 52 fraudes) jamais utilisé pour les choix :
recall 0,75 [0,62 ; 0,87], precision 0,75 [0,63 ; 0,87] (intervalles de confiance à 95 % par bootstrap).
Méthode et limites : voir le rapport d'évaluation du dépôt principal.

## ⏱️ Démarrage à froid
L'API est hébergée sur l'offre gratuite de Render : elle se met en veille après 15 minutes sans requête et met environ une minute à redémarrer. Le dashboard la réveille avant l'analyse et affiche un message pendant l'attente. En cas d'échec, il propose d'afficher des **résultats précalculés en lecture seule**.

## 🚀 Lancer en local

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # puis renseigner la clé
streamlit run streamlit_app.py
```

Réglages (secrets Streamlit ou variables d'environnement) :

| Nom | Rôle |
|---|---|
| `FRAUD_API_URL` | URL de l'API (défaut : l'API publique) |
| `FRAUD_API_KEY` | clé d'API envoyée dans l'en-tête `X-API-Key` (**ne jamais la commiter**) |
| `DEMO_OFFLINE` | `1` : mode hors ligne, lecture seule, à partir de `data/sample_scored.csv` |

## 📁 Structure

```
streamlit_app.py           # dashboard, client de l'API
data/sample_demo.csv       # 200 transactions du jeu de test (avec la vraie classe)
data/sample_scored.csv     # mêmes transactions + réponses précalculées (mode hors ligne)
.streamlit/secrets.toml.example
requirements.txt
```

## 📁 Données
Échantillon du jeu de test du dataset public
[Credit Card Fraud Detection, Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## 👤 Auteur
**Alhassane Baldé**
Étudiant en Master IA & Big Data
[LinkedIn](https://www.linkedin.com/in/alhassane-bald%C3%A9-0007bb19a/) ·
[GitHub](https://github.com/Balde-ui)
