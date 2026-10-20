"""
Dashboard de démonstration : client de l'API de scoring de fraude.

Le dépôt ne contient aucun modèle. Les transactions de data/sample_demo.csv sont
envoyées à l'API (POST /predict/batch, 100 transactions par requête), qui renvoie
pour chacune la probabilité, la décision et le niveau de risque. Aucun seuil n'est
codé ici : is_fraud, risk_level et threshold_used viennent de l'API.

Configuration (secrets Streamlit Cloud, ou variables d'environnement en local) :
  FRAUD_API_URL   URL de l'API (défaut : l'API publique sur Render)
  FRAUD_API_KEY   clé envoyée dans l'en-tête X-API-Key (jamais commitée)
  DEMO_OFFLINE    "1" : mode hors ligne, en lecture seule, à partir des réponses
                  précalculées de data/sample_scored.csv (aucun appel à l'API)
"""

import os
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Fraud Detection System", page_icon="🏦", layout="wide")


def setting(name, default=""):
    """Lit un réglage dans les secrets Streamlit, sinon dans l'environnement."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # pas de fichier secrets.toml en local : ce n'est pas une erreur
        pass
    return os.getenv(name, default)


API_URL = setting("FRAUD_API_URL", "https://fraud-detection-eh3v.onrender.com").rstrip("/")
API_KEY = setting("FRAUD_API_KEY")
OFFLINE = setting("DEMO_OFFLINE", "0") == "1"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
BATCH_SIZE = 100          # maximum accepté par /predict/batch
WAKE_BUDGET = 180         # Render (offre gratuite) met environ 50 s à sortir de veille
WAKE_ATTEMPT_TIMEOUT = 30
WAKE_RETRY_DELAY = 3
REQUEST_TIMEOUT = 60
FP_COST = 5.0             # coût d'une fausse alerte, même convention que le rapport d'évaluation

# Render répond en 5xx pendant que le service redémarre : ce n'est pas une panne
RETRYABLE_STATUS = {500, 502, 503, 504}
RISK_LABELS = {"HIGH": "🔴 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🟢 LOW"}


class ApiUnavailable(Exception):
    """L'API ne répond pas : veille Render, réseau, ou erreur serveur."""


class ApiRefused(Exception):
    """L'API répond mais refuse la requête : clé absente ou invalide, quota dépassé."""


def headers():
    return {"X-API-Key": API_KEY} if API_KEY else {}


def wake_api():
    """
    Réveille l'API (route publique /health) en réessayant pendant WAKE_BUDGET secondes :
    pendant le réveil de Render, un timeout, une coupure ou un 502 sont attendus.
    """
    deadline = time.monotonic() + WAKE_BUDGET
    last_error = "l'API n'a pas répondu à temps"
    while True:
        try:
            r = requests.get(f"{API_URL}/health", timeout=WAKE_ATTEMPT_TIMEOUT)
            if r.status_code == 200:
                return
            if r.status_code not in RETRYABLE_STATUS:
                raise ApiUnavailable(f"/health a répondu avec le code HTTP {r.status_code}.")
            last_error = f"l'API a répondu avec le code HTTP {r.status_code}"
        except requests.exceptions.Timeout:
            last_error = "l'API n'a pas répondu à temps"
        except requests.exceptions.ConnectionError:
            last_error = "impossible de joindre l'API (réseau ou DNS)"
        if time.monotonic() >= deadline:
            raise ApiUnavailable(f"{last_error}, après {WAKE_BUDGET} s de tentatives.")
        time.sleep(WAKE_RETRY_DELAY)


def check_response(r):
    if r.status_code == 401:
        raise ApiRefused("clé d'API absente ou invalide : vérifier le secret FRAUD_API_KEY.")
    if r.status_code == 429:
        raise ApiRefused("trop de requêtes vers l'API pour le moment, réessayer dans une minute.")
    if r.status_code == 503 and "configur" in r.text:
        raise ApiRefused("l'API n'a pas de clé configurée côté serveur (variable API_KEY sur Render).")
    if r.status_code in RETRYABLE_STATUS:
        raise ApiUnavailable(f"l'API a répondu avec le code HTTP {r.status_code}.")
    r.raise_for_status()


def score_batch(rows):
    """Score une liste de transactions. Repli sur /predict si l'API ne connaît pas encore /predict/batch."""
    r = requests.post(f"{API_URL}/predict/batch", json={"transactions": rows},
                      headers=headers(), timeout=REQUEST_TIMEOUT)
    if r.status_code == 404:  # ancienne version de l'API, pendant la transition
        out = []
        for row in rows:
            r1 = requests.post(f"{API_URL}/predict", json=row, headers=headers(), timeout=REQUEST_TIMEOUT)
            check_response(r1)
            out.append(r1.json())
        return out
    check_response(r)
    return r.json()["results"]


@st.cache_data
def load_sample():
    return pd.read_csv("data/sample_demo.csv")


@st.cache_data
def load_offline():
    return pd.read_csv("data/sample_scored.csv")


@st.cache_data(show_spinner=False, ttl=3600)
def score_with_api(df, api_url):
    """Envoie l'échantillon à l'API par lots de 100. api_url ne sert qu'à invalider le cache."""
    rows = df[FEATURES].to_dict(orient="records")
    results = []
    progress = st.progress(0.0, text="Analyse des transactions par l'API…")
    for start in range(0, len(rows), BATCH_SIZE):
        results.extend(score_batch(rows[start:start + BATCH_SIZE]))
        progress.progress(len(results) / len(rows), text=f"Analyse par l'API… {len(results)}/{len(rows)}")
    progress.empty()

    scored = df.copy()
    scored["fraud_probability"] = [r["fraud_probability"] for r in results]
    scored["is_fraud"] = [int(r["is_fraud"]) for r in results]
    scored["risk_level"] = [r["risk_level"] for r in results]
    # Champs ajoutés par l'API V3 ; absents de l'ancienne API pendant la transition
    scored["model_version"] = [r.get("model_version", "inconnue") for r in results]
    scored["threshold_used"] = [r.get("threshold_used", np.nan) for r in results]
    scored["top_features"] = [
        ", ".join(f"{t['feature']} ({t['contribution']:+.2f})" for t in r.get("top_features", []))
        for r in results
    ]
    return scored


def metrics_at(df, threshold):
    """Recall, precision, fausses alertes et coût sur l'échantillon, pour un seuil donné."""
    alert = df["fraud_probability"] >= threshold
    fraud = df["Class"] == 1
    tp, fp = int((alert & fraud).sum()), int((alert & ~fraud).sum())
    fn = int((~alert & fraud).sum())
    cost = float(df.loc[~alert & fraud, "Amount"].sum() + FP_COST * fp)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "cost": cost,
    }


# ── EN-TÊTE ──────────────────────────────────────────────────────────────────
st.title("🏦 Fraud Detection System")
st.markdown("Analyse automatique de transactions bancaires par Machine Learning.")

# ── DONNÉES ET SCORING ───────────────────────────────────────────────────────
if OFFLINE or st.session_state.get("offline"):
    df = load_offline()
    df["top_features"] = ""
    st.warning(
        "📴 **Mode hors ligne, lecture seule** : réponses précalculées par le modèle "
        f"{df['model_version'].iloc[0]}, sans appel à l'API. Le curseur de seuil reste utilisable."
    )
else:
    try:
        with st.spinner("Connexion à l'API de scoring… elle peut mettre jusqu'à une minute "
                        "à sortir de veille (hébergement gratuit Render)."):
            wake_api()
        df = score_with_api(load_sample(), API_URL)
    except (ApiUnavailable, ApiRefused, requests.exceptions.RequestException) as error:
        if isinstance(error, ApiRefused):
            st.error(f"🔒 **Accès refusé par l'API** : {error}")
        else:
            st.error(f"🔌 **API indisponible** : {error}")
            st.info("L'API est hébergée sur Render (offre gratuite) : elle se met en veille après "
                    "15 minutes d'inactivité et met environ 50 secondes à redémarrer.")
        c1, c2 = st.columns(2)
        if c1.button("🔄 Réessayer"):
            st.cache_data.clear()
            st.rerun()
        if c2.button("📴 Voir les résultats précalculés (lecture seule)"):
            st.session_state["offline"] = True
            st.rerun()
        st.stop()

model_version = df["model_version"].iloc[0]
api_threshold = df["threshold_used"].dropna().iloc[0] if df["threshold_used"].notna().any() else 0.5
n_fraud, n_total = int(df["Class"].sum()), len(df)

st.info(
    f"**Échantillon** : {n_total} transactions tirées du **jeu de test** (dernières heures du dataset), "
    f"jamais vues par le modèle à l'entraînement : {n_total - n_fraud} légitimes et {n_fraud} fraudes, "
    f"soit {n_fraud / n_total:.0%} de fraudes. Dans la réalité du dataset, c'est environ 0,12 % : "
    "la proportion est gonflée ici pour avoir des fraudes à montrer, ce qui rend la precision "
    "de cet échantillon bien plus flatteuse qu'en conditions réelles."
)
st.divider()

# ── INDICATEURS (décisions de l'API) ─────────────────────────────────────────
alerts = int(df["is_fraud"].sum())
caught = int(((df["is_fraud"] == 1) & (df["Class"] == 1)).sum())
k1, k2, k3, k4 = st.columns(4)
k1.metric("📊 Transactions analysées", n_total)
k2.metric("🚨 Alertes émises par l'API", alerts)
k3.metric("✅ Fraudes réelles détectées", f"{caught} / {n_fraud}")
k4.metric("💰 Montant bloqué", f"{df.loc[df['is_fraud'] == 1, 'Amount'].sum():,.2f}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Niveaux de risque renvoyés par l'API")
    levels = df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
    levels.index = [RISK_LABELS[i] for i in levels.index]
    st.bar_chart(levels)
with col2:
    st.subheader("Matrice de confusion de l'échantillon")
    tp = caught
    fp = alerts - caught
    fn = n_fraud - caught
    tn = n_total - n_fraud - fp
    st.table(pd.DataFrame(
        {"Prédit légitime": [tn, fn], "Prédit fraude": [fp, tp]},
        index=["Réel légitime", "Réel fraude"],
    ))
    st.caption(f"Décision de l'API : fraude si probabilité ≥ {api_threshold:.3f} (seuil renvoyé par l'API, "
               f"modèle {model_version}).")

st.divider()

# ── CURSEUR DE SEUIL ─────────────────────────────────────────────────────────
st.subheader("🎚️ Et avec un autre seuil ?")
st.markdown(
    "Le seuil de l'API a été choisi sur des données de validation pour minimiser un coût : "
    f"montant des fraudes manquées + {FP_COST:.0f} par fausse alerte. Déplacez le curseur pour voir "
    "le compromis sur cet échantillon : un seuil bas attrape plus de fraudes, mais crée plus de "
    "fausses alertes."
)
grid = sorted({round(float(x), 4) for x in np.geomspace(1e-4, 0.99, 40)} | {round(float(api_threshold), 4)})
threshold = st.select_slider("Seuil de décision sur la probabilité", options=grid,
                             value=round(float(api_threshold), 4),
                             format_func=lambda x: f"{x:.4f}")
m, ref = metrics_at(df, threshold), metrics_at(df, api_threshold)
s1, s2, s3, s4 = st.columns(4)
s1.metric("Recall (fraudes trouvées)", f"{m['recall']:.0%}", f"{m['recall'] - ref['recall']:+.0%}")
s2.metric("Precision (alertes justifiées)",
          "—" if np.isnan(m["precision"]) else f"{m['precision']:.0%}")
s3.metric("Fausses alertes", m["fp"], m["fp"] - ref["fp"], delta_color="inverse")
s4.metric("Coût sur l'échantillon", f"{m['cost']:,.0f}", f"{m['cost'] - ref['cost']:+,.0f}",
          delta_color="inverse")
st.caption("Écarts calculés par rapport au seuil de l'API. Coût exprimé dans l'unité des montants "
           "du dataset (non précisée par l'éditeur).")

st.divider()

# ── TABLEAU ──────────────────────────────────────────────────────────────────
st.subheader("📋 Détail des transactions")
filtre = st.selectbox("Filtrer par niveau de risque :", ["Toutes", "HIGH", "MEDIUM", "LOW"])
view = df if filtre == "Toutes" else df[df["risk_level"] == filtre]
table = pd.DataFrame({
    "Montant": view["Amount"].round(2),
    "Probabilité de fraude": (view["fraud_probability"] * 100).round(3).astype(str) + " %",
    "Niveau de risque": view["risk_level"].map(RISK_LABELS),
    "Décision de l'API": view["is_fraud"].map({1: "🚨 Alerte", 0: "✅ Acceptée"}),
    "Réalité": view["Class"].map({1: "Fraude", 0: "Légitime"}),
    "Variables les plus influentes": view["top_features"],
})
st.dataframe(table, width="stretch", hide_index=True)

st.divider()
st.caption(
    "Données : échantillon du jeu de test du dataset public "
    "[Credit Card Fraud Detection, Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). "
    f"Modèle XGBoost {model_version}, servi par [l'API]({API_URL}/apidocs/). "
    "Sur le jeu de test complet (52 fraudes), il retrouve 75 % des fraudes avec 75 % d'alertes justifiées."
)
