import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import streamlit as st

# -- Configuration --
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🏦",
    layout="wide"
)

API_URL = os.getenv(
    "FRAUD_API_URL",
    "https://fraud-detection-eh3v.onrender.com"
).rstrip("/")

# Features attendues par l'API (l'ordre n'importe pas, les noms si)
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

MAX_WORKERS = 8        # l'API score une transaction par requête
WAKE_BUDGET = 180      # Render (plan gratuit) met ~50s à sortir de veille
WAKE_ATTEMPT_TIMEOUT = 30   # une tentative de réveil isolée
WAKE_RETRY_DELAY = 3
PREDICT_TIMEOUT = 30

# Render répond en 5xx pendant que le service redémarre : ce n'est pas une
# panne, juste la fin du réveil.
RETRYABLE_STATUS = {500, 502, 503, 504}

RISK_LABELS = {"HIGH": "🔴 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🟢 LOW"}


class ApiUnavailable(Exception):
    """L'API ne répond pas : veille Render, réseau, ou erreur serveur."""


# -- Client API --
_local = threading.local()


def _session():
    """Une session HTTP par thread : connexions réutilisées, rien de partagé."""
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def wake_api():
    """
    Réveille l'API et vérifie qu'elle répond, sinon lève ApiUnavailable.

    On réessaie pendant WAKE_BUDGET secondes plutôt que d'attendre une seule
    réponse : pendant le réveil de Render, un timeout, une coupure réseau
    passagère ou un 502 sont attendus et ne doivent pas faire échouer la page.
    """
    deadline = time.monotonic() + WAKE_BUDGET
    last_error = "l'API n'a pas répondu à temps"

    while True:
        try:
            response = requests.get(
                f"{API_URL}/health", timeout=WAKE_ATTEMPT_TIMEOUT
            )
            if response.status_code in RETRYABLE_STATUS:
                last_error = (
                    f"l'API a répondu avec une erreur HTTP "
                    f"{response.status_code}"
                )
            else:
                # Une 4xx signale une mauvaise URL : inutile de réessayer.
                response.raise_for_status()
                return True
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            raise ApiUnavailable(f"l'API a répondu avec une erreur HTTP {code}.")
        except requests.exceptions.Timeout:
            last_error = "l'API n'a pas répondu à temps"
        except requests.exceptions.ConnectionError:
            last_error = "impossible de joindre l'API (réseau ou DNS)"

        if time.monotonic() >= deadline:
            raise ApiUnavailable(
                f"{last_error}, après {WAKE_BUDGET}s de tentatives. Elle est "
                "probablement encore en train de sortir de veille."
            )
        time.sleep(WAKE_RETRY_DELAY)


def predict_one(payload):
    """Score une transaction. Retourne None si la requête échoue."""
    try:
        response = _session().post(
            f"{API_URL}/predict", json=payload, timeout=PREDICT_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, ValueError):
        return None


# -- Chargement des données --
@st.cache_data
def load_data():
    return pd.read_csv("data/sample_demo.csv")


@st.cache_data(show_spinner=False)
def score_transactions(df, api_url):
    """
    Envoie chaque transaction à l'API et retourne le dataframe enrichi ainsi
    que le nombre de lignes que l'API n'a pas pu scorer.

    api_url ne sert qu'à invalider le cache si l'API change.
    """
    payloads = df[FEATURES].to_dict(orient="records")
    total = len(payloads)
    results = [None] * total

    progress = st.progress(0.0, text="Analyse des transactions par l'API…")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for done, (index, result) in enumerate(
            enumerate(executor.map(predict_one, payloads)), start=1
        ):
            results[index] = result
            progress.progress(
                done / total,
                text=f"Analyse des transactions par l'API… {done}/{total}",
            )
    progress.empty()

    # Une API qui s'arrête en cours de route ne renvoie que des échecs
    if all(result is None for result in results):
        raise ApiUnavailable(
            "aucune transaction n'a pu être scorée, l'API a cessé de répondre "
            "pendant l'analyse."
        )

    scored = df.copy()
    scored["fraud_probability"] = [
        r["fraud_probability"] if r else None for r in results
    ]
    scored["is_fraud_predicted"] = [r["is_fraud"] if r else None for r in results]
    scored["risk_level"] = [
        RISK_LABELS.get(r["risk_level"], r["risk_level"]) if r else None
        for r in results
    ]

    failed = int(scored["fraud_probability"].isna().sum())
    scored = scored.dropna(subset=["fraud_probability"])
    scored["is_fraud_predicted"] = scored["is_fraud_predicted"].astype(int)

    return scored, failed


# ── HEADER ──
st.title("🏦 Fraud Detection System")
st.markdown("Analyse automatique de transactions bancaires par Machine Learning.")
st.divider()

raw_df = load_data()

# ── SCORING VIA L'API (avec gestion de la veille Render) ──
try:
    with st.spinner(
        "Connexion à l'API de scoring… "
        "(elle peut mettre jusqu'à une minute à sortir de veille)"
    ):
        wake_api()
    df, failed = score_transactions(raw_df, API_URL)
except ApiUnavailable as error:
    st.error(f"🔌 **API indisponible** — {error}")
    st.info(
        "L'API est hébergée sur Render (plan gratuit) : elle se met en veille "
        "après 15 minutes d'inactivité et met environ 50 secondes à redémarrer. "
        "Relance l'analyse dans un instant."
    )
    if st.button("🔄 Réessayer"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

if failed:
    st.warning(
        f"⚠️ {failed} transaction(s) n'ont pas pu être scorées par l'API "
        "et sont exclues des statistiques ci-dessous."
    )

# ── KPIs ──
total = len(df)
fraudes = int(df['is_fraud_predicted'].sum())
legit = total - fraudes
montant_risque = df[df['is_fraud_predicted'] == 1]['Amount'].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("📊 Transactions analysées", total)
k2.metric("✅ Légitimes", legit)
k3.metric("🚨 Fraudes détectées", fraudes)
k4.metric("💰 Montant à risque", f"${montant_risque:,.2f}")

st.divider()

# ── GRAPHIQUES ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution des scores de risque")
    hist_data = pd.cut(
        df['fraud_probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['LOW 🟢', 'MEDIUM 🟡', 'HIGH 🔴']
    ).value_counts()
    st.bar_chart(hist_data)

with col2:
    st.subheader("Répartition des transactions")
    pie_data = pd.DataFrame({
        'Type': ['Légitimes ✅', 'Fraudes 🚨'],
        'Nombre': [legit, fraudes]
    }).set_index('Type')
    st.bar_chart(pie_data)

st.divider()

# ── TABLEAU DES TRANSACTIONS ──
st.subheader("📋 Détail des transactions")

# Filtre
filtre = st.selectbox(
    "Filtrer par niveau de risque :",
    ["Toutes", "🔴 HIGH", "🟡 MEDIUM", "🟢 LOW"]
)

display_df = df.copy()
if filtre != "Toutes":
    display_df = display_df[display_df['risk_level'] == filtre]

# Colonnes à afficher
display_df = display_df[[
    'Amount', 'fraud_probability', 'risk_level', 'is_fraud_predicted'
]].rename(columns={
    'Amount': 'Montant ($)',
    'fraud_probability': 'Probabilité de fraude',
    'risk_level': 'Niveau de risque',
    'is_fraud_predicted': 'Fraude détectée'
})

display_df['Probabilité de fraude'] = (
    display_df['Probabilité de fraude'] * 100
).round(2).astype(str) + '%'

display_df['Fraude détectée'] = display_df['Fraude détectée'].apply(
    lambda x: "🚨 OUI" if x == 1 else "✅ NON"
)

st.dataframe(display_df, width='stretch')

st.divider()

# ── FOOTER ──
st.caption(
    "Données : échantillon du dataset public "
    "[Credit Card Fraud Detection — Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · "
    "Modèle : XGBoost · AUC-ROC : 0.976 · "
    f"Scoring temps réel via [l'API de prédiction]({API_URL})"
)
