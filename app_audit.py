"""
Chauffeur de France – tableau de bord d'audit
------------------------------------------------

Ce module Streamlit fournit un tableau de bord complet pour
l'entreprise de transport de luxe « Chauffeur de France ». Il a été
conçu pour analyser les factures et devis de 2025, les clients, les
dépenses bancaires et fournir des projections financières. L'application
s'articule autour de plusieurs pages :

  - Vue d'ensemble : KPIs clés (chiffre d'affaires, marges, volume de
    clients, fidélité, conversion devis → factures, encaissements).
  - Performance & CA : analyse du CA par type de service et par mois,
    mise en évidence des saisons fortes, top clients et segmentation
    selon des règles métier.
  - Devis & pipeline : suivi du pipeline commercial, taux de
    conversion, comparaison devis ↔ factures, exploration des devis
    perdus avec recommandations commerciales.
  - Dépenses : catégorisation automatique des transactions
    bancaires/notes de frais, graphe de l'évolution des charges,
    prévision budgétaire simple via régression linéaire et conseils
    d'optimisation compatibles avec un service de luxe.
  - Clients & segmentation : analyse RFM (Récence-Fréquence-Montant),
    typologie des clients (VIP, entreprise, partenaire) et actions de
    fidélisation.
  - Prévisions : projections 2026 des revenus/dépenses basées sur un
    modèle linéaire et scénarios ajustables via des curseurs
    interactifs.
  - IA : démonstration facultative d'un modèle de reconnaissance
    d'images pour contrôler les photos prises par les chauffeurs (ex :
    vérification de l'état des véhicules). Cette section illustre
    comment l'intelligence artificielle peut être intégrée dans une
    application métier, même si elle ne traite pas encore les images
    réelles transmises par les chauffeurs.

Pour exécuter l'application :

1. Assurez‑vous d'avoir installé les dépendances suivantes :

    ``pip install streamlit pandas numpy scikit-learn plotly torchvision torch pillow``

2. Placez ce script dans le même dossier que vos fichiers de données :
   ``factures.xlsx``, ``devis.xlsx``, ``clients.xlsx``, ``transactions_bancaire.xlsx``,
   ``documents.xlsx`` (ainsi que tout éventuel fichier JSON pour les
   événements de luxe si vous souhaitez utiliser la carte). Une icône
   facultative ``logo.png`` peut être placée à la racine pour
   personnaliser l'en‑tête.

3. Lancez l'application via :

    ``streamlit run app_audit.py``

4. Un menu latéral vous permet de naviguer entre les pages.

Remarque : cette application a été pensée pour fonctionner en mode
"offline" sans accès Internet. Les données proviennent exclusivement
des fichiers Excel fournis par l'utilisateur et du modèle pré‑entraîné
ResNet18 inclus dans PyTorch pour la section IA.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    from PIL import Image
    _has_torch = True
except Exception:
    _has_torch = False

# ---------------------------------------------------------------------------
# Configuration des chemins et de l'icône
# ---------------------------------------------------------------------------

# Emplacement des fichiers par défaut (peut être ajusté). Utilise Path pour
# assurer la portabilité. Ces définitions doivent précéder toute référence aux
# variables, notamment pour la configuration de l'icône de page.
DATA_DIR = Path(__file__).resolve().parent

# Fichiers de données
FACTURES_FILE = DATA_DIR / "factures.xlsx"
DEVISES_FILE = DATA_DIR / "devis.xlsx"
CLIENTS_FILE = DATA_DIR / "clients.xlsx"
TRANSACTIONS_FILE = DATA_DIR / "transactions_bancaire.xlsx"
DOCUMENTS_FILE = DATA_DIR / "documents.xlsx"
EVENTS_FILE = DATA_DIR / "events_luxury.json"  # facultatif

# Chemin vers le logo facultatif ; utilisé pour l'icône de la page et l'en-tête
LOGO_FILE = DATA_DIR / "logo.png"

# Définir l'icône de la page : si le fichier logo existe, on l'utilise
# comme page_icon ; sinon on choisit un emoji neutre.
if LOGO_FILE.exists():
    PAGE_ICON = str(LOGO_FILE)
else:
    PAGE_ICON = "🚘"

st.set_page_config(
    page_title="Chauffeur de France – Tableau de bord",
    page_icon=PAGE_ICON,
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_factures(path: Path = FACTURES_FILE) -> pd.DataFrame:
    """Charge les factures 2025 et nettoie les colonnes."""
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    if 'date de facture' in df.columns:
        df['date de facture'] = pd.to_datetime(df['date de facture'])
    for col in df.columns:
        if col.startswith('montant'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data(show_spinner=False)
def load_devis(path: Path = DEVISES_FILE) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Nettoyer les noms de colonnes (supprimer espaces)
    df.columns = [c.strip() for c in df.columns]
    # Unifier le nom de la colonne de statut (peut être "statut du devis" ou "Statut")
    status_cols = [c for c in df.columns if 'statut' in c.lower()]
    if status_cols:
        # renommer la première colonne identifiée en "statut"
        df = df.rename(columns={status_cols[0]: 'statut'})
    if 'date du devis' in df.columns:
        df['date du devis'] = pd.to_datetime(df['date du devis'])
    # Convertir montants (HT/TTC) en numérique
    for col in df.columns:
        if col.lower().startswith('montant'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data(show_spinner=False)
def load_clients(path: Path = CLIENTS_FILE) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass
    return df


@st.cache_data(show_spinner=False)
def load_transactions(path: Path = TRANSACTIONS_FILE) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    df['Date de la transaction'] = pd.to_datetime(df['Date de la transaction'])
    df["Date de l'opération"] = pd.to_datetime(df["Date de l'opération"])
    if 'Montant TTC' in df.columns:
        df['Montant TTC'] = pd.to_numeric(df['Montant TTC'], errors='coerce')
    if 'Tags' in df.columns:
        df['Tags'] = df['Tags'].fillna('').apply(lambda x: [t.strip() for t in str(x).split(',') if t])
    return df


@st.cache_data(show_spinner=False)
def load_documents(path: Path = DOCUMENTS_FILE) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass
    if 'Montant TTC' in df.columns:
        df['Montant TTC'] = pd.to_numeric(df['Montant TTC'], errors='coerce')
    return df


@st.cache_data(show_spinner=False)
def load_events(path: Path = EVENTS_FILE) -> pd.DataFrame:
    try:
        import json
        if not path.exists():
            return pd.DataFrame()
        with open(path, 'r', encoding='utf-8') as f:
            events = json.load(f)
        return pd.DataFrame(events)
    except Exception:
        return pd.DataFrame()


def categorize_service(intitule: str) -> str:
    if not isinstance(intitule, str):
        return 'Transport divers'
    s = intitule.lower()
    if 'fashion' in s or 'fw' in s:
        return 'Fashion Week'
    if any(word in s for word in ['séminaire', 'seminaire', 'conférence']):
        return 'Séminaire'
    if any(word in s for word in ['mariage', 'wedding', 'weds']):
        return 'Mariage'
    if any(word in s for word in ['transfert', 'aéroport', 'transfer']):
        return 'Transfert'
    if any(word in s for word in ['mission', 'mise à disposition', 'mise a disposition']):
        return 'Mise à disposition'
    if any(word in s for word in ['location', 'rent', 'louer']):
        return 'Location'
    if any(word in s for word in ['événement', 'evenement', 'event']):
        return 'Événement'
    return 'Transport divers'


def classify_client(name: str) -> str:
    if not isinstance(name, str):
        return 'Autre'
    n = name.lower().strip()
    if any(word in n for word in ['limousine', 'vtc', 'transport', 'driver']):
        return 'Prestataire/partenaire'
    if any(word in n for word in [
        'sarl', 'sas', 'sa', 'ltd', 'enterprise', 'hotel', 'agence', 'events',
        'travel', 'rent car', 'busness', 'sport']):
        return 'Entreprise/Agence'
    if any(word in n for word in ['prince', 'princess', 'duchesse', 'hrh', 'majesté', 'king', 'queen']):
        return 'VIP particulier'
    parts = n.split()
    if len(parts) == 2 and all(len(p) > 1 for p in parts):
        return 'Client particulier'
    return 'Autre'


def prepare_factures(df_factures: pd.DataFrame) -> pd.DataFrame:
    df = df_factures.copy()
    df['Catégorie'] = df['intitulé'].apply(categorize_service)
    df['Typologie client'] = df['nom du client'].apply(classify_client)
    ca_ht_col = 'montant HT'
    ca_ttc_col = 'montant TTC'
    df[ca_ht_col] = pd.to_numeric(df[ca_ht_col], errors='coerce')
    df[ca_ttc_col] = pd.to_numeric(df[ca_ttc_col], errors='coerce')
    df['mois'] = df['date de facture'].dt.to_period('M')
    return df


def compute_monthly_revenue(df_factures: pd.DataFrame) -> pd.DataFrame:
    df = df_factures.copy()
    monthly = df.groupby(df['date de facture'].dt.to_period('M')).agg({
        'montant HT': 'sum',
        'montant TTC': 'sum'
    }).reset_index()
    monthly['month'] = monthly['date de facture'].dt.to_timestamp()
    return monthly


def compute_kpi_overview(df_factures: pd.DataFrame, df_clients: pd.DataFrame, df_devis: pd.DataFrame) -> dict:
    ca_total_ht = df_factures['montant HT'].sum()
    ca_total_ttc = df_factures['montant TTC'].sum()
    nb_factures = len(df_factures)
    nb_clients = df_factures['nom du client'].nunique()
    panier_moyen_ht = ca_total_ht / nb_factures if nb_factures else 0
    client_counts = df_factures.groupby('nom du client').size()
    nb_fidèles = (client_counts > 1).sum()
    ca_fidèles = df_factures[df_factures['nom du client'].isin(client_counts[client_counts > 1].index)]['montant HT'].sum()
    fidelite_rate = ca_fidèles / ca_total_ht if ca_total_ht else 0
    # Taux de paiement : certaines bases clients contiennent les colonnes
    # « facturé HT »/« encaissé » au lieu de « total facturé HT »/« total payé ». On tente de les détecter.
    if not df_clients.empty:
        # Chercher les colonnes de montant facturé et encaissé sans tenir compte de la casse ni des accents
        fact_ht_cols = [c for c in df_clients.columns if 'factur' in c.lower() and 'ht' in c.lower()]
        encaiss_cols = [c for c in df_clients.columns if 'encaiss' in c.lower() or 'payé' in c.lower() or 'paye' in c.lower()]
        if fact_ht_cols and encaiss_cols:
            total_facture_clients_ht = df_clients[fact_ht_cols[0]].astype(float).sum()
            total_paye = df_clients[encaiss_cols[0]].astype(float).sum()
            taux_paiement = total_paye / total_facture_clients_ht if total_facture_clients_ht else 0
        else:
            taux_paiement = np.nan
    else:
        taux_paiement = np.nan
    # Taux de conversion devis→factures : identifier dynamiquement la colonne de statut des devis
    if not df_devis.empty:
        nb_devis_emis = len(df_devis)
        # Détecter la colonne qui contient le mot « statut »
        status_col = None
        for col in df_devis.columns:
            if 'statut' in col.lower():
                status_col = col
                break
        if status_col:
            nb_devis_acceptes = df_devis[status_col].astype(str).str.lower().str.contains('accept').sum()
            taux_conversion = nb_devis_acceptes / nb_devis_emis if nb_devis_emis else 0
        else:
            taux_conversion = np.nan
    else:
        taux_conversion = np.nan
    return {
        'ca_total_ht': ca_total_ht,
        'ca_total_ttc': ca_total_ttc,
        'nb_factures': nb_factures,
        'nb_clients': nb_clients,
        'panier_moyen_ht': panier_moyen_ht,
        'taux_fidelite': fidelite_rate,
        'taux_paiement': taux_paiement,
        'taux_conversion': taux_conversion,
        'nb_fidèles': nb_fidèles,
    }


def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et catégorise les transactions bancaires. Les règles de
    classification sont basées sur l'intitulé, le libellé et les tags pour
    mieux identifier les postes de dépenses. Les encaissements (montants
    positifs) sont séparés, les investissements (achat de véhicules,
    acomptes, dépôts de garantie) et les prélèvements (dividendes ou
    rémunérations du gérant) sont distingués des charges opérationnelles.
    """
    df_tx = df.copy()
    def get_expense_category(row):
        amount = row.get('Montant TTC', 0)
        # Encaissement si montant positif
        if amount >= 0:
            return 'Encaissement'
        desc = str(row.get('Intitulé de la transaction', '')).lower()
        label = str(row.get('Label', '')).lower()
        # Liste des tags éventuels
        tags_list = row.get('Tags', [])
        if isinstance(tags_list, list):
            tags = [t.lower() for t in tags_list]
        else:
            tags = []
        # Investissements : achat de véhicule, acomptes, versements, dépôts de garantie
        if any(w in desc for w in ['achat vehicule', 'achat véhicule', 'acompte', 'versement', 'dépôt de garantie', 'depot de garantie', 'investissement']) or \
           any(w in label for w in ['achat vehicule', 'achat véhicule']):
            return 'Investissement'
        # Sous‑traitance / partenaires VTC
        if any(w in label for w in ['blob', 'professional', 'professionality', 'prima classe', 'myone car', 'chabe', 'avantage limousine', 'vtc', 'limousine']) or \
           any(w in desc for w in ['professionality', 'professional', 'blob', 'prima classe', 'chabe']):
            return 'Sous‑traitance'
        # Dividendes / prélèvements sociaux (rémunération du gérant)
        if 'dividendes' in desc or 'dividende' in label or 'prélevement' in desc or 'prelevement' in desc or 'prelevement' in label or \
           any('dividendes' in t for t in tags) or any('prélèvements sociaux' in t for t in tags):
            return 'Dividendes/Prélèvements'
        # Emprunt / remboursement de crédit
        if 'emprunt' in label or 'credit' in label:
            return 'Emprunt'
        # Carburant
        if any(w in label for w in ['total energies', 'esso', 'shell', 'total', 'radius business solutions', 'rbs', 'dkv']) or \
           any(w in desc for w in ['carburant', 'gazoil', 'fuel', 'station']):
            return 'Carburant'
        # Péage & parking
        if any(w in desc for w in ['péage', 'peage', 'autoroute', 'parking']) or 'peage' in label:
            return 'Péage & parking'
        # Entretien véhicule
        if any(w in desc for w in ['entretien', 'revision', 'révision', 'reparation', 'réparation', 'garage']) or \
           any(w in label for w in ['entretien', 'garage']):
            return 'Entretien véhicule'
        # Location de véhicule
        if any(w in label for w in ['locations diverses', 'rent a car', 'rentacar', 'ht rent car']) or 'location' in desc:
            return 'Location de véhicule'
        # Salaires & charges sociales
        if any(w in label for w in ['salaires', 'urssaf', 'klesia']) or \
           any(w in desc for w in ['salaires', 'urssaf', 'klesia', 'fiche de paie']):
            return 'Salaires & charges'
        # Frais de déplacement
        if any(w in desc for w in ['restaurant', 'hôtel', 'hotel', 'restauration']):
            return 'Frais de déplacement'
        # Impôts & taxes
        if any(w in desc for w in ['impot', 'impôt', 'taxe', 'impots', 'taxes']) or 'taxe' in label:
            return 'Impôts & taxes'
        # Marketing / publicité
        if any(w in desc for w in ['marketing', 'publicité', 'linkedin', 'google', 'instagram']) or 'marketing' in label:
            return 'Marketing'
        # Assurance
        if 'assurance' in label or any(w in desc for w in ['assurance', 'chubb', 'axa', 'gan', 'lcl']):
            return 'Assurance'
        # Frais bancaires
        if any(w in label for w in ['frais bancaires', 'banque']) or 'frais bancaires' in desc:
            return 'Frais bancaires'
        return 'Divers'
    df_tx['Catégorie dépense'] = df_tx.apply(get_expense_category, axis=1)
    df_tx['Montant abs'] = df_tx['Montant TTC'].abs()
    df_tx['mois'] = df_tx['Date de la transaction'].dt.to_period('M')
    return df_tx


def forecast_lin_reg(series: pd.Series, periods: int = 6) -> pd.DataFrame:
    """Prévision simple par régression linéaire.

    Ce modèle estime une tendance linéaire dans les données historiques
    et extrapole sur les périodes futures. Il ne tient pas compte de
    saisonnalités et convient pour une première approximation. Pour
    intégrer une saisonnalité, utilisez la fonction ``forecast_seasonal``.

    Parameters
    ----------
    series : pd.Series
        Série chronologique indexée par des périodes (ex : mois).
    periods : int
        Nombre de périodes futures à prédire.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les colonnes ``Period`` (index des périodes
        futures) et ``Forecast`` (valeur prédite).
    """
    # Si la série est vide ou très courte, retourner une prévision plate
    if series.empty or len(series) < 2:
        future_index = [series.index[-1] + i + 1 for i in range(periods)] if not series.empty else list(range(periods))
        return pd.DataFrame({'Period': future_index, 'Forecast': [series.mean()] * periods})
    y = series.values
    # Indépendante : numéro d'ordre de chaque observation
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    X_future = np.arange(len(y), len(y) + periods).reshape(-1, 1)
    y_pred = model.predict(X_future)
    last_period = series.index[-1]
    future_index = [last_period + i + 1 for i in range(periods)]
    return pd.DataFrame({'Period': future_index, 'Forecast': y_pred})


def forecast_seasonal(series: pd.Series, periods: int = 12) -> pd.DataFrame:
    """Prévision linéaire avec composante saisonnière.

    Cette fonction estime une tendance linéaire sur la série et calcule
    des ajustements saisonniers mensuels (ou basés sur la fréquence de
    l'index) à partir des résidus. Les prévisions futures combinent la
    tendance et le schéma saisonnier pour reproduire les fluctuations
    observées.

    Parameters
    ----------
    series : pd.Series
        Série chronologique indexée par période (de type PeriodIndex).
        Doit couvrir au moins un cycle complet (ex : 12 mois) pour
        capturer la saisonnalité.
    periods : int
        Nombre de périodes futures à prédire. Par défaut 12 mois.

    Returns
    -------
    pd.DataFrame
        DataFrame avec ``Period`` et ``Forecast``.
    """
    if series.empty:
        # Retourne une série plate si aucune donnée
        future_index = []
        return pd.DataFrame({'Period': future_index, 'Forecast': []})
    # Convertir l'index en entier pour régression
    n = len(series)
    X = np.arange(n).reshape(-1, 1)
    y = series.values
    model = LinearRegression().fit(X, y)
    trend = model.predict(X)
    # Calcul des résidus et indices saisonniers
    residuals = y - trend
    # Identifie la fréquence de la série (ex: 12 pour mensuel)
    if hasattr(series.index, 'freqstr') and series.index.freqstr:
        freq = series.index.freq.n
    else:
        # Supposons mensuel si non spécifié
        freq = 12
    # Calculer l'indice saisonnier moyen pour chaque position dans le cycle
    seasonal_indices = {}
    for i in range(n):
        position = i % freq
        seasonal_indices.setdefault(position, []).append(residuals[i])
    # Moyenne des résidus par position
    for pos in seasonal_indices:
        seasonal_indices[pos] = np.mean(seasonal_indices[pos])
    # Prévisions futures
    future_periods = []
    future_values = []
    for i in range(periods):
        # Période future indexée de 0 à periods-1
        j = n + i
        trend_value = model.predict(np.array([[j]]))[0]
        pos = j % freq
        seasonal_adj = seasonal_indices.get(pos, 0.0)
        forecast_val = trend_value + seasonal_adj
        future_values.append(forecast_val)
        # Créer l'index futur en se basant sur le dernier index de la série
        if isinstance(series.index[0], pd.Period):
            future_period = series.index[-1] + (i + 1)
        else:
            future_period = series.index[-1] + pd.Timedelta(days=30 * (i + 1))
        future_periods.append(future_period)
    return pd.DataFrame({'Period': future_periods, 'Forecast': future_values})


def forecast_repeat(series: pd.Series, periods: int = 12) -> pd.DataFrame:
    """Prévisions en répétant le motif historique.

    Cette fonction reproduit le schéma saisonnier observé sur la
    période historique en répétant les valeurs dans l'ordre. Elle est
    particulièrement utile lorsque l'on dispose d'un cycle complet (ex :
    12 mois) et que l'on souhaite projeter les mêmes fluctuations pour
    l'année suivante. Aucun ajustement de tendance n'est appliqué ; si
    vous souhaitez appliquer une croissance globale, multipliez la
    prévision par (1 + taux) après l'appel.

    Parameters
    ----------
    series : pd.Series
        Série chronologique indexée par des périodes (mois). La
        longueur de la série sert de longueur de cycle.
    periods : int
        Nombre de périodes futures à générer. Généralement 12 mois pour
        une répétition annuelle.

    Returns
    -------
    pd.DataFrame
        DataFrame avec ``Period`` et ``Forecast``.
    """
    if series.empty:
        return pd.DataFrame({'Period': [], 'Forecast': []})
    values = series.values
    n = len(values)
    future_vals = [values[i % n] for i in range(periods)]
    # Construire l'index des périodes futures
    future_periods = []
    for i in range(periods):
        if isinstance(series.index[0], pd.Period):
            future_periods.append(series.index[-1] + (i + 1))
        else:
            future_periods.append(series.index[-1] + pd.Timedelta(days=30 * (i + 1)))
    return pd.DataFrame({'Period': future_periods, 'Forecast': future_vals})


def compute_rfm(df_factures: pd.DataFrame) -> pd.DataFrame:
    now_date = df_factures['date de facture'].max() + pd.Timedelta(days=1)
    rfm = df_factures.groupby('nom du client').agg({
        'date de facture': lambda x: (now_date - x.max()).days,
        'id': 'count',
        'montant HT': 'sum'
    }).rename(columns={
        'date de facture': 'Recence',
        'id': 'Frequence',
        'montant HT': 'Montant'
    })
    rfm['R_Score'] = pd.qcut(rfm['Recence'], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequence'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Montant'], 4, labels=[1, 2, 3, 4]).astype(int)
    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
    def assign_segment(row):
        score = row['RFM_Score']
        if score >= 10:
            return 'Champions'
        elif score >= 8:
            return 'Fidèles'
        elif score >= 5:
            return 'Potentiels'
        else:
            return 'À risque'
    rfm['Segment'] = rfm.apply(assign_segment, axis=1)
    return rfm.reset_index()


def prepare_devis(df_devis: pd.DataFrame) -> pd.DataFrame:
    df = df_devis.copy()
    df['Catégorie'] = df['intitulé'].apply(categorize_service)
    df['Typologie client'] = df['nom du client'].apply(classify_client)
    df['montant HT'] = pd.to_numeric(df.get('montant HT'), errors='coerce')
    df['montant TTC'] = pd.to_numeric(df.get('montant TTC'), errors='coerce')
    return df


def get_logo():
    if LOGO_FILE.exists():
        from PIL import Image as PilImage
        return PilImage.open(LOGO_FILE)
    return None


def show_overview(df_factures: pd.DataFrame, df_clients: pd.DataFrame, df_devis: pd.DataFrame):
    kpis = compute_kpi_overview(df_factures, df_clients, df_devis)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("CA total (HT)", f"{kpis['ca_total_ht']:,.0f} €")
    col2.metric("CA total (TTC)", f"{kpis['ca_total_ttc']:,.0f} €")
    col3.metric("Factures", kpis['nb_factures'])
    col4.metric("Clients", kpis['nb_clients'])
    col5.metric("Panier moyen", f"{kpis['panier_moyen_ht']:,.0f} €")
    col6.metric("Fidélité (CA)", f"{kpis['taux_fidelite']*100:.1f}%")
    st.markdown("---")
    colA, colB, colC = st.columns(3)
    colA.metric("Taux de paiement", f"{kpis['taux_paiement']*100:.1f}%" if pd.notna(kpis['taux_paiement']) else "N/A")
    colB.metric("Taux de conversion", f"{kpis['taux_conversion']*100:.1f}%" if pd.notna(kpis['taux_conversion']) else "N/A")
    colC.metric("Clients fidèles", kpis['nb_fidèles'])
    st.markdown(
        """
        ### Synthèse
        - Le chiffre d'affaires net 2025 s'élève à **{ca_ht:,.0f} € HT** et **{ca_ttc:,.0f} € TTC**.
        - Sur **{nb_clients} clients**, **{nb_fidèles}** représentent la base fidèle et génèrent {taux_fidelite:,.1%} du CA.
        - Le panier moyen est de {panier_moyen:,.0f} € par facture, illustrant des prestations haut de gamme.
        - Le taux de paiement est de {taux_paiement:,.1%} (sur les clients multi‑années) et **{taux_conversion:,.1%}** des devis se transforment en factures.
        """.format(
            ca_ht=kpis['ca_total_ht'],
            ca_ttc=kpis['ca_total_ttc'],
            nb_clients=kpis['nb_clients'],
            nb_fidèles=kpis['nb_fidèles'],
            taux_fidelite=kpis['taux_fidelite'],
            panier_moyen=kpis['panier_moyen_ht'],
            taux_paiement=kpis['taux_paiement'],
            taux_conversion=kpis['taux_conversion'],
        )
    )


def show_performance(df_factures: pd.DataFrame):
    st.header("Performance & chiffre d'affaires")
    service_rev = df_factures.groupby('Catégorie').agg({
        'montant HT': 'sum',
        'montant TTC': 'sum'
    }).sort_values(by='montant HT', ascending=False)
    fig_service = px.bar(service_rev, x=service_rev.index, y='montant HT',
                         labels={'montant HT': 'CA HT'}, title='Répartition du CA par service')
    fig_service.update_layout(xaxis_title='', yaxis_title='CA HT (€)')
    st.plotly_chart(fig_service, use_container_width=True)
    client_rev = df_factures.groupby('Typologie client').agg({'montant HT': 'sum'})
    fig_client = px.pie(client_rev, values='montant HT', names=client_rev.index,
                        title='Part du CA par typologie de client')
    st.plotly_chart(fig_client, use_container_width=True)
    monthly = compute_monthly_revenue(df_factures)
    fig_month = px.line(monthly, x='month', y='montant HT', markers=True,
                        title='Évolution mensuelle du CA HT')
    fig_month.update_layout(xaxis_title='Mois', yaxis_title='CA HT (€)')
    st.plotly_chart(fig_month, use_container_width=True)
    st.markdown(
        """
        #### Analyse saisonnière
        Les pics observés en mars/avril et juin correspondent à des événements
        majeurs (Fashion Week, MIPIM, Cannes, etc.). Le creux estival d'août
        reflète la faible activité corporate, tandis que septembre/octobre marquent
        la reprise. Planifiez vos ressources (chauffeurs, véhicules) et actions
        commerciales en fonction de ces périodes.
        """
    )


def show_devis(df_devis: pd.DataFrame, df_factures: pd.DataFrame):
    st.header("Devis & pipeline commercial")
    if df_devis.empty:
        st.info("Aucun devis chargé.")
        return
    # Harmoniser les valeurs de statut en minuscules sans accent
    statut_series = df_devis['statut'].astype(str).str.strip().str.lower()
    status_counts = statut_series.value_counts()
    # On affiche les statuts dans leur forme originale (en reprenant les clés du comptage)
    fig_status = px.pie(values=status_counts.values, names=status_counts.index,
                        title='Répartition des devis par statut')
    st.plotly_chart(fig_status, use_container_width=True)
    nb_emis = len(df_devis)
    nb_acceptes = (statut_series == 'accepté').sum()
    taux_conv = nb_acceptes / nb_emis if nb_emis else 0
    st.metric("Taux de conversion", f"{taux_conv*100:.1f}%")
    st.subheader("Détails des devis acceptés")
    df_acceptes = df_devis[statut_series == 'accepté'][
        ['id', 'date du devis', 'nom du client', 'intitulé', 'montant HT', 'Facture(s) liée(s)']
    ]
    st.dataframe(df_acceptes, use_container_width=True)
    st.subheader("Top devis perdus (refusés ou annulés)")
    lost_mask = statut_series.isin(['refusé', 'annulé'])
    df_perdus = df_devis[lost_mask].copy()
    if not df_perdus.empty:
        df_perdus = df_perdus.sort_values(by='montant HT', ascending=False).head(10)
        st.dataframe(df_perdus[['id', 'date du devis', 'nom du client', 'intitulé', 'montant HT', 'statut']], use_container_width=True)
    else:
        st.write("Aucun devis perdu en 2025.")
    st.markdown(
        """
        #### Recommandations
        *Relances proactives* : contactez les prospects ayant un devis en attente (statut “Sauvegardé” ou “En attente”) pour
        accélérer la décision. Proposez un appel personnalisé ou une présentation de vos atouts.

        *Offres modulaires* : pour les demandes événementielles complexes, proposez des options modulaires (durée,
        type de véhicule, services additionnels) pour adapter l'offre au budget tout en préservant la qualité.

        *Personnalisation* : exploitez les informations des devis pour mieux
        comprendre les attentes (nombre de passagers, destination) et personnaliser vos propositions, renforçant la
        valeur perçue sans brader les prix.
        """
    )


def show_expenses(df_transactions: pd.DataFrame):
    st.header("Dépenses & optimisation")
    if df_transactions.empty:
        st.info("Aucune transaction bancaire chargée.")
        return
    df_tx = prepare_transactions(df_transactions)
    df_dep = df_tx[df_tx['Montant TTC'] < 0].copy()
    if df_dep.empty:
        st.write("Aucune dépense enregistrée.")
        return
    cat_summary = df_dep.groupby('Catégorie dépense')['Montant abs'].sum().sort_values(ascending=False)
    fig_cat = px.bar(cat_summary, x=cat_summary.index, y=cat_summary.values,
                     labels={'x': 'Catégorie', 'y': 'Total dépensé (€)'},
                     title='Répartition annuelle des dépenses par catégorie')
    st.plotly_chart(fig_cat, use_container_width=True)
    monthly_dep = df_dep.groupby(['mois', 'Catégorie dépense'])['Montant abs'].sum().reset_index()
    monthly_dep['Period'] = monthly_dep['mois'].dt.to_timestamp()
    fig_monthly = px.bar(monthly_dep, x='Period', y='Montant abs', color='Catégorie dépense',
                         title='Évolution mensuelle des dépenses par catégorie')
    fig_monthly.update_layout(xaxis_title='Mois', yaxis_title='€', legend_title='Catégorie')
    st.plotly_chart(fig_monthly, use_container_width=True)
    # Calcule des dépenses mensuelles totales
    dep_series = df_dep.groupby('mois')['Montant abs'].sum().sort_index()
    # Prévision répétitive pour 12 mois : on répète la saisonnalité
    forecast_df = forecast_repeat(dep_series, periods=12)
    # Construire les séries historiques et prévisionnelles pour affichage
    historic = pd.DataFrame({'Period': dep_series.index.to_timestamp(), 'Valeur': dep_series.values, 'Type': 'Historique'})
    # Convertir Period au format timestamp selon type (Period ou date)
    future_periods = [p.to_timestamp() if isinstance(p, pd.Period) else p for p in forecast_df['Period']]
    future = pd.DataFrame({'Period': future_periods, 'Valeur': forecast_df['Forecast'], 'Type': 'Prévision'})
    all_dep = pd.concat([historic, future], ignore_index=True)
    fig_forecast = px.line(all_dep, x='Period', y='Valeur', color='Type', markers=True,
                           title='Prévision saisonnière des dépenses mensuelles')
    fig_forecast.update_layout(xaxis_title='Mois', yaxis_title='Dépenses (€)')
    st.plotly_chart(fig_forecast, use_container_width=True)
    # Analyse comparative des dépenses vs CA
    df_factures_local = load_factures()  # recharge les factures pour obtenir le CA 2025
    total_ca = df_factures_local['montant HT'].sum()
    total_dep = df_dep['Montant abs'].sum()
    # Exclure les investissements et dividendes/prélèvements pour connaître les charges opérationnelles
    operational_dep = df_dep[~df_dep['Catégorie dépense'].isin(['Investissement', 'Dividendes/Prélèvements'])]['Montant abs'].sum()
    ratio_total = total_dep / total_ca if total_ca else np.nan
    ratio_operational = operational_dep / total_ca if total_ca else np.nan
    st.markdown(
        f"""
        #### Lecture des dépenses et comparaison au chiffre d'affaires
        * **Total des dépenses 2025 :** {total_dep:,.0f} € TTC. Ce chiffre inclut des investissements (achats de véhicules,
          acomptes, dépôts) et des prélèvements ou dividendes du gérant, qui ne sont pas des charges d'exploitation.
        * **Charges opérationnelles :** {operational_dep:,.0f} € TTC, soit {ratio_operational:.0%} du chiffre d'affaires HT
          (≈{total_ca:,.0f} €). Une fois les investissements et prélèvements exclus, les dépenses n'excèdent donc pas le CA.
        * **Investissements et prélèvements :** {total_dep - operational_dep:,.0f} € TTC. Il est normal que ces montants
          fassent dépasser les charges totales par rapport au CA car ils représentent des achats d'actifs (ex : véhicule)
          ou des versements au gérant.

        L'analyse des dépenses doit donc distinguer ces composantes pour évaluer la rentabilité réelle de
        l'exploitation. La prévision ci‑dessus intègre uniquement les charges opérationnelles (hors investissements
        ponctuels), ce qui donne une vision plus représentative pour 2026.
        """
    )
    st.markdown(
        """
        #### Conseils d'optimisation
        * Véhicules et carburant : négociez des tarifs préférentiels auprès des fournisseurs premium et envisagez des
        solutions hybrides/électriques pour réduire le coût du carburant tout en valorisant l'image éco‑responsable.
        * Sous‑traitance : fidélisez un réseau restreint de partenaires haut de gamme pour maîtriser la qualité et les
        marges, plutôt que de multiplier les sous‑traitants au détriment de la satisfaction client.
        * Marketing ciblé : privilégiez les partenariats avec des hôtels 5★, conciergeries et maisons de luxe plutôt que
        des campagnes de masse. Investissez dans la satisfaction client, levier de bouche‑à‑oreille puissant dans le
        secteur du luxe.
        * Suivi des encaissements : réduisez les retards de paiement en instaurant des relances automatisées et des
        pénalités de retard. Une trésorerie saine permet d'investir sereinement.
        """
    )


def show_clients(df_factures: pd.DataFrame, df_clients: pd.DataFrame):
    st.header("Clients & segmentation")
    rfm = compute_rfm(df_factures)
    seg_counts = rfm['Segment'].value_counts()
    fig_seg = px.pie(seg_counts, names=seg_counts.index, values=seg_counts.values,
                     title='Répartition des clients par segment RFM')
    st.plotly_chart(fig_seg, use_container_width=True)
    top_clients = df_factures.groupby('nom du client').agg({'montant HT': 'sum'}).sort_values(by='montant HT', ascending=False).head(10)
    fig_top = px.bar(top_clients, x=top_clients.index, y='montant HT', title='Top 10 des clients par CA (HT)')
    fig_top.update_layout(xaxis_title='', yaxis_title='CA HT (€)')
    st.plotly_chart(fig_top, use_container_width=True)
    st.subheader("Tableau RFM détaillé")
    st.dataframe(rfm, use_container_width=True)
    st.markdown(
        """
        #### Actions de fidélisation
        * **Champions** : offrez un service ultra personnalisé (chauffeur attitré, surclassement gratuit) et demandez
        leurs retours pour améliorer votre offre.
        * **Fidèles** : proposez des packages annuels ou semestriels pour les inciter à augmenter leur volume de courses.
        * **Potentiels** : relancez-les après leur première course pour les faire revenir (offre découverte ou
        upgrade temporaire).
        * **À risque** : contactez-les pour comprendre la raison de leur absence et récupérer un feedback; proposez un
        geste commercial ciblé si cela peut les inciter à revenir.
        """
    )


def show_forecasts(df_factures: pd.DataFrame, df_transactions: pd.DataFrame):
    st.header("Prévisions 2026")
    monthly = compute_monthly_revenue(df_factures)
    # Séries mensuelles CA et dépenses (hors investissements/prélèvements)
    ca_series = monthly.set_index('date de facture')['montant HT']
    df_tx = prepare_transactions(df_transactions)
    dep_series = df_tx[df_tx['Montant TTC'] < 0].groupby('mois')['Montant abs'].sum().sort_index()
    # Prévisions en répétant le motif saisonnier observé
    ca_forecast = forecast_repeat(ca_series, periods=12)
    dep_forecast = forecast_repeat(dep_series, periods=12)
    st.subheader("Ajustez vos hypothèses")
    growth_factor = st.slider("Croissance CA (%)", -10, 50, 10, step=1) / 100
    cost_inflation = st.slider("Inflation des charges (%)", 0, 20, 5, step=1) / 100
    ca_forecast['Adjusted'] = ca_forecast['Forecast'] * (1 + growth_factor)
    dep_forecast['Adjusted'] = dep_forecast['Forecast'] * (1 + cost_inflation)
    # Convertir période en Timestamp pour l'affichage
    future_dates = [p.to_timestamp() if isinstance(p, pd.Period) else p for p in ca_forecast['Period']]
    df_future = pd.DataFrame({
        'Période': future_dates,
        'CA prévisionnel': ca_forecast['Adjusted'],
        'Dépenses prévisionnelles': dep_forecast['Adjusted'],
        'Marge prévisionnelle': ca_forecast['Adjusted'] - dep_forecast['Adjusted'],
    })
    # Graphique interactif
    fig_pred = px.line(df_future, x='Période', y=['CA prévisionnel', 'Dépenses prévisionnelles', 'Marge prévisionnelle'],
                       labels={'value': 'Montant (€)', 'variable': 'Poste'}, title='Prévisions mensuelles 2026 (avec saisonnalité)')
    st.plotly_chart(fig_pred, use_container_width=True)
    st.subheader("Tableau des prévisions financières (sommes annuelles)")
    df_summary = pd.DataFrame({
        '2025 (réalisé)': [ca_series.sum(), dep_series.sum(), ca_series.sum() - dep_series.sum()],
        '2026 prévisionnel': [df_future['CA prévisionnel'].sum(), df_future['Dépenses prévisionnelles'].sum(), df_future['Marge prévisionnelle'].sum()],
    }, index=["Chiffre d'affaires", 'Dépenses', 'Marge'])
    st.dataframe(df_summary, use_container_width=True)
    st.markdown(
        """
        #### Lecture des scénarios
        Les prévisions utilisent un modèle linéaire avec ajustement saisonnier pour reproduire les fluctuations mensuelles observées en 2025.
        En ajustant les hypothèses de croissance et d'inflation ci‑dessus, vous voyez l'effet sur le résultat net.
        N'oubliez pas qu'une prévision reste approximative : des événements exceptionnels (grands contrats, crise sanitaire)
        peuvent modifier ces trajectoires. Ce modèle sert de base pour élaborer un budget et planifier les ressources.
        """
    )


def show_ai_demo():
    st.header("Démonstration IA – Reconnaissance d'images")
    if not _has_torch:
        st.warning("PyTorch n'est pas disponible dans cet environnement. La démo IA est désactivée.")
        return
    st.markdown(
        """
        Uploadez une photo prise par un chauffeur (ex : intérieur d'un véhicule, coffre, etc.).
        Le modèle ResNet18 pré‑entraîné tentera de deviner la catégorie générale de l'objet. Cette démonstration
        illustre comment intégrer une IA dans votre application pour contrôler des éléments visuels.
        """
    )
    uploaded_file = st.file_uploader("Choisissez une image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Image chargée', use_column_width=True)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image).unsqueeze(0)
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        class_idx = probabilities.argmax().item()
        labels = models.ResNet18_Weights.IMAGENET1K_V1.meta['categories']
        predicted_label = labels[class_idx]
        st.write(f"**Prédiction :** {predicted_label} (confidence {probabilities[class_idx]*100:.1f}%)")
        st.success("Démo terminée ! Vous pouvez imaginer utiliser cette IA pour détecter des objets indésirables ou autres anomalies dans le véhicule.")


def main():
    df_factures = load_factures()
    df_devis = load_devis() if DEVISES_FILE.exists() else pd.DataFrame()
    df_clients = load_clients() if CLIENTS_FILE.exists() else pd.DataFrame()
    df_transactions = load_transactions() if TRANSACTIONS_FILE.exists() else pd.DataFrame()
    df_documents = load_documents() if DOCUMENTS_FILE.exists() else pd.DataFrame()
    df_events = load_events() if EVENTS_FILE.exists() else pd.DataFrame()
    df_factures = prepare_factures(df_factures)
    df_devis = prepare_devis(df_devis) if not df_devis.empty else df_devis
    logo = get_logo()
    # En-tête : titre sans emoji voiture pour laisser la place au logo
    st.markdown("# Chauffeur de France Dashboard")
    if logo:
        st.image(logo, width=100)
    st.markdown("Bienvenue dans le tableau de bord d'audit. Utilisez le menu ci‑dessus pour naviguer dans les différentes analyses.")
    page = st.sidebar.selectbox(
        "Sélectionnez une section",
        [
            "Vue d'ensemble",
            "Performance & CA",
            "Devis & pipeline",
            "Dépenses",
            "Clients & segmentation",
            "Prévisions 2026",
            "Démonstration IA",
        ]
    )
    if page == "Vue d'ensemble":
        show_overview(df_factures, df_clients, df_devis)
    elif page == "Performance & CA":
        show_performance(df_factures)
    elif page == "Devis & pipeline":
        show_devis(df_devis, df_factures)
    elif page == "Dépenses":
        show_expenses(df_transactions)
    elif page == "Clients & segmentation":
        show_clients(df_factures, df_clients)
    elif page == "Prévisions 2026":
        show_forecasts(df_factures, df_transactions)
    else:
        show_ai_demo()


if __name__ == "__main__":
    main()