# 🏀 SportSee — NBA RAG Analyst AI

Assistant IA pour l'analyse de performances NBA, combinant **RAG** (Retrieval-Augmented Generation), **SQL dynamique** et **visualisation automatique** via un agent LangChain orchestrant plusieurs outils.

---

## Architecture du système

```
┌──────────────────────────────────────────────────────┐
│                Interface Streamlit                    │
│                  MistralChat.py                       │
└───────────────────────┬──────────────────────────────┘
                        │ question utilisateur
                        ▼
┌──────────────────────────────────────────────────────┐
│              Agent LangChain (agent.py)               │
│         Mistral LLM — raisonnement + orchestration    │
└──────────┬─────────────────┬────────────────┬────────┘
           │                 │                │
           ▼                 ▼                ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │   RAG Tool   │  │   SQL Tool   │  │  Plot Tool   │
   │  (FAISS +    │  │  (SQLite +   │  │ (matplotlib) │
   │   chunks)    │  │  few-shot)   │  │              │
   └──────────────┘  └──────────────┘  └──────────────┘
           │                 │
           ▼                 ▼
   ┌──────────────┐  ┌──────────────┐
   │  Textes &    │  │  Statistiques │
   │  rapports    │  │  NBA (Excel) │
   └──────────────┘  └──────────────┘
```

## Structure du projet

```
sportsee/
├── MistralChat.py          # Interface Streamlit (entrée utilisateur)
├── agent.py                # Agent LangChain orchestrant les tools
├── indexer.py              # Indexation des documents texte → FAISS
├── load_excel_to_db.py     # Pipeline ETL : Excel NBA → SQLite
├── sql_tool.py             # Tool SQL : NL → SQL → résultats
├── plot_tool.py            # Tool visualisation : données → graphique
├── evaluate_ragas.py       # Évaluation RAGAS (baseline + post-SQL)
│
├── utils/
│   ├── config.py           # Configuration centralisée (.env)
│   ├── schemas.py          # Modèles Pydantic (validation des données)
│   ├── database.py         # Modèles SQLAlchemy (schéma relationnel)
│   └── vector_store.py     # Gestion index FAISS + embeddings
│
├── inputs/                 # Documents texte sources (rapports, analyses)
├── vector_db/              # Index FAISS (généré par indexer.py)
├── database/               # sportsee.db (généré par load_excel_to_db.py)
├── outputs/                # Rapports RAGAS, graphiques générés
│
├── .env.example            # Template variables d'environnement
├── .gitignore
└── requirements.txt
```

## Prérequis

- Python 3.10+
- Clé API Mistral ([console.mistral.ai](https://console.mistral.ai/))

## Installation

```bash
# 1. Cloner le repo
git clone <url> && cd sportsee

# 2. Environnement virtuel
python -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate        # Windows

# 3. Dépendances
pip install -r requirements.txt

# 4. Variables d'environnement
cp .env.example .env
# Éditez .env et renseignez votre MISTRAL_API_KEY
```

## Utilisation

### Étape 1 — Charger les données Excel en base

```bash
python load_excel_to_db.py --excel-path inputs/regular_NBA.xlsx
```

Ce script valide chaque ligne avec Pydantic avant insertion.
Sortie attendue : `✅ 569 joueurs insérés.`

### Étape 2 — Indexer les documents texte

```bash
python indexer.py --input-dir inputs/
```

Génère l'index FAISS dans `vector_db/`.

### Étape 3 — Lancer l'interface

```bash
streamlit run MistralChat.py
```

Accessible sur [http://localhost:8501](http://localhost:8501)

### Étape 4 — Évaluer le système (RAGAS)

```bash
# Baseline (avant SQL Tool)
python evaluate_ragas.py --mode baseline

# Post-intégration SQL
python evaluate_ragas.py --mode post-sql
```

Les rapports sont générés dans `outputs/`.

---

## Évaluation RAGAS

### Métriques mesurées

| Métrique | Ce qu'elle mesure | Problème détecté si bas |
|---|---|---|
| `faithfulness` | Fidélité aux documents récupérés | Hallucinations |
| `answer_relevancy` | Pertinence de la réponse | Réponse hors-sujet |
| `context_precision` | Qualité des chunks récupérés | Bruit dans le retrieval |
| `context_recall` | Exhaustivité des chunks | Informations manquées |

### Cas de test (12 cas, 3 catégories)

- **Simple (4 cas)** : questions directes avec réponse factuelle
- **Complex (4 cas)** : agrégations, comparaisons multi-critères
- **Noisy (4 cas)** : questions mal formulées, ambiguës

### Résultats types attendus

```
── BASELINE (avant SQL Tool) ──
  faithfulness         ██████████░░░░░░░░░░ ~0.52
  answer_relevancy     ████████████░░░░░░░░ ~0.61
  context_precision    ████████░░░░░░░░░░░░ ~0.44
  context_recall       ██████████░░░░░░░░░░ ~0.51

── POST-SQL (après SQL Tool) ──
  faithfulness         ███████████████░░░░░ ~0.78
  answer_relevancy     ████████████████░░░░ ~0.81
  context_precision    █████████████░░░░░░░ ~0.69
  context_recall       ███████████████░░░░░ ~0.77
```

---

## Schéma de la base de données

```sql
TABLE teams (
    code      TEXT PRIMARY KEY,   -- "OKC", "LAL", ...
    full_name TEXT                -- "Oklahoma City Thunder"
)

TABLE players (
    id         INTEGER PRIMARY KEY,
    player     TEXT,    team TEXT,
    pts REAL,  reb REAL, ast REAL, ...
    fg3_pct REAL,       -- 3-Point %
    netrtg  REAL,       -- Net Rating
    ts_pct  REAL,       -- True Shooting %
    -- (45 colonnes statistiques au total)
)
```

### Exemples de requêtes SQL types

```sql
-- Top 5 scoreurs (min. 50 matchs)
SELECT player, team, pts, gp FROM players
WHERE gp >= 50 ORDER BY pts DESC LIMIT 5;

-- Meilleure équipe défensive
SELECT team, AVG(defrtg) as avg_defrtg FROM players
GROUP BY team ORDER BY avg_defrtg ASC LIMIT 5;

-- Joueurs avec 20+ pts, 8+ reb, 5+ ast
SELECT player, team, pts, reb, ast FROM players
WHERE pts >= 20 AND reb >= 8 AND ast >= 5;
```

---

## Checklist de validation

- [ ] `.env` créé avec `MISTRAL_API_KEY` valide
- [ ] `pip install -r requirements.txt` sans erreur
- [ ] `python load_excel_to_db.py` : 500+ joueurs insérés
- [ ] `python indexer.py` : index FAISS créé dans `vector_db/`
- [ ] `streamlit run MistralChat.py` : interface accessible
- [ ] Question stats NBA → sql_tool appelé (visible dans les logs)
- [ ] `python evaluate_ragas.py --mode baseline` : rapport généré

---

## Limites et axes d'amélioration

1. **Mapping NL→SQL** : des questions très complexes ou ambiguës peuvent générer du SQL incorrect. Les exemples few-shot couvrent les cas courants mais pas tous.
2. **Couverture des données** : les données Excel représentent la saison régulière ; les playoffs ne sont pas inclus.
3. **Logfire** : la traçabilité Pydantic Logfire est préparée mais nécessite un compte Logfire actif.
4. **PostgreSQL** : passer à PostgreSQL en production nécessite uniquement de changer `DB_PATH` en URL PostgreSQL dans `.env`.
