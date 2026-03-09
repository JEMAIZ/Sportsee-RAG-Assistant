# 🏀 SportSee — NBA RAG Assistant

Assistant IA pour analyser les statistiques NBA. Pose une question en langage naturel, l'agent interroge la base de données, génère des graphiques et recherche dans les documents textuels.

## Stack

- **LangChain + Mistral** — agent IA orchestrant les tools
- **SQLite + SQLAlchemy** — stats de 569 joueurs (saison régulière)
- **FAISS** — recherche sémantique sur les documents texte
- **Streamlit** — interface web
- **Pydantic Logfire** — observabilité des traces

## Installation

```bash
git clone https://github.com/JEMAIZ/Sportsee-RAG-Assistant.git
cd Sportsee-RAG-Assistant
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env         # puis renseigner MISTRAL_API_KEY
```

## Démarrage

```bash
# 1. Charger les données
python load_excel_to_db.py --excel-path inputs/regular_NBA.xlsx

# 2. Indexer les documents (optionnel)
python indexer.py --input-dir inputs/

# 3. Lancer l'interface
python -m streamlit run MistralChat.py
```

## Structure

```
├── MistralChat.py        # Interface Streamlit
├── agent.py              # Agent LangChain (5 tools)
├── sql_tool.py           # NL → SQL → résultats
├── plot_tool.py          # Génération graphiques matplotlib
├── team_tool.py          # Stats complètes par équipe
├── logfire_setup.py      # Observabilité Logfire
├── evaluate_ragas.py     # Évaluation qualité RAG
├── load_excel_to_db.py   # ETL Excel → SQLite
├── indexer.py            # Indexation FAISS
├── utils/
│   ├── config.py
│   ├── database.py       # Modèles SQLAlchemy
│   ├── vector_store.py   # Gestion FAISS
│   └── schemas.py        # Validation Pydantic
├── inputs/               # regular_NBA.xlsx + documents texte
├── database/             # sportsee.db (généré)
├── vector_db/            # Index FAISS (généré)
└── outputs/              # Graphiques + rapports RAGAS
```

## Variables d'environnement

```bash
MISTRAL_API_KEY=...        # obligatoire
LOGFIRE_TOKEN=...          # optionnel — observabilité
```

## Évaluation RAGAS

Score global mesuré sur 12 cas (simple / complex / noisy) :

| Catégorie | Score |
|-----------|-------|
| Simple    | 0.66  |
| Complex   | 0.91  |
| Noisy     | 0.83  |
| **Global**| **0.80** |

```bash
python evaluate_ragas.py --mode baseline
```

## Base de données

569 joueurs, 30 équipes, 45 colonnes statistiques par joueur (pts, reb, ast, fg_pct, ts_pct, netrtg...).

## Limites

- Questions très ambiguës peuvent générer du SQL incorrect
