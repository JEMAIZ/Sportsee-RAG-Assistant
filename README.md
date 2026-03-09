# 🏀 SportSee — NBA RAG Assistant

Assistant IA pour analyser les statistiques NBA. Pose une question en langage naturel, l'agent interroge la base de données, génère des graphiques et recherche dans les documents textuels.

## Stack

- **LangChain + Mistral** — agent orchestrant 5 tools
- **SQLite + SQLAlchemy** — stats de 569 joueurs (saison régulière)
- **FAISS** — recherche sémantique sur documents texte
- **Redis** — cache sémantique (similarité cosine, seuil 0.85)
- **Streamlit** — interface web avec FAQ et stats par équipe
- **Pydantic Logfire** — observabilité des traces
- **GitHub Actions** — évaluation RAGAS automatique à chaque push

## Installation

```bash
git clone https://github.com/JEMAIZ/Sportsee-RAG-Assistant.git
cd Sportsee-RAG-Assistant
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env         # renseigner MISTRAL_API_KEY
```

## Démarrage

```bash
# 1. Charger les données
python load_excel_to_db.py --excel-path inputs/regular_NBA.xlsx

# 2. Indexer les documents (optionnel — nécessite des PDF/TXT dans inputs/)
python indexer.py --input-dir inputs/

# 3. Lancer Redis (cache sémantique)
docker-compose up -d
# ou : redis-server (si Redis installé localement)

# 4. Lancer l'interface
python -m streamlit run MistralChat.py
```

## Structure

```
├── MistralChat.py        # Interface Streamlit
├── agent.py              # Agent LangChain (5 tools) + Logfire + cache
├── sql_tool.py           # NL → SQL → résultats
├── plot_tool.py          # Génération graphiques matplotlib (cache MD5)
├── team_tool.py          # Stats complètes par équipe
├── semantic_cache.py     # Cache sémantique Redis
├── logfire_setup.py      # Observabilité Logfire
├── evaluate_ragas.py     # Évaluation RAGAS (baseline, mode CI)
├── load_excel_to_db.py   # ETL Excel → SQLite
├── indexer.py            # Indexation FAISS
├── docker-compose.yml    # Redis local
├── utils/
│   ├── config.py
│   ├── database.py       # Modèles SQLAlchemy
│   ├── vector_store.py   # Gestion FAISS
│   └── schemas.py        # Validation Pydantic
├── .github/workflows/
│   └── ragas_eval.yml    # CI/CD — bloque merge si score < 0.65
├── inputs/               # regular_NBA.xlsx + documents texte
├── database/             # sportsee.db (généré)
├── vector_db/            # Index FAISS (généré)
└── outputs/              # Graphiques + rapports RAGAS
```

## Variables d'environnement

```bash
MISTRAL_API_KEY=...        # obligatoire
LOGFIRE_TOKEN=...          # optionnel — observabilité
REDIS_HOST=localhost       # optionnel — défaut localhost
REDIS_PORT=6379            # optionnel — défaut 6379
```

## Cache sémantique

Les réponses texte sont mises en cache dans Redis. Si une question similaire (similarité cosine > 0.85) a déjà été posée, la réponse est retournée instantanément sans appel LLM.

```bash
# Vérifier le cache
python -c "from semantic_cache import get_cache; print(get_cache().stats())"
```

## Évaluation RAGAS

Score mesuré sur 12 cas (simple / complex / noisy) :

| Catégorie | Score |
|-----------|-------|
| Simple    | 0.70  |
| Complex   | 0.90  |
| Noisy     | 0.81  |
| **Global**| **0.80** |

```bash
# Lancement manuel
python evaluate_ragas.py --mode baseline

# Mode CI (exit 1 si score < seuil)
python evaluate_ragas.py --mode baseline --ci --threshold 0.65
```

Le workflow GitHub Actions se déclenche automatiquement sur chaque push touchant `agent.py`, `sql_tool.py` ou `utils/`.

## Base de données

569 joueurs, 30 équipes, 45 colonnes statistiques (pts, reb, ast, fg_pct, ts_pct, netrtg, pie...).

## Limites

- Données saison régulière uniquement (pas les playoffs)
- Questions très ambiguës peuvent générer du SQL incorrect
- Python 3.14 : warnings Pydantic V1 non bloquants
