# 🏀 Sportsee RAG Assistant – Analytics IA (ESN / Consulting Ready)

[![Streamlit](https://img.shields.io/badge/Streamlit-Live-blue?logo=streamlit)](https://share.streamlit.io/JEMAIZ/Sportsee-RAG-Assistant/main/MistralChat.py)
[![Python](https://img.shields.io/badge/Python-3.11-green)]
[![Docker](https://img.shields.io/badge/Docker-Deploy-blue)]
[![RAGAS](https://img.shields.io/badge/RAGAS-Score_0.80-orange)]()

**Assistant IA NBA stats** : NL → SQL/graphiques/RAG docs. **ESN use cases** : reporting projets, insights clients, knowledge base.

**[BOOK Demo ESN Pack → TON CALENDLY]**
Calendly Demo : [BOOK DEMO → https://calendly.com/jemai/demo-ia-esn-15min
## 🎯 Use Cases ESN / Cabinets Conseil
- **Reporting KPI** : RAG analytics projets (comme sport → business metrics).
- **Insights data** : Recommandations automatisées clients PME.
- **Knowledge base** : Q&A datasets internes (score RAGAS 0.80).

**ROI** : -8h analyse manuelle → +CA consulting.

## 🛠 Stack Production
- **Agent** : LangChain + Mistral (5 tools : SQL, plot, team, semantic cache).
- **DB** : SQLite/SQLAlchemy (569 joueurs, 45 stats).
- **RAG** : FAISS sémantique + Redis cache (cosine >0.85).
- **UI** : Streamlit (FAQ, stats équipe).
- **Observabilité** : Pydantic Logfire.
- **CI/CD** : GitHub Actions RAGAS eval (seuil 0.65).

## 📊 RAGAS Scores (12 cas)
| Catégorie | Score |
|-----------|-------|
| Simple    | 0.70  |
| Complex   | 0.90  |
| Noisy     | 0.81  |
| **Global**| **0.80** |

## 🚀 Quick Start (5 min)
```bash
git clone https://github.com/JEMAIZ/Sportsee-RAG-Assistant
cd Sportsee-RAG-Assistant
python -m venv venv && source venv/bin/activate  # ou venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # MISTRAL_API_KEY=...
python load_excel_to_db.py --excel-path inputs/regular_NBA.xlsx
docker-compose up -d  # Redis
streamlit run MistralChat.py
Structure
text
├── MistralChat.py     # Streamlit UI
├── agent.py           # LangChain agent + Logfire + Redis cache
├── sql_tool.py        # NL → SQL
├── plot_tool.py       # Matplotlib graphs (MD5 cache)
├── semantic_cache.py  # Redis cosine >0.85
├── evaluate_ragas.py  # CI/CD eval
├── .github/workflows/ragas_eval.yml
├── inputs/regular_NBA.xlsx
└── database/sportsee.db (généré)
Env
text
MISTRAL_API_KEY=...  # Obligatoire
LOGFIRE_TOKEN=...    # Observabilité
REDIS_HOST=localhost
Projets ESN Complémentaires
n8n-leadgen-esn-demo : Prospection auto.

n8n-reporting-esn : KPI projets.

**[BOOK Demo ESN Pack → ]**
Calendly Demo : [BOOK DEMO → https://calendly.com/jemai/demo-ia-esn-15min

⭐ Star si testé ! #RAG #ESN #IA #Streamlit
