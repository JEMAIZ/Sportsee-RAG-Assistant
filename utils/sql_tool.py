# sql_tool.py
import re
import logging
from typing import Optional

from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text
from sqlalchemy.orm import Session

from utils.config import MISTRAL_API_KEY, MODEL_NAME, DB_PATH
from utils.database import get_engine
from utils.schemas import SQLQueryResult

logger = logging.getLogger(__name__)

DB_SCHEMA = """
Tables disponibles :

TABLE teams (
    code      TEXT PRIMARY KEY,
    full_name TEXT
)

TABLE players (
    id         INTEGER PRIMARY KEY,
    player     TEXT,
    team       TEXT,
    age        REAL,
    gp         REAL,
    w          REAL,
    l          REAL,
    min_pg     REAL,
    pts        REAL,
    fgm        REAL,
    fga        REAL,
    fg_pct     REAL,
    fg3m       REAL,
    fg3a       REAL,
    fg3_pct    REAL,
    ftm        REAL, fta REAL, ft_pct REAL,
    oreb REAL, dreb REAL, reb REAL,
    ast REAL, tov REAL, stl REAL, blk REAL, pf REAL,
    fp         REAL,
    dd2        REAL,
    td3        REAL,
    plus_minus REAL,
    offrtg REAL, defrtg REAL, netrtg REAL,
    ast_pct REAL, ast_to REAL, ast_ratio REAL,
    oreb_pct REAL, dreb_pct REAL, reb_pct REAL,
    to_ratio REAL, efg_pct REAL, ts_pct REAL,
    usg_pct REAL, pace REAL, pie REAL, poss REAL
)
"""

FEW_SHOT_EXAMPLES = """
EXEMPLES (question -> requete SQL) :

Q: Qui a le meilleur pourcentage a 3 points (minimum 100 tentatives) ?
SQL: SELECT player, team, fg3_pct, fg3a FROM players WHERE fg3a >= 100 ORDER BY fg3_pct DESC LIMIT 5;

Q: Quels sont les 10 meilleurs scoreurs de la saison ?
SQL: SELECT player, team, pts, gp FROM players ORDER BY pts DESC LIMIT 10;

Q: Compare les rebonds moyens de LeBron James et Nikola Jokic.
SQL: SELECT player, team, reb, oreb, dreb FROM players WHERE player IN ('LeBron James', 'Nikola Jokic');

Q: Quelle equipe a la meilleure defense (DEFRTG le plus bas) ?
SQL: SELECT team, AVG(defrtg) as avg_defrtg FROM players GROUP BY team ORDER BY avg_defrtg ASC LIMIT 5;

Q: Qui a le plus de passes decisives avec un ratio AST/TO > 3 ?
SQL: SELECT player, team, ast, tov, ast_to FROM players WHERE ast_to > 3 ORDER BY ast DESC LIMIT 10;

Q: Combien de joueurs ont realise plus de 10 triple-doubles ?
SQL: SELECT COUNT(*) as nb_players FROM players WHERE td3 > 10;

Q: Quelles sont les statistiques completes de Stephen Curry ?
SQL: SELECT * FROM players WHERE player LIKE '%Curry%';
"""

SQL_SYSTEM_PROMPT = f"""Tu es un expert SQL specialise dans les statistiques NBA.
Tu dois generer des requetes SQLite valides et securisees.

REGLES ABSOLUES :
- Retourne UNIQUEMENT la requete SQL, sans explication ni markdown
- N'utilise que les tables et colonnes du schema fourni
- Limite toujours les resultats avec LIMIT (max 20 lignes)
- N'utilise jamais DROP, DELETE, UPDATE, INSERT, CREATE
- Pour les noms de joueurs, utilise LIKE avec % pour etre tolerant

SCHEMA DE LA BASE :
{DB_SCHEMA}

{FEW_SHOT_EXAMPLES}
"""


def _generate_sql(question: str, llm: ChatMistralAI) -> str:
    messages = [
        SystemMessage(content=SQL_SYSTEM_PROMPT),
        HumanMessage(content=f"Genere une requete SQL pour : {question}")
    ]
    response = llm.invoke(messages)
    raw_sql  = response.content.strip()
    raw_sql  = re.sub(r"```sql\s*", "", raw_sql)
    raw_sql  = re.sub(r"```\s*",    "", raw_sql)
    return raw_sql.strip()


def _is_safe_query(sql: str) -> bool:
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER", "TRUNCATE"]
    sql_upper = sql.upper()
    return not any(keyword in sql_upper for keyword in forbidden)


def _execute_sql(sql: str, db_path: str = DB_PATH) -> SQLQueryResult:
    engine = get_engine(db_path)
    try:
        with Session(engine) as session:
            result  = session.execute(text(sql))
            columns = list(result.keys())
            rows    = [dict(zip(columns, row)) for row in result.fetchall()]
            return SQLQueryResult(query=sql, rows=rows)
    except Exception as e:
        logger.error(f"Erreur SQL: {e} | Requete: {sql}")
        return SQLQueryResult(query=sql, rows=[], error=str(e))


@tool
def sql_tool(question: str) -> str:
    """
    Repond aux questions sur les statistiques NBA en interrogeant
    la base de donnees SQLite. A utiliser pour les questions chiffrees :
    classements, comparaisons de joueurs, agregations, pourcentages.
    """
    logger.info(f"SQL Tool appele — question: {question}")

    llm = ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model=MODEL_NAME,
        temperature=0
    )

    sql = _generate_sql(question, llm)
    logger.info(f"SQL genere : {sql}")

    if not _is_safe_query(sql):
        return "Requete refusee : operation non autorisee detectee."

    result = _execute_sql(sql)

    if result.error:
        return f"Erreur SQL : {result.error}\nRequete : {sql}"

    if result.row_count == 0:
        return f"Aucun resultat pour : {question}"

    lines = [f"Resultats ({result.row_count} ligne(s)) pour : {question}",
             f"Requete SQL : {sql}", ""]
    for i, row in enumerate(result.rows, 1):
        lines.append(f"  {i}. " + " | ".join(f"{k}: {v}" for k, v in row.items()))

    return "\n".join(lines)