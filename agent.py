# agent.py
# agent.py  — tout en haut
from dotenv import load_dotenv
load_dotenv()  # ← charge .env AVANT logfire_setup
import logging
import os
from typing import Optional
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai import ChatMistralAI
from utils.config import MISTRAL_API_KEY, MODEL_NAME

logger = logging.getLogger(__name__)

# ── Initialisation Logfire (avant tout import LangChain) ─────────────────────
from logfire_setup import setup_logfire
_logfire_ok = setup_logfire()

# Import logfire pour les spans manuels (no-op si absent)
try:
    import logfire
except ImportError:
    logfire = None

# ── Logfire + variables OTEL AVANT tout import LangChain ─────────────────────
import logfire
logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    service_name="sportsee-rag",
    send_to_logfire=True,
)

# Ces 3 variables DOIVENT être définies avant "from langchain..."
os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
os.environ["LANGSMITH_OTEL_ONLY"]    = "true"
os.environ["LANGSMITH_TRACING"]      = "true"

# Instrumentation SQLAlchemy
from utils.database import get_engine
logfire.instrument_sqlalchemy(engine=get_engine())


# ── RAG Tool ─────────────────────────────────────────────────────────────────
@tool
def rag_tool(question: str) -> str:
    """
    Recherche dans les documents textuels via le vector store FAISS.
    A utiliser pour les questions sur le contexte des matchs,
    les analyses qualitatives, les commentaires de matchs.
    """
    span = logfire.span("rag_tool", question=question) if logfire else None
    if span:
        span.__enter__()
    try:
        from utils.vector_store import VectorStoreManager
        from utils.config import SEARCH_K
        vsm = VectorStoreManager()
        if vsm.index is None:
            return "Index vectoriel non disponible. Executez d'abord python indexer.py"
        results = vsm.search(question, k=SEARCH_K)
        if not results:
            return "Aucun document pertinent trouve pour cette question."
        context_parts = []
        for r in results:
            source = r["metadata"].get("source", "Inconnue")
            score  = r.get("score", 0)
            context_parts.append(
                f"[Source: {source} | Pertinence: {score:.1f}%]\n{r['text']}"
            )
        output = "\n\n---\n\n".join(context_parts)
        if logfire and span:
            logfire.info("rag_tool résultat", nb_results=len(results))
        return output
    except Exception as e:
        logger.error(f"Erreur RAG Tool: {e}")
        if logfire and span:
            logfire.error("rag_tool erreur", error=str(e))
        return f"Erreur lors de la recherche documentaire : {e}"
    finally:
        if span:
            span.__exit__(None, None, None)


# ── Imports des autres tools ──────────────────────────────────────────────────
from sql_tool  import sql_tool
from plot_tool import plot_tool
from team_tool import team_tool, list_teams_tool


# ── Prompt système ────────────────────────────────────────────────────────────
AGENT_SYSTEM_PROMPT = """Tu es NBA Analyst AI, un assistant expert en statistiques NBA pour SportSee.
Tu as acces a 5 outils :

1. rag_tool        : Pour les questions sur le contenu textuel (commentaires de matchs,
                     analyses qualitatives, contexte narratif).

2. sql_tool        : Pour les questions chiffrees (stats, classements, comparaisons numeriques).
                     TOUJOURS utiliser ce tool pour les statistiques NBA generales.

3. plot_tool       : Pour generer des graphiques a partir de donnees chiffrees.
                     Utilise-le quand l'utilisateur demande un graphique ou une visualisation.

4. team_tool       : Pour obtenir les statistiques completes d'une equipe NBA specifique :
                     top scoreurs, rebondeurs, passeurs et roster complet.
                     Utilise-le quand l'utilisateur mentionne une equipe precise.

5. list_teams_tool : Pour lister toutes les equipes disponibles dans la base.
                     Utilise-le si l'utilisateur demande quelles equipes sont disponibles
                     ou si team_tool ne trouve pas l'equipe demandee.

REGLES :
- Pour les statistiques d'UNE equipe specifique, utilise TOUJOURS team_tool
- Pour les statistiques generales (ligue entiere, classements), utilise sql_tool
- Pour les graphiques, utilise plot_tool apres avoir obtenu les donnees
- Synthetise toujours la reponse en langage naturel clair
- Si tu n'as pas assez d'informations, dis-le clairement
- Reponds en francais
"""


# ── Construction de l'agent ───────────────────────────────────────────────────
def build_agent() -> AgentExecutor:
    llm = ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model=MODEL_NAME,
        temperature=0.1,
    )
    tools = [rag_tool, sql_tool, plot_tool, team_tool, list_teams_tool]
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
    )


_agent_executor: Optional[AgentExecutor] = None

def get_agent_response(question: str, chat_history: list = None) -> str:
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = build_agent()
    try:
        with (logfire.span("agent.invoke", question=question) if logfire else __import__('contextlib').nullcontext()):
            result = _agent_executor.invoke({
                "input":        question,
                "chat_history": chat_history or []
            })
        return result.get("output", "Desole, je n'ai pas pu generer de reponse.")
    except Exception as e:
        logger.error(f"Erreur agent: {e}")
        if logfire:
            logfire.error("agent.invoke erreur", error=str(e))
        return f"Erreur lors du traitement de votre question : {e}"
