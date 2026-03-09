# logfire_setup.py
"""
Initialisation de Pydantic Logfire pour SportSee RAG Assistant.
Instrumente automatiquement :
  - LangChain (traces des appels agent, tools, LLM)
  - SQLAlchemy (traces des requêtes SQL)
  - Spans manuels sur les tools RAG, Plot, Team

Appelé UNE SEULE FOIS au démarrage depuis agent.py.
"""
import os
import logging

logger = logging.getLogger(__name__)


def setup_logfire():
    try:
        import logfire
    except ImportError:
        logger.warning("logfire non installé.")
        return False

    token = os.getenv("LOGFIRE_TOKEN", "").strip()

    if not token:
        logger.warning("LOGFIRE_TOKEN absent — traces désactivées.")
        logfire.configure(send_to_logfire=False, service_name="sportsee-rag")
        return True

    # Force le token explicitement
    logfire.configure(
        service_name="sportsee-rag",
        service_version="1.0.0",
        #environment=os.getenv("ENV", "development"),
        send_to_logfire=True,   
    )
    logger.info(f"Logfire initialisé avec token : {token[:12]}...")
    return True

# ── Décorateurs utilitaires pour les tools ───────────────────────────────────
def logfire_span(name: str):
    """
    Décorateur qui enveloppe une fonction dans un span Logfire.
    Usage :
        @logfire_span("sql_tool.query")
        def ma_fonction(...): ...
    """
    def decorator(func):
        try:
            import logfire
            import functools

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with logfire.span(name, input=str(args[0]) if args else ""):
                    return func(*args, **kwargs)
            return wrapper
        except ImportError:
            return func  # Logfire absent → pas de span, fonction inchangée
    return decorator
