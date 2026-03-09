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
    """
    Configure Logfire et instrumente les packages du projet.
    Retourne True si l'initialisation a réussi, False sinon.
    """
    try:
        import logfire
    except ImportError:
        logger.warning("logfire non installé — observabilité désactivée. "
                       "Installez-le avec : pip install logfire")
        return False

    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        logger.warning("LOGFIRE_TOKEN absent dans .env — traces envoyées en local seulement.")

    # ── 1. Configuration principale Logfire ──────────────────────────────────
    logfire.configure(
        token=os.getenv("LOGFIRE_TOKEN"), 
        service_name="sportsee-rag",
        service_version="1.0.0",
        environment=os.getenv("ENV", "development"),
        # Si pas de token, affiche les traces dans la console
        send_to_logfire=bool(token),
    )

    # ── 2. Instrumentation LangChain via OpenTelemetry ───────────────────────
    # Ces variables DOIVENT être définies AVANT l'import de langchain
    os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
    os.environ.setdefault("LANGSMITH_OTEL_ONLY",    "true")
    os.environ.setdefault("LANGSMITH_TRACING",      "true")

    # ── 3. Instrumentation SQLAlchemy ────────────────────────────────────────
    try:
        from utils.database import get_engine
        engine = get_engine()
        logfire.instrument_sqlalchemy(engine=engine)
        logger.info("Logfire : SQLAlchemy instrumenté.")
    except Exception as e:
        logger.warning(f"Logfire : impossible d'instrumenter SQLAlchemy : {e}")

    logger.info("Logfire initialisé — service: sportsee-rag")
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
