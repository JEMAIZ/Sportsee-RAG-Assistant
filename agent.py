# agent.py
import logging
from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai import ChatMistralAI

from utils.config import MISTRAL_API_KEY, MODEL_NAME

logger = logging.getLogger(__name__)


@tool
def rag_tool(question: str) -> str:
    """
    Recherche dans les documents textuels (rapports de matchs, analyses)
    via le vector store FAISS. A utiliser pour les questions sur le contexte