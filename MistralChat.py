# MistralChat.py
import streamlit as st
import logging
import base64
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config page (DOIT être le 1er appel Streamlit) ──────────────────────────
try:
    from utils.config import APP_TITLE, NAME, MODEL_NAME
except ImportError as e:
    st.error(f"Erreur d'importation config: {e}")
    st.stop()

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏀",
    layout="wide"
)

# ── Chargement agent mis en cache ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agent():
    """Charge l'agent une seule fois et le met en cache."""
    try:
        from agent import build_agent
        agent = build_agent()
        logger.info("Agent LangChain chargé.")
        return agent
    except Exception as e:
        logger.error(f"Erreur chargement agent: {e}")
        return None

# ── Chargement avec barre de progression ─────────────────────────────────────
if "agent_loaded" not in st.session_state:
    with st.spinner("⏳ Chargement de l'agent NBA (première fois ~30s)..."):
        agent_executor = load_agent()
    st.session_state.agent_loaded = True
    st.session_state.agent = agent_executor
else:
    agent_executor = st.session_state.get("agent") or load_agent()

# ── Initialisation historique ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            f"Bonjour ! Je suis **{APP_TITLE}**, votre analyste IA pour la {NAME}. 🏀\n\n"
            "Je peux répondre à :\n"
            "- 📊 **Questions statistiques** : *Qui a le meilleur 3P% ?*\n"
            "- 📝 **Analyses qualitatives** : basées sur les rapports de matchs\n"
            "- 📈 **Visualisations** : *Montre-moi un graphique des top scoreurs*\n\n"
            "Comment puis-je vous aider ?"
        ),
        "image_bytes": None  # on stocke des BYTES, pas une string base64
    }]

# ── Header ───────────────────────────────────────────────────────────────────
st.title(f"🏀 {APP_TITLE}")
st.caption(f"{NAME} | Modèle: {MODEL_NAME} | Powered by LangChain + Mistral")

# ── Affichage historique ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # FIX : on stocke des bytes → st.image() les accepte nativement
        if message.get("image_bytes"):
            st.image(message["image_bytes"], use_column_width=True)

# ── Traitement nouvelle question ──────────────────────────────────────────────
if prompt := st.chat_input("Posez votre question NBA..."):
    # Ajout message utilisateur
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "image_bytes": None
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse assistant
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyse en cours..."):
            if agent_executor is None:
                response_text = "❌ L'agent n'est pas disponible. Vérifiez la configuration."
                img_bytes = None
            else:
                from langchain_core.messages import HumanMessage, AIMessage

                # Construction historique LangChain
                history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        history.append(AIMessage(content=msg["content"]))

                try:
                    result = agent_executor.invoke({
                        "input": prompt,
                        "chat_history": history
                    })
                    raw_output = result.get("output", "Désolé, pas de réponse.")

                    # Extraction graphique Base64 si présent
                    if "GRAPH_BASE64:" in raw_output:
                        parts = raw_output.split("GRAPH_BASE64:", 1)
                        response_text = parts[0].strip() or "Voici le graphique :"
                        b64_raw = parts[1].strip()

                        # Nettoyage préfixe data URI si présent
                        if "base64," in b64_raw:
                            b64_raw = b64_raw.split("base64,", 1)[1]

                        # FIX : décoder en bytes immédiatement
                        img_bytes = base64.b64decode(b64_raw)
                    else:
                        response_text = raw_output
                        img_bytes = None

                except Exception as e:
                    response_text = f"❌ Erreur : {e}"
                    img_bytes = None

        # Affichage réponse
        st.markdown(response_text)
        if img_bytes:
            st.image(img_bytes, use_column_width=True)

    # Sauvegarde dans l'historique avec BYTES (pas base64 string)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "image_bytes": img_bytes  # bytes ou None
    })

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("SportSee — Powered by LangChain + Mistral AI + FAISS")
