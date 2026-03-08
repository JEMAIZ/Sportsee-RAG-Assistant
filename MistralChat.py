# MistralChat.py
import streamlit as st
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config page ──────────────────────────────────────────────────────────────
try:
    from utils.config import APP_TITLE, NAME, MODEL_NAME
except ImportError as e:
    st.error(f"Erreur d'importation config: {e}")
    st.stop()

st.set_page_config(page_title=APP_TITLE, page_icon="🏀", layout="wide")

# ── Chargement agent (une seule fois, mis en cache) ──────────────────────────
@st.cache_resource(show_spinner=False)
def load_agent():
    try:
        from agent import build_agent
        agent = build_agent()
        logger.info("Agent LangChain chargé.")
        return agent
    except Exception as e:
        logger.error(f"Erreur chargement agent: {e}")
        return None

if "agent_ready" not in st.session_state:
    with st.spinner("⏳ Chargement de l'agent NBA (première fois ~30s)..."):
        st.session_state["agent_executor"] = load_agent()
    st.session_state["agent_ready"] = True

agent_executor = st.session_state.get("agent_executor") or load_agent()

# ── Historique ────────────────────────────────────────────────────────────────
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
        "image_path": None
    }]

# ── Fonction utilitaire : extraire proprement le chemin PNG ──────────────────
def extract_img_path(raw: str):
    """
    Extrait le chemin absolu vers le .png dans l'output LangChain.
    Gère les cas où LangChain colle du texte après le chemin.
    Retourne (img_path, texte_supplementaire).
    """
    match = re.search(r'([A-Za-z]:[^\n]*?\.png|/[^\n]*?\.png)', raw)
    if match:
        img_path = match.group(1).rstrip(")").strip()
        # Texte éventuel après le chemin
        end      = match.end()
        extra    = raw[end:].lstrip(")\n ").strip()
        return img_path, extra
    return None, raw.strip()

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"🏀 {APP_TITLE}")
st.caption(f"{NAME} | Modèle: {MODEL_NAME} | Powered by LangChain + Mistral")

# ── Affichage historique ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        img_path = message.get("image_path")
        if img_path:
            try:
                st.image(Path(img_path).read_bytes(), use_column_width=True)
            except Exception as e:
                st.warning(f"Image non disponible : {e}")

# ── Nouvelle question ─────────────────────────────────────────────────────────
if prompt := st.chat_input("Posez votre question NBA..."):
    st.session_state.messages.append({
        "role": "user", "content": prompt, "image_path": None
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyse en cours..."):
            if agent_executor is None:
                response_text = "❌ L'agent n'est pas disponible. Vérifiez la configuration."
                img_path = None
            else:
                from langchain_core.messages import HumanMessage, AIMessage
                history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        history.append(AIMessage(content=msg["content"]))
                try:
                    result     = agent_executor.invoke({"input": prompt, "chat_history": history})
                    raw_output = result.get("output", "Désolé, pas de réponse.")

                    if "GRAPH_FILE:" in raw_output:
                        parts        = raw_output.split("GRAPH_FILE:", 1)
                        text_before  = parts[0].strip()
                        img_path, extra = extract_img_path(parts[1])
                        # Assemble le texte de réponse
                        response_text = " ".join(filter(None, [text_before, extra])) or "Voici le graphique :"

                    elif "GRAPH_BASE64:" in raw_output:
                        import base64
                        from utils.config import OUTPUTS_DIR
                        parts         = raw_output.split("GRAPH_BASE64:", 1)
                        response_text = parts[0].strip() or "Voici le graphique :"
                        b64_raw = parts[1].strip()
                        if "base64," in b64_raw:
                            b64_raw = b64_raw.split("base64,", 1)[1]
                        b64_raw = b64_raw.split("\n")[0].rstrip(")")
                        tmp_path = Path(OUTPUTS_DIR) / "graphs" / "temp_display.png"
                        tmp_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_path.write_bytes(base64.b64decode(b64_raw))
                        img_path = str(tmp_path)

                    else:
                        response_text = raw_output
                        img_path      = None

                except Exception as e:
                    response_text = f"❌ Erreur : {e}"
                    img_path      = None

        st.markdown(response_text)
        if img_path:
            try:
                st.image(Path(img_path).read_bytes(), use_column_width=True)
            except Exception as e:
                st.warning(f"Impossible d'afficher l'image : {e}")

    st.session_state.messages.append({
        "role": "assistant", "content": response_text, "image_path": img_path
    })

st.markdown("---")
st.caption("SportSee — Powered by LangChain + Mistral AI + FAISS")
