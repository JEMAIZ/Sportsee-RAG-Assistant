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

# ── Questions fréquentes générales ──────────────────────────────────────────
FAQ_GENERAL = [
    ("📊 Top 5 scoreurs",       "Qui sont les 5 meilleurs scoreurs de la saison ? Montre un graphique."),
    ("🎯 Meilleur 3P%",         "Quel joueur a le meilleur pourcentage à 3 points avec au moins 100 tentatives ?"),
    ("🏆 Top rebondeurs",       "Qui sont les 5 meilleurs rebondeurs de la saison ? Montre un graphique."),
    ("⚡ Meilleur passeur",     "Qui a le plus grand nombre de passes décisives cette saison ?"),
    ("🛡️ Meilleur défenseur",  "Qui a le plus de blocks et de steals combinés cette saison ?"),
    ("📈 Efficacité offensive", "Quels joueurs ont le meilleur True Shooting % avec au moins 500 points ?"),
    ("🔥 Double-doubles",       "Quels joueurs ont le plus de double-doubles cette saison ?"),
    ("⏱️ Top minutes jouées",   "Quels joueurs ont joué le plus de minutes cette saison ?"),
]

# ── Questions fréquentes par équipe ─────────────────────────────────────────
FAQ_TEAM = [
    ("🏀 Stats équipe complètes", "Donne-moi les statistiques complètes de l'équipe {team} : top scoreurs, rebondeurs, passeurs et roster complet."),
    ("🔥 Top scoreurs équipe",    "Qui sont les 5 meilleurs scoreurs des {team} cette saison ? Montre un graphique."),
    ("💪 Top rebondeurs équipe",  "Qui sont les meilleurs rebondeurs des {team} ? Montre un graphique."),
    ("🎯 Top passeurs équipe",    "Qui sont les meilleurs passeurs des {team} cette saison ?"),
    ("🛡️ Défense équipe",        "Quels joueurs des {team} ont le plus de blocks et steals cette saison ?"),
    ("📋 Roster complet",         "Liste tous les joueurs des {team} avec leurs statistiques clés cette saison."),
]

# ── Liste équipes pour le selectbox ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_teams():
    try:
        from utils.database import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT full_name, code FROM teams ORDER BY full_name")
            ).mappings().all()
        return [f"{r['full_name']} ({r['code']})" for r in rows]
    except Exception:
        return []

# ── Chargement agent ──────────────────────────────────────────────────────────
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
            "- 🏀 **Stats par équipe** : sélectionnez une équipe dans la sidebar\n"
            "- 📈 **Visualisations** : *Montre-moi un graphique des top scoreurs*\n\n"
            "Utilisez les **questions fréquentes** dans la sidebar ou tapez votre question."
        ),
        "image_path": None
    }]

if "faq_prompt" not in st.session_state:
    st.session_state["faq_prompt"] = None

# ── Fonction utilitaire : extraire proprement le chemin PNG ──────────────────
def extract_img_path(raw: str):
    match = re.search(r'([A-Za-z]:[^\n]*?\.png|/[^\n]*?\.png)', raw)
    if match:
        img_path = match.group(1).rstrip(")").strip()
        extra    = raw[match.end():].lstrip(")\n ").strip()
        return img_path, extra
    return None, raw.strip()

# ── Fonction principale : envoyer un prompt à l'agent ────────────────────────
def run_prompt(prompt: str):
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
                        parts           = raw_output.split("GRAPH_FILE:", 1)
                        text_before     = parts[0].strip()
                        img_path, extra = extract_img_path(parts[1])
                        response_text   = " ".join(filter(None, [text_before, extra])) or "Voici le graphique :"
                    elif "GRAPH_BASE64:" in raw_output:
                        import base64
                        from utils.config import OUTPUTS_DIR
                        parts         = raw_output.split("GRAPH_BASE64:", 1)
                        response_text = parts[0].strip() or "Voici le graphique :"
                        b64_raw = parts[1].strip()
                        if "base64," in b64_raw:
                            b64_raw = b64_raw.split("base64,", 1)[1]
                        b64_raw  = b64_raw.split("\n")[0].rstrip(")")
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
                st.image(Path(img_path).read_bytes(), use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible d'afficher l'image : {e}")

    st.session_state.messages.append({
        "role": "assistant", "content": response_text, "image_path": img_path
    })

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🏀 SportSee Assistant")

    # ── Section Stats générales ──────────────────────────────────────────────
    st.subheader("📊 Stats générales")
    for label, question in FAQ_GENERAL:
        if st.button(label, use_container_width=True, key=f"gen_{label}"):
            st.session_state["faq_prompt"] = question

    st.divider()

    # ── Section Stats par équipe ─────────────────────────────────────────────
    st.subheader("🏀 Stats par équipe")

    teams = get_teams()
    if teams:
        selected_team = st.selectbox(
            "Choisir une équipe",
            options=teams,
            index=None,
            placeholder="Sélectionner une équipe...",
            key="team_select"
        )

        if selected_team:
            # Extrait juste le nom court (ex: "Los Angeles Lakers (LAL)" → "Lakers")
            team_label = selected_team.split("(")[0].strip()

            for label, question_template in FAQ_TEAM:
                question = question_template.replace("{team}", team_label)
                if st.button(label, use_container_width=True, key=f"team_{label}"):
                    st.session_state["faq_prompt"] = question
        else:
            st.caption("👆 Sélectionnez une équipe pour voir les questions disponibles")
    else:
        st.warning("Base de données non disponible.")

    st.divider()

    # ── Effacer conversation ─────────────────────────────────────────────────
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Conversation effacée. Comment puis-je vous aider ? 🏀",
            "image_path": None
        }]
        st.rerun()

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
                st.image(Path(img_path).read_bytes(), use_container_width=True)
            except Exception as e:
                st.warning(f"Image non disponible : {e}")

# ── Déclenchement question FAQ ────────────────────────────────────────────────
if st.session_state["faq_prompt"]:
    prompt = st.session_state["faq_prompt"]
    st.session_state["faq_prompt"] = None
    run_prompt(prompt)

# ── Saisie manuelle ───────────────────────────────────────────────────────────
if prompt := st.chat_input("Posez votre question NBA..."):
    run_prompt(prompt)

st.markdown("---")
st.caption("SportSee — Powered by LangChain + Mistral AI + FAISS")
