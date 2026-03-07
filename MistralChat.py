# MistralChat.py
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


@st.cache_resource
def load_agent():
    try:
        from agent import build_agent
        agent = build_agent()
        logger.info("Agent LangChain charge.")
        return agent
    except Exception as e:
        logger.error(f"Erreur chargement agent: {e}")
        return None


agent_executor = load_agent()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role":    "assistant",
        "content": (
            f"Bonjour ! Je suis **{APP_TITLE}**, votre analyste IA pour la {NAME}. 🏀\n\n"
            "Je peux repondre a :\n"
            "- 📊 **Questions statistiques** : *Qui a le meilleur 3P% ?*\n"
            "- 📝 **Analyses qualitatives** : basees sur les rapports de matchs\n"
            "- 📈 **Visualisations** : *Montre-moi un graphique des top scoreurs*\n\n"
            "Comment puis-je vous aider ?"
        ),
        "image": None
    }]

st.title(f"🏀 {APP_TITLE}")
st.caption(f"{NAME} | Modele: {MODEL_NAME} | Powered by LangChain + Mistral")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image"):
            st.image(message["image"], use_column_width=True)

if prompt := st.chat_input("Posez votre question NBA..."):

    st.session_state.messages.append({"role": "user", "content": prompt, "image": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyse en cours..."):

            if agent_executor is None:
                response_text = "L'agent n'est pas disponible. Verifiez la configuration."
                graph_data    = None
            else:
                from langchain_core.messages import HumanMessage, AIMessage
                history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        history.append(AIMessage(content=msg["content"]))
                try:
                    result     = agent_executor.invoke({
                        "input":        prompt,
                        "chat_history": history
                    })
                    raw_output = result.get("output", "Desole, pas de reponse.")

                    if "GRAPH_BASE64:" in raw_output:
                        parts         = raw_output.split("GRAPH_BASE64:", 1)
                        response_text = parts[0].strip() or "Voici le graphique :"
                        graph_data    = parts[1].strip()
                    else:
                        response_text = raw_output
                        graph_data    = None

                except Exception as e:
                    response_text = f"Erreur : {e}"
                    graph_data    = None

        st.markdown(response_text)
      
        if graph_data:
            import base64, io 
            from PIL import Image
            # Supprime le prefixe data:image/png;base64,
            b64_data = graph_data.replace("data:image/png;base64,", "")
            img_bytes = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_bytes))
            st.image(img, use_column_width=True)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": response_text,
        "image":   graph_data
    })

st.markdown("---")
st.caption("SportSee — Powered by LangChain + Mistral AI + FAISS")