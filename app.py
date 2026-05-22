import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import plotly.graph_objects as go
import time
import json
import os
from deep_translator import GoogleTranslator
from langdetect import detect


# ==========================
# Configuration de la page
# ==========================

st.set_page_config(
    page_title="Détecteur d'Émotions",
    page_icon="🎭",
    layout="centered"
)


# ==========================
# Constantes
# ==========================

MODEL_PATH = "emotion_distilbert_model"
CONFIDENCE_THRESHOLD = 0.60

EMOJI_MAP = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲"
}

COLOR_MAP = {
    "sadness": "#1E88E5",
    "joy": "#FDD835",
    "love": "#E53935",
    "anger": "#FB8C00",
    "fear": "#8E24AA",
    "surprise": "#00ACC1"
}

LABEL_FR = {
    "sadness": "Tristesse",
    "joy": "Joie",
    "love": "Amour",
    "anger": "Colère",
    "fear": "Peur",
    "surprise": "Surprise",
    "uncertain": "Incertain"
}


# ==========================
# Chargement des labels
# ==========================

def load_labels():
    labels_path = os.path.join(MODEL_PATH, "labels.json")

    if not os.path.exists(labels_path):
        st.error("Erreur : fichier labels.json introuvable dans le dossier du modèle.")
        st.stop()

    with open(labels_path, "r", encoding="utf-8") as f:
        id2label = json.load(f)

    label_names = [id2label[str(i)] for i in range(len(id2label))]
    return id2label, label_names


id2label, LABEL_NAMES = load_labels()


# ==========================
# Chargement du modèle
# ==========================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


# ==========================
# Gestion du texte
# ==========================

def set_example_text(example_text):
    st.session_state["text_input"] = example_text


def clear_text():
    st.session_state["text_input"] = ""


# ==========================
# Traduction FR → EN
# ==========================

def translate_to_english_if_needed(text: str):
    try:
        detected_lang = detect(text)

        if detected_lang == "fr":
            translated_text = GoogleTranslator(source="fr", target="en").translate(text)
            return translated_text, detected_lang

        return text, detected_lang

    except Exception:
        return text, "unknown"


# ==========================
# Fonction de prédiction
# ==========================

def predict(text: str, tokenizer, model, device, threshold: float = CONFIDENCE_THRESHOLD):
    processed_text, detected_lang = translate_to_english_if_needed(text)

    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.detach().cpu().numpy()[0]

    probs = softmax(logits)

    pred_idx = int(np.argmax(probs))
    pred_label = LABEL_NAMES[pred_idx]
    pred_score = float(probs[pred_idx])

    if pred_score < threshold:
        final_label = "uncertain"
    else:
        final_label = pred_label

    return final_label, pred_score, probs, processed_text, detected_lang


# ==========================
# CSS
# ==========================

st.markdown("""
<style>
    .result-box {
        padding: 20px 24px;
        border-radius: 16px;
        text-align: center;
        margin: 16px 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .uncertain-box {
        background: #F5F5F5;
        border: 2px dashed #9E9E9E;
        color: #616161;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ==========================
# Interface principale
# ==========================

st.title("🎭 Détecteur d'Émotions")
st.caption("DistilBERT fine-tuné pour la classification multiclasses des émotions textuelles.")
st.divider()

with st.spinner("Chargement du modèle..."):
    tokenizer, model, device = load_model()

st.success("✅ Modèle chargé !", icon="🤖")


# ==========================
# Zone de saisie
# ==========================

st.subheader("Entrez votre texte")

if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""

text_input = st.text_area(
    label="Texte à analyser",
    placeholder="Exemple : Je suis très heureux aujourd’hui.",
    height=120,
    label_visibility="collapsed",
    key="text_input"
)


# ==========================
# Paramètres avancés
# ==========================

with st.expander("⚙️ Paramètres avancés"):
    threshold = st.slider(
        "Seuil de confiance minimum",
        min_value=0.40,
        max_value=0.95,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="En-dessous de ce seuil, le modèle retourne 'incertain'."
    )


# ==========================
# Boutons principaux
# ==========================

col_btn, col_clear = st.columns([3, 1])

with col_btn:
    analyze = st.button(
        "🔍 Analyser",
        type="primary",
        use_container_width=True,
        key="analyze_button"
    )

with col_clear:
    st.button(
        "🗑️ Effacer",
        use_container_width=True,
        on_click=clear_text,
        key="clear_button"
    )


# ==========================
# Résultats
# ==========================

if analyze and text_input.strip():

    with st.spinner("Analyse en cours..."):
        time.sleep(0.3)
        label, score, probs, processed_text, detected_lang = predict(
            text_input,
            tokenizer,
            model,
            device,
            threshold
        )

    st.divider()

    if detected_lang == "fr":
        st.info(f"Texte détecté en français. Traduction utilisée : **{processed_text}**")
    elif detected_lang == "en":
        st.info("Texte détecté en anglais.")
    else:
        st.info(f"Langue détectée : {detected_lang}")

    if label == "uncertain":
        st.markdown(
            f"""
            <div class="result-box uncertain-box">
                🤔 Incertain — confiance trop faible ({score * 100:.1f}%)
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        emoji = EMOJI_MAP.get(label, "❓")
        color = COLOR_MAP.get(label, "#616161")
        label_fr = LABEL_FR.get(label, label)

        st.markdown(
            f"""
            <div class="result-box" style="
                background:{color}22;
                border: 3px solid {color};
                color:{color};
            ">
                {emoji} {label_fr.upper()}
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Émotion détectée", f"{emoji} {label_fr}")
        c2.metric("Confiance", f"{score * 100:.1f}%")
        c3.metric("Statut", "✅ Fiable" if score >= threshold else "⚠️ Limite")

    # Graphique des probabilités
    st.subheader("📊 Distribution des probabilités")

    sorted_idx = np.argsort(probs)[::-1]
    sorted_labels = [LABEL_NAMES[i] for i in sorted_idx]
    sorted_probs = [float(probs[i]) for i in sorted_idx]
    sorted_emojis = [EMOJI_MAP.get(label, "❓") for label in sorted_labels]
    sorted_colors = [COLOR_MAP.get(label, "#616161") for label in sorted_labels]
    sorted_labels_fr = [LABEL_FR.get(label, label) for label in sorted_labels]

    fig = go.Figure(go.Bar(
        x=[f"{emoji} {label}" for emoji, label in zip(sorted_emojis, sorted_labels_fr)],
        y=[prob * 100 for prob in sorted_probs],
        marker_color=sorted_colors,
        text=[f"{prob * 100:.1f}%" for prob in sorted_probs],
        textposition="outside"
    ))

    fig.update_layout(
        yaxis_title="Probabilité (%)",
        yaxis_range=[0, 115],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=20, b=20),
        height=350
    )

    fig.add_hline(
        y=threshold * 100,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Seuil ({threshold * 100:.0f}%)"
    )

    st.plotly_chart(fig, use_container_width=True)

elif analyze and not text_input.strip():
    st.warning("⚠️ Veuillez entrer du texte avant d’analyser.")


# ==========================
# Exemples français uniquement
# ==========================

st.divider()
st.subheader("Exemples")

french_examples = {
    "😄 Joie": "Je suis très heureux aujourd’hui, tout se passe bien.",
    "😢 Tristesse": "Je me sens triste et seul ce soir.",
    "😡 Colère": "Je suis vraiment en colère à cause de ce qui s’est passé.",
    "😨 Peur": "J’ai peur de l’examen de demain.",
    "❤️ Amour": "J’aime beaucoup ma famille, elle compte énormément pour moi.",
    "😲 Surprise": "Je ne m’attendais pas du tout à cette nouvelle, c’est incroyable !",
}

cols_fr = st.columns(3)

for idx, (btn_label, example_text) in enumerate(french_examples.items()):
    with cols_fr[idx % 3]:
        st.button(
            btn_label,
            use_container_width=True,
            on_click=set_example_text,
            args=(example_text,),
            key=f"fr_example_{idx}"
        )


# ==========================
# Footer
# ==========================

st.divider()
st.caption(
    "Modèle : DistilBERT · Traduction automatique FR→EN pour les textes français · "
    "6 classes : tristesse, joie, amour, colère, peur, surprise."
)
