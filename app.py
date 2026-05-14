
import streamlit as st

from src.prediction_pipeline import DialogueSummarizer


st.set_page_config(
    page_title="Dialogue Summarization System",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource

def load_model():
    return DialogueSummarizer()


summarizer = load_model()


st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .title-text {
        font-size: 42px;
        font-weight: 700;
        color: #4F46E5;
        margin-bottom: 0.2rem;
    }

    .subtitle-text {
        font-size: 18px;
        color: #6B7280;
        margin-bottom: 2rem;
    }

    .stTextArea textarea {
        border-radius: 12px;
        font-size: 16px;
    }

    .summary-box {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid #E5E7EB;
        font-size: 17px;
        line-height: 1.7;
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="title-text">🧠 Dialogue Summarization System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle-text">Transformer-based conversational summarization using PEGASUS.</div>',
    unsafe_allow_html=True,
)


col1, col2 = st.columns([1.2, 1])


with col1:

    user_input = st.text_area(
        "💬 Enter Conversation",
        placeholder="Paste conversation here...",
        height=420,
    )

