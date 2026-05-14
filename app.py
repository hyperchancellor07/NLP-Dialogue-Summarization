import time

import streamlit as st

from src.prediction_pipeline import (
    DialogueSummarizer,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="Dialogue Summarization System",

    page_icon="🧠",

    layout="wide",
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return DialogueSummarizer()


summarizer = load_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .title-text {
        font-size: 48px;
        font-weight: 800;
        color: #6366F1;
        margin-bottom: 0.2rem;
    }

    .subtitle-text {
        font-size: 18px;
        color: #9CA3AF;
        margin-bottom: 2rem;
    }

    .stTextArea textarea {
        border-radius: 14px;
        font-size: 16px;
        padding: 1rem;
    }

    .summary-container {

        background-color: #111827;

        padding: 1.5rem;

        border-radius: 18px;

        border: 1px solid #374151;

        margin-top: 1rem;
    }

    .summary-title {

        font-size: 22px;

        font-weight: 700;

        color: #F9FAFB;

        margin-bottom: 1rem;
    }

    .summary-text {

        font-size: 18px;

        line-height: 1.8;

        color: #E5E7EB;
    }

    .feature-card {

        background-color: #1F2937;

        padding: 1rem;

        border-radius: 14px;

        margin-bottom: 1rem;

        border: 1px solid #374151;
    }

    .metric-container {

        background-color: #111827;

        padding: 1rem;

        border-radius: 16px;

        border: 1px solid #374151;

        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(

    '<div class="title-text">🧠 Dialogue Summarization System</div>',

    unsafe_allow_html=True,
)


st.markdown(

    '''
    <div class="subtitle-text">

    Transformer-based conversational summarization
    with semantic robustness analysis.

    </div>
    ''',

    unsafe_allow_html=True,
)


# ============================================================
# LAYOUT
# ============================================================

left_col, right_col = st.columns(
    [1.5, 1]
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_col:

    user_input = st.text_area(

        "💬 Enter Conversation",

        placeholder="""
John: Are we still meeting tomorrow?

Sarah: Yes, around 5 PM.

John: Great, I will bring the project files.

Sarah: Perfect.
        """,

        height=420,
    )


    generate_button = st.button(

        "🚀 Generate Summary",

        use_container_width=True,
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with right_col:

    st.markdown(
        "## 📌 System Information"
    )


    st.markdown(
        """
        <div class="feature-card">

        <h4>🤖 Model</h4>

        PEGASUS Transformer

        </div>
        """,

        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <div class="feature-card">

        <h4>📚 Dataset</h4>

        SAMSum Dialogue Dataset

        </div>
        """,

        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <div class="feature-card">

        <h4>⚡ Features</h4>

        • Dialogue Summarization <br>
        • Semantic Robustness Analysis <br>
        • HDRS Scoring <br>
        • Hugging Face Deployment <br>
        • Streamlit Interface

        </div>
        """,

        unsafe_allow_html=True,
    )
# ============================================================
# GENERATION SECTION
# ============================================================
if generate_button:

    if user_input.strip() == "":

        st.warning(
            "⚠️ Please enter dialogue text."
        )

    else:

        with st.spinner(
            "🧠 Generating summary and robustness analysis..."
        ):

            time.sleep(1)

            summary = summarizer.summarize(
                user_input
            )

            scores = summarizer.compute_hdrs(
                user_input,
                summary,
            )
        st.markdown("---")
        # ====================================================
        # SUMMARY OUTPUT
        # ====================================================
        st.markdown(
            """
            <div class="summary-container">

            <div class="summary-title">

            📝 Generated Summary

            </div>
            """,

            unsafe_allow_html=True,
        )
        st.markdown(

            f'<div class="summary-text">{summary}</div>',

            unsafe_allow_html=True,
        )
        st.markdown(
            "</div>",

            unsafe_allow_html=True,
        )
        # ====================================================
        # HDRS ANALYSIS
        # ====================================================
        st.markdown("## 📊 Robustness Analysis")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric(

            "Semantic Similarity",

            scores["semantic_similarity"],
        )
        metric_col2.metric(

            "Sparse Similarity",

            scores["sparse_similarity"],
        )
        metric_col3.metric(

            "HDRS Score",

            scores["HDRS"],
        )
# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(

    "Built using Transformers, PyTorch, Hugging Face, Sentence Transformers, and Streamlit 🚀"
)
