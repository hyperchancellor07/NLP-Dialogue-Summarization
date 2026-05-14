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
        margin-bottom: 0.2rem;
    }

    .subtitle-text {
        font-size: 18px;
        opacity: 0.8;
        margin-bottom: 2rem;
    }

    .stTextArea textarea {

        border-radius: 14px;

        font-size: 16px;

        padding: 1rem;
    }

    div[data-testid="metric-container"] {

        border-radius: 14px;

        padding: 10px;
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
    with lightweight robustness analysis.

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
        • Lightweight Robustness Analysis <br>
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
            "🧠 Generating summary..."
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
        st.markdown(
            "## 📊 Robustness Analysis"
        )
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric(

            "Sparse Similarity",

            scores["Sparse Similarity"],
        )
        metric_col2.metric(

            "HDRS Score",

            scores["HDRS"],
        )
# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(

    "Built using Transformers, PyTorch, Hugging Face, BM25, and Streamlit 🚀"
)
