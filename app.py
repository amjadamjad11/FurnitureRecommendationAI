
import sys
from pathlib import Path

import requests
import streamlit as st

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# BACKEND
# ============================================================

from backend.model_loader import load_all_models
from backend.search_pipeline import search_products


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Furniture Recommendation AI",
    page_icon="🪑",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def get_models():
    return load_all_models()


with st.spinner("Loading Furniture Recommendation AI..."):
    models = get_models()


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = None

if "preferences" not in st.session_state:
    st.session_state.preferences = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# MISTRAL
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral:latest"


def ask_mistral(question, products):

    if products is None or len(products) == 0:
        product_context = "No products were found."

    else:
        lines = []

        for i, (_, row) in enumerate(
            products.iterrows(),
            start=1
        ):
            lines.append(
                f"""
Product {i}:
Title: {row.get("title", "Unknown")}
Price: ${row.get("price", "Unknown")}
Rating: {row.get("average_rating", "Unknown")}
Reviews: {row.get("rating_number", "Unknown")}
"""
            )

        product_context = "\n".join(lines)

    system_prompt = f"""
You are a shopping assistant for Furniture Recommendation AI.

Use ONLY the product information supplied below.

Do not invent:
- prices
- ratings
- review counts
- materials
- colors
- features
- availability
- specifications

If information is unavailable, say so.

You can compare the products using the supplied information.
Keep your answer concise and helpful.

CURRENT SEARCH RESULTS:
{product_context}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        return response.json()["message"]["content"]

    except Exception as e:

        return f"Could not connect to Mistral: {e}"


# ============================================================
# HEADER
# ============================================================

st.title("🪑 Furniture Recommendation AI")

st.markdown(
    "### Find furniture using natural-language search"
)

st.caption(
    "Mistral • MiniLM • Image Embeddings • Review-Aware Ranking"
)


# ============================================================
# SEARCH BOX
# ============================================================

query = st.text_input(
    "🔎 What are you looking for?",
    placeholder=(
        "Example: Show me brown wooden dining tables under $100"
    )
)

if st.button(
    "🔍 Search Products",
    type="primary",
    use_container_width=True
):

    if not query.strip():

        st.warning("Please enter a search query.")

    else:

        with st.spinner(
            "Understanding your request and searching..."
        ):

            try:

                preferences, results = search_products(
                    query.strip(),
                    models,
                    top_k=10
                )

                st.session_state.preferences = preferences
                st.session_state.results = results
                st.session_state.messages = []

                st.rerun()

            except Exception as e:

                st.error(
                    f"Search failed: {e}"
                )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.results is not None:

    results = st.session_state.results
    preferences = st.session_state.preferences

    st.divider()

    products_col, assistant_col = st.columns(
        [2.1, 1],
        gap="large"
    )


    # ========================================================
    # PRODUCTS
    # ========================================================

    with products_col:

        st.subheader(
            f"🛍️ Recommendations ({len(results)})"
        )

        if len(results) == 0:

            st.warning(
                "No products matched your search."
            )

        else:

            for number, (_, product) in enumerate(
                results.iterrows(),
                start=1
            ):

                image_col, info_col = st.columns(
                    [1, 2.5]
                )

                with image_col:

                    image_url = product.get(
                        "image_url"
                    )

                    if (
                        image_url
                        and str(image_url) != "nan"
                    ):

                        try:
                            st.image(
                                image_url,
                                use_container_width=True
                            )
                        except:
                            st.write("🪑")

                    else:
                        st.write("🪑")


                with info_col:

                    title = product.get(
                        "title",
                        "Unknown product"
                    )

                    price = product.get(
                        "price",
                        "N/A"
                    )

                    rating = product.get(
                        "average_rating",
                        "N/A"
                    )

                    reviews = product.get(
                        "rating_number",
                        "N/A"
                    )

                    st.markdown(
                        f"### {number}. {title}"
                    )

                    st.write(
                        f"💰 **${price}**"
                    )

                    st.write(
                        f"⭐ **{rating}** "
                        f"({reviews} reviews)"
                    )

                    description = str(
                        product.get(
                            "description",
                            ""
                        )
                    )

                    if (
                        description
                        and description != "nan"
                    ):

                        if len(description) > 300:
                            description = (
                                description[:300]
                                + "..."
                            )

                        st.caption(description)

                st.divider()


    # ========================================================
    # AI ASSISTANT
    # ========================================================

    with assistant_col:

        st.subheader(
            "🤖 AI Shopping Assistant"
        )

        st.caption(
            "Ask questions about the products above."
        )

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):
                st.markdown(
                    message["content"]
                )

        question = st.chat_input(
            "Ask about these products..."
        )

        if question:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                with st.spinner(
                    "Mistral is thinking..."
                ):

                    answer = ask_mistral(
                        question,
                        st.session_state.results
                    )

                st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                })


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Search Information")

    if st.session_state.preferences:

        st.write("**Detected preferences**")

        for key, value in (
            st.session_state.preferences.items()
        ):

            if value is not None and value != "":

                st.write(
                    f"**{key.replace('_', ' ').title()}:** "
                    f"{value}"
                )

    st.divider()

    st.caption(
        "Furniture Recommendation AI"
    )

    st.caption(
        "Natural-language multimodal furniture discovery."
    )
