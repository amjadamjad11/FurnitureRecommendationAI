# ==========================================
# SEMANTIC PRODUCT SEARCH
# ==========================================

import numpy as np


# ==========================================
# LOAD TEXT EMBEDDINGS
# ==========================================

def load_text_embeddings(
    embeddings_path
):

    embeddings = np.load(
        embeddings_path
    )

    # --------------------------------------
    # Normalize embeddings
    # --------------------------------------

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms = np.maximum(
        norms,
        1e-12
    )

    normalized_embeddings = (
        embeddings / norms
    )

    return normalized_embeddings


# ==========================================
# SEARCH PRODUCTS
# ==========================================

def semantic_search(
    query_embedding,
    product_embeddings,
    top_k=100
):

    # --------------------------------------
    # Normalize query
    # --------------------------------------

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    query_norm = np.linalg.norm(
        query_embedding
    )

    if query_norm == 0:

        raise ValueError(
            "Query embedding has zero norm."
        )

    query_embedding = (
        query_embedding / query_norm
    )


    # --------------------------------------
    # Calculate cosine similarity
    # --------------------------------------

    scores = (
        product_embeddings
        @ query_embedding
    )


    # --------------------------------------
    # Get top products
    # --------------------------------------

    top_k = min(
        top_k,
        len(scores)
    )

    top_indices = np.argpartition(
        scores,
        -top_k
    )[-top_k:]


    # Sort properly
    top_indices = top_indices[
        np.argsort(
            scores[top_indices]
        )[::-1]
    ]


    return (
        top_indices,
        scores[top_indices]
    )