# ==========================================
# RECOMMENDER ENGINE
# Furniture Recommendation AI
# ==========================================

import numpy as np


# ==========================================
# MIN-MAX NORMALIZATION
# ==========================================

def minmax_normalize(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:

        return np.zeros_like(scores)

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


# ==========================================
# CREATE USER TEXT PROFILE
# ==========================================

def create_user_text_profile(
    user_id,
    models
):

    user_to_index = models[
        "user_to_index"
    ]

    catalog_asin_to_index = models[
        "catalog_asin_to_index"
    ]

    text_norm_all = models[
        "text_norm_all"
    ]

    train_matrix = models[
        "train_interaction_matrix"
    ]

    # --------------------------------------
    # Check user
    # --------------------------------------

    if user_id not in user_to_index:

        return None

    user_idx = user_to_index[
        user_id
    ]

    # --------------------------------------
    # Get products interacted with
    # --------------------------------------

    product_indices = (
        train_matrix[user_idx]
        .indices
    )

    if len(product_indices) == 0:

        return None

    # --------------------------------------
    # Convert ALS product indices
    # to catalog indices
    # --------------------------------------

    asins = []

    product_ids_array = models[
        "product_ids_array"
    ]

    for product_idx in product_indices:

        asin = product_ids_array[
            product_idx
        ]

        if asin in catalog_asin_to_index:

            asins.append(asin)

    # --------------------------------------
    # Get text embedding indices
    # --------------------------------------

    text_indices = [
        catalog_asin_to_index[asin]
        for asin in asins
    ]

    if len(text_indices) == 0:

        return None

    # --------------------------------------
    # Create profile
    # --------------------------------------

    profile = np.mean(
        text_norm_all[text_indices],
        axis=0
    )

    norm = np.linalg.norm(profile)

    if norm == 0:

        return None

    profile = profile / norm

    return profile

# ==========================================
# CREATE USER IMAGE PROFILE
# ==========================================

def create_user_image_profile(
    user_id,
    models
):

    user_to_index = models[
        "user_to_index"
    ]

    train_matrix = models[
        "train_interaction_matrix"
    ]

    product_ids_array = models[
        "product_ids_array"
    ]

    catalog_asin_to_index = models[
        "catalog_asin_to_index"
    ]

    image_full_index_to_row = models[
        "image_full_index_to_row"
    ]

    successful_indices = models[
        "successful_indices"
    ]

    image_embeddings = models[
        "image_embeddings"
    ]

    # --------------------------------------
    # Check user
    # --------------------------------------

    if user_id not in user_to_index:

        return None

    user_idx = user_to_index[
        user_id
    ]

    # --------------------------------------
    # Get user's training products
    # --------------------------------------

    product_indices = (
        train_matrix[user_idx]
        .indices
    )

    if len(product_indices) == 0:

        return None

    image_vectors = []

    # --------------------------------------
    # Collect image embeddings
    # --------------------------------------

    for product_idx in product_indices:

        asin = product_ids_array[
            product_idx
        ]

        if asin not in catalog_asin_to_index:

            continue

        catalog_index = (
            catalog_asin_to_index[asin]
        )

        image_row = (
            image_full_index_to_row.get(
                int(catalog_index)
            )
        )

        if image_row is None:

            continue

        full_index = successful_indices[
            image_row
        ]

        image_vector = image_embeddings[
            full_index
        ]

        image_vectors.append(
            image_vector
        )

    # --------------------------------------
    # No images available
    # --------------------------------------

    if not image_vectors:

        return None

    # --------------------------------------
    # Create image profile
    # --------------------------------------

    profile = np.mean(
        image_vectors,
        axis=0
    )

    norm = np.linalg.norm(profile)

    if norm == 0:

        return None

    profile = profile / norm

    return profile

# ==========================================
# RECOMMEND PRODUCTS
# ==========================================

def recommend_products(
    user_id,
    models,
    top_k=10
):

    # --------------------------------------
    # Get model components
    # --------------------------------------

    als_model = models[
        "als_model"
    ]

    user_to_index = models[
        "user_to_index"
    ]

    product_ids_array = models[
        "product_ids_array"
    ]

    text_als = models[
        "text_als"
    ]

    products_df = models[
        "products_df"
    ]

    image_embeddings = models[
        "image_embeddings"
    ]

    successful_indices = models[
        "successful_indices"
    ]

    image_full_index_to_row = models[
        "image_full_index_to_row"
    ]

    catalog_asin_to_index = models[
        "catalog_asin_to_index"
    ]

    # --------------------------------------
    # Check user
    # --------------------------------------

    if user_id not in user_to_index:

        raise ValueError(
            f"User '{user_id}' was not found."
        )

    user_idx = user_to_index[
        user_id
    ]

    # --------------------------------------
    # ALS scores
    # --------------------------------------

    als_scores = (
        als_model.user_factors[user_idx]
        @
        als_model.item_factors.T
    )

    # --------------------------------------
    # Text profile
    # --------------------------------------

    text_profile = (
        create_user_text_profile(
            user_id,
            models
        )
    )

    if text_profile is not None:

        text_scores = (
            text_als
            @
            text_profile
        )

    else:

        text_scores = np.zeros(
            len(product_ids_array),
            dtype=np.float32
        )

    # --------------------------------------
    # Image profile
    # --------------------------------------

    image_profile = (
        create_user_image_profile(
            user_id,
            models
        )
    )

    image_scores = np.zeros(
        len(product_ids_array),
        dtype=np.float32
    )

    if image_profile is not None:

        successful_image_scores = (
            image_embeddings[
                successful_indices
            ]
            @
            image_profile
        )

        for als_idx, full_index in enumerate(
            product_ids_array
        ):

            catalog_index = (
                catalog_asin_to_index.get(
                    full_index
                )
            )

            if catalog_index is None:

                continue

            image_row = (
                image_full_index_to_row.get(
                    int(catalog_index)
                )
            )

            if image_row is not None:

                image_scores[als_idx] = (
                    successful_image_scores[
                        image_row
                    ]
                )

    # --------------------------------------
    # Normalize scores
    # --------------------------------------

    als_normalized = (
        minmax_normalize(
            als_scores
        )
    )

    text_normalized = (
        minmax_normalize(
            text_scores
        )
    )

    image_normalized = (
        minmax_normalize(
            image_scores
        )
    )

    # --------------------------------------
    # Hybrid score
    # --------------------------------------

    config = models[
        "hybrid_config"
    ]

    als_weight = config[
        "als_weight"
    ]

    text_weight = config[
        "text_weight"
    ]

    image_weight = config[
        "image_weight"
    ]

    hybrid_scores = (
        als_weight * als_normalized
        +
        text_weight * text_normalized
        +
        image_weight * image_normalized
    )

    # --------------------------------------
    # Remove already seen products
    # --------------------------------------

    # --------------------------------------
    # Get already-seen products
    # --------------------------------------

    user_idx = user_to_index[user_id]

    seen_product_indices = (
        models["train_interaction_matrix"]
        [user_idx]
        .indices
    )

    seen_products = set(
        product_ids_array[
            seen_product_indices
        ]
    )

    hybrid_scores = hybrid_scores.copy()

    for index, asin in enumerate(
        product_ids_array
    ):

        if asin in seen_products:

            hybrid_scores[index] = -np.inf

    # --------------------------------------
    # Top K
    # --------------------------------------

    top_indices = np.argsort(
        -hybrid_scores
    )[:top_k]

    # --------------------------------------
    # Build results
    # --------------------------------------

    results = []

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        asin = product_ids_array[
            index
        ]

        rows = products_df[
            products_df["parent_asin"]
            == asin
        ]

        if len(rows) > 0:

            row = rows.iloc[0]

            title = row.get(
                "title",
                "Unknown"
            )

        else:

            title = "Unknown"

        results.append({

            "rank": rank,

            "parent_asin": str(
            asin
            ),

        "title": str(
            title
        ),

        "image_url": str(
            row.get(
                "image_url",
                ""
            )
        ),

        "als_score": float(
            als_normalized[index]
        ),

        "text_score": float(
            text_normalized[index]
        ),

        "image_score": float(
            image_normalized[index]
        ),

        "hybrid_score": float(
            hybrid_scores[index]
        )
    })

    return results