# ==========================================
# PRODUCT SEARCH ENGINE
# ==========================================

import numpy as np
import pandas as pd


# ==========================================
# LOAD PRODUCT METADATA
# ==========================================

def load_product_search_data(
    metadata_path
):

    products = pd.read_parquet(
        metadata_path
    )

    # Make a copy so the original data
    # is never modified
    products = products.copy()

    # --------------------------------------
    # Clean price
    # --------------------------------------

    products["price"] = pd.to_numeric(
        products["price"],
        errors="coerce"
    )

    # --------------------------------------
    # Clean ratings
    # --------------------------------------

    products["average_rating"] = pd.to_numeric(
        products["average_rating"],
        errors="coerce"
    )

    products["rating_number"] = pd.to_numeric(
        products["rating_number"],
        errors="coerce"
    )

    # --------------------------------------
    # Replace missing values
    # --------------------------------------

    products["average_rating"] = (
        products["average_rating"]
        .fillna(0)
    )

    products["rating_number"] = (
        products["rating_number"]
        .fillna(0)
    )

    products["price"] = (
        products["price"]
        .fillna(np.inf)
    )

    return products


# ==========================================
# CALCULATE REVIEW QUALITY SCORE
# ==========================================

def calculate_review_score(
    products
):

    products = products.copy()

    # --------------------------------------
    # Bayesian-style weighted rating
    # --------------------------------------

    ratings = products[
        "average_rating"
    ]

    review_counts = products[
        "rating_number"
    ]

    # Global average rating
    global_rating = (
        ratings[ratings > 0].mean()
    )

    if pd.isna(global_rating):

        global_rating = 0.0

    # Minimum reviews required
    # before trusting rating strongly
    minimum_reviews = 100

    weighted_rating = (
        (
            review_counts
            /
            (review_counts + minimum_reviews)
        )
        * ratings
        +
        (
            minimum_reviews
            /
            (review_counts + minimum_reviews)
        )
        * global_rating
    )

    products[
        "review_score"
    ] = weighted_rating

    return products


# ==========================================
# PREPARE SEARCH TEXT
# ==========================================

def prepare_search_text(
    products
):

    products = products.copy()

    products["search_text"] = (

        products["title"]
        .fillna("")
        .astype(str)

        + " "

        +

        products["description"]
        .fillna("")
        .astype(str)

        + " "

        +

        products["categories"]
        .fillna("")
        .astype(str)

        + " "

        +

        products["main_category"]
        .fillna("")
        .astype(str)
    )

    return products