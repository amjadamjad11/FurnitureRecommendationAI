"""
AI Shopping Agent

This module analyzes products returned by our semantic search
system and provides simple buying advice.

Important:
The agent uses the actual product data from our dataset.
It does not invent prices, ratings, or review counts.
"""


import pandas as pd
import numpy as np


def find_cheapest(products):
    """
    Find the product with the lowest available price.

    Parameters
    ----------
    products : pandas.DataFrame
        Products returned by the product search system.

    Returns
    -------
    pandas.Series or None
        The cheapest product.
    """

    available = products[
        np.isfinite(products["price"])
    ]

    if len(available) == 0:
        return None

    return available.loc[
        available["price"].idxmin()
    ]


def find_highest_rated(products):
    """
    Find the product with the highest average rating.

    If two products have the same rating,
    the one with more reviews is preferred.
    """

    available = products[
        products["average_rating"] > 0
    ].copy()

    if len(available) == 0:
        return None

    available = available.sort_values(
        ["average_rating", "rating_number"],
        ascending=[False, False]
    )

    return available.iloc[0]


def find_most_reviewed(products):
    """
    Find the product with the largest number of reviews.
    """

    if len(products) == 0:
        return None

    return products.loc[
        products["rating_number"].idxmax()
    ]


def find_best_value(products):
    """
    Find the product with the best balance of:

    - Semantic relevance
    - Rating
    - Number of reviews
    - Price

    The product search system has already calculated
    'final_score', so we use that score here.
    """

    if len(products) == 0:
        return None

    return products.loc[
        products["final_score"].idxmax()
    ]


def create_product_summary(product):
    """
    Convert one product row into a human-readable summary.
    """

    if product is None:
        return "No suitable product was found."

    title = str(product["title"])

    price = product["price"]

    if np.isfinite(price):
        price_text = f"${price:,.2f}"
    else:
        price_text = "Price unavailable"

    rating = product["average_rating"]

    reviews = int(
        product["rating_number"]
    )

    semantic_score = product.get(
        "semantic_score",
        0
    )

    return (
        f"{title}\n"
        f"Price: {price_text}\n"
        f"Rating: {rating:.1f}/5\n"
        f"Reviews: {reviews:,}\n"
        f"Relevance: {semantic_score:.3f}"
    )


def generate_buying_advice(products, query):
    """
    Generate a complete buying analysis.

    This function identifies:

    1. Best overall value
    2. Cheapest product
    3. Highest-rated product
    4. Most-reviewed product

    It then creates a simple recommendation.
    """

    if products is None or len(products) == 0:

        return {
            "query": query,
            "message": (
                "I could not find products matching "
                "your requirements."
            ),
            "best_value": None,
            "cheapest": None,
            "highest_rated": None,
            "most_reviewed": None
        }

    best_value = find_best_value(products)

    cheapest = find_cheapest(products)

    highest_rated = find_highest_rated(products)

    most_reviewed = find_most_reviewed(products)

    return {
        "query": query,

        "message": (
            f"I found {len(products)} products matching "
            f"your search for '{query}'."
        ),

        "best_value": best_value,

        "cheapest": cheapest,

        "highest_rated": highest_rated,

        "most_reviewed": most_reviewed
    }