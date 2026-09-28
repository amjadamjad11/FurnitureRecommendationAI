# ==========================================
# TEST PRODUCT SEARCH DATA
# ==========================================

from pathlib import Path

from product_search import (
    load_product_search_data,
    calculate_review_score,
    prepare_search_text
)


# ==========================================
# PATH
# ==========================================

FINAL_MODEL_DIR = Path(
    r"E:\FurnitureRecommendationAI_ImageEmbeddings\FINAL_MODEL"
)

metadata_path = (
    FINAL_MODEL_DIR /
    "product_metadata.parquet"
)


# ==========================================
# LOAD
# ==========================================

print(
    "===== LOADING PRODUCT SEARCH DATA ====="
)

products = load_product_search_data(
    metadata_path
)

print(
    "Products:",
    products.shape
)


# ==========================================
# REVIEW SCORE
# ==========================================

print(
    "\n===== CALCULATING REVIEW SCORES ====="
)

products = calculate_review_score(
    products
)


# ==========================================
# SEARCH TEXT
# ==========================================

print(
    "\n===== PREPARING SEARCH TEXT ====="
)

products = prepare_search_text(
    products
)


# ==========================================
# DISPLAY SAMPLE
# ==========================================

print(
    "\n===== SAMPLE PRODUCTS ====="
)

print(
    products[
        [
            "parent_asin",
            "title",
            "price",
            "average_rating",
            "rating_number",
            "review_score"
        ]
    ].head(10).to_string(
        index=False
    )
)


# ==========================================
# VALIDATION
# ==========================================

print(
    "\n===== VALIDATION ====="
)

print(
    "Price column:",
    products["price"].dtype
)

print(
    "Rating column:",
    products["average_rating"].dtype
)

print(
    "Review count column:",
    products["rating_number"].dtype
)

print(
    "Review score column:",
    products["review_score"].dtype
)

print(
    "Search text created:",
    "search_text" in products.columns
)


print(
    "\n✅ PRODUCT SEARCH DATA TEST PASSED!"
)