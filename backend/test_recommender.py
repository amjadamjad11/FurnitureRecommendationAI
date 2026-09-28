# ==========================================
# TEST STANDALONE RECOMMENDER
# Furniture Recommendation AI
# ==========================================

from model_loader import load_all_models
from recommender import recommend_products


# ==========================================
# LOAD MODEL
# ==========================================

print("===== STARTING RECOMMENDER TEST =====")

models = load_all_models()


# ==========================================
# TEST USER
# ==========================================

test_user = "AE2222VGHFD37G7NOCWXASYY3ZDQ"


print("\n===== GENERATING RECOMMENDATIONS =====")

print(
    "User:",
    test_user
)


# ==========================================
# GENERATE RECOMMENDATIONS
# ==========================================

recommendations = recommend_products(
    user_id=test_user,
    models=models,
    top_k=10
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print(
    "\n===== TOP 10 RECOMMENDATIONS ====="
)

for item in recommendations:

    print(
        f"\n{item['rank']}. "
        f"{item['title']}"
    )

    print(
        f"   ASIN: "
        f"{item['parent_asin']}"
    )

    print(
        f"   ALS: "
        f"{item['als_score']:.4f}"
    )

    print(
        f"   Text: "
        f"{item['text_score']:.4f}"
    )

    print(
        f"   Image: "
        f"{item['image_score']:.4f}"
    )

    print(
        f"   Hybrid: "
        f"{item['hybrid_score']:.4f}"
    )


# ==========================================
# FINAL CHECK
# ==========================================

print(
    "\n=========================================="
)

print(
    "Number of recommendations:",
    len(recommendations)
)

if len(recommendations) == 10:

    print(
        "✅ STANDALONE RECOMMENDER TEST PASSED!"
    )

else:

    print(
        "⚠️ Expected 10 recommendations."
    )

print(
    "=========================================="
)