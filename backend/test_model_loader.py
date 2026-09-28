from model_loader import load_all_models


models = load_all_models()

print("\n===== LOADER TEST =====")

print(
    "ALS:",
    models["als_model"].user_factors.shape,
    models["als_model"].item_factors.shape
)

print(
    "Text:",
    models["text_embeddings"].shape
)

print(
    "Image:",
    models["image_embeddings"].shape
)

print(
    "Products:",
    models["products_df"].shape
)

print(
    "\n✅ MODEL LOADER TEST PASSED!"
)