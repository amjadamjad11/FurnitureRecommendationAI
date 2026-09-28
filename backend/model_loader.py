# ==========================================
# MODEL LOADER
# Furniture Recommendation AI
# ==========================================

from pathlib import Path
import pickle
from scipy.sparse import load_npz
import numpy as np
import pandas as pd


# ==========================================
# PATHS
# ==========================================

FINAL_MODEL_DIR = Path(
    r"E:\FurnitureRecommendationAI_ImageEmbeddings\FINAL_MODEL"
)

IMAGE_EMBEDDING_FILE = Path(
    r"E:\FurnitureRecommendationAI_ImageEmbeddings\image_embeddings.dat"
)


# ==========================================
# CHECK FILES
# ==========================================

def check_required_files():

    required_files = [
        "als_model_v2.npz",
        "text_embeddings.npy",
        "text_norm_all.npy",
        "text_als.npy",
        "products.parquet",
        "user_to_index.pkl",
        "product_to_index.pkl",
        "product_ids_array.npy",
        "catalog_asins.npy",
        "catalog_asin_to_index.pkl",
        "image_full_index_to_row.pkl",
        "successful_indices.npy",
        "hybrid_config.pkl",
    ]

    missing_files = []

    for filename in required_files:

        path = FINAL_MODEL_DIR / filename

        if not path.exists():
            missing_files.append(filename)

    if not IMAGE_EMBEDDING_FILE.exists():
        missing_files.append(
            "image_embeddings.dat"
        )

    if missing_files:

        raise FileNotFoundError(
            "Missing model files:\n"
            + "\n".join(
                f" - {file}"
                for file in missing_files
            )
        )


# ==========================================
# LOAD ALS MODEL
# ==========================================

def load_als_model():

    from implicit.cpu.als import (
        AlternatingLeastSquares as CPU_ALS
    )

    model_path = (
        FINAL_MODEL_DIR / "als_model_v2.npz"
    )

    # Read saved parameters
    with np.load(
        model_path,
        allow_pickle=False
    ) as data:

        user_factors = data[
            "user_factors"
        ]

        item_factors = data[
            "item_factors"
        ]

        factors = int(
            data["factors"]
        )

        regularization = float(
            data["regularization"]
        )

        iterations = int(
            data["iterations"]
        )

        num_threads = int(
            data["num_threads"]
        )

    # Create ALS object
    model = CPU_ALS(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        num_threads=num_threads
    )

    # Restore learned factors
    model.user_factors = user_factors
    model.item_factors = item_factors

    return model


# ==========================================
# LOAD PICKLE
# ==========================================

def load_pickle(filename):

    with open(
        FINAL_MODEL_DIR / filename,
        "rb"
    ) as f:

        return pickle.load(f)


# ==========================================
# LOAD ALL MODEL COMPONENTS
# ==========================================

def load_all_models():

    # --------------------------------------
    # TRAINING INTERACTION MATRIX
    # --------------------------------------

    train_interaction_matrix = load_npz(
        FINAL_MODEL_DIR /
        "train_interaction_matrix.npz"
    )

    print(
        "✅ Training interaction matrix loaded"
    )

    print(
        "Training matrix:",
        train_interaction_matrix.shape
    )

    print(
        "Training interactions:",
        train_interaction_matrix.nnz
    )
    

    print(
        "===== LOADING FINAL RECOMMENDATION MODEL ====="
    )

    # Check everything first
    check_required_files()

    print("✅ All required files found")


    # --------------------------------------
    # ALS
    # --------------------------------------

    # --------------------------------------
# ALS
# --------------------------------------

# ALS is optional for the final natural-language
# product search application.
# The current app uses semantic search with
# MiniLM + BLIP instead.
    als_model = None

    print(
        "⚠️ ALS model skipped "
        "(not required for semantic product search)"
    )

    # --------------------------------------
    # TEXT
    # --------------------------------------

    text_embeddings = np.load(
        FINAL_MODEL_DIR /
        "text_embeddings.npy"
    )

    text_norm_all = np.load(
        FINAL_MODEL_DIR /
        "text_norm_all.npy"
    )

    text_als = np.load(
        FINAL_MODEL_DIR /
        "text_als.npy"
    )

    print(
        "✅ Text embeddings loaded"
    )


    # --------------------------------------
    # PRODUCTS
    # --------------------------------------

    products_df = pd.read_parquet(
        FINAL_MODEL_DIR /
        "products.parquet"
    )

    print(
        "✅ Product data loaded"
    )

    # --------------------------------------
    # ALS MAPPINGS
    # --------------------------------------
    # ALS is not used by the final natural-language
    # product search pipeline, so these user/product
    # mappings are optional.
    user_to_index = {}
    product_to_index = {}

    product_ids_array = np.load(
        FINAL_MODEL_DIR / "product_ids_array.npy",
        allow_pickle=True
    )

    print(
        "⚠️ ALS mappings skipped "
        "(not required for semantic product search)"
    )


    # --------------------------------------
    # CATALOG MAPPINGS
    # --------------------------------------

    catalog_asins = np.load(
        FINAL_MODEL_DIR /
        "catalog_asins.npy",
        allow_pickle=True
    )

    catalog_asin_to_index = load_pickle(
        "catalog_asin_to_index.pkl"
    )

    print(
        "✅ Catalog mappings loaded"
    )


    # --------------------------------------
    # IMAGE LOOKUP
    # --------------------------------------

    image_full_index_to_row = load_pickle(
        "image_full_index_to_row.pkl"
    )

    successful_indices = np.load(
        FINAL_MODEL_DIR /
        "successful_indices.npy"
    )

    print(
        "✅ Image lookup loaded"
    )


    # --------------------------------------
    # IMAGE EMBEDDINGS
    # --------------------------------------

    image_embeddings = np.memmap(
        IMAGE_EMBEDDING_FILE,
        dtype=np.float32,
        mode="r",
        shape=(30000, 2048)
    )

    print(
        "✅ Image embeddings loaded"
    )


    # --------------------------------------
    # HYBRID CONFIG
    # --------------------------------------

    hybrid_config = load_pickle(
        "hybrid_config.pkl"
    )

    print(
        "✅ Hybrid configuration loaded"
    )


    # ======================================
    # RETURN EVERYTHING
    # ======================================

    models = {

        "train_interaction_matrix": train_interaction_matrix,

        "als_model": als_model,

        "text_embeddings": text_embeddings,

        "text_norm_all": text_norm_all,

        "text_als": text_als,

        "products_df": products_df,

        "user_to_index": user_to_index,

        "product_to_index": product_to_index,

        "product_ids_array": product_ids_array,

        "catalog_asins": catalog_asins,

        "catalog_asin_to_index":
            catalog_asin_to_index,

        "image_full_index_to_row":
            image_full_index_to_row,

        "successful_indices":
            successful_indices,

        "image_embeddings":
            image_embeddings,

        "hybrid_config":
            hybrid_config,
    }


    print(
        "\n=========================================="
    )

    print(
        "FINAL MODEL LOADED SUCCESSFULLY"
    )

    print(
        "=========================================="
    )

    print(
        "Users:",
        len(user_to_index)
    )

    print(
        "ALS products:",
        len(product_ids_array)
    )

    print(
        "Catalog products:",
        len(products_df)
    )

    print(
        "Text embeddings:",
        text_embeddings.shape
    )

    print(
        "Image embeddings:",
        image_embeddings.shape
    )

    print(
        "Successful images:",
        len(successful_indices)
    )

    print(
        "=========================================="
    )

    return models