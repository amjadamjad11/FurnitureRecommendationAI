
import re
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel

from backend.preference_extractor import (
    extract_product_preferences,
    clean_search_query
)


# ============================================================
# CONFIGURATION
# ============================================================

MINILM_PATH = r"C:\Users\amjad\FurnitureRecommendationAI_Models\all-MiniLM-L6-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MINI-LM QUERY MODEL
# ============================================================

_tokenizer = None
_text_model = None


def load_query_model():
    """
    Load MiniLM once for converting the user's search query
    into a 384-dimensional embedding.

    We use the same MiniLM model that was used to create
    the saved product embeddings.
    """

    global _tokenizer, _text_model

    if _tokenizer is None or _text_model is None:

        print("Loading MiniLM query model...")

        _tokenizer = AutoTokenizer.from_pretrained(
            MINILM_PATH,
            local_files_only=True
        )

        _text_model = AutoModel.from_pretrained(
            MINILM_PATH,
            local_files_only=True
        ).to(DEVICE)

        _text_model.eval()

        print(f"MiniLM query model loaded on {DEVICE}")

    return _tokenizer, _text_model


# ============================================================
# QUERY EMBEDDING
# ============================================================

def create_query_embedding(query):
    """
    Convert a natural-language query into the same
    384-dimensional MiniLM embedding space used by
    the product embeddings.
    """

    tokenizer, model = load_query_model()

    encoded = tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        output = model(**encoded)

        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"]

        mask = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()

        summed = torch.sum(
            token_embeddings * mask,
            dim=1
        )

        counts = torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )

        embedding = summed / counts

        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

    return embedding.cpu().numpy()[0]


# ============================================================
# CATEGORY HELPERS
# ============================================================

def get_category_list(value):

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, str):

        if not value.strip():
            return []

        return [value]

    return []


FURNITURE_CATEGORIES = {
    "Accent Furniture",
    "Armchairs",
    "Armoires & Dressers",
    "Bar Cabinets",
    "Bar Tables",
    "Bars & Wine Cabinets",
    "Barstools",
    "Bathroom Furniture",
    "Bed Frames",
    "Bed Frames, Headboards & Footboards",
    "Bookcases",
    "Chairs",
    "Desks",
    "Dining Room Furniture",
    "Dressers",
    "End Tables",
    "Furniture",
    "Kitchen Furniture",
    "Living Room Furniture",
    "Nightstands",
    "Ottomans",
    "Recliners",
    "Rocking Chairs",
    "Sofas & Couches",
    "Stools",
    "Tables",
    "Wine Racks & Cabinets"
}


AVAILABLE_FURNITURE_CATEGORIES = {
    x.lower()
    for x in FURNITURE_CATEGORIES
}


def is_furniture_product(categories):

    categories = get_category_list(categories)

    return any(
        str(category).lower()
        in AVAILABLE_FURNITURE_CATEGORIES
        for category in categories
    )


# ============================================================
# ATTRIBUTE ALIASES
# ============================================================

ATTRIBUTE_ALIASES = {

    "color": {

        "brown": [
            "brown",
            "dark brown",
            "light brown",
            "medium brown",
            "rustic brown",
            "chocolate brown",
            "espresso",
            "chestnut"
        ],

        "black": [
            "black",
            "jet black",
            "matte black"
        ],

        "white": [
            "white",
            "off white",
            "ivory",
            "cream"
        ],

        "pink": [
            "pink",
            "blush pink",
            "rose pink"
        ],

        "blue": [
            "blue",
            "navy blue",
            "royal blue"
        ],

        "red": [
            "red",
            "burgundy",
            "maroon"
        ],

        "gray": [
            "gray",
            "grey",
            "charcoal"
        ],

        "green": [
            "green",
            "olive green"
        ]
    },

    "material": {

        "wood": [
            "wood",
            "wooden",
            "oak",
            "walnut",
            "pine",
            "mahogany",
            "cherry wood",
            "maple",
            "bamboo",
            "rubberwood",
            "solid wood",
            "engineered wood",
            "mdf",
            "particle board",
            "plywood"
        ],

        "metal": [
            "metal",
            "steel",
            "stainless steel",
            "iron",
            "aluminum",
            "aluminium"
        ],

        "leather": [
            "leather",
            "faux leather",
            "pu leather",
            "vegan leather"
        ],

        "glass": [
            "glass",
            "tempered glass"
        ],

        "plastic": [
            "plastic",
            "polypropylene",
            "acrylic"
        ]
    }
}


def get_attribute_terms(attribute, attribute_type):

    if not attribute:
        return []

    attribute = str(attribute).lower().strip()

    aliases = ATTRIBUTE_ALIASES.get(
        attribute_type,
        {}
    )

    if attribute in aliases:
        return aliases[attribute]

    return [attribute]


def contains_whole_word(text, word):

    text = str(text).lower()
    word = str(word).lower()

    pattern = rf"\b{re.escape(word)}\b"

    return bool(
        re.search(pattern, text)
    )


def attribute_evidence_score(
    row,
    attribute,
    attribute_type
):

    if not attribute:
        return 0.0

    title = str(
        row.get("title", "")
    ).lower()

    description = str(
        row.get("description", "")
    ).lower()

    categories = " ".join(
        str(x).lower()
        for x in get_category_list(
            row.get("categories", [])
        )
    )

    terms = get_attribute_terms(
        attribute,
        attribute_type
    )

    if any(
        contains_whole_word(title, term)
        for term in terms
    ):
        return 1.0

    if any(
        contains_whole_word(categories, term)
        for term in terms
    ):
        return 0.7

    if any(
        contains_whole_word(description, term)
        for term in terms
    ):
        return 0.4

    return 0.0


# ============================================================
# PRODUCT TYPE
# ============================================================

def dining_table_evidence(row):

    title = str(
        row.get("title", "")
    ).lower()

    categories = get_category_list(
        row.get("categories", [])
    )

    category_text = " ".join(
        str(x).lower()
        for x in categories
    )

    if not is_furniture_product(categories):
        return 0.0

    exclusions = [
        "tablecloth",
        "table cloth",
        "table runner",
        "table linen",
        "table protector",
        "table cover",
        "table covers",
        "table mat",
        "table pad",
        "placemat",
        "placemats",
        "chair cover",
        "chair covers",
        "chair cushion",
        "chair cushions",
        "trivet",
        "napkin",
        "napkins",
        "paper towel holder",
        "utensil holder",
        "silverware holder",
        "silverware caddy",
        "flatware organizer",
        "organizer",
        "storage cart",
        "storage rack",
        "countertop",
        "candle holder",
        "vase",
        "coaster",
        "kitchen towel",
        "dishcloth"
    ]

    if any(
        term in title
        for term in exclusions
    ):
        return 0.0

    furniture_exclusions = [
        "bench",
        "benches",
        "dining chair",
        "dining chairs",
        "accent chair",
        "accent chairs",
        "side table",
        "side tables",
        "end table",
        "end tables",
        "coffee table",
        "coffee tables",
        "accent table",
        "accent tables",
        "console table",
        "console tables",
        "nightstand",
        "nightstands",
        "tv stand",
        "tv stands",
        "tray table",
        "tray tables",
        "snack table",
        "snack tables",
        "computer desk",
        "writing desk",
        "office desk",
        "desk",
        "desks",
        "table top",
        "tabletop"
    ]

    if any(
        term in title
        for term in furniture_exclusions
    ):
        return 0.0

    strong_terms = [
        "dining table",
        "dining tables",
        "dining room table",
        "dining room tables",
        "farmhouse dining table",
        "farmhouse dining tables",
        "farmhouse table",
        "farmhouse tables",
        "kitchen dining table",
        "kitchen dining tables",
        "kitchen table",
        "kitchen tables",
        "dining table set",
        "dining table sets",
        "dining room set",
        "dining room sets"
    ]

    if any(
        term in title
        for term in strong_terms
    ):
        return 1.0

    has_table = (
        contains_whole_word(title, "table")
        or contains_whole_word(title, "tables")
    )

    has_dining = (
        contains_whole_word(title, "dining")
        or "dining room" in title
    )

    if has_table and has_dining:
        return 0.9

    has_table_category = (
        "tables" in category_text
        or "table" in category_text
    )

    if has_dining and has_table_category:
        return 0.85

    if (
        contains_whole_word(title, "farmhouse")
        and has_table
    ):
        return 0.9

    if has_table:
        return 0.2

    return 0.0


def sofa_evidence(row):

    title = str(
        row.get("title", "")
    ).lower()

    categories = get_category_list(
        row.get("categories", [])
    )

    category_text = " ".join(
        str(x).lower()
        for x in categories
    )

    competing_products = [
        "sleeper chair",
        "convertible chair",
        "accent chair",
        "lounge chair",
        "rocking chair",
        "recliner chair",
        "reclining chair",
        "lazy chair",
        "dining chair",
        "office chair",
        "desk chair",
        "folding chair",
        "bean bag",
        "beanbag",
        "ottoman",
        "coffee table",
        "side table",
        "end table",
        "console table",
        "sofa table",
        "sofa tray",
        "sofa organizer",
        "replacement mattress",
        "mattress",
        "bed frame",
        "bed",
        "blanket",
        "blankets",
        "throw blanket",
        "pillow",
        "pillow cover",
        "cushion cover",
        "slipcover",
        "slip cover",
        "sofa cover",
        "couch cover",
        "furniture protector",
        "protector",
        "replacement"
    ]

    if any(
        term in title
        for term in competing_products
    ):
        return 0.0

    sofa_terms = [
        "sofa",
        "sofas",
        "couch",
        "couches",
        "sectional",
        "sectionals",
        "loveseat",
        "loveseats"
    ]

    has_sofa_term = any(
        contains_whole_word(title, term)
        for term in sofa_terms
    )

    if not has_sofa_term:
        return 0.0

    sofa_category = (
        "sofas" in category_text
        or "sofas & couches" in category_text
        or "couches" in category_text
    )

    if sofa_category:
        return 1.0

    return 0.0


def product_type_evidence(row, product_type):

    if not product_type:
        return 0.0

    product_type = str(
        product_type
    ).lower().strip()

    if product_type == "dining table":
        return dining_table_evidence(row)

    if product_type == "sofa":
        return sofa_evidence(row)

    title = str(
        row.get("title", "")
    ).lower()

    generic_aliases = {
        "coffee table": [
            "coffee table"
        ],
        "chair": [
            "chair"
        ],
        "desk": [
            "desk"
        ],
        "bed": [
            "bed"
        ],
        "dresser": [
            "dresser",
            "dressers"
        ],
        "nightstand": [
            "nightstand",
            "nightstands"
        ],
        "table": [
            "table",
            "tables"
        ]
    }

    aliases = generic_aliases.get(
        product_type,
        [product_type]
    )

    for alias in aliases:

        if contains_whole_word(
            title,
            alias
        ):
            return 1.0

    return 0.0


# ============================================================
# REVIEW SCORE
# ============================================================

def calculate_review_score(
    rating,
    rating_number,
    global_mean=4.4648,
    minimum_reviews=100
):

    try:
        rating = float(rating)
        rating_number = float(rating_number)
    except:
        return 0.0

    if np.isnan(rating) or np.isnan(rating_number):
        return 0.0

    return (
        (
            rating_number /
            (rating_number + minimum_reviews)
        ) * rating
        +
        (
            minimum_reviews /
            (rating_number + minimum_reviews)
        ) * global_mean
    )


# ============================================================
# MAIN SEARCH
# ============================================================

def search_products(
    user_query,
    models,
    top_k=10
):

    print("\n" + "=" * 70)
    print("NATURAL LANGUAGE PRODUCT SEARCH")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Extract structured preferences with Mistral
    # --------------------------------------------------------

    preferences = extract_product_preferences(
        user_query
    )

    print("\nPreferences:")
    print(preferences)

    # --------------------------------------------------------
    # 2. Clean semantic query
    # --------------------------------------------------------

    semantic_query = clean_search_query(
        preferences
    )

    if not semantic_query:
        semantic_query = user_query

    print(
        "\nSemantic query:",
        semantic_query
    )

    # --------------------------------------------------------
    # 3. Query embedding
    # --------------------------------------------------------

    query_embedding = create_query_embedding(
        semantic_query
    )

    # --------------------------------------------------------
    # 4. MiniLM semantic similarity
    # --------------------------------------------------------

    product_embeddings = models[
        "text_norm_all"
    ]

    semantic_scores = (
        product_embeddings @ query_embedding
    )

    # --------------------------------------------------------
    # 5. Start with complete catalog
    # --------------------------------------------------------

    products_df = models[
        "products_df"
    ].copy()

    products_df["_semantic_score"] = (
        semantic_scores
    )

    # --------------------------------------------------------
    # 6. Price filter
    # --------------------------------------------------------

    max_price = preferences.get(
        "max_price"
    )

    min_price = preferences.get(
        "min_price"
    )

    if max_price is not None:

        products_df = products_df[
            pd.to_numeric(
                products_df["price"],
                errors="coerce"
            ) <= float(max_price)
        ]

    if min_price is not None:

        products_df = products_df[
            pd.to_numeric(
                products_df["price"],
                errors="coerce"
            ) >= float(min_price)
        ]

    # --------------------------------------------------------
    # 7. Rating filter
    # --------------------------------------------------------

    min_rating = preferences.get(
        "min_rating"
    )

    if min_rating is not None:

        products_df = products_df[
            pd.to_numeric(
                products_df["average_rating"],
                errors="coerce"
            ) >= float(min_rating)
        ]

    # --------------------------------------------------------
    # 8. Furniture category
    # --------------------------------------------------------

    category = preferences.get(
        "category"
    )

    if category == "furniture":

        furniture_mask = products_df[
            "categories"
        ].apply(
            is_furniture_product
        )

        products_df = products_df[
            furniture_mask
        ]

    # --------------------------------------------------------
    # 9. Product type
    # --------------------------------------------------------

    product_type = preferences.get(
        "product_type"
    )

    if product_type:

        products_df["_product_type_score"] = (
            products_df.apply(
                lambda row:
                product_type_evidence(
                    row,
                    product_type
                ),
                axis=1
            )
        )

        # Specific product requests require
        # strong evidence.
        if product_type in [
            "dining table",
            "sofa"
        ]:

            products_df = products_df[
                products_df[
                    "_product_type_score"
                ] >= 0.85
            ]

        else:

            products_df = products_df[
                products_df[
                    "_product_type_score"
                ] > 0
            ]

    else:

        products_df[
            "_product_type_score"
        ] = 0.0

    # --------------------------------------------------------
    # 10. Color
    # --------------------------------------------------------

    color = preferences.get(
        "color"
    )

    if color:

        products_df["_color_score"] = (
            products_df.apply(
                lambda row:
                attribute_evidence_score(
                    row,
                    color,
                    "color"
                ),
                axis=1
            )
        )

    else:

        products_df["_color_score"] = 0.0

    # --------------------------------------------------------
    # 11. Material
    # --------------------------------------------------------

    material = preferences.get(
        "material"
    )

    if material:

        products_df["_material_score"] = (
            products_df.apply(
                lambda row:
                attribute_evidence_score(
                    row,
                    material,
                    "material"
                ),
                axis=1
            )
        )

    else:

        products_df["_material_score"] = 0.0

    # --------------------------------------------------------
    # 12. Review score
    # --------------------------------------------------------

    products_df["_review_score"] = (
        products_df.apply(
            lambda row:
            calculate_review_score(
                row.get("average_rating"),
                row.get("rating_number")
            ),
            axis=1
        )
    )

    products_df["_review_normalized"] = (
        products_df["_review_score"] / 5.0
    )

    # --------------------------------------------------------
    # 13. Final ranking
    # --------------------------------------------------------

    products_df["_final_score"] = (
        0.65 *
        products_df["_semantic_score"]
        +
        0.15 *
        products_df["_review_normalized"]
        +
        0.10 *
        products_df["_color_score"]
        +
        0.10 *
        products_df["_material_score"]
    )

    products_df = products_df.sort_values(
        "_final_score",
        ascending=False
    )

    results = products_df.head(
        top_k
    ).copy()

    # --------------------------------------------------------
    # 14. Clean output
    # --------------------------------------------------------

    output_columns = [
        "parent_asin",
        "title",
        "image_url",
        "price",
        "average_rating",
        "rating_number",
        "main_category",
        "description",
        "categories",
        "_semantic_score",
        "_final_score"
    ]

    output_columns = [
        col
        for col in output_columns
        if col in results.columns
    ]

    results = results[
        output_columns
    ]

    results = results.reset_index(
        drop=True
    )

    print(
        f"\nFound {len(results)} results."
    )

    return preferences, results
