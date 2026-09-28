import json
import re
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral:latest"


def normalize_preferences(preferences):
    """Convert Mistral's textual null values into Python None."""

    preferences = dict(preferences)

    null_values = {
        "null",
        "none",
        "",
        "n/a",
        "na"
    }

    for key, value in preferences.items():

        if isinstance(value, str):
            if value.strip().lower() in null_values:
                preferences[key] = None

    return preferences


def extract_product_preferences(user_query):

    prompt = f"""
You are a product search preference extractor.

Extract only information explicitly stated or clearly implied
by the user's shopping request.

Return ONLY valid JSON.

Required JSON structure:

{{
    "category": null,
    "product_type": null,
    "color": null,
    "material": null,
    "max_price": null,
    "min_price": null,
    "min_rating": null,
    "query": ""
}}

Rules:

1. category:
   Use "furniture" when the user clearly requests furniture.
   Otherwise use null.

2. product_type:
   Extract a specific product type such as:
   sofa, dining table, coffee table, chair, bed,
   dresser, nightstand.

   Broad categories such as:
   bedroom furniture
   living room furniture
   dining room furniture
   should also be preserved.

3. color:
   Extract explicit colors such as:
   black, white, brown, blue, red, pink, gray, green.

4. material:
   Extract explicit materials such as:
   wood, wooden, metal, leather, glass, plastic.

5. max_price:
   Extract maximum price from phrases such as:
   under $100
   below $150
   less than $200
   up to $300

6. min_price:
   Extract minimum price from phrases such as:
   above $100
   over $100
   more than $100

7. min_rating:
   Extract rating requirements such as:
   rating above 4.3
   rated 4.5 or higher
   at least 4 stars

8. query:
   Keep the important semantic product description.
   Remove price and rating constraints.

9. Never invent missing information.

User request:

{user_query}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "format": "json"
        },
        timeout=120
    )

    response.raise_for_status()

    content = response.json()["message"]["content"]

    preferences = json.loads(content)

    return normalize_preferences(preferences)


def clean_search_query(preferences):

    query = str(
        preferences.get("query", "")
    ).lower().strip()

    # Remove rating expressions
    rating_patterns = [
        r"\bwith\s+(a\s+)?rating\s+(of\s+)?\d+(?:\.\d+)?",
        r"\bwith\s+(a\s+)?ratings?\s+(of\s+)?\d+(?:\.\d+)?",
        r"\bwith\s+(a\s+)?ratings?\s+(above|over|at\s+least|greater\s+than)\s*\d+(?:\.\d+)?",
        r"\bratings?\s+(above|over|at\s+least|greater\s+than)\s*\d+(?:\.\d+)?",
        r"\brated\s+\d+(?:\.\d+)?\s*(and\s+above|or\s+higher|\+)?"
    ]

    for pattern in rating_patterns:
        query = re.sub(pattern, " ", query)

    # Remove price expressions
    price_patterns = [
        r"\b(under|below|less\s+than|up\s+to|maximum|max)\s*\$?\s*\d+(?:\.\d+)?",
        r"\b(over|above|more\s+than|minimum|min)\s*\$?\s*\d+(?:\.\d+)?"
    ]

    for pattern in price_patterns:
        query = re.sub(pattern, " ", query)

    removable_words = [
        "show me",
        "find me",
        "find",
        "show",
        "give me",
        "i want",
        "i need",
        "looking for",
        "please",
        "products",
        "product",
        "furniture"
    ]

    for word in removable_words:
        query = query.replace(word, " ")

    replacements = {
        "sofas": "sofa",
        "couches": "couch",
        "dining tables": "dining table",
        "coffee tables": "coffee table",
        "tables": "table",
        "chairs": "chair",
        "desks": "desk",
        "beds": "bed",
        "dressers": "dresser",
        "nightstands": "nightstand"
    }

    for old, new in replacements.items():
        query = query.replace(old, new)

    query = re.sub(r"\s+", " ", query).strip()

    return query