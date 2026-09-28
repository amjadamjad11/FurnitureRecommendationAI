User Query
    │
    ▼
Mistral Preference Extraction
    │
    ├── Category
    ├── Product Type
    ├── Color
    ├── Material
    ├── Price
    └── Rating
    │
    ▼
Query Cleaning
    │
    ▼
MiniLM Query Embedding
    │
    ▼
Semantic Retrieval
    │
    ├───────────────┐
    ▼               ▼
Text Similarity   BLIP Visual Semantics
    │               │
    └───────┬───────┘
            ▼
Attribute Evidence
            │
            ▼
Price / Rating Filtering
            │
            ▼
Review-Aware Ranking
            │
            ▼
Top-K Products
            │
      ┌─────┴─────┐
      ▼           ▼
 Streamlit      Mistral
 Product UI     Assistant