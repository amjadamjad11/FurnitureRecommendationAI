                    ┌──────────────────────────┐
                    │       User Query         │
                    │ "brown wooden table..."  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Mistral Preference     │
                    │       Extraction         │
                    │                          │
                    │ category                  │
                    │ product_type              │
                    │ color                     │
                    │ material                  │
                    │ price                     │
                    │ rating                    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │        Query Representation        │
              └──────────────┬─────────────────────┘
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
     ┌──────────────────┐        ┌──────────────────┐
     │     MiniLM       │        │      BLIP        │
     │ Text Embeddings  │        │ Image Captioning │
     └────────┬─────────┘        └────────┬─────────┘
              │                           │
              ▼                           ▼
     ┌──────────────────┐        ┌──────────────────┐
     │ Semantic Search  │        │ Visual Semantics │
     └────────┬─────────┘        └────────┬─────────┘
              │                           │
              └────────────┬──────────────┘
                           ▼
                ┌──────────────────────┐
                │ Attribute Processing │
                │                      │
                │ Color                │
                │ Material             │
                │ Product Type         │
                │ Furniture Category   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Constraint Filtering │
                │                      │
                │ Price                │
                │ Rating               │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Review-Aware Ranking │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    Top 10 Products   │
                └──────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌──────────────────┐      ┌──────────────────┐
     │ Streamlit UI     │      │ Mistral Chatbot  │
     │ Product Cards    │      │ Grounded Q&A     │
     └──────────────────┘      └──────────────────┘