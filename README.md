# 🪑 Furniture Recommendation AI

> A multimodal AI-powered furniture recommendation system that understands natural-language shopping queries and retrieves relevant products using semantic search, visual semantics, attribute-aware filtering, and review-aware ranking.

## 🚀 Overview

Furniture Recommendation AI is an intelligent product discovery system inspired by modern e-commerce recommendation platforms.

Instead of requiring users to select a predefined category or enter a User ID, the system accepts natural-language queries such as:

> "Show me brown wooden dining tables under $100"

The system extracts the user's preferences, performs semantic product retrieval, applies structured constraints, ranks the results, and presents the most relevant furniture products through a Streamlit interface.

An AI shopping assistant powered by Mistral can then answer questions about the currently retrieved products.

---

## ✨ Key Features

- Natural-language furniture search
- Mistral-based preference extraction
- MiniLM semantic embeddings
- DistilBERT experimentation
- BLIP image captioning
- Image-based semantic experimentation
- Attribute-aware retrieval
- Product-type detection
- Furniture-category filtering
- Price constraints
- Rating constraints
- Review-aware ranking
- Product image display
- Grounded Mistral shopping assistant
- Streamlit web interface
- GPU acceleration with CUDA
- 30,000-product catalog

---

## 🧠 AI Architecture

The system combines pretrained language and vision models with a custom retrieval and ranking pipeline.

### 1. Natural Language Understanding

Mistral converts a user's natural-language request into structured preferences:

```json
{
  "category": "furniture",
  "product_type": "dining table",
  "color": "brown",
  "material": "wooden",
  "max_price": 100
}