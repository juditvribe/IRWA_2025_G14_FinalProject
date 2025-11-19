# Ranking and Filtering Lab – Information Retrieval

This lab implements several ranking and filtering techniques for product search. The goal is to experiment with different relevance scoring algorithms and compare their behavior on a dataset of fashion products.

## Features Implemented

### Conjunctive Search
- **`conjunctive_search_terms(query, index, build_terms_fn)`**  
  Retrieves a list of documents that contain **all query terms** (AND semantics). Used as a first filtering step for all ranking methods.

### TF-IDF + Cosine Similarity
- **`tfidf_cosine_rank(query, docs, tf, idf, index, title_index, vocab, build_terms_fn, top_k)`**  
  Ranks candidate documents based on cosine similarity between the query and document TF-IDF vectors. Rewards documents whose term distribution matches the query.

### BM25 Ranking
- **`bm25_rank(query, docs, index, tf, idf, doc_lengths, k1, b, top_k, build_terms_fn)`**  
  Implements BM25 ranking. Scores documents based on term frequency saturation, inverse document frequency, and document length normalization.

### Custom Scoring
- **`ourscore_cosine(query, inverted_index, docs, vocab, fashion_df, idf, tf, top_k, build_terms_fn)`**  
  Combines TF-IDF relevance with product metadata such as rating, discount, price, stock availability, and title matches. Provides a more user-oriented ranking that balances textual relevance and practical attributes.

- **Helper functions**:
  - `min_max_scaling`: Normalizes numeric fields between 0 and 1.  
  - `processing_title`: Counts how many query terms appear in the product title.

### Word2Vec + Cosine Similarity
- **`word2vec_cosine_rank(query, model, products, tf, idf, index, vocab, build_terms_fn, top_k)`**  
  Uses Word2Vec embeddings to rank documents semantically. Each document is represented by the average of its word vectors. PCA reduces dimensionality, and cosine similarity is computed between query and document vectors.

- **Helper function**:
  - `document_vector(model, words)`: Computes the average Word2Vec vector for a list of words.

## Usage
1. First, filter documents with `conjunctive_search_terms`.
2. Then rank them using one of the methods: `tfidf_cosine_rank`, `bm25_rank`, `ourscore_cosine`, or `word2vec_cosine_rank`.
3. Top-k results can be displayed for any query.  
