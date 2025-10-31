**Instructions**

Place the preprocessed dataset file (e.g., fashion_df_processed.csv) and validation labels (validation_labels.csv) inside the data/ directory.
Open the main notebook (IRWA_Part2_Indexing_and_Evaluation.ipynb) and execute it sequentially.

**Functions Overview**

create_index_fashion_fields(fashion_df):
Builds a basic inverted index for the dataset. Iterates through selected fields (category, sub_category, brand, product_details, seller) and maps each term to the documents and fields in which it appears. Returns the inverted index (index) and a dictionary mapping product IDs to their titles (title_index).

search(index):
Performs a conjunctive (AND) search on the inverted index. Retrieves documents containing all query terms, after preprocessing them with build_terms(). Returns a list of document IDs that match the query.

create_index_tfidf(dataset):
Creates a TF-IDF-based inverted index for ranking documents. Processes multiple text fields per document, computes Term Frequency (TF), Document Frequency (DF), and Inverse Document Frequency (IDF). Stores the position of each term for reference. Returns the index, tf, df, idf, and title_index structures.

rank_documents(query_terms, candidate_docs, index, idf, tf, title_index):
Ranks candidate documents by cosine similarity between the query vector and each document vector. Uses precomputed TF-IDF weights to calculate document relevance scores. Returns a ranked list of document IDs and their similarity scores.

search_tf_idf(query, index, tf, idf, title_index):
Processes a user query, searches for all documents containing the query terms, and calls the ranking function to order them by TF-IDF similarity. Returns ranked document IDs and their respective scores.

precision_at_k(y_true, y_score, k=10):
Measures how many of the top-K retrieved documents are relevant. Of the documents retrieved, how much are relevant.

recall_at_k(y_true, y_score, k=10):
Measures how many of the relevant documents were retrieved in the top-K. Of what was relevant, how much did I retrieve.

avg_precision_at_k(y_true, y_score, k=10):
Computes the average precision at all ranks where relevant documents appear. Evaluates how well the relevant items are distributed among the top results.

f1_at_k(y_true, y_score, k=10):
Calculates the harmonic mean of Precision@K and Recall@K. Provides a balanced metric for ranking quality in a single query.

map_at_k(search_res, k=10):
Computes the Mean Average Precision (MAP) across all queries. For each query, the average precision is calculated and averaged over all queries to provide an overall system performance score.

rr_at_k(y_true, y_score, k=10):
Calculates the Reciprocal Rank (RR) for a single query, which measures how early the first relevant document appears in the ranked list.

mrr_at_k(search_res, k=10):
Computes the Mean Reciprocal Rank (MRR) across multiple queries. It is the average of the reciprocal ranks of the first relevant results for each query.

dcg_at_k(y_true, y_score, k=10):
Calculates the Discounted Cumulative Gain (DCG) metric, which measures ranking quality by considering both the relevance of documents and their position in the ranked list. Highly relevant items appearing earlier contribute more to the score.

ndcg_at_k(y_true, y_score, k=10):
Computes the Normalized Discounted Cumulative Gain (nDCG), which normalizes the DCG score by the ideal ranking. Values range from 0 to 1, where 1 indicates a perfect ranking.
