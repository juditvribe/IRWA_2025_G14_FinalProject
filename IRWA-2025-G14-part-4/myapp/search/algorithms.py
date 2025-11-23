import re
import nltk
nltk.download('stopwords')

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import numpy as np
import math 
from math import log, sqrt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


#def search_in_corpus(query):
    # 1. create create_tfidf_index

    # 2. apply ranking
    #return ""



# Function to process each product registered by eliminating stop words and punctuation marks, as well as doing stemming and tokenization
def build_terms(line):
    """
    Preprocess the specific text related to an article by removing stop words, stemming,
    transforming in lowercase, removing punctuation and return the tokens of the text.

    Argument:
    line -- string (text) to be preprocessed

    Returns:
    line - a list of tokens corresponding to the input text after the preprocessing
    """

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    line=  line.lower()  # Transform in lowercase
    line= re.sub(r'[^\w\s]', '', line)  # Remove punctuation
    line= re.sub(r'[0-9]', '', line)  # Remove numbers
    line =  line.split()
    line= [word for word in line if word not in stop_words]  # Eliminate the stopwords
    line= [stemmer.stem(word) for word in line]  # Perform stemming

    return line




def create_index_tfidf(processed_corpus):
    num_registers = len(processed_corpus.keys())
    index = defaultdict(list)
    tf = defaultdict(dict)  # term -> {doc_id: tf}
    df = defaultdict(int)
    idf = defaultdict(float)

    for pid, doc_terms in processed_corpus.items():
        
        current_page_index = {} # stores words and positions of CURRENT doc
        for position, term in enumerate(doc_terms):
            try:
                current_page_index[term][1].append(position)
            except KeyError:
                current_page_index[term] = [pid, [position]]

        # Compute norm for term frequency
        norm = math.sqrt(sum(len(posting[1])**2 for posting in current_page_index.values()))
        if norm == 0: norm = 1.0

        # Fill tf and df
        for term, posting in current_page_index.items():
            tf[term][pid] = len(posting[1]) / norm
            df[term] += 1

        # Merge with main index
        for term, posting in current_page_index.items():
            index[term].append(posting) # main index stores words and positions from ALL docs

    # Compute idf
    for term in df:
        idf[term] = math.log(num_registers / df[term])

    return index, tf, df, idf


# to get the candidate docs before ranking
def conjunctive_search_terms(query, index, build_terms_fn=build_terms):
  """Return list of doc_ids that contain ALL query terms (using stemmed tokens)."""
  terms = build_terms_fn(query)
  if not terms:
    return []
  # start with postings of first term
  first = terms[0]
  try:
    docs = set(posting[0] for posting in index[first])
  except KeyError:
    return []
  for t in terms[1:]:
    try:
      docs &= set(posting[0] for posting in index[t])
    except KeyError:
      return []
  return list(docs)



def query_to_vector(query, vocabulary, idf):
    terms = build_terms(query)

    # Compute raw term frequencies for the query
    tf_query = {}
    for term in terms:
        if term in vocabulary:
            tf_query[term] = tf_query.get(term, 0) + 1

    # Normalize term frequency like in create_index_tfidf()
    norm = math.sqrt(sum(tf**2 for tf in tf_query.values()))
    if norm == 0:
        norm = 1.0

    vector = np.zeros(len(vocabulary))
    for i, term in enumerate(vocabulary):
        if term in terms:
            vector[i] = tf_query[term]/norm * idf.get(term, 0)
    return vector






def tfidf_cosine_rank(query, docs, tf, idf, vocabulary):
    """
    Rank documents using cosine similarity over TF-IDF vectors.
    """

    # --- 1. Build query vector ---
    query_vector = query_to_vector(query, vocabulary, idf).reshape(1, -1)

    # --- 2. Build TF-IDF matrix for documents ---
    tfidf_matrix = np.zeros((len(docs), len(vocabulary)))
    for i, term in enumerate(vocabulary):
      for j, doc_id in enumerate(docs):
        if doc_id in tf[term].keys():
          tf_value = tf[term][doc_id]
          tfidf_matrix[j, i] = tf_value * idf[term]

    # --- 3. Compute cosine similarity manually or with sklearn ---
    # Compute similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    scores = {key: score for key, score in zip(docs, similarity_scores)}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked, scores



def compute_doc_lengths_from_tf(tf):
  # tf: term -> {doc_id: tf}
  doc_lengths = defaultdict(int)
  for t, posting in tf.items():
    for d, freq in posting.items():
      doc_lengths[d] += freq
  return doc_lengths


def bm25_rank(query, docs, index, tf, idf, doc_lengths=None, k1=1.5, b=0.75, build_terms_fn=build_terms):
  terms = build_terms_fn(query)
  if not terms:
    return [], {}
  if doc_lengths is None:
    doc_lengths = compute_doc_lengths_from_tf(tf)
  avgdl = np.mean(list(doc_lengths.values())) if doc_lengths else 1.0


  # document frequency for a term: number of docs containing the term
  # compute BM25 score
  scores = defaultdict(float)
  N = len(doc_lengths)
  for t in terms:
    # df_t
    postings = index.get(t, [])
    df_t = len(postings)
    if df_t == 0:
      continue #if there are 0 postings, skip the following code and start again with next item t
    # idf as used in BM25 (with small smoothing)
    idf_bm25 = max(0.0, log((N - df_t + 0.5) / (df_t + 0.5) + 1e-9))


    for doc_id, *rest in postings:
      if doc_id not in docs:
        continue
      f = tf[t].get(doc_id, 0)
      denom = f + k1 * (1 - b + b * (doc_lengths.get(doc_id, 0) / avgdl))
      score = idf_bm25 * ((f * (k1 + 1)) / (denom + 1e-9))
      scores[doc_id] += score


  ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
  return ranked, scores



def processing_title(title, query):
  title_vec = np.array(title)
  query_vec = np.array(query)
  counter = 0
  for t in title_vec:
    if t in query_vec:
      positions = np.where(query_vec==t)[0]
      counter += len(positions)
  return counter




def ourscore_cosine(query, docs, vocabulary, corpus, idf, tf, build_terms_fn=build_terms):
    # 1. Query vector
    query_vector = query_to_vector(query, vocabulary, idf).reshape(1, -1)

    # --- 2. Build TF-IDF matrix for documents ---
    tfidf_matrix = np.zeros((len(docs), len(vocabulary)))
    for i, term in enumerate(vocabulary):
      for j, doc_id in enumerate(docs):
        if doc_id in tf[term].keys():
          tf_value = tf[term][doc_id]
          tfidf_matrix[j, i] = tf_value * idf[term]

    # 3. VECTOR dot products
    dot_products = tfidf_matrix.dot(query_vector.T).flatten()

    # 4. Load document metadata in vector form
    actual_price_vec = np.array([corpus[doc_id].actual_price if corpus[doc_id].actual_price is not None else 0 for doc_id in docs])
    avg_rating_vec = np.array([corpus[doc_id].average_rating if corpus[doc_id].average_rating is not None else 0 for doc_id in docs])
    discount_vec = np.array([corpus[doc_id].discount if corpus[doc_id].discount is not None else 0 for doc_id in docs])
    out_of_stock_vec = np.array([corpus[doc_id].out_of_stock for doc_id in docs], dtype=bool)

    processed_query = build_terms_fn(query)
    in_title = np.array([processing_title(build_terms_fn(corpus[doc_id].title), query=processed_query) for doc_id in docs])

    # 5. Vector min-max scaling
    actual_price_vec = (actual_price_vec - actual_price_vec.min()) / (actual_price_vec.max() - actual_price_vec.min())
    avg_rating_vec = (avg_rating_vec - avg_rating_vec.min()) / (avg_rating_vec.max() - avg_rating_vec.min())
    discount_vec = (discount_vec - discount_vec.min()) / (discount_vec.max() - discount_vec.min())

    title_vec = np.zeros(len(in_title))
    if in_title.max() != in_title.min():
      title_vec = (in_title - in_title.min()) / (in_title.max() - in_title.min())

    # 6. VECTOR final scoring
    bonus_title = 0.3 * title_vec
    base = 0.5 * dot_products + bonus_title
    bonus = 0.2 * (discount_vec*0.6 + avg_rating_vec*0.6 - actual_price_vec*0.2)

    scores_vec = np.where(out_of_stock_vec, base, base + bonus)

    # 7. Convert to dict
    scores = dict(zip(docs, scores_vec))

    # 8. Sort and return
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked, scores



def doc2vec(model, words):
    words = [w for w in words if w in model.wv.key_to_index]
    if not words:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[words], axis=0)

# rank documents using Word2Vec and cosine similarity
def word2vec_cosine_rank(query, documents, query_docs, model, build_terms_fn=build_terms):
  query = build_terms_fn(query)
  query_vector = doc2vec(model, query)
  doc_vectors = [doc2vec(model, doc) for doc_id, doc in documents.items() if doc_id in query_docs]

  similarity_scores = cosine_similarity([query_vector], doc_vectors).flatten()
  scores = {key: score for key, score in zip(query_docs, similarity_scores)}
  ranked_docs = sorted(zip(query_docs, similarity_scores), key=lambda x: x[1], reverse=True)

  return ranked_docs, scores