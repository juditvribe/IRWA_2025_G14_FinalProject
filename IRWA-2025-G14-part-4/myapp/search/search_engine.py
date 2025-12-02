import random
import numpy as np
from myapp.search.algorithms import *
from gensim.models import Word2Vec

from myapp.search.objects import Document, ResultItem


def build_demo_results(corpus: dict, search_id, doc_scores):
    """
    Method that returns the ranking of results: documents from corpues sorted by score
    :param corpus: the documents corpus
    :param search_id: the search id
    :param doc_scores: dictionary of documents wiitht their corresponding scores
    :return: a list of documents from the corpus sorted by score
    """
    res = []
    for rank, (doc_id, score) in enumerate(doc_scores): 
        doc = corpus.get(doc_id) # We get the information for the specific doc_id
        if doc:
            # We save the information needed in ResultItem form
            res.append(ResultItem(pid=doc.pid, title=doc.title, description=doc.description,
                            brand=doc.brand, selling_price=doc.selling_price, average_rating=doc.average_rating,
                            url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), ranking=rank))
    return res


class SearchEngine:
    """Class that implements the search engine logic"""

    def search(self, search_query, search_id, corpus, selected_option):
        print("Search query:", search_query)
        
        # Process all fields that need to be normalized or tokenized
        processed_corpus = {}
        for pid, doc in corpus.items():
            corpus[pid].product_details = corpus[pid].normalize_product_details(corpus[pid].product_details)
            corpus[pid].selling_price = corpus[pid].parse_price(corpus[pid].selling_price)
            corpus[pid].actual_price = corpus[pid].parse_price(corpus[pid].actual_price)
            corpus[pid].discount = corpus[pid].parse_discount(corpus[pid].discount)
            corpus[pid].average_rating = corpus[pid].parse_rating(corpus[pid].average_rating)

            text_fields = [doc.title, doc.description, doc.brand, doc.category, doc.sub_category, doc.seller]

            if isinstance(doc.product_details, dict):
                pd_text = " ".join(f"{v}" for k, v in doc.product_details.items())
                text_fields.append(pd_text)
            
            joined_text = " ".join([t for t in text_fields if t])
            processed_corpus[pid] = build_terms(joined_text)

        # Compute inverted index
        index, tf, df, idf = create_index_tfidf(processed_corpus)

        vocabulary = list(tf.keys())
        docs = conjunctive_search_terms(search_query, index)

        results = []
        if not docs:
            print("No documents found containing all query terms.")
        else:
            # For each search option, we use its respective function and save it to doc_scores which is a dictionary: key = id | value = score
            if selected_option == "TF-IDF Search":
                ranked_docs, doc_scores = tfidf_cosine_rank(search_query, docs, tf, idf, vocabulary)
            
            elif selected_option == "Own Search Method":
                ranked_docs, doc_scores = ourscore_cosine(search_query, docs, vocabulary, corpus, idf, tf, build_terms_fn=build_terms)

            elif selected_option == 'BM25':
                ranked_docs, doc_scores = bm25_rank(search_query, docs, index, tf, idf, doc_lengths=None, k1=1.5, b=0.75, build_terms_fn=build_terms)

            elif selected_option == 'Word2Vec':
                model = Word2Vec(list(processed_corpus.values()), vector_size=100, window=5, min_count=1)
                ranked_docs, doc_scores = word2vec_cosine_rank(search_query, processed_corpus, docs, model, build_terms_fn=build_terms)

            results = build_demo_results(corpus, search_id, ranked_docs)  


        return results
