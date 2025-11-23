import random
import numpy as np
from myapp.search.algorithms import *
from gensim.models import Word2Vec

from myapp.search.objects import Document, ResultItem


'''def dummy_search(corpus: dict, search_id, num_results=20):
    """
    Just a demo method, that returns random <num_results> documents from the corpus
    :param corpus: the documents corpus
    :param search_id: the search id
    :param num_results: number of documents to return
    :return: a list of random documents from the corpus
    """
    res = []
    doc_ids = list(corpus.keys())
    docs_to_return = np.random.choice(doc_ids, size=num_results, replace=False)
    for doc_id in docs_to_return:
        doc = corpus[doc_id]
        res.append(Document(pid=doc.pid, title=doc.title, description=doc.description,
                            url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), ranking=random.random()))
    return res
'''


def build_demo_results(corpus: dict, search_id, doc_scores):
    """
    Helper method, just to demo the app
    :return: a list of demo docs sorted by ranking
    """
    res = []
    for rank, (doc_id, score) in enumerate(doc_scores): 
        doc = corpus.get(doc_id) # We get the information for the specific doc_id
        if doc:
            # We save the information needed in ResultItem form
            res.append(Document(pid=doc.pid, title=doc.title, description=doc.description,
                            url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), ranking=rank))
    return res

class SearchEngine:
    """Class that implements the search engine logic"""

    def search(self, search_query, search_id, corpus, selected_option):
        print("Search query:", search_query)

        results = []
        ### You should implement your search logic here:
        # results = dummy_search(corpus, search_id)  # replace with call to search algorithm
        # results = search_in_corpus(search_query)
        
        # Process all fields
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

        if not docs:
            print("No documents found containing all query terms.")
        else:
            # For each search option, we use its respective function and save it to doc_scores wich is a dictionary
            # with the form id -> score
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
