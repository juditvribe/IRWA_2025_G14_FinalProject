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


def build_demo_results(fashion_df: dict, search_id, doc_scores):
    """
    Helper method, just to demo the app
    :return: a list of demo docs sorted by ranking
    """
    res = []
    for doc_id, rank in doc_scores.items(): 
        doc = fashion_df.get(doc_id) # We get the information for the specific doc_id
        if doc:
            # We save the information needed in ResultItem form
            res.append(Document(pid=doc.pid, title=doc.title, description=doc.description,
                            url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), ranking=random.random()))
    return res

class SearchEngine:
    """Class that implements the search engine logic"""

    def search(self, search_query, search_id, fashion_df, selected_option):
        print("Search query:", search_query)

        results = []
        ### You should implement your search logic here:
        # results = dummy_search(corpus, search_id)  # replace with call to search algorithm


        # results = search_in_corpus(search_query)
        
        # First, process "product details" in order to have them as a unique string
        fashion_df["product_details"] = fashion_df["product_details"].apply(flatten_product_details)

        # Select the text fields that should be processed
        fashion_df_text = fashion_df[["pid", "title", "description", "category", "sub_category", "brand", "seller", "product_details"]]
        fashion_df_text.set_index("pid", inplace=True)

        # Preprocess the text in each field by applying the function "build_terms"
        for col in fashion_df_text.columns:
            fashion_df_text[col] = fashion_df_text[col].apply(build_terms)

        
        index, tf, df, idf, title_index = create_index_tfidf(fashion_df) # inverted index

        vocab = list(tf.keys())

        docs = conjunctive_search_terms(search_query, index)
        if not docs:
            print("No documents found containing all query terms.")
        else:
            # For each search option, we use its respective function and save it to doc_scores wich is a dictionary
            # with the form id -> score
            if selected_option == "TF-IDF Search":
                ranked_docs, doc_scores = tfidf_cosine_rank(search_query, docs, tf, idf, index, title_index, vocab, build_terms_fn=build_terms, top_k=20)
            
            elif selected_option == "Own Search Method":
                ranked_docs, doc_scores = ourscore_cosine(search_query, index, docs, vocab, fashion_df, idf, tf, top_k=20, build_terms_fn=build_terms)

            elif selected_option == 'BM25':
                ranked_docs, doc_scores = bm25_rank(search_query, docs, index, tf, idf, doc_lengths=None, k1=1.5, b=0.75, top_k=20, build_terms_fn=build_terms)

            elif selected_option == 'Word2Vec':
                products = {}
                for _, row in fashion_df.iterrows():
                    combined_text = row.get('title', '') \
                                    + row.get('description', '') \
                                    + row.get('category', '') \
                                    + row.get('sub_category', '') \
                                    + row.get('brand', '') \
                                    + row.get('seller', '') \
                                    + row.get('product_details', '')
                    doc_id = row.get('pid')
                    products[doc_id] = build_terms(combined_text)

                model = Word2Vec(list(products.values()), vector_size=100, window=5, min_count=1)
                ranked_docs, doc_scores = word2vec_cosine_rank(search_query, products, docs, model, build_terms_fn=build_terms, top_k=20)

            results = build_demo_results(fashion_df, search_id, doc_scores)  


        return results
