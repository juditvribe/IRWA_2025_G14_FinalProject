import os
from json import JSONEncoder

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session, jsonify
from flask import request
import nltk

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import *
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

from datetime import datetime


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")
# instantiate our search engine
search_engine = SearchEngine()
# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()

# load documents corpus into memory.
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)
# Log first element of corpus to verify it loaded correctly:
print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    
    # We save the browser and the operating system
    analytics_data.fact_browser = agent['browser']['name']
    analytics_data.fact_os = agent['os']['name']

    print(session)

    # We save the session and the time that it was established and store data in statistics table 4
    if session['some_var'] not in analytics_data.fact_http_sessions:
        analytics_data.fact_http_sessions[session['some_var']] = datetime.now().time()

    # We save the user and store data in statistics table 2
    if request.remote_addr in analytics_data.fact_http_requests.keys():
        analytics_data.fact_http_requests[request.remote_addr] += 1
    else:
        analytics_data.fact_http_requests[request.remote_addr] = 1
    
    # We add to total number of clicks
    analytics_data.fact_tot_num_clicks += 1

    # These are the available search options that the user will have to see the results
    search_options = ['TF-IDF Search', 'Own Search Method', 'BM25', 'Word2Vec']


    return render_template('index.html', search_options=search_options, page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    
    search_query = request.form['search-query']

    selected_option = request.form['search-option'] #added

    session['last_search_query'] = search_query

    search_id = analytics_data.save_query_terms(search_query)

    results = search_engine.search(search_query, search_id, corpus, selected_option)

    # generate RAG response based on user query and retrieved results
    rag_response = rag_generator.generate_response(search_query, results)
    print("RAG response:", rag_response)

    found_count = len(results)
    session['last_found_count'] = found_count

    print(session)

    # We save the query and store data in statistics table 5
    if search_query in analytics_data.fact_queries.keys():
        analytics_data.fact_queries[search_query] += 1
    else:
        analytics_data.fact_queries[search_query] = 1

    # We save the selected options for search algorithm
    # and store data in statistics table 3
    if selected_option in analytics_data.fact_search_options:
        analytics_data.fact_search_options[selected_option] += 1
    else:
        analytics_data.fact_search_options[selected_option] = 1
    
    # We link the search query to the documents that were found to be able to display the query that each document
    # is related to
    if search_query not in analytics_data.fact_queries_to_docs:
        analytics_data.fact_queries_to_docs[search_query] = []

    for result in results:
        document_id = result.pid 
        if document_id not in analytics_data.fact_queries_to_docs[search_query]:
            analytics_data.fact_queries_to_docs[search_query].append(document_id)

    #We add to total number of clicks
    analytics_data.fact_tot_num_clicks += 1

    return render_template('results.html', results_list=results, page_title="Results", query=search_query, found_counter=found_count, rag_response=rag_response)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """
    Show document details page
    ### Replace with your custom logic ###
    """

    # getting request parameters:
    # user = request.args.get('user')
    print("doc details session: ")
    print(session)

    res = session["some_var"]
    print("recovered var from session:", res)

    # get the query string parameters from request
    clicked_doc_id = request.args["pid"]
    print("click in id={}".format(clicked_doc_id))

    result_query = corpus.get(clicked_doc_id)

    # store data in statistics table 1
    if clicked_doc_id in analytics_data.fact_clicks.keys():
        analytics_data.fact_clicks[clicked_doc_id] += 1
    else:
        analytics_data.fact_clicks[clicked_doc_id] = 1

    analytics_data.fact_tot_num_clicks += 1

    print("fact_clicks count for id={} is {}".format(clicked_doc_id, analytics_data.fact_clicks[clicked_doc_id]))
    print(analytics_data.fact_clicks)
    return render_template('doc_details.html', page_title="Product Page", clicked_doc_id=result_query)


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with yourdashboard ###
    :return:
    """
    #We add to total number of clicks
    analytics_data.fact_tot_num_clicks += 1

    docs = []
    users = []
    sessions = []
    queries = []
    docs_for_queries = []

    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[doc_id]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
        docs.append(doc)
    
    # We save in array 'users' the amount of http requests for each user
    for u_id in analytics_data.fact_http_requests:
        count = analytics_data.fact_http_requests[u_id]
        user = UserReq(u_id, count)
        users.append(user)

    # We save in array 'sessions' the sessions that were opened with the structure Session
    for ses_id in  analytics_data.fact_http_sessions:
        # We compute start and current time to now the time elapsed
        start_time = analytics_data.fact_http_sessions[ses_id] 
        current_time = datetime.now().time()
        total_clicks = analytics_data.fact_tot_num_clicks
        browser = analytics_data.fact_browser
        os = analytics_data.fact_os
        session = Session(ses_id, start_time, current_time, total_clicks, browser, os)
        sessions.append(session)
    
    # We save in array 'queries' the information about the queries that were searched 
    for q in  analytics_data.fact_queries:
        count = analytics_data.fact_queries[q]       
        query = Query(q,count)
        queries.append(query)
    
    # We save in array 'docs_for_queries' the docs associated with each query searched to 
    # display the associated query for each doc
    for query in analytics_data.fact_queries_to_docs:
        associated_docs = analytics_data.fact_queries_to_docs[query]
        docs_for_queries.append((query, associated_docs))

    # We rank the documents that were clicked on by number of clicks
    docs.sort(key=lambda doc: doc.count, reverse=True)
    if len(docs) > 5:
        top5docs = docs[:5]
    else:
        top5docs = docs

    # We rank the queries that were searched by amount of times searched
    queries.sort(key=lambda query: query.count, reverse=True)
    if len(queries) > 5:
        top5queries = queries[:5]
    else:
        top5queries = queries

    # Function created to get the average of tearms of the queries searched
    def get_avg_terms(arr):
        terms = 0
        if len(arr) == 0:
            return 0
        for q in top5queries:
            terms += len(q.query.split())
        return round(terms/len(arr),2)
    
    top5_avg_terms = get_avg_terms(top5queries) # Average of terms top 5 queries 
    queries_avg_terms = get_avg_terms(queries) # Average of terms top all queries searched

    average_t = [top5_avg_terms, queries_avg_terms]

    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    #return render_template('stats.html', clicks_data=docs)
    return render_template('stats.html', page_title='Statistics', clicks_data=top5docs, request_data = users, session_data = sessions, queries_data = top5queries, averages = average_t, docs_for_queries= docs_for_queries)
    

@app.route('/dashboard', methods=['GET'])
def dashboard():
    #We add to total number of clicks
    analytics_data.fact_tot_num_clicks += 1
    
    visited_docs = []
    queries = []
    options_chosen = []
    term_frequency = []
    docs_for_queries = []

    print(analytics_data.fact_clicks.keys())

    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[doc_id]
        doc = ClickedDoc(doc_id, d.title, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # We save the searched queries as an structure Query
    for q in  analytics_data.fact_queries:
        count = analytics_data.fact_queries[q]       
        query = Query(q, count)
        queries.append(query)

    # We get the docs related to each query to be able to retrieved the associated query to each doc
    for query in analytics_data.fact_queries_to_docs:
        associated_docs = analytics_data.fact_queries_to_docs[query]
        docs_for_queries.append((query, associated_docs))
        
    # We save the selected options for searching 
    for option in analytics_data.fact_search_options:
        value = analytics_data.fact_search_options[option]
        options_chosen.append(Query(option, value))

    # We get the tearm frequency for each term
    tf_dict = {}
    for query, count in analytics_data.fact_queries.items():
        terms = query.split()
        for term in terms:
            if term in tf_dict:
                tf_dict[term] += count
            else:
                tf_dict[term] = count

    for term in tf_dict:
        value = tf_dict[term]
        term_frequency.append(Query(term, value))

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    for doc in visited_docs: print(doc)

    #return render_template('dashboard.html', visited_docs=visited_docs)
    return render_template('dashboard.html',
                           page_title='Dashboard',
                           visited_docs =jsonify([doc.to_json() for doc in visited_docs]).json,
                           queries_data = jsonify([query.to_json() for query in queries]).json,
                           search_options_data = jsonify([option.to_json() for option in options_chosen]).json,
                           tf_data = jsonify([term.to_json() for term in term_frequency]).json,
                           d_and_q = docs_for_queries)

# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
