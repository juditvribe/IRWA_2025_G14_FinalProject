import json
import random
import altair as alt
import pandas as pd


class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    # ----------- Statistics tables -----------------
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    fact_clicks = dict([])

    # fact_http_requests is a dictionary with the IP addresses that have accessed the web app (and how many times): key = IP address | value = counter
    fact_http_requests = dict([])

    # fact_search_options is a dictionary with the search methods used and their corresponding counter: key = method used | value = counter
    fact_search_options = dict([])

    # fact_http_sessions is a dictionary with the active sessions of the web app and their starting times: key = session | value = start time
    fact_http_sessions = dict([])

    # fact_queries is a dictionary with the queries counters: key = query | value = query counter
    fact_queries = dict([])

    # fact_queries_to_docs is a dictionary with the queries and their corresponding documents in the results: key = query | value = list of document ids
    fact_queries_to_docs = dict([])

    # Clicks start being -1 because when return to the main page you add 1 click, and we do not want to add when we first enter
    fact_tot_num_clicks = -1
    fact_browser = 0
    fact_os = 0


    def save_query_terms(self, terms: str) -> int:
        print(self)
        return random.randint(0, 100000)
    
    def plot_number_of_views(self):
        # Prepare data
        data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)
        # Create Altair chart
        chart = alt.Chart(df).mark_bar().encode(
            x='Document ID',
            y='Number of Views'
        ).properties(
            title='Number of Views per Document'
        )
        # Render the chart to HTML
        return chart.to_html()


class ClickedDoc:
    def __init__(self, pid, title, description, counter):
        self.pid = pid
        self.title = title
        self.description = description  # Added description to show it in the Dashboard and Statistics pages
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
