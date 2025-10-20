**Instructions**
Place the dataset files (fashion_products_dataset.json and validation_labels.csv) inside the data/ directory.
Open the main notebook (IRWA_Part1_Preprocessing_and_EDA.ipynb) and execute it sequentially.

**Functions Overview**
_build_terms(line):_
Cleans and normalizes text by converting to lowercase, removing punctuation and digits, tokenizing, removing stopwords, and applying stemming using the PorterStemmer. Returns a list of processed tokens.

_flatten_product_details(details):_
Converts the list of dictionaries contained in the product_details field into a single string of key–value pairs (e.g., “Fabric Cotton”). This preserves attribute information for indexing.

_word_count(df, fields):_
Iterates through specified text fields in the dataset and counts token occurrences across all documents. Returns a dictionary of word frequencies sorted in descending order.

_plot_top_10(dic, xlabel, ylabel, color):_
Visualizes the top 10 most frequent elements from a given frequency dictionary. Produces a bar chart with labeled axes and a line plot overlay for readability.

_avg_length(df, field):_
Calculates the average number of tokens per document for a specified text field. Useful for analyzing the relative verbosity of product titles and descriptions.

_extract_domain_entities(text):_
Performs rule-based entity detection to identify fashion-related attributes such as colors, fabrics, and sizes within product titles and descriptions. Complements the general spaCy NER model.
