### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:

    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import string
    
    nlp = spacy.load("en_core_web_sm")

###### Sample documents stored in a dictionary
    def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.text not in nlp.Defaults.stop_words and token.text not in string.punctuation]
    return " ".join(tokens)

    }

###### Preprocessing function to tokenize and remove stopwords/punctuation
    def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.text not in nlp.Defaults.stop_words and token.text not in string.punctuation]
    return " ".join(tokens)


###### Preprocess documents and store them in a dictionary
    preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}
    
###### Construct TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())
###### Calculate cosine similarity between query and documents
    def search(query, tfidf_matrix, tfidf_vectorizer):
    preprocessed_query = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([preprocessed_query])
    
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    
    sorted_indexes = similarity_scores.argsort()[0][::-1]
    
    results = [(list(preprocessed_docs.keys())[i], list(documents.values())[i], similarity_scores[0, i]) for i in sorted_indexes]
    
    return results
    
###### Get input from user
    query = input("Enter your query: ")

###### Perform search
    search_results = search(query, tfidf_matrix, tfidf_vectorizer)

###### Display search results
    print("Query:", query)
    for i, result in enumerate(search_results, start=1):
    print(f"\nRank: {i}")
    print("Document ID:", result[0])
    print("Document:", result[1])
    print("Similarity Score:", result[2])
    print("----------------------")

###### Get the highest rank cosine score
    highest_rank_score = max(result[2] for result in search_results)
    print("The highest rank cosine score is:", highest_rank_score)

### Output:
![Screenshot 2025-04-24 152208](https://github.com/user-attachments/assets/49130add-384e-4c54-a11a-40e1db3ab3c6)

### Result:
Thus, the implementation of Information Retrieval Using Vector Space Model in Python is executed successfully.
