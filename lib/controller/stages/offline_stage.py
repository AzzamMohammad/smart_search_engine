from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
from joblib import dump


class OfflineStage :
    def __init__(self , tfidf_matrix_file_path, vectorizer_file_path,clustering_file_path):
        self._document_body_tfidf_matrix_file_path = tfidf_matrix_file_path
        self._document_body_tfidf_vectorizer_file_path = vectorizer_file_path
        self._document_body_tfidf_clustering_file_path = clustering_file_path
        
    # calculate and save tfidf of documents body , vectorizer and cluster
    def calculate_and_save_tfidf_of_documents_body(self, document): 
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(document)
        # save the tfidfMatrix object to a file
        with open(self._document_body_tfidf_matrix_file_path, 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        # save the TfidfVectorizer object to a file
        with open(self._document_body_tfidf_vectorizer_file_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        # save the clustering object to a file
        kmeans = KMeans(n_clusters= 7, n_init=20 , random_state=30)
        kmeans.fit(tfidf_matrix)
        dump(kmeans, self._document_body_tfidf_clustering_file_path)    

