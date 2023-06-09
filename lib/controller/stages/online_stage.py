import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load

class OnlineStage:
    
    def __init__(self, tfidf_matrix_file_path, vectorizer_file_path,clustering_file_path):
        self._document_body_tfidf_matrix_file_path = tfidf_matrix_file_path
        self._document_body_tfidf_vectorizer_file_path = vectorizer_file_path
        self._document_body_tfidf_clustering_file_path = clustering_file_path
        
    # load vectorize and trancform query
    def apply_tfidf_online(self, query): 
        with open(self._document_body_tfidf_vectorizer_file_path, 'rb') as f:
            vectorizer = pickle.load(f)
        query_tfidf_vector = vectorizer.transform([query])
        return query_tfidf_vector


    # load tfdif from file and calculate cosine similarities
    def match_document_query(self, query_tfidf_vector):
        with open(self._document_body_tfidf_matrix_file_path, 'rb') as f:
            tfidf_matrix = pickle.load(f)
        cosine_similarities = cosine_similarity(
            query_tfidf_vector,
            tfidf_matrix
        ).flatten()
        doc_indices_sorted = np.argsort(cosine_similarities)[::-1]
        doc_indices_sorted = doc_indices_sorted[cosine_similarities[doc_indices_sorted] > 0]
        return doc_indices_sorted
    

    # load tfdif from file and clustering file and calculate cosine similarities
    def match_document_query_with_clustering(self, query_tfidf_vector):
        with open(self._document_body_tfidf_matrix_file_path, 'rb') as f:
            tfidf_matrix = pickle.load(f)
        kmeans = load(self._document_body_tfidf_clustering_file_path)
        label = kmeans.predict(query_tfidf_vector)[0]
        cluster_indices = [i for i, l in enumerate(kmeans.labels_) if l == label]
        similarity_scores = cosine_similarity(query_tfidf_vector, tfidf_matrix[cluster_indices]).flatten()
        doc_indices_sorted = np.argsort(similarity_scores)[::-1]
        doc_indices_sorted = doc_indices_sorted[similarity_scores[doc_indices_sorted] > 0]
        return doc_indices_sorted
    
    
    def GetDocumentResultIDsList(self , doc_indices , dataset ):  
        doc_result_IDs_list = [] 
        for i in doc_indices:
            doc_result_IDs_list.append(dataset['id'][i])
        return doc_result_IDs_list
    
    
    def DisplayFirstKResult(self,K,doc_indices,docs_dataset , query):
        print(f"\n\n **Top {K} most similar documents to query --'{query}'--:\n")
        count = 0
        for i in doc_indices:
            if count < K:
                print(f"{count+1} - Document {i}  : {docs_dataset['document'][i]}")
            count+=1
        
