from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from text_processing.TextProcessor import TextProcessor
from stages.offline_stage import OfflineStage
from stages.online_stage import OnlineStage
from stages.evaluation_stage import EvaluationStage
from statics.init_read_stored_files import InitReadStoredFiles


import pandas as pd
app = Flask(__name__)
CORS(app)



docs_data2 =  pd.read_csv('storege/saved_lotte_en_dataset.csv')

# Load the dataset only once, when the application starts
docs_data1 = pd.read_csv('storege/saved_en_dataset.csv')





text_processor = TextProcessor()

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    dataset_lang = request.form['dataset']
    

    if dataset_lang == "b":
        FilesName = InitReadStoredFiles("lotteDS") 
        online_stage2 = OnlineStage(FilesName.GetTFDIFFileName(),FilesName.GetVectorisFileName(),FilesName.GetClusteringFileName())
        query = text_processor.ProcessText(query)
        query_tfidf_vector = online_stage2.apply_tfidf_online(query)
        doc_indices = online_stage2.match_document_query_with_clustering(query_tfidf_vector)
        j= 0
        doc_results_list = [] 
        for i in doc_indices:
            if(j==11):
                break
            j = j +1
            doc_results_list.append({
            'id': int(docs_data2['id'][i]),
            'title': docs_data2['document'][i]
            })
        doc_results_list = doc_results_list[:j]
        result_list = [doc['id'] for doc in doc_results_list[:j]]
    #####
    if dataset_lang == "a":
        FilesName = InitReadStoredFiles("Antique") 
        online_stage1 = OnlineStage(FilesName.GetTFDIFFileName(),FilesName.GetVectorisFileName(),FilesName.GetClusteringFileName())
        query = text_processor.ProcessText(query)
        query_tfidf_vector = online_stage1.apply_tfidf_online(query)
        doc_indices = online_stage1.match_document_query_with_clustering(query_tfidf_vector)
        doc_indices = online_stage1.match_document_query(query_tfidf_vector)
        j= 0
        doc_results_list = [] 
        for i in doc_indices:
            if(j==11):
                break
            j = j +1
            doc_results_list.append({
                'id': docs_data1['id'][i],
                'title': docs_data1['document'][i]
            })
        doc_results_list = doc_results_list[:10]
        result_list = [doc['id'] for doc in doc_results_list[:10]]
    
    #####
    
    
    
    
    
    response = jsonify({
        "result_list": result_list,
        "doc_results_list": doc_results_list
    })
    response.headers.add('Access-Control-Allow-Origin', 'http://127.0.0.1:5500')
    return response
   

if __name__ == '__main__':
    app.run(debug=True)