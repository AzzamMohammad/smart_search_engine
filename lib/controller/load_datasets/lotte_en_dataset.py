import ir_datasets
import pandas as pd
import os
import csv


class LotteEnDataset:
    def __init__(self):
        self._dataset = ir_datasets.load("lotte/lifestyle/test/search")
        self._csv_lotte_dataset_file_path = 'storege/saved_lotte_en_dataset.csv'
        self._csv_loote_query_file_path = 'storege/saved_lotte_en_query.csv'
        self._csv_loote_quel_file_path = 'storege/saved_lotte_en_quel.csv'
        
    # init document Dataframe 
    def GetDocumentsDataframe(self):
        if os.path.isfile(self._csv_lotte_dataset_file_path):
            docs_data = pd.read_csv(self._csv_lotte_dataset_file_path)
            print('**** Loading saved documents from CSV file is complete ****')
        else:
            with open(self._csv_lotte_dataset_file_path,'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id','document'])
                for doc in self._dataset.docs_iter():
                    doc_id, text = doc.doc_id, doc.text
                    if text.strip():
                        writer.writerow([doc_id,text])
            docs_data = pd.read_csv(self._csv_lotte_dataset_file_path)
            print('**** Loading document from scratch is complete ****')
        return docs_data

    # init query Dataframe
    def GetQueriesDataframe(self):
        if os.path.isfile(self._csv_loote_query_file_path):
            query_data = pd.read_csv(self._csv_loote_query_file_path)
            print('**** Loading saved queries from CSV file is complete ****')
        else:
            with open(self._csv_loote_query_file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['query_id', 'text'])
                for query in self._dataset.queries_iter():
                    query_id, text = query.query_id, query.text
                    if text.strip():
                        writer.writerow([query_id,text])
            query_data = pd.read_csv(self._csv_loote_query_file_path)
            print('**** Loading queries from scratch is complete ****')
        return query_data
    
        
    # init quele Dataframe
    def GetQuelsDataframe(self):
        if os.path.isfile(self._csv_loote_quel_file_path):
            quels_data = pd.read_csv(self._csv_loote_quel_file_path)
            print('**** Loading saved quels from CSV file is complete ****')
        else:
            with open(self._csv_loote_quel_file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['query_id', 'doc_id', 'relevance'])
                for qrels in self._dataset.qrels_iter():
                    query_id, doc_id ,relevance  = qrels.query_id, qrels.doc_id,qrels.relevance
                    writer.writerow([query_id, doc_id ,relevance])
            quels_data = pd.read_csv(self._csv_loote_quel_file_path)
            print('**** Loading quels from scratch is complete ****')
        return quels_data
    