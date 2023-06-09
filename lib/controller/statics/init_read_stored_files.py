class InitReadStoredFiles:
    _VectorisFileName = None
    _TFDIFFileName = None
    _DocumentFileName = None
    _QueriesFileName = None
    _QuelsFileName = None
    _ClusteringFileName = None
    
    def __init__(self,DatasetName):
        self._initFilesName(DatasetName)
        
    
    def _initFilesName(self , DatasetName):
        if DatasetName == "lotteDS":
            self._TFDIFFileName = 'storege/document_lotte_tfidf_matrix.pickle'
            self._VectorisFileName = 'storege/document_lotte_tfidf_vectorizer.pickle'
            self._ClusteringFileName = 'storege/saved_lotte_en_clustering.joblib'
            self._DocumentFileName = 'storege/saved_lotte_en_dataset.csv'
            self._QueriesFileName = 'storege/saved_lotte_en_query.csv'
            self._QuelsFileName = 'storege/saved_lotte_en_quel.csv'
            
            
        elif DatasetName == "antiqueDS":
            self._TFDIFFileName = 'storege/document_body_tfidf_matrix.pickle'
            self._VectorisFileName = 'storege/document_body_tfidf_vectorizer.pickle'
            self._ClusteringFileName = 'storege/saved_en_clustering.joblib'
            self._DocumentFileName = 'storege/saved_en_dataset.csv'
            self._QueriesFileName = 'storege/saved_en_query.csv'
            self._QuelsFileName = 'storege/saved_en_quel.csv'
            
    
    
    def GetTFDIFFileName(self):
        return self._TFDIFFileName
        
    def GetVectorisFileName(self):
        return self._VectorisFileName
        
    def GetDocumentFileName(self):
        return self._DocumentFileName
        
    def GetQueriesFileName(self):
        return self._QueriesFileName
        
    def GetQuelsFileName(self):
        return self._QuelsFileName
        
        
    def GetClusteringFileName(self):
        return self._ClusteringFileName