from load_datasets.antique_en_dataset import Antique
from text_processing.TextProcessor import TextProcessor
from stages.offline_stage import OfflineStage
from stages.online_stage import OnlineStage
from stages.evaluation_stage import EvaluationStage
from statics.init_read_stored_files import InitReadStoredFiles

if __name__ == '__main__':
    evaluation_stage = EvaluationStage()
 
    
    ## load dataset , query and quils 
    initDatasetModel = Antique() # LotteEnDataset() or Antique()
    FilesName = InitReadStoredFiles("antiqueDS") # lotteDS or antiqueDS 
    dataset = initDatasetModel.GetDocumentsDataframe()
    queries = initDatasetModel.GetQueriesDataframe()
    quils = initDatasetModel.GetQuelsDataframe()
    offline_stage = OfflineStage(FilesName.GetTFDIFFileName(),FilesName.GetVectorisFileName(),FilesName.GetClusteringFileName())
    online_stage = OnlineStage(FilesName.GetTFDIFFileName(),FilesName.GetVectorisFileName(),FilesName.GetClusteringFileName())
    text_processor = TextProcessor()
    
        
    ###########################
    ########## OFFLINE ########
    ###########################  
    
    # Document text processing of document
    print("Apply TF-IDF document offline")
    dataset['document'] = dataset['document'].apply(
        lambda x: text_processor.ProcessText(x)
    )

    # Apply TF-IDF offline
    print("Apply TF-IDFdocument offline")
    tfidf_matrix = offline_stage.calculate_and_save_tfidf_of_documents_body(dataset['document'])
    

    ###########################
    ####### EVALUATION ########
    ###########################
    
    evaluation_stage.DesplayEvaluationResults(dataset , queries , quils , online_stage)
