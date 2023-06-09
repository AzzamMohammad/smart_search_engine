from text_processing.TextProcessor import TextProcessor

class EvaluationStage :
    text_processor = TextProcessor()
    _PrecisionList = []
    _RecallList = []
    _MeanAvgPrecisionList = []
    _MeanReciprocalRankList = []
    
    
    def DesplayEvaluationResults(self ,dataset, queries,quils ,online_stage):
        self._DatasetEvaluation(dataset, queries,quils , self.text_processor , online_stage)
        print("\n\n\t\t\t *** EVALUATION RESULTS *** \n")
        #Precision@10 result
        precisionAVG = self._GetAVG(self._PrecisionList)
        print("The Precision@10 result is : --> {0}".format(precisionAVG))
        # Recall result
        RecallAVG = self._GetAVG(self._RecallList)
        print("The Recall result is : --> {0}".format(RecallAVG))
        
        # Mean Avg Precision result
        MeanAvgPrecisionAVG = self._GetAVG(self._MeanAvgPrecisionList)
        print("The Mean Avg Precision result is : --> {0}".format(MeanAvgPrecisionAVG))
        
        # Mean Reciprocal Rank result
        MeanReciprocalRankAVG = self._GetAVG(self._MeanReciprocalRankList)
        print("The Mean Reciprocal Rank result is : --> {0}".format(MeanReciprocalRankAVG))


    
    
    def _DatasetEvaluation(self ,dataset, queries,quils , text_processor , online_stage ):
        for i, query in queries.iterrows():
            query_id = query.query_id
            query = query.text
            # query processing
            query = text_processor.ProcessText(query)
            # query TF-IDF 
            query_tfidf_vector = online_stage.apply_tfidf_online(query)
            # Match document query
            # to conversion between match_document_query and match_document_query_with_clustering
            # doc_indices = online_stage.match_document_query(query_tfidf_vector)
            doc_indices = online_stage.match_document_query_with_clustering(query_tfidf_vector)
            # Display results
            relevant_items = online_stage.GetDocumentResultIDsList(doc_indices , dataset)
            recommended_items = self._Get_Recommended_qurls_list(query_id ,quils)
            
            
            #Precision@10 precision of query
            Precision = self._PrecisionEvaluation(recommended_items,relevant_items)
            self._PrecisionList.append(Precision)
            print("\n\nThe Precision value of query {0} is: --> {1:.2f}".format(query_id, Precision))
            
            
            # Recall Evaluation of query
            Recall = self._RecallEvaluation(recommended_items,relevant_items)
            self._RecallList.append(Recall)
            print("The Recall value of query {0} is: --> {1:.2f}".format(query_id, Recall))
            
            
            # Mean Avg Precision Evaluationof query
            MeanAvgPrecision = self._MeanAvgPrecisionEvaluation(recommended_items,relevant_items)
            self._MeanAvgPrecisionList.append(MeanAvgPrecision)
            print("The Mean Avg Precision value of query {0} is: --> {1:.2f}".format(query_id, MeanAvgPrecision))
            
            
            #  Mean Reciprocal Rank Evaluation of query
            MeanReciprocalRank = self._MeanReciprocalRankEvaluation(recommended_items,relevant_items)
            self._MeanReciprocalRankList.append(MeanReciprocalRank)
            print("The Mean Avg Precision value of query {0} is: --> {1:.2f}".format(query_id, MeanReciprocalRank))
        
        
    def _Get_Recommended_qurls_list( self,id , quils):
        return quils[quils['query_id'] == int(id)]['doc_id'].tolist()

       
    
    def _GetAVG(self , list):
        if len(list) == 0:
            return None
        total = sum(list)
        avg = total / len(list)
        return avg
    
    def _PrecisionEvaluation(self,recommended_items, relevant_items):
        k = 10 
        recommended_items = recommended_items[:k]  
        relevant_and_recommended = set(recommended_items).intersection(set(relevant_items[:k]))
        precision = len(relevant_and_recommended) / len(recommended_items)
        return precision
    
    
    def _RecallEvaluation(self,recommended_items, relevant_items):
        relevant_and_recommended = set(recommended_items).intersection(set(relevant_items))
        recall = len(relevant_and_recommended) / len(recommended_items)
        return recall
    
    
    def _MeanAvgPrecisionEvaluation(self ,recommended_items, relevant_items):
        avg_precision = 0.0
        num_hits = 0.0
        for i, item in enumerate(relevant_items):
            if item in recommended_items and item not in relevant_items[:i]:
                num_hits += 1.0
                precision = num_hits / (i+1)
                avg_precision += precision
        if num_hits > 0:
            avg_precision = avg_precision /num_hits
        return avg_precision
    
    
    def _MeanReciprocalRankEvaluation(self,recommended_items, relevant_items):
        reciprocal_rank = 0.0
        for i, item in enumerate(relevant_items):
            if item in recommended_items:
                reciprocal_rank = 1.0 / (i+1)
                break
        return reciprocal_rank
        
    

   
