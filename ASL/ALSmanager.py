from implicit.cpu.als import AlternatingLeastSquares

import numpy as np

import scipy
import pandas as pd

class ALSmanager:

    def __init__(self,path_to_model,inter, new = True):

        self.sp_ma = self.process_dataset(inter)

        if new == True:
        
            self.model = AlternatingLeastSquares(factors=64,
            regularization=0.5,
            iterations=7)
            
            self.model.fit(self.sp_ma, show_progress=True)
        
            self.model.save(path_to_model)

        else:
            with open(path_to_model+".npz", 'rb') as f:
                self.model = AlternatingLeastSquares.load(path_to_model+".npz")
    

    def process_dataset(self, inter):
            inter['rating'] = inter['rating'].fillna(0)

            inter['rating'] = [r if r >= 4 else -(3-r) for r in inter['rating']]


            return scipy.sparse.csr_matrix((inter['rating'], 
                                            (inter['user_id'],inter['item_id'])))
    
    def predict_for_user(self,user_id,show_first = 10):
        ids, scores = self.model.recommend(user_id, self.sp_ma[user_id], N=show_first, filter_already_liked_items=True)
        return ids
    
    def update_users(self,users,inter):
        self.sp_ma = self.process_dataset(inter)
        self.model.partial_fit_users(users,self.sp_ma[users])