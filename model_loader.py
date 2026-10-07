import pickle
import numpy as np
from scipy.sparse import csr_matrix

class RecommenderModel:
    def __init__(self, model_path='model.pkl'):
        with open(model_path, 'rb') as f:
            self.model, self.matrix, self.user_to_idx, self.item_to_idx = pickle.load(f)
            self.idx_to_item = {i: item_id for item_id, i in self.item_to_idx.items()}

    def recommend(self, user_id, top_n=3):
        user_id = str(user_id)
        if user_id not in self.user_to_idx:
            return None
        user_idx = self.user_to_idx[user_id]
        # Берём только строку этого пользователя (CSR-матрица с одной строкой)
        user_row = self.matrix[user_idx]  # shape: (1, n_items)
        ids, scores = self.model.recommend(user_idx, user_row, N=top_n)
        return [(self.idx_to_item[i], float(score)) for i, score in zip(ids, scores)]