'''
'''

import numpy as np
from word2vec_mt.model.data import DataSplit


#########################################
def cosine_similarity(
    vec: np.ndarray,
    mat: np.ndarray,
) -> np.ndarray:
    return (vec@mat.T)/(np.linalg.norm(vec)*np.linalg.norm(mat, axis=1))


#########################################
def average_precision(
    source_vec: np.ndarray,
    targets_mat: np.ndarray,
    expected_most_similar_rows: list[int],
) -> float:
    '''
    https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    '''
    similarity_sorted_indexes = np.argsort(-cosine_similarity(source_vec, targets_mat))
    similars_ranks = np.where(np.isin(similarity_sorted_indexes, expected_most_similar_rows))[0] + 1
    average_precision = (np.arange(1, len(expected_most_similar_rows) + 1)/similars_ranks).mean()
    return average_precision


#########################################
def synonym_mean_average_precision(
    embedding_matrix_mt: np.ndarray,
    data: DataSplit,
) -> float:
    '''
    '''
    total_average_precision = 0.0
    for i in range(len(data.source_token_indexes)):
        total_average_precision += average_precision(
            embedding_matrix_mt[data.source_token_indexes[i]],
            embedding_matrix_mt,
            data.similars_token_indexes[i],
        )
    return total_average_precision/len(data.source_token_indexes)


#########################################
def translation_mean_average_precision(
    embedding_matrix_mt: np.ndarray,
    embedding_matrix_en: np.ndarray,
    data: DataSplit,
) -> float:
    '''
    '''
    total_average_precision = 0.0
    for i in range(len(data.source_token_indexes)):
        total_average_precision += average_precision(
            embedding_matrix_mt[data.source_token_indexes[i]],
            embedding_matrix_en,
            data.similars_token_indexes[i],
        )
    return total_average_precision/len(data.source_token_indexes)
