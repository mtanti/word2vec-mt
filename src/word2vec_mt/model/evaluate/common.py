'''
'''

import numpy as np


#########################################
def cosine_similarity(
    vec: np.ndarray,
    mat: np.ndarray,
) -> np.ndarray:
    '''
    '''
    return (vec@mat.T)/(np.linalg.norm(vec)*np.linalg.norm(mat, axis=1))


#########################################
def get_average_precision(
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
