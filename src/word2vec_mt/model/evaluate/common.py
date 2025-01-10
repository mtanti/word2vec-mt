'''
'''

from dataclasses import dataclass
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
    expected_most_similar_indexes: list[int],
) -> float:
    '''
    https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    '''
    similarity_sorted_indexes = np.argsort(-cosine_similarity(source_vec, targets_mat))
    similars_ranks = np.where(
        np.isin(similarity_sorted_indexes, expected_most_similar_indexes)
    )[0] + 1
    average_precision = (np.arange(1, len(expected_most_similar_indexes) + 1)/similars_ranks).mean()
    return average_precision


#########################################
@dataclass
class Report:
    '''
    '''

    top_5_tokens: list[str]
    similars_ranks: list[tuple[str, int]]


#########################################
def get_report_for_one_source(
    source_vec: np.ndarray,
    targets_mat: np.ndarray,
    expected_most_similar_indexes: list[int],
    vocab: list[str],
) -> Report:
    '''
    '''
    similarity_sorted_indexes = np.argsort(-cosine_similarity(source_vec, targets_mat)).tolist()
    similars_ranks = (np.where(
        np.isin(similarity_sorted_indexes, expected_most_similar_indexes)
    )[0] + 1).tolist()
    return Report(
        top_5_tokens=[vocab[index] for index in similarity_sorted_indexes[:5]],
        similars_ranks=list(sorted([
            (vocab[index], rank)
            for (index, rank) in zip(expected_most_similar_indexes, similars_ranks)
        ], key=lambda pair: pair[1])),
    )
