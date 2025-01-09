'''
'''

import numpy as np
from word2vec_mt.model.data import DataSplit
from word2vec_mt.model.evaluate.common import get_average_precision


#########################################
def synonym_mean_average_precision(
    embedding_matrix_mt: np.ndarray,
    data: DataSplit,
) -> float:
    '''
    '''
    total_average_precision = 0.0
    for (source_index, targets_indexes) in zip(
        data.source_token_indexes,
        data.targets_token_indexes,
    ):
        total_average_precision += get_average_precision(
            embedding_matrix_mt[source_index],
            embedding_matrix_mt,
            targets_indexes,
        )
    return total_average_precision/len(data.source_token_indexes)