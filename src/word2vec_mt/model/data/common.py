'''
'''

from dataclasses import dataclass
import numpy as np


#########################################
@dataclass
class FlatDataSplit:
    '''
    '''
    source_token_indexes: np.ndarray
    similar_token_indexes: np.ndarray


#########################################
@dataclass
class DataSplit:
    '''
    '''
    source_token_indexes: list[int]
    targets_token_indexes: list[list[int]]

    def flatten(
        self,
    ) -> FlatDataSplit:
        '''
        '''
        return FlatDataSplit(
            source_token_indexes=np.fromiter((
                self.source_token_indexes[i]
                for i in range(len(self.source_token_indexes))
                for _ in self.targets_token_indexes[i]
            ), np.int32),
            similar_token_indexes=np.fromiter((
                similar
                for i in range(len(self.source_token_indexes))
                for similar in self.targets_token_indexes[i]
            ), np.int32),
        )
