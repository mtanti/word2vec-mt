'''
'''

import random
import json
from dataclasses import dataclass
import numpy as np
from word2vec_mt.paths import (
    vocab_mt_path, vocab_en_path,
    synonyms_mt_val_path, synonyms_mt_dev_path, synonyms_mt_test_path,
    translations_mten_train_path, translations_mten_val_path, translations_mten_dev_path,
    translations_mten_test_path,
)
from word2vec_mt.data_maker.load_data import load_file


#########################################
@dataclass
class DataSplit:
    '''
    '''
    source_token_indexes: list[int]
    targets_token_indexes: list[list[int]]


#########################################
@dataclass
class SynonymDataSplits:
    '''
    '''
    val: DataSplit
    dev: DataSplit
    test: DataSplit


#########################################
@dataclass
class TranslationDataSplits:
    '''
    '''
    train: DataSplit
    val: DataSplit
    dev: DataSplit
    test: DataSplit


#########################################
def load_synonym_data_set(
) -> SynonymDataSplits:
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        token2index_mt = {token: i for (i, token) in enumerate(line.strip() for line in f)}
    with open(synonyms_mt_val_path, 'r', encoding='utf-8') as f:
        val = json.load(f)
    with open(synonyms_mt_dev_path, 'r', encoding='utf-8') as f:
        dev = json.load(f)
    with open(synonyms_mt_test_path, 'r', encoding='utf-8') as f:
        test = json.load(f)

    return SynonymDataSplits(
        val=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in val['source']],
            targets_token_indexes=[[token2index_mt[target] for target in targets] for targets in val['targets']],
        ),
        dev=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in dev['source']],
            targets_token_indexes=[[token2index_mt[target] for target in targets] for targets in dev['targets']],
        ),
        test=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in test['source']],
            targets_token_indexes=[[token2index_mt[target] for target in targets] for targets in test['targets']],
        ),
    )


#########################################
def load_translation_data_set(
    seed: int,
) -> TranslationDataSplits:
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        token2index_mt = {token: i for (i, token) in enumerate(line.strip() for line in f)}
    with open(vocab_en_path, 'r', encoding='utf-8') as f:
        token2index_en = {token: i for (i, token) in enumerate(line.strip() for line in f)}
    with open(translations_mten_train_path, 'r', encoding='utf-8') as f:
        train = json.load(f)
    with open(translations_mten_val_path, 'r', encoding='utf-8') as f:
        val = json.load(f)
    with open(translations_mten_dev_path, 'r', encoding='utf-8') as f:
        dev = json.load(f)
    with open(translations_mten_test_path, 'r', encoding='utf-8') as f:
        test = json.load(f)

    return SynonymDataSplits(
        train=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in train['source']],
            targets_token_indexes=[[token2index_en[target] for target in targets] for targets in train['targets']],
        ),
        val=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in val['source']],
            targets_token_indexes=[[token2index_en[target] for target in targets] for targets in val['targets']],
        ),
        dev=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in dev['source']],
            targets_token_indexes=[[token2index_en[target] for target in targets] for targets in dev['targets']],
        ),
        test=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in test['source']],
            targets_token_indexes=[[token2index_en[target] for target in targets] for targets in test['targets']],
        ),
    )


#########################################
@dataclass
class FlatDataSplit:
    '''
    '''
    source_token_indexes: np.ndarray
    similar_token_indexes: np.ndarray

    #########################################
    @staticmethod
    def flatten(
        data: DataSplit,
    ) -> list[int]:
        '''
        '''
        return FlatDataSplit(
            source_token_indexes=np.fromiter(
                data.source_token_indexes[i]
                for i in range(len(data.source_token_indexes))
                for _ in data.targets_token_indexes[i]
            ),
            similar_token_indexes=np.fromiter(
                similar
                for i in range(len(data.source_token_indexes))
                for similar in data.targets_token_indexes[i]
            ),
        )
