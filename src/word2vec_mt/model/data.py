'''
'''

import random
from dataclasses import dataclass
import numpy as np
from word2vec_mt.constants import vocab_mt_path, vocab_en_path, synonyms_mt_path, translations_mten_path
from word2vec_mt.data_maker.load_data import load_file


#########################################

SYNONYM_VAL_FRAC = 0.2
'''
'''

SYNONYM_DEV_FRAC = 0.4
'''
'''

SYNONYM_TEST_FRAC = 0.4
'''
'''

assert SYNONYM_VAL_FRAC + SYNONYM_DEV_FRAC + SYNONYM_TEST_FRAC == 1.0

TRANSLATION_TRAIN_FRAC = 0.5
'''
'''

TRANSLATION_VAL_FRAC = 0.1
'''
'''

TRANSLATION_DEV_FRAC = 0.2
'''
'''

TRANSLATION_TEST_FRAC = 0.2
'''
'''

assert TRANSLATION_TRAIN_FRAC + TRANSLATION_VAL_FRAC + TRANSLATION_DEV_FRAC + TRANSLATION_TEST_FRAC == 1.0


#########################################
@dataclass
class DataSplit:
    '''
    '''
    source_token_indexes: list[int]
    similars_token_indexes: list[list[int]]


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
    seed: int,
) -> SynonymDataSplits:
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = set(line.strip() for line in f)
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        token2index_mt = {token: i for (i, token) in enumerate(line.strip() for line in f)}

    data = load_file(vocab_mt, vocab_mt, synonyms_mt_path)

    rng = random.Random(seed)
    rng.shuffle(data)

    val_size = int(len(data)*(SYNONYM_VAL_FRAC))
    dev_size = int(len(data)*(SYNONYM_DEV_FRAC))
    val = data[0:val_size]
    dev = data[val_size:val_size+dev_size]
    test = data[val_size+dev_size:]

    return SynonymDataSplits(
        val=DataSplit(
            source_token_indexes=[token2index_mt[entry.source] for entry in val],
            similars_token_indexes=[[token2index_mt[similar] for similar in entry.similars] for entry in val],
        ),
        dev=DataSplit(
            source_token_indexes=[token2index_mt[entry.source] for entry in dev],
            similars_token_indexes=[[token2index_mt[similar] for similar in entry.similars] for entry in dev],
        ),
        test=DataSplit(
            source_token_indexes=[token2index_mt[entry.source] for entry in test],
            similars_token_indexes=[[token2index_mt[similar] for similar in entry.similars] for entry in test],
        ),
    )


#########################################
def load_translation_data_set(
    seed: int,
) -> TranslationDataSplits:
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = set(line.strip() for line in f)
    with open(vocab_en_path, 'r', encoding='utf-8') as f:
        vocab_en = set(line.strip() for line in f)
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        token2index_mt = {token: i for (i, token) in enumerate(line.strip() for line in f)}
    with open(vocab_en_path, 'r', encoding='utf-8') as f:
        token2index_en = {token: i for (i, token) in enumerate(line.strip() for line in f)}

    data = load_file(vocab_mt, vocab_en, translations_mten_path)

    rng = random.Random(seed)
    rng.shuffle(data)

    train_size = int(len(data)*(TRANSLATION_TRAIN_FRAC))
    val_size = int(len(data)*(TRANSLATION_VAL_FRAC))
    dev_size = int(len(data)*(TRANSLATION_DEV_FRAC))
    train = data[0:train_size]
    val = data[train_size:train_size+val_size]
    dev = data[train_size+val_size:train_size+val_size+dev_size]
    test = data[train_size+val_size+dev_size:]

    return SynonymDataSplits(
        train=DataSplit(
            source_token_indexes=[token2index_mt[entry.source] for entry in train],
            similars_token_indexes=[[token2index_en[similar] for similar in entry.similars] for entry in train],
        ),
        val=DataSplit(
            source_token_indexes=[token2index_mt[entry.source] for entry in val],
            similars_token_indexes=[[token2index_en[similar] for similar in entry.similars] for entry in val],
        ),
        dev=DataSplit(
            source_token_indexes=[token2index_mt[entry.source] for entry in dev],
            similars_token_indexes=[[token2index_en[similar] for similar in entry.similars] for entry in dev],
        ),
        test=DataSplit(
            source_token_indexes=[token2index_mt[entry.source] for entry in test],
            similars_token_indexes=[[token2index_en[similar] for similar in entry.similars] for entry in test],
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
                for _ in data.similars_token_indexes[i]
            ),
            similar_token_indexes=np.fromiter(
                similar
                for i in range(len(data.source_token_indexes))
                for similar in data.similars_token_indexes[i]
            ),
        )
