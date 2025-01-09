'''
'''

import json
from dataclasses import dataclass
from word2vec_mt.model.data.common import DataSplit
from word2vec_mt.paths import vocab_mt_path, vocab_en_path, translations_mten_split_path


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
def load_translation_data_set(
) -> TranslationDataSplits:
    '''
    '''
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        token2index_mt = {token: i for (i, token) in enumerate(line.strip() for line in f)}
    with open(vocab_en_path, 'r', encoding='utf-8') as f:
        token2index_en = {token: i for (i, token) in enumerate(line.strip() for line in f)}
    with open(translations_mten_split_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        train = data['train']
        val = data['val']
        dev = data['dev']
        test = data['test']

    return TranslationDataSplits(
        train=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in train['source']],
            targets_token_indexes=[
                [token2index_en[target] for target in targets]
                for targets in train['targets']
            ],
        ),
        val=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in val['source']],
            targets_token_indexes=[
                [token2index_en[target] for target in targets]
                for targets in val['targets']
            ],
        ),
        dev=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in dev['source']],
            targets_token_indexes=[
                [token2index_en[target] for target in targets]
                for targets in dev['targets']
            ],
        ),
        test=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in test['source']],
            targets_token_indexes=[
                [token2index_en[target] for target in targets]
                for targets in test['targets']
            ],
        ),
    )
