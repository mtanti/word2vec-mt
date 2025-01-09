'''
'''

import json
from dataclasses import dataclass
from word2vec_mt.model.data.common import DataSplit
from word2vec_mt.paths import vocab_mt_path, synonyms_mt_split_path


#########################################
@dataclass
class SynonymDataSplits:
    '''
    '''
    val: DataSplit
    dev: DataSplit
    test: DataSplit


#########################################
def load_synonym_data_set(
) -> SynonymDataSplits:
    '''
    '''
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        token2index_mt = {token: i for (i, token) in enumerate(line.strip() for line in f)}
    with open(synonyms_mt_split_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        val = data['val']
        dev = data['dev']
        test = data['test']

    return SynonymDataSplits(
        val=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in val['source']],
            targets_token_indexes=[
                [token2index_mt[target] for target in targets]
                for targets in val['targets']
            ],
        ),
        dev=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in dev['source']],
            targets_token_indexes=[
                [token2index_mt[target] for target in targets]
                for targets in dev['targets']
            ],
        ),
        test=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in test['source']],
            targets_token_indexes=[
                [token2index_mt[target] for target in targets]
                for targets in test['targets']
            ],
        ),
    )
