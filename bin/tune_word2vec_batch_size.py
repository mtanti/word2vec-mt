#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2025 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Tune the batch size of the word2vec model for Maltese such that a batch uses all available VRAM.
'''

import argparse
from word2vec_mt.paths import (
    vocab_mt_path, synonyms_mt_split_path, proccorpus_mt_path, skipgram_hyperparams_config_path,
)
from word2vec_mt.model import optimise_skipgram_batch_size



#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Tune the batch size of the word2vec model for Maltese such that a batch uses all'
            ' available VRAM.'
            ' | Input files:'
            f' * {skipgram_hyperparams_config_path} (manually set config file),'
            f' * {vocab_mt_path} (extract_vocab.py),'
            f' * {synonyms_mt_split_path} (split_synonym_data_set.py),'
            f' * {proccorpus_mt_path} (preprocess_corpus_to_train_set.py)'
            ' | Output files:'
            f' * {skipgram_hyperparams_config_path};'
            ' Note that this overwrites the batch_size hyperparameter in the config file.'
        ),
    )
    parser.add_argument(
        'max_batch_size',
        type=int,
        help=(
            'The maximum batch size to use.'
            ' The more VRAM in your GPU, the bigger this number can be.'
        ),
    )
    args = parser.parse_args()

    optimise_skipgram_batch_size(args.max_batch_size)


#########################################
if __name__ == '__main__':
    main()
