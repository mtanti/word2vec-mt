#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2024 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Split the Maltese synonyms and translation data sets.
'''

import argparse
from word2vec_mt.data_maker.data_splitter import split_synonym_data_set, split_translation_data_set


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Split the Maltese synonyms and translation data sets that were extracted using'
            ' help_make_synonym_data_set and help_make_translation_data_set in a train, val,'
            ' dev, and test splits. Save the splits in output/.'
        ),
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        required=False,
        help='The random seed to use for randomly splitting the data sets (defaults to 0).'
    )
    parser.parse_args()

    split_synonym_data_set(seed=0)
    split_translation_data_set(seed=0)


#########################################
if __name__ == '__main__':
    main()
