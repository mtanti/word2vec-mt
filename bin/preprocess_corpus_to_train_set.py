#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2024 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Convert the Maltese corpus into a word2vec skip-gram model train set.
'''

import argparse
from word2vec_mt.corpus_preprocessor.corpus_to_train_set import (
    preprocess_corpus,
)


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Convert the Maltese corpus into a word2vec skip-gram model train'
            ' set and save it in data/.'
        ),
    )
    parser.add_argument(
        'radius',
        type=int,
        help='The context window radius to use.',
    )
    parser.add_argument(
        '--buffer',
        type=int,
        default=10000,
        help='The number of data items to buffer before flushing into file.',
    )
    args = parser.parse_args()

    if args.radius <= 0:
        print('Radius must be a positive number.')
        return
    if args.buffer <= 0:
        print('Buffer must be a positive number.')
        return

    preprocess_corpus(args.radius, args.buffer)


#########################################
if __name__ == '__main__':
    main()
