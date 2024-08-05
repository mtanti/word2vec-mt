#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2024 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Extract a vocabulary from the Maltese corpus.
'''

import argparse
from word2vec_mt.vocab_extractor import (
    extract_vocab,
)


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Extract a vocabulary from the Maltese corpus and save it in'
            ' output/.'
        )
    )

    parser.parse_args()

    extract_vocab()


#########################################
if __name__ == '__main__':
    main()
