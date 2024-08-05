#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2024 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Download the Maltese and English data sources to be used for creating word2vec
embeddings.
'''

import argparse
from word2vec_mt.data_downloader import (
    download_mt, download_en,
)


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Download the Maltese and English data sources to be used for'
            ' creating word2vec embeddings and save them in data/.'
        )
    )

    parser.parse_args()

    download_mt()
    download_en()


#########################################
if __name__ == '__main__':
    main()
