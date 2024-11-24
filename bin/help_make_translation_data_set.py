#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2024 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Help make a Maltese translation data set.
'''

import argparse
from word2vec_mt.data_maker import (
    help_make_translations_data_set,
)


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description='Help make a Maltese translation data set and save it in output/.'
    )

    parser.parse_args()

    help_make_translations_data_set()


#########################################
if __name__ == '__main__':
    main()
