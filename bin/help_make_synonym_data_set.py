#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2024 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Help make a Maltese synonym data set.
'''

import argparse
import word2vec_mt.data_maker


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description='Help make a Maltese synonym data set and save it in data/.'
    )

    parser.parse_args()

    word2vec_mt.data_maker.help_make_synonym_data_set()


#########################################
if __name__ == '__main__':
    main()
