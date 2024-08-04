'''
Data making helper package.

Used to assist with manually constructing a similar word data set consisting of
a sample of words called source words together with and their similar words
taken from one of the vocabularies.
Each source word - similar words pair is called an entry.

An entry is validated by checking that:

* The source word is in the source vocabulary.
* The similar words are in the similars vocabulary.
* There is at least one similar word.
* The source word and its similars are all unique.
* The source word is different from the words in previous entries.
'''

from word2vec_mt.data_maker.similar_word_data_maker_helper import (
    help_make_synonym_data_set,
    help_make_translations_data_set,
)
