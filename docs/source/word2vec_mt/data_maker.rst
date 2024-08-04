data_maker
==========

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

.. toctree::
    :maxdepth: 1

    data_maker/entry.rst
    data_maker/load_data.rst
    data_maker/similar_word_data_maker_helper.rst
    data_maker/user_input_logic.rst
