'''
Constants that are common throughout the project.
'''

import os
import word2vec_mt


#########################################

corpus_mt_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'corpus_mt.txt',
))
'''
The path to the Maltese corpus text file.
'''

vocab_mt_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'output', 'vocab_mt.txt',
))
'''
The path to the Maltese corpus text file.
'''

word2vec_mt_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'output', 'word2vec_mt.npy',
))
'''
The path to the English word2vec NumPy file.
'''

synonyms_mt_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'synonyms_mt.jsonl',
))
'''
The path to the Maltese text file of synonyms.
'''

vocab_en_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'vocab_en.txt',
))
'''
The path to the English vocabulary text file.
'''

word2vec_en_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'word2vec_en.npy',
))
'''
The path to the English word2vec NumPy file.
'''

translations_mten_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'translations_mten.jsonl',
))
'''
The path to the Maltese-English text file of translated words.
'''
