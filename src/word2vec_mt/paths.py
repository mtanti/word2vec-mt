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
The path to the Maltese word2vec NumPy file.
'''

synonyms_mt_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'output', 'synonyms_mt.jsonl',
))
'''
The path to the Maltese text file of synonyms.
'''

synonyms_mt_val_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'synonyms_mt_val.json',
))
'''
The path to the Maltese text file of synonyms validation split.
'''

synonyms_mt_dev_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'synonyms_mt_dev.json',
))
'''
The path to the Maltese text file of synonyms development split.
'''

synonyms_mt_test_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'synonyms_mt_test.json',
))
'''
The path to the Maltese text file of synonyms test split.
'''

proccorpus_mt_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'proccorpus_mt.hdf5',
))
'''
The path to the processed Maltese corpus file.
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

word2vec_mten_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'output', 'word2vec_mten.npy',
))
'''
The path to the English-aligned Maltese word2vec NumPy file.
'''

translations_mten_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'output', 'translations_mten.jsonl',
))
'''
The path to the Maltese-English text file of translated words.
'''

translations_mten_train_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'translations_mten_train.json',
))
'''
The path to the Maltese-English text file of translated words train split.
'''

translations_mten_val_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'translations_mten_val.json',
))
'''
The path to the Maltese-English text file of translated words validation split.
'''

translations_mten_dev_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'translations_mten_dev.json',
))
'''
The path to the Maltese-English text file of translated words development split.
'''

translations_mten_test_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'translations_mten_test.json',
))
'''
The path to the Maltese-English text file of translated words test split.
'''

skipgram_hyperparams_config_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'skipgram_hyperparams.json',
))
'''
The path to the skipgram model's hyperparameters configuration file.
'''

skipgram_hyperparams_record_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'skipgram_hyperparams.sqlite3',
))
'''
The path to the skipgram model's hyperparameter search history file.
'''

skipgram_hyperparams_best_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'skipgram_best_hyperparams.json',
))
'''
The path to the skipgram model's best hyperparameters found from search file.
'''

skipgram_model_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'output', 'skipgram_model.pt',
))
'''
The path to the saved final skipgram model (after tuning).
'''

linear_hyperparams_config_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'linear_hyperparams.json',
))
'''
The path to the linear model's hyperparameters configuration file.
'''

linear_hyperparams_record_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'linear_hyperparams.sqlite3',
))
'''
The path to the linear model's hyperparameter search history file.
'''

linear_hyperparams_best_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'linear_best_hyperparams.json',
))
'''
The path to the linear model's best hyperparameters found from search file.
'''

linear_model_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'output', 'linear_model.pt',
))
'''
The path to the saved final linear model (after tuning).
'''