'''
'''

import json
import csv
import torch
import h5py
import numpy as np
import optuna
from word2vec_mt.model.tuner.common import Listener
from word2vec_mt.model.trainer import train_skipgram_model
from word2vec_mt.model.data import load_synonym_data_set
from word2vec_mt.model.evaluate import synonym_mean_average_precision
from word2vec_mt.paths import (
    vocab_mt_path,
    proccorpus_mt_path, word2vec_mt_path,
    skipgram_hyperparams_config_path, skipgram_hyperparams_db_path,
    skipgram_hyperparams_result_path, skipgram_hyperparams_best_path, skipgram_model_path,
)


#########################################
def optimise_skipgram_batch_size(
    max_batch_size: int,
) -> None:
    '''
    '''
    data_set = load_synonym_data_set()
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = f.read().strip().split('\n')
    proccorp = h5py.File(proccorpus_mt_path, 'r')
    with open(skipgram_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    upper_batch_size = max_batch_size
    lower_batch_size = 1
    while True:
        batch_size = (upper_batch_size + lower_batch_size)//2
        if batch_size == lower_batch_size:
            batch_size = upper_batch_size
        print(
            f'best batch size is between {lower_batch_size} and {upper_batch_size};'
             ' now trying {batch_size}'
        )

        try:
            train_skipgram_model(
                vocab_size=len(vocab_mt),
                embedding_size=hyperparams['embedding_size'],
                init_stddev=hyperparams['init_stddev'][0],
                dropout_rate=hyperparams['dropout_rate'][0],
                learning_rate=hyperparams['learning_rate'][0],
                max_epochs=hyperparams['max_epochs'],
                train_data=proccorp['radius_'+str(hyperparams['radius'])],
                val_data=data_set.val,
                superbatch_size=batch_size*hyperparams['superbatch_size_multiple'],
                batch_size=batch_size,
                patience=hyperparams['patience'],
                device=hyperparams['device'],
                seed=hyperparams['seed'],
                test_mode=True,
            )
            lower_batch_size = batch_size
        except torch.OutOfMemoryError:
            upper_batch_size = batch_size - 1

        if hyperparams['device'].startswith('cuda'):
            torch.cuda.empty_cache()
        if abs(lower_batch_size - upper_batch_size) <= 1:
            break

    print('settled on', lower_batch_size)
    print()

    hyperparams['batch_size'] = lower_batch_size
    with open(skipgram_hyperparams_config_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, ensure_ascii=False, indent=4)


#########################################
def skipgram_model_objective(
    trial: optuna.Trial,
) -> float:
    '''
    '''
    data_set = load_synonym_data_set()
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = f.read().strip().split('\n')
    proccorp = h5py.File(proccorpus_mt_path, 'r')
    with open(skipgram_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    init_stddev = trial.suggest_categorical('init_stddev', hyperparams['init_stddev'])
    dropout_rate = trial.suggest_categorical('dropout_rate', hyperparams['dropout_rate'])
    learning_rate = trial.suggest_categorical('learning_rate', hyperparams['learning_rate'])

    print()
    print('===========================================')
    print(
        f'Now training model with init_stddev: {init_stddev}, dropout_rate: {dropout_rate},'
         ' learning_rate: {learning_rate}'
    )
    model = train_skipgram_model(
        vocab_size=len(vocab_mt),
        embedding_size=hyperparams['embedding_size'],
        init_stddev=init_stddev,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        max_epochs=hyperparams['max_epochs'],
        train_data=proccorp['radius_'+str(hyperparams['radius'])],
        val_data=data_set.val,
        superbatch_size=hyperparams['batch_size']*hyperparams['superbatch_size_multiple'],
        batch_size=hyperparams['batch_size'],
        patience=hyperparams['patience'],
        device=hyperparams['device'],
        seed=hyperparams['seed'],
        one_superbatch=True,
        listener=Listener(),
    )

    dev_map = synonym_mean_average_precision(model.get_embeddings(), data_set.dev)
    return dev_map


#########################################
def tune_skipgram_model(
) -> None:
    '''
    '''
    with open(skipgram_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    study = optuna.create_study(
        direction='maximize',
        study_name='word2vec_mt',
        storage='sqlite:///' + skipgram_hyperparams_db_path,
        load_if_exists=True,
    )
    num_complete_trials = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
    study.optimize(
        skipgram_model_objective,
        n_trials=max(0, hyperparams['tuning_trials'] - num_complete_trials),
    )

    with open(skipgram_hyperparams_result_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'init_stddev',
            'dropout_rate',
            'learning_rate',
            'dev_map',
        ])
        complete_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        for trial in complete_trials:
            writer.writerow([
                trial.params['init_stddev'],
                trial.params['dropout_rate'],
                trial.params['learning_rate'],
                trial.value,
            ])


#########################################
def train_best_skipgram_model(
) -> None:
    '''
    '''
    data_set = load_synonym_data_set()
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = f.read().strip().split('\n')
    proccorp = h5py.File(proccorpus_mt_path, 'r')
    with open(skipgram_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    study = optuna.create_study(
        direction='maximize',
        study_name='word2vec_mt',
        storage='sqlite:///' + skipgram_hyperparams_db_path,
        load_if_exists=True,
    )

    print('training model')
    model = train_skipgram_model(
        vocab_size=len(vocab_mt),
        embedding_size=hyperparams['embedding_size'],
        init_stddev=study.best_params['init_stddev'],
        dropout_rate=study.best_params['dropout_rate'],
        learning_rate=study.best_params['learning_rate'],
        max_epochs=hyperparams['max_epochs'],
        train_data=proccorp['radius_'+str(hyperparams['radius'])],
        val_data=data_set.val,
        superbatch_size=hyperparams['batch_size']*hyperparams['superbatch_size_multiple'],
        batch_size=hyperparams['batch_size'],
        patience=hyperparams['patience'],
        device=hyperparams['device'],
        seed=hyperparams['seed'],
        listener=Listener(),
    )

    print('saving model')
    torch.save(model, skipgram_model_path)

    print('saving word2vec embeddings')
    np.save(word2vec_mt_path, model.get_embeddings(), allow_pickle=False)

    print('evaluating model')
    test_map = synonym_mean_average_precision(model.get_embeddings(), data_set.test)
    hyperparams.update(study.best_params)
    hyperparams['test_set_map'] = test_map

    print('saving model hyperparameters')
    with open(skipgram_hyperparams_best_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f)
