'''
'''

import json
import torch
import h5py
import numpy as np
import optuna
from word2vec_mt.model.trainer import train_skipgram_model, train_linear_model
from word2vec_mt.model.data import load_synonym_data_set, load_translation_data_set, FlatDataSplit
from word2vec_mt.model.trainer import train_skipgram_model, train_linear_model, TrainListener
from word2vec_mt.model.evaluate import synonym_mean_average_precision, translation_mean_average_precision
from word2vec_mt.paths import (
    vocab_mt_path,
    proccorpus_mt_path, word2vec_mt_path,
    skipgram_hyperparams_config_path, skipgram_hyperparams_record_path,
    skipgram_hyperparams_best_path, skipgram_model_path,
    word2vec_en_path, word2vec_mten_path,
    linear_hyperparams_config_path, linear_hyperparams_record_path,
    linear_hyperparams_best_path, linear_model_path,
)


#########################################
def optimise_skipgram_batch_size(
) -> None:
    '''
    '''
    data_set = load_synonym_data_set()
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = f.read().strip().split('\n')
    proccorp = h5py.File(proccorpus_mt_path, 'r')
    with open(skipgram_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    upper_batch_size = 1024
    lower_batch_size = 1
    while True:
        batch_size = (upper_batch_size + lower_batch_size)//2
        if batch_size == lower_batch_size:
            batch_size = upper_batch_size
        print('best batch size is between', lower_batch_size, 'and', upper_batch_size, '; now trying', batch_size)

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
class Listener(TrainListener):
    '''
    '''

    #########################################
    def __init__(self):
        '''
        '''
        super().__init__()

    #########################################
    def started_training(self):
        print()

    #########################################
    def started_epoch(self, epoch_num):
        '''
        '''
        print('-----------')
        print('epoch', epoch_num)

    #########################################
    def ended_batch(self, batch_num, num_batches, train_error):
        if batch_num%1000 == 0:
            print('batch', batch_num, 'of', num_batches)

    #########################################
    def ended_epoch(self, epoch_num, val_map, new_best, num_bad_epochs):
        print('ended epoch with val map:', val_map)
        print()


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
    print('Now training model with init_stddev', init_stddev, ', dropout_rate', dropout_rate, ', learning_rate', learning_rate)
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
def linear_model_objective(
    trial: optuna.Trial,
) -> float:
    '''
    '''
    data_set = load_translation_data_set()
    word2vec_mt = np.load(word2vec_mt_path)
    word2vec_en = np.load(word2vec_en_path)
    with open(linear_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    init_stddev = trial.suggest_categorical('init_stddev', hyperparams['init_stddev'])
    use_bias = trial.suggest_categorical('use_bias', hyperparams['use_bias'])
    weight_decay = trial.suggest_categorical('weight_decay', hyperparams['weight_decay'])
    learning_rate = trial.suggest_categorical('learning_rate', hyperparams['learning_rate'])
    batch_size = trial.suggest_categorical('batch_size', hyperparams['batch_size'])

    print()
    print('===========================================')
    print('Now training model with init_stddev', init_stddev, ', use_bias', use_bias, ', weight_decay', weight_decay, ', learning_rate', learning_rate)
    model = train_linear_model(
        source_embedding_size=hyperparams['source_embedding_size'],
        target_embedding_matrix=hyperparams['target_embedding_size'],
        init_stddev=init_stddev,
        use_bias=use_bias,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        max_epochs=hyperparams['max_epochs'],
        source_embedding_matrix=word2vec_mt,
        target_embedding_size=word2vec_en,
        train_data=FlatDataSplit.flatten(data_set.train),
        val_data=data_set.val,
        batch_size=batch_size,
        patience=hyperparams['patience'],
        device=hyperparams['device'],
        seed=hyperparams['seed'],
        one_superbatch=True,
        listener=Listener(),
    )

    dev_map = translation_mean_average_precision(word2vec_mt, word2vec_en, data_set.dev)
    return dev_map


#########################################
def tune_skipgram_model(
) -> None:
    with open(skipgram_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    study = optuna.create_study(
        direction='maximize',
        study_name='word2vec_mt',
        storage='sqlite:///' + skipgram_hyperparams_record_path,
        load_if_exists=True,
    )
    num_complete_trials = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
    study.optimize(skipgram_model_objective, n_trials=max(0, hyperparams['tuning_trials'] - num_complete_trials))


#########################################
def tune_linear_model(
) -> None:
    with open(linear_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    study = optuna.create_study(
        direction='maximize',
        study_name='word2vec_mt',
        storage='sqlite:///' + skipgram_hyperparams_record_path,
        load_if_exists=True,
    )
    num_complete_trials = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
    study.optimize(linear_model_objective, n_trials=max(0, hyperparams['tuning_trials'] - num_complete_trials))


#########################################
def train_best_skipgram_model(
) -> None:
    data_set = load_synonym_data_set()
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = f.read().strip().split('\n')
    proccorp = h5py.File(proccorpus_mt_path, 'r')
    with open(skipgram_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    study = optuna.create_study(
        direction='maximize',
        study_name='word2vec_mt',
        storage='sqlite:///' + skipgram_hyperparams_record_path,
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


#########################################
def train_best_linear_model(
) -> None:
    data_set = load_translation_data_set()
    word2vec_mt = np.load(word2vec_mt_path)
    word2vec_en = np.load(word2vec_en_path)
    with open(linear_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    study = optuna.create_study(
        direction='maximize',
        study_name='word2vec_mt',
        storage='sqlite:///' + linear_hyperparams_record_path,
        load_if_exists=True,
    )

    print('training model')
    model = train_skipgram_model(
        source_embedding_size=hyperparams['source_embedding_size'],
        target_embedding_matrix=hyperparams['target_embedding_size'],
        init_stddev=study.best_params['init_stddev'],
        use_bias=study.best_params['use_bias'],
        weight_decay=study.best_params['weight_decay'],
        learning_rate=study.best_params['learning_rate'],
        max_epochs=hyperparams['max_epochs'],
        train_data=FlatDataSplit.flatten(data_set.train),
        val_data=data_set.val,
        batch_size=study.best_params['batch_size'],
        patience=hyperparams['patience'],
        device=hyperparams['device'],
        seed=hyperparams['seed'],
        one_superbatch=True,
        listener=Listener(),
    )

    print('saving model')
    torch.save(model, linear_model_path)

    print('saving word2vec embeddings')
    with torch.no_grad():
        word2vec_mten = model(
            torch.from_numpy(word2vec_en).to(hyperparams['device'])
        ).cpu().numpy()
    np.save(word2vec_mten_path, word2vec_mten, allow_pickle=False)

    print('evaluating model')
    test_map = translation_mean_average_precision(word2vec_mt, word2vec_en, data_set.test)
    hyperparams.update(study.best_params)
    hyperparams['test_set_map'] = test_map

    print('saving model hyperparameters')
    with open(linear_hyperparams_best_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f)
