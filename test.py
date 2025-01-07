from word2vec_mt.model.tuner import optimise_batch_size, tune_skipgram_model, train_best_skipgram_model
from codecarbon import OfflineEmissionsTracker

'''
with OfflineEmissionsTracker(
    'word2vec_mt',
    country_iso_code='MLT',
    log_level='error',
    tracking_mode='process',
) as tracker:
'''
if True:
    print('optimising batch sizes')
    optimise_batch_size()

    print('tuning model')
    tune_skipgram_model()

    print('training best model')
    train_best_skipgram_model()
