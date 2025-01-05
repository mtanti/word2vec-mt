from word2vec_mt.model.data import load_synonym_data_set
from word2vec_mt.model.trainer import train_skipgram_model
from word2vec_mt.model.trainer import TrainListener
import word2vec_mt.constants
import h5py
from codecarbon import OfflineEmissionsTracker


with OfflineEmissionsTracker(country_iso_code='MLT') as tracker:
    data_set = load_synonym_data_set(0)
    with open(word2vec_mt.constants.vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = f.read().strip().split('\n')
    proccorp = h5py.File(word2vec_mt.constants.proccorpus_mt_path)

    class Listener(TrainListener):
        def ended_batch(batch_num, train_error):
            if batch_num%100 == 0:
                print('>', batch_num, train_error)
        def ended_epoch(epoch_num, val_error, new_best, num_bad_epochs):
            print(epoch_num, val_error)

    model = train_skipgram_model(
        vocab_size=len(vocab_mt),
        embedding_size=300,
        init_stddev=0.1,
        dropout_rate=0.5,
        learning_rate=0.1,
        max_epochs=10,
        train_data=proccorp['radius_5'],
        val_data=data_set.val,
        superbatch_size=1000000,
        batch_size=1000,
        patience=3,
        device='cuda',
        seed=0,
        listener=Listener(),
    )
