from word2vec_mt.model.data import load_synonym_data_set
from word2vec_mt.model.trainer import train_skipgram_model
from word2vec_mt.model.trainer import TrainListener, SkipgramDataSet
import word2vec_mt.constants
import h5py
import timeit
import torch


data_set = load_synonym_data_set(0)
with open(word2vec_mt.constants.vocab_mt_path, 'r', encoding='utf-8') as f:
    vocab_mt = f.read().strip().split('\n')
proccorp = h5py.File(word2vec_mt.constants.proccorpus_mt_path, 'r')

class Listener(TrainListener):

    def __init__(self):
        super().__init__()
        self.start_time = 0.0

    def started_epoch(self, epoch_num):
        self.start_time = timeit.default_timer()

    def ended_batch(self, batch_num, num_batches, train_error):
        if batch_num%1000 == 0:
            print('>', batch_num, num_batches, timeit.default_timer() - self.start_time)
            self.start_time = timeit.default_timer()

    def ended_epoch(self, epoch_num, val_error, new_best, num_bad_epochs):
        print(epoch_num, val_error)

'''
upper_batch_size = 1000000
lower_batch_size = 1
while True:
    try:
        batch_size = (upper_batch_size + lower_batch_size)//2
        if batch_size == lower_batch_size:
            batch_size = upper_batch_size
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
            batch_size=batch_size,
            patience=3,
            device='cuda',
            seed=0,
            test_mode=True,
        )
        print('OK', lower_batch_size, batch_size, upper_batch_size)
        lower_batch_size = batch_size
    except torch.OutOfMemoryError:
        print('NO', lower_batch_size, batch_size, upper_batch_size)
        upper_batch_size = batch_size - 1
    if lower_batch_size == upper_batch_size:
        break

'''
model = train_skipgram_model(
            vocab_size=len(vocab_mt),
            embedding_size=300,
            init_stddev=0.1,
            dropout_rate=0.5,
            learning_rate=0.1,
            max_epochs=10,
            train_data=proccorp['radius_5'],
            val_data=data_set.val,
            superbatch_size=244*100000,
            batch_size=244,
            patience=3,
            device='cuda',
            seed=0,
            listener=Listener(),
            one_superbatch=True,
        )
#'''
