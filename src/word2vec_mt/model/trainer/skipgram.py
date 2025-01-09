'''
'''

import os
import tempfile
import random
import math
import numpy as np
import torch
import h5py
from word2vec_mt.model.trainer.common import TrainListener
from word2vec_mt.model.model import SkipgramModel
from word2vec_mt.model.data import DataSplit
from word2vec_mt.model.evaluate import synonym_mean_average_precision


#########################################
class SkipgramDataSet(torch.utils.data.Dataset):
    '''
    '''

    #########################################
    def __init__(
        self,
        input_token_indexes: np.ndarray,
        target_token_indexes: np.ndarray,
    ) -> None:
        '''
        '''
        super().__init__()
        self.input_token_indexes = input_token_indexes
        self.target_token_indexes = target_token_indexes

    #########################################
    def __len__(
        self,
    ) -> int:
        '''
        '''
        return len(self.input_token_indexes)

    #########################################
    def __getitem__(
        self,
        index: int,
    ) -> dict[str, np.ndarray]:
        '''
        '''
        return {
            'input': self.input_token_indexes[index],
            'target': self.target_token_indexes[index],
        }


#########################################
def train_skipgram_model(
    vocab_size: int,
    embedding_size: int,
    init_stddev: float,
    dropout_rate: float,
    learning_rate: float,
    max_epochs: int,
    train_data: h5py.Dataset,
    val_data: DataSplit,
    superbatch_size: int,
    batch_size: int,
    patience: int,
    device: str,
    seed: int,
    listener: TrainListener = TrainListener(),
    test_mode: bool = False,
    one_superbatch: bool = False,
) -> SkipgramModel:
    '''
    '''
    assert superbatch_size%batch_size == 0

    model = SkipgramModel(vocab_size, embedding_size, init_stddev, dropout_rate)
    model.initialise(seed)
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    error_func = torch.nn.CrossEntropyLoss()
    best_val_map = 0.0
    num_bad_epochs = 0
    if not one_superbatch:
        num_batches = math.ceil(len(train_data)/batch_size)
    else:
        num_batches = math.ceil(min(len(train_data), superbatch_size)/batch_size)
    seed_rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as tmp_dir:
        listener.started_training()

        for epoch_num in range(1, max_epochs+1):
            listener.started_epoch(epoch_num, num_batches)

            batch_num = 0
            for superbatch_index in range(0, train_data.shape[0], superbatch_size):
                generator = torch.Generator()
                generator.manual_seed(seed_rng.randrange(0, 2**32 - 1))
                data_loader = torch.utils.data.DataLoader(
                    SkipgramDataSet(
                        input_token_indexes=train_data[
                            superbatch_index:superbatch_index+superbatch_size,
                            0,
                        ],
                        target_token_indexes=train_data[
                            superbatch_index:superbatch_index+superbatch_size,
                            1,
                        ],
                    ),
                    batch_size, shuffle=True, generator=generator,
                )

                model.train()
                for batch in data_loader:
                    batch_num += 1
                    listener.started_batch(batch_num)

                    batch_input = batch['input'].to(device)
                    batch_target = batch['target'].to(device)
                    optimiser.zero_grad()
                    logits = model(batch_input)
                    train_error = error_func(logits, batch_target)
                    train_error.backward()
                    optimiser.step()

                    listener.ended_batch(
                        batch_num,
                        num_batches,
                        train_error.detach().cpu().tolist(),
                    )

                    if test_mode:
                        if batch_num == 2:
                            break

                if test_mode:
                    if batch_num == 2:
                        break
                if one_superbatch:
                    break

            model.eval()
            val_map = synonym_mean_average_precision(model.get_embeddings(), val_data)
            if val_map > best_val_map:
                torch.save(model.state_dict(), os.path.join(tmp_dir, 'model.pt'))
                best_val_map = val_map
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
                if num_bad_epochs == patience:
                    listener.ended_epoch(epoch_num, val_map, num_bad_epochs == 0, num_bad_epochs)
                    break

            listener.ended_epoch(epoch_num, val_map, num_bad_epochs == 0, num_bad_epochs)

            if test_mode:
                break

        model.load_state_dict(torch.load(os.path.join(tmp_dir, 'model.pt'), weights_only=True))

        listener.ended_training()

        return model
