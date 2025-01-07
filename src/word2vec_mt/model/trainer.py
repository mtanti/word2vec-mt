'''
'''

import os
import tempfile
import random
import math
import numpy as np
import torch
import h5py
from word2vec_mt.model.model import SkipgramModel, LinearModel
from word2vec_mt.model.data import DataSplit, FlatDataSplit
from word2vec_mt.model.evaluate import synonym_mean_average_precision, translation_mean_average_precision


#########################################
class TrainListener:
    '''
    '''

    #########################################
    def __init__(
        self,
    ) -> None:
        '''
        '''

    #########################################
    def started_training(
        self,
    ) -> None:
        '''
        '''

    #########################################
    def started_epoch(
        self,
        epoch_num: int,
    ) -> None:
        '''
        '''

    #########################################
    def started_batch(
        self,
        batch_num: int,
    ) -> None:
        '''
        '''

    #########################################
    def ended_batch(
        self,
        batch_num: int,
        num_batches: int,
        train_error: float,
    ) -> None:
        '''
        '''

    #########################################
    def ended_epoch(
        self,
        epoch_num: int,
        val_map: float,
        new_best: bool,
        num_bad_epochs: int,
    ) -> None:
        '''
        '''

    #########################################
    def ended_training(
        self,
    ) -> None:
        '''
        '''


#########################################
class SkipgramDataSet(torch.utils.data.Dataset):

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
    num_batches = math.ceil(len(train_data)/batch_size) if not one_superbatch else math.ceil(min(len(train_data), superbatch_size)/batch_size)
    seed_rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as tmp_dir:
        listener.started_training()

        for epoch_num in range(1, max_epochs+1):
            listener.started_epoch(epoch_num)

            batch_num = 0
            for superbatch_index in range(0, train_data.shape[0], superbatch_size):
                generator = torch.Generator()
                generator.manual_seed(seed_rng.randrange(0, 2**32 - 1))
                data_loader = torch.utils.data.DataLoader(
                    SkipgramDataSet(
                        input_token_indexes=train_data[superbatch_index:superbatch_index+superbatch_size, 0],
                        target_token_indexes=train_data[superbatch_index:superbatch_index+superbatch_size, 1],
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

                    listener.ended_batch(batch_num, num_batches, train_error.detach().cpu().tolist())

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


#########################################
class LinearDataSet(torch.utils.data.Dataset):

    #########################################
    def __init__(
        self,
        input_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
    ) -> None:
        '''
        '''
        super().__init__()
        self.input_embeddings = input_embeddings
        self.target_embeddings = target_embeddings

    #########################################
    def __len__(
        self,
    ) -> int:
        '''
        '''
        return len(self.input_embeddings)

    #########################################
    def __getitem__(
        self,
        index: int,
    ) -> dict[str, np.ndarray]:
        '''
        '''
        return {
            'input': self.input_embeddings[index, :],
            'target': self.target_embeddings[index, :],
        }


#########################################
def train_linear_model(
    source_embedding_size: int,
    target_embedding_size: int,
    init_stddev: float,
    use_bias: float,
    weight_decay: float,
    learning_rate: float,
    max_epochs: int,
    source_embedding_matrix: np.ndarray,
    target_embedding_matrix: np.ndarray,
    train_data: FlatDataSplit,
    val_data: DataSplit,
    batch_size: int,
    patience: int,
    device: str,
    seed: int,
    listener: TrainListener = TrainListener(),
) -> LinearModel:
    '''
    '''
    model = LinearModel(source_embedding_size, target_embedding_size, init_stddev, use_bias)
    model.initialise(seed)
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    error_func = torch.nn.MSELoss()
    generator = torch.Generator()
    generator.manual_seed(seed)
    data_loader = torch.utils.data.DataLoader(
        LinearDataSet(
            input_token_indexes=source_embedding_matrix[train_data.source_token_indexes, :],
            target_token_indexes=target_embedding_matrix[train_data.similar_token_indexes, :],
        ),
        batch_size, shuffle=True, generator=generator,
    )
    best_val_map = 0.0
    num_bad_epochs = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        listener.started_training()

        for epoch_num in range(1, max_epochs+1):
            listener.started_epoch(epoch_num)

            model.train()
            for (batch_num, batch) in enumerate(data_loader, start=1):
                listener.started_batch(batch_num)

                batch_input = batch['input'].to(device)
                batch_target = batch['target'].to(device)
                optimiser.zero_grad()
                outputs = model(batch_input)
                train_error = error_func(outputs, batch_target)
                train_error.backward()
                optimiser.step()

                listener.ended_batch(batch_num, train_error.detach().cpu().tolist())

            model.eval()
            val_map = translation_mean_average_precision(source_embedding_matrix, target_embedding_matrix, val_data)
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

        model.load_state_dict(torch.load(os.path.join(tmp_dir, 'model.pt'), weights_only=True))

        listener.ended_training()

        return model
