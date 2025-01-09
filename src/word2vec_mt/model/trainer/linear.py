'''
'''

import os
import tempfile
import numpy as np
import torch
from word2vec_mt.model.trainer.common import TrainListener
from word2vec_mt.model.model import LinearModel
from word2vec_mt.model.data import DataSplit, FlatDataSplit
from word2vec_mt.model.evaluate import translation_mean_average_precision


#########################################
class LinearDataSet(torch.utils.data.Dataset):
    '''
    '''

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
    use_bias: bool,
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
            input_embeddings=source_embedding_matrix[train_data.source_token_indexes, :],
            target_embeddings=target_embedding_matrix[train_data.similar_token_indexes, :],
        ),
        batch_size, shuffle=True, generator=generator,
    )
    best_val_map = 0.0
    num_bad_epochs = 0
    num_batches = len(train_data.source_token_indexes)
    with tempfile.TemporaryDirectory() as tmp_dir:
        listener.started_training()

        for epoch_num in range(1, max_epochs+1):
            listener.started_epoch(epoch_num, num_batches)

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

                listener.ended_batch(
                    batch_num,
                    num_batches,
                    train_error.detach().cpu().tolist(),
                )

            model.eval()
            val_map = translation_mean_average_precision(
                source_embedding_matrix,
                target_embedding_matrix,
                val_data,
            )
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
