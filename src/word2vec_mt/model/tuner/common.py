'''
'''

from typing import Optional
import tqdm
from word2vec_mt.model.trainer import TrainListener


#########################################
class Listener(TrainListener):
    '''
    '''

    #########################################
    def __init__(
        self,
    ) -> None:
        '''
        '''
        self.progbar: Optional[tqdm.tqdm] = None

    #########################################
    def started_training(
        self,
    ) -> None:
        print()

    #########################################
    def started_epoch(
        self,
        epoch_num: int,
        num_batches: int,
    ) -> None:
        '''
        '''
        print('-----------')
        print('epoch', epoch_num)
        self.progbar = tqdm.tqdm(total=num_batches)

    #########################################
    def ended_batch(
        self,
        batch_num: int,
        num_batches: int,
        train_error: float,
    ) -> None:
        assert self.progbar is not None
        self.progbar.update()
        if batch_num == num_batches:
            self.progbar.close()
            self.progbar = None

    #########################################
    def ended_epoch(
        self,
        epoch_num: int,
        val_map: float,
        new_best: bool,
        num_bad_epochs: int,
    ) -> None:
        print('ended epoch with val map:', val_map)
        print()
