'''
'''

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

    #########################################
    def ended_batch(
        self,
        batch_num: int,
        num_batches: int,
        train_error: float,
    ) -> None:
        if batch_num%1000 == 0:
            print('batch', batch_num, 'of', num_batches)

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
