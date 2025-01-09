'''
'''


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
        num_batches: int,
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
