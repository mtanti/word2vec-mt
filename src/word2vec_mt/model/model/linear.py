'''
'''

import torch


#########################################
class LinearModel(torch.nn.Module):
    '''
    '''

    #########################################
    def __init__(
        self,
        source_embedding_size: int,
        target_embedding_size: int,
        init_stddev: float,
        use_bias: bool,
    ) -> None:
        '''
        '''
        super().__init__()
        self.init_stddev = init_stddev
        self.layer = torch.nn.Linear(source_embedding_size, target_embedding_size, bias=use_bias)

    #########################################
    def initialise(
        self,
        seed: int,
    ) -> None:
        '''
        '''
        g = torch.Generator()
        g.manual_seed(seed)
        for (_, param) in self.named_parameters():
            torch.nn.init.normal_(param, std=self.init_stddev, generator=g)

    #########################################
    def forward(
        self,
        source_vectors: torch.Tensor,
    ) -> torch.Tensor:
        '''
        '''
        return self.layer(source_vectors)
