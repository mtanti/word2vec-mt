'''
'''

import torch
import numpy as np


#########################################
class SkipgramModel(torch.nn.Module):
    '''
    '''

    #########################################
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        init_stddev: float,
        dropout_rate: float,
    ) -> None:
        '''
        '''
        super().__init__()
        self.init_stddev = init_stddev
        self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.output_layer = torch.nn.Linear(embedding_size, vocab_size)

    #########################################
    def initialise(
        self,
        seed: int,
    ) -> None:
        '''
        '''
        g = torch.Generator()
        g.manual_seed(seed)
        for (name, param) in self.named_parameters():
            torch.nn.init.normal_(param, std=self.init_stddev, generator=g)

    #########################################
    def forward(
        self,
        input_token_index: torch.Tensor,
    ) -> torch.Tensor:
        '''
        '''
        embedded = self.embedding_layer(input_token_index)
        embedded = self.dropout(embedded)
        return self.output_layer(embedded)

    #########################################
    def get_embeddings(
        self,
    ) -> np.ndarray:
        return self.embedding_layer.weight.data.detach().cpu().numpy()


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
        for (name, param) in self.named_parameters():
            torch.nn.init.normal_(param, std=self.init_stddev, generator=g)

    #########################################
    def forward(
        self,
        source_vectors: torch.Tensor,
    ) -> torch.Tensor:
        '''
        '''
        return self.layer(source_vectors)
