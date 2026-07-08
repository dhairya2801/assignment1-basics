import torch.nn as nn
import torch

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.embedding_matrix = torch.nn.init.trunc_normal_(embedding_matrix, mean=0, std=1, a=-3, b=3, generator=None)


    def forward(self, token_ids):
        '''
        1. for every token_id, find its embedding vector of size d_model by indexing the embedding matrix.
        (batch, seq_len) -> (batch, seq_len, d_model)
        
        '''
        return self.embedding_matrix[token_ids]