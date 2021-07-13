import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pos_i = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        
        # pos_j = torch.tensor([1e4**-(2*(i//2)/embed_dim) for i in range(embed_dim)]).unsqueeze(0)  # bad method (1, embed_dim) 
        pos_j = torch.tensor([1e4**-(i/embed_dim) for i in range(0, embed_dim, 2)]).unsqueeze(0)  # (1, embed_dim/2)

        pos = pos_i * pos_j  # (max_len, embed_dim/2)      (max_len, embed_dim)时: sin(pos[:, 0::2])
        pe[0, :, 0::2] = torch.sin(pos)
        pe[0, :, 1::2] = torch.cos(pos)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        output = self.dropout(x + self.pe[:, :S, :])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        ############################################################################
        # TODO: Initialize any remaining layers and parameters to perform the      #
        # attention operation as defined in Transformer_Captioning.ipynb. We will  #
        # also apply dropout just after the softmax step. For reference, our       #
        # solution is less than 5 lines.                                           #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to bque used as the ery, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (T, S) where mask[i,j] == 0 indicates token
          i in the target should not be influenced by token j in the source.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, D = query.shape
        N, T, D = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, T, D))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        H = self.num_heads
        dk = D // H   # E//H

        Q = self.query(query).view(N, S, H, dk).transpose(1, 2)     # (N, S, E) -> (N, S, H, E//H) -> (N, H, S, E//H)
        K = self.key(key).view(N, T, H, dk).transpose(1, 2)       # (N, T, E) -> (N, T, H, E//H) -> (N, H, T, E//H)
        V = self.value(value).view(N, T, H, dk).transpose(1, 2)     # (N, T, E) -> (N, T, H, E//H) -> (N, H, T, E//H)
        
        attn = torch.matmul(Q, K.transpose(2, 3)) / torch.sqrt(torch.tensor(dk))   # (N, H, S, T)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask==0, float('-inf'))  
            # attn_mask的size为(T, S)是不是有问题，调用时T刚好等于S所以才没出错，应该(S, T)才对
            # 搭建的整个框架调用有mask的这个函数时，q=k=v，所以S和T是相等的，不会出错

        attn = self.dropout(F.softmax(attn, dim=-1))   # 即T(目标长度)维度
        output = torch.matmul(attn, V)   # (N, H, S, T) @ (N, H, T, E//H) -> (N, H, S, E//H)

        # multi-head concat
        # output = output.transpose(1, 2).reshape(N, S, -1)  # (N, S, E)
        output = output.transpose(1, 2).contiguous().view(N, S, -1)  # (N, S, E)
        # .contiguous()用在view之前，改变在内存中的存储方式才能view
        output = self.proj(output)  # concat之后的linear层，即使concat之后维度(H*dk大于E时)大于了E，也可以用FC层转回E

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


