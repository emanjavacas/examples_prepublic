"""
Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                          a

Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

"""

import torch
import torch.nn as nn
import math


class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, dec_output, enc_outputs):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        dec_att = self.linear_in(dec_output).unsqueeze(2)  # batch x dim x 1
        # Get attention
        weights = torch.bmm(enc_outputs, dec_att).squeeze(2)  # batch x sourceL
        if self.mask is not None:
            weights.data.masked_fill_(self.mask, -math.inf)
        weights = self.sm(weights)
        # batch x dim
        weighted = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        combined = torch.cat([weighted, dec_output], 1)
        output = self.tanh(self.linear_out(combined))
        return output, weights
