import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, input_size, output_size=1, n_heads=8, num_enc_layers=6, num_dec_layers=6, batch_first=True):
        super(Transformer, self).__init__()
        # print("d_model",input_size*n_heads)
        # print(input_size,n_heads)
        self.n_heads = n_heads
        self.transformer = nn.Transformer(d_model=input_size * n_heads, batch_first=batch_first, nhead=n_heads,
                                          num_encoder_layers=num_enc_layers, num_decoder_layers=num_dec_layers)
        self.linear = nn.Linear(input_size * n_heads, output_size)

    def forward(self, src, tgt):
        src = torch.tile(src, (1, 1, self.n_heads))
        tgt = torch.tile(tgt, (1, 1, self.n_heads))
        # print(src.shape,tgt.shape)
        # exit()
        # print("src, tgt shape",src.shape, tgt.shape)
        output = self.transformer(src, tgt)
        output = self.linear(output)
        return output
