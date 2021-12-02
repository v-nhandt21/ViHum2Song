import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init

torch.manual_seed(1999)
random.seed(1999)


class SRU_Formula_Cell(nn.Module):
     def __init__(self, n_in, n_out, layer_numbers=1, dropout=0.0, bias=True):
          super(SRU_Formula_Cell, self).__init__()
          self.layer_numbers = layer_numbers
          self.n_in = n_in
          self.n_out = n_out
          self.dropout = dropout
          # Linear
          self.x_t = nn.Linear(self.n_in, self.n_out, bias=False)
          self.ft = nn.Linear(self.n_in, self.n_out, bias=bias)
          self.rt = nn.Linear(self.n_in, self.n_out, bias=bias)
          # self.convert_x = nn.Linear(self.n_in, self.n_out, bias=True)
          self.convert_x = self.init_Linear(self.n_in, self.n_out, bias=True)
          self.convert_x_layer = self.init_Linear(self.n_out, self.n_in, bias=True)
          self.convert_dim = self.init_Linear(self.n_in, self.n_out, bias=True)
          # dropout
          self.dropout = nn.Dropout(dropout)


     def init_Linear(self, in_fea, out_fea, bias=True):
          linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
          return linear.cuda()

     def forward(self, xt, ct_forward):
          layer = self.layer_numbers
          for layers in range(layer):
               if xt.size(2) == self.n_out:
                    xt = self.convert_x_layer(xt)
               # xt = self.convert_x(xt)
               xt, ct = SRU_Formula_Cell.calculate_one_layer(self, xt, ct_forward[layers])
          if self.dropout is not None:
               ht = self.dropout(xt)
               ct = self.dropout(ct)
          return ht.cuda(), ct.cuda()

     def calculate_one_layer(self, xt, ct_forward):
          ct = ct_forward
          ht_list = []
          for i in range(xt.size(0)):
               x_t = self.x_t(xt[i])
               ft = F.sigmoid(self.ft(xt[i]).cuda()).cuda()
               rt = F.sigmoid(self.rt(xt[i]))
               self.convert_dim = self.init_Linear(in_fea=ct_forward.size(0), out_fea=ft.size(0), bias=True)
               ct = torch.add(torch.mul(ft, ct), torch.mul((1 - ft), x_t))
               con_xt = self.convert_x(xt[i])
               ht = torch.add(torch.mul(rt, F.tanh(ct)), torch.mul((1 - rt), con_xt))
               ht_list.append(ht.unsqueeze(0))
          ht = torch.cat(ht_list, 0)
          return ht.cuda(), ct.cuda()


class Song2Hum(nn.Module):
     def __init__(self, input_dim, sru_hidden_dim, sru_num_layers, batch_size, dropout=0.2):
          super(Song2Hum, self).__init__()

          self.hidden_dim = sru_hidden_dim
          self.num_layers = sru_num_layers
          self.batch_size = batch_size
          self.dropout = nn.Dropout(dropout)

          self.sru = SRU_Formula_Cell(n_in=input_dim, n_out=self.hidden_dim, layer_numbers=self.num_layers,
                                        dropout=dropout, bias=True)
          self.sru.cuda()

          self.hidden = self.init_hidden(self.num_layers, batch_size).cuda()

     def init_hidden(self, num_layers, batch_size):
          # the first is the hidden h
          # the second is the cell  c
          return Variable(torch.zeros(num_layers, batch_size, self.hidden_dim)).cuda()

     def init_hidden_c(self, length, batch_size):
          # the first is the hidden h
          # the second is the cell  c
          return Variable(torch.zeros(length, batch_size, self.hidden_dim)).cuda()

     def forward(self, x):

          x = torch.permute(x, (1, 0, 2))
          h0 = self.init_hidden(self.num_layers, batch_size=self.batch_size).cuda()

          sru_out, self.hidden = self.sru(x, h0)

          # sru_out = torch.transpose(sru_out, 0, 1)
          # sru_out = torch.transpose(sru_out, 1, 2)
          # sru_out = F.tanh(sru_out)
          # sru_out = F.max_pool1d(sru_out, sru_out.size(2)).squeeze(2)
          # sru_out = F.tanh(sru_out)

          sru_out = torch.permute(sru_out, (1, 0, 2))

          return sru_out

if __name__ == "__main__":


     # input has length 20, batch size 32 and dimension 128
     x = torch.FloatTensor(200, 32, 128).cuda()

     input_size, hidden_size = 128, 128

     SRU = Song2Hum(input_size, hidden_size, sru_num_layers = 2, batch_size=32).cuda()

     output = SRU(x)      # forward pass

     print(output.shape)