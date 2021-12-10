import torch
import torch.nn as nn


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EmbedderGRU(nn.Module):

     def __init__(self, n_hid=64, n_mels=12, n_layers=2, fc_dim=128, bidir=False):
          super().__init__()    
          self.rnn_stack = nn.GRU(n_mels, n_hid, num_layers=n_layers, 
                              batch_first=True, bidirectional=bidir, dropout=0.8)
          for name, param in self.rnn_stack.named_parameters():
               if 'bias' in name:
                    nn.init.constant_(param, 0.0)
               elif 'weight' in name:
                    nn.init.xavier_normal_(param)
          self.projection = nn.Linear(n_hid, fc_dim)
          self.drop = nn.Dropout(p=0.2)
          
     def forward(self, x):
          """ Takes in a set of mel spectrograms in shape (batch, frames, n_mels) """
          self.rnn_stack.flatten_parameters()            
          
          x, _ = self.rnn_stack(x) #(batch, frames, n_mels)
          #only use last frame
          x = x[:,-1,:]
          x = self.projection(x)
          x = x / torch.norm(x, p=2, dim=-1, keepdim=True)

          return x
class ConvNet(nn.Module):
     def __init__(self, num_classes=10):
          super(ConvNet, self).__init__()
          self.layer1 = nn.Sequential(
               nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
               nn.BatchNorm2d(16),
               nn.ReLU(),
               nn.MaxPool2d(kernel_size=2, stride=2))
          self.layer2 = nn.Sequential(
               nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
               nn.BatchNorm2d(32),
               nn.ReLU(),
               nn.MaxPool2d(kernel_size=2, stride=2))
          
     def forward(self, x):
          out = self.layer1(x)
          out = self.layer2(out)
          return out

class BiRNN(nn.Module):
     def __init__(self, input_size=12, hidden_size=12, num_layers=1):
          super(BiRNN, self).__init__()
          self.hidden_size = hidden_size
          self.input_size = input_size
          self.num_layers = num_layers
          self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=0.5)

          self.conv1d = torch.nn.Conv1d(12, 12,
                                    kernel_size=5, stride=1,
                                    padding=2, dilation=1,
                                    bias=True)

          self.conv1d_norm1 = torch.nn.Conv1d(12, 12,
                                    kernel_size=11, stride=2,
                                    padding=5, dilation=2,
                                    bias=True)

          self.conv1d_norm2 = torch.nn.Conv1d(12, 12,
                                    kernel_size=3, stride=2,
                                    padding=1, dilation=1,
                                    bias=True)

          self.linear_layer = torch.nn.Linear(12, 12, bias=True)
          self.drop = nn.Dropout(p=0.2)
          self.sigmoid = nn.Sigmoid()
          
     
     def forward(self, x):
          # Set initial states
          h0 = torch.zeros(self.num_layers, x.size(0), self.input_size).to(device)
          c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
          
          # Forward propagate LSTM
          out, _ = self.lstm(x, (h0,c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
          out1 = self.conv1d(out.permute(0,2,1))

          out1 = self.drop(out1)

          out2 = self.conv1d_norm1(out1)
          out = self.conv1d_norm2(out2)

          out = self.linear_layer(out.permute(0,2,1))

          out = self.sigmoid(out)
          
          return out

class Hum2Song(nn.Module):
     def __init__(self):
          super(Hum2Song, self).__init__()
          self.emb_model = EmbedderGRU().to(device)
          self.chroma_model = BiRNN().to(device)

     def forward(self, x):
          return self.chroma_model(x)#, self.emb_model(x)

if __name__ == "__main__":
     from torch.autograd import Variable
     model = ConvNet().to(device)

     inputs = Variable(torch.rand(8, 100, 12)) # batch_size seq_len feat_dim
     outputs = model(inputs.to(device))
     print('outputs', outputs.size()) # conv_seq_len x batch_size x output_size