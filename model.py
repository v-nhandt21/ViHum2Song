import torch
import torch.nn as nn


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
          self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
     
     def forward(self, x):
          # Set initial states
          h0 = torch.zeros(self.num_layers, x.size(0), self.input_size).to(device)
          c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
          
          # Forward propagate LSTM
          out, _ = self.lstm(x, (h0,c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
          return out

if __name__ == "__main__":
     from torch.autograd import Variable
     model = BiRNN().to(device)

     inputs = Variable(torch.rand(10, 80, 12)) # batch_size seq_len feat_dim
     outputs = model(inputs.to(device))
     print('outputs', outputs.size()) # conv_seq_len x batch_size x output_size