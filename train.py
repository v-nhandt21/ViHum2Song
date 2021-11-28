import librosa
import librosa.display
import matplotlib
import matplotlib.pylab as plt
from scipy.io.wavfile import read
from tqdm import tqdm
from soft_dtw_cuda import SoftDTW
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import glob
from utils import get_chromagram, get_score, get_top_10
from dataloader import AudioDataset, AudioCollateInfer, AudioCollateTrain

from model import BiRNN
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
     batch_size = 8
     num_workers = 8
     hop_length=516
     gamma = 0.1 #Acuracy of DTW
     train_call = True
     pad_hum = False
     n_pitch = 12
     epochs = 1000

     device = "cuda" #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     logs = open("logs.txt","w+", encoding="utf-8")

     sdtw = SoftDTW(use_cuda=True, gamma=gamma).to("cuda")

     songs_list = glob.glob("/home/noahdrisort/Desktop/ViHum2Song/data/train/song/????.wav")
     songs_list.sort()

     librarySet = AudioDataset(songs_list, hop_length=hop_length, shuffle=False, train_call = train_call)

     audio_collate = AudioCollateTrain(n_pitch=n_pitch)
     
     loader = DataLoader(librarySet, num_workers=num_workers, shuffle=False, sampler=None, batch_size=batch_size, pin_memory=True, drop_last=True, collate_fn=audio_collate)

     model = BiRNN().to(device)

     model.train()

     optimizer = optim.Adam(model.parameters(), lr=0.001)

     sw = SummaryWriter("Outdir/logdir")
     steps = 0

     for epoch in range(epochs):
          running_loss = []
          for batch in tqdm(loader):
               
               hums, songs = batch
               songs = songs.to(device)

               SongFeatures = model(songs)

               losses = sdtw(SongFeatures, hums.to(device))

               losses.mean().backward()
               optimizer.step()

               running_loss.append(losses.cpu().detach().numpy())

               logs.write("Epoch: {}/{} - Loss: {:.4f}\n".format(epoch, epochs, np.mean(running_loss)))

               sw.add_scalar("Loss", losses, steps)
               steps+=1

          if epoch%10==0:
               torch.save({'model': model.state_dict()}, "Outdir/"+str(epoch))