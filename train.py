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
from utils import get_chromagram, get_score, get_top_10, to_gpu
from dataloader import AudioDataset, AudioCollateInfer, AudioCollateTrain
import sys
from model import BiRNN

from SRU import Song2Hum
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
     batch_size = 8
     num_workers = 8
     hop_length=1024
     gamma = 0.1 #Acuracy of DTW
     train_call = True
     pad_hum = False
     n_pitch = 12
     epochs = 1000

     device = "cuda" #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     logs = open("logs.txt","w+", encoding="utf-8")

     sdtw = SoftDTW(use_cuda=True, gamma=gamma).to("cuda")

     # songs_list = glob.glob("/home/noahdrisort/Desktop/ViHum2Song/data/train/song/????.wav")
     songs_set = glob.glob("/home/nhandt23/Desktop/Hum2Song/data/train/song/????.wav")

     RM = ["/home/nhandt23/Desktop/Hum2Song/data/train/song/1356.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/0806.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/0489.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/2461.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/2560.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/0371.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/0375.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/0376.wav",

     "/home/nhandt23/Desktop/Hum2Song/data/train/song/0378.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/0489.wav",

     "/home/nhandt23/Desktop/Hum2Song/data/train/song/0497.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/0795.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/1409.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/2297.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/2302.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/2307.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/2527.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/2543.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/2560.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/2836.wav",
     "/home/nhandt23/Desktop/Hum2Song/data/train/song/2837.wav"]

     songs_list = [ss for ss in songs_set if ss not in RM]

     songs_list.sort()

     librarySet = AudioDataset(songs_list, hop_length=hop_length, shuffle=False, train_call = train_call)

     audio_collate = AudioCollateTrain(n_pitch=n_pitch)
     
     loader = DataLoader(librarySet, num_workers=num_workers, shuffle=False, sampler=None, batch_size=batch_size, pin_memory=True, drop_last=True, collate_fn=audio_collate)

     model_sru = Song2Hum(12, 12, sru_num_layers = 2, batch_size=batch_size).cuda()

     model_lstm = BiRNN().to(device)

     model = model_sru

     model.train()

     optimizer = optim.Adam(model.parameters(), lr=0.001)

     sw = SummaryWriter("Outdir/logdir")

     for epoch in range(epochs):
          running_loss = []
          train_set = tqdm(loader)
          for batch in train_set:

               optimizer.zero_grad()
               
               hums, songs = batch
               hums= to_gpu(hums).float()
               songs = to_gpu(songs).float()

               try:

                    songs = model(songs)
                    # hums = model(hums)

                    losses = sdtw(songs, hums)
                    losses.mean().backward()
                    optimizer.step()

               except RuntimeError as e:
                    if 'out of memory' in str(e):
                         print('| WARNING: ran out of memory, retrying batch')
                         for p in model.parameters():
                              if p.grad is not None:
                                   del p.grad  # free some memory
                         torch.cuda.empty_cache()
                         continue
                    else:
                         raise e

               running_loss.append(losses.cpu().detach().numpy())
               train_set.set_description("Loss %s" % losses.cpu().detach().numpy())

          logs.write("Epoch: {}/{} - Loss: {:.4f}\n".format(epoch, epochs, float(np.mean(running_loss))) )

          sw.add_scalar("Loss", np.mean(running_loss), epoch)

               

          if epoch%20==0:
               torch.save({'model': model.state_dict()}, "Outdir/"+str(epoch))