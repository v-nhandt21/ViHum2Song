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
import glob
from utils import get_chromagram
from dataloader import AudioDataset, AudioCollate

def get_top_10(top10, batch_candidate):
     merge_candidate = {**top10, **batch_candidate}
     if len(merge_candidate) <=10:
          return merge_candidate
     return dict(sorted(merge_candidate.items(), key=lambda item: item[1]))

def get_score(target, top10):
     for i, song in enumerate(top10.keys()):
          if target == song:
               return 1/(i+1)
     return 0

if __name__ == "__main__":
     #get_chromagram("/home/nhandt23/Desktop/Hum2Song/data/train/song/2895.wav", True)
     batch_size = 1
     num_workers = 1
     hop_length=256
     gamma = 0.1 #Acuracy of DTW

     sdtw = SoftDTW(use_cuda=True, gamma=gamma).to("cuda")

     hums_list = glob.glob("/home/nhandt23/Desktop/Hum2Song/data/train/hum/????.wav")
     hums_list.sort()
     top10 = {}
     
     total_score = 0

     hums_list = hums_list

     for idxH, hum in enumerate(hums_list):
          target_song = hum.split("/")[-1]

          hum = torch.from_numpy(get_chromagram(hum, plot=False, inbatch=True, hop_length=hop_length))

          songs_list = glob.glob("/home/nhandt23/Desktop/Hum2Song/data/train/song/????.wav")
          songs_list.sort()

          librarySet = AudioDataset(songs_list, hop_length=256, shuffle=False)

          audio_collate = AudioCollate(hum_len=hum.shape[1], n_pitch=hum.shape[2], pad_hum=True)
          
          loader = DataLoader(librarySet, num_workers=num_workers, shuffle=False, sampler=None, batch_size=batch_size, pin_memory=True, drop_last=True, collate_fn=audio_collate)

          for batch in tqdm(loader):
               
               file_id, songs = batch
               songs = songs.to("cuda")
               hums = hum.repeat(batch_size,1,1).to("cuda")
               losses = sdtw(songs, hums).tolist()

               batch_candidate = {}
               for id, loss in zip(file_id, losses):
                    batch_candidate[id] = loss

               top10 = get_top_10(top10, batch_candidate)
               # print(top10)
               # print("===================")
          
          score = get_score(target_song, top10)
          total_score = total_score + score
          print("TMP score: ",total_score/(idxH+1))
     
     total_score = total_score/len(hums_list)
     print(total_score)