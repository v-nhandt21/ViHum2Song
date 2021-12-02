# Signal Processing to select the best signal parameter for feature chromagram which is used for training
# song_chroma <- song
# hum_chroma <- hum
# softDTW(song_chroma, hum_chroma) -> distance -> score

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
from utils import get_chromagram, get_score, get_top_10
from dataloader import AudioDataset, AudioCollate

def optimize_top_10(target, top10, path="/home/nhandt23/Desktop/Hum2Song/data/train/song/", hop_length=64, gamma = 0.005):
     optimize = {}
     tg = torch.from_numpy(get_chromagram("/home/nhandt23/Desktop/Hum2Song/data/train/hum/"+target, hop_length=hop_length, plot=False, inbatch=True)).to("cuda")
     print("===========================", top10.keys())
     for i, song in enumerate(top10.keys()):
          sg = torch.from_numpy(get_chromagram(path+song, hop_length=hop_length, plot=False, inbatch=True)).to("cuda")
          loss = sdtw(sg, tg).tolist()[0]
          optimize[song] = loss
     return dict(sorted(optimize.items(), key=lambda item: item[1])[:10])

if __name__ == "__main__":
     batch_size = 16
     num_workers = 1
     hop_length=64
     gamma = 0.1 #Acuracy of DTW
     pad_hum = True

     sdtw = SoftDTW(use_cuda=True, gamma=gamma).to("cuda")

     hums_list = glob.glob("/home/nhandt23/Desktop/Hum2Song/data/train/hum/????.wav")
     hums_list.sort()
     top10 = {}
     
     total_score = 0

     hums_list = hums_list

     for idxH, hum in enumerate(hums_list):
          target_song = hum.split("/")[-1]

          hum = torch.from_numpy(get_chromagram(hum, hop_length=hop_length, plot=False, inbatch=True))

          songs_list = glob.glob("/home/nhandt23/Desktop/Hum2Song/data/train/song/????.wav")
          songs_list.sort()

          librarySet = AudioDataset(songs_list, hop_length=hop_length, shuffle=False)

          audio_collate = AudioCollateInfer(hum_len=hum.shape[1], n_pitch=hum.shape[2], pad_hum=pad_hum)
          
          loader = DataLoader(librarySet, num_workers=num_workers, shuffle=False, sampler=None, batch_size=batch_size, pin_memory=True, drop_last=True, collate_fn=audio_collate)

          top10 = {} # TODO check

          for batch in tqdm(loader):
               
               file_id, songs = batch
               songs = songs.to("cuda")
               hums = hum.repeat(batch_size,1,1).to("cuda")
               losses = sdtw(songs, hums).tolist()

               batch_candidate = {}
               for id, loss in zip(file_id, losses):
                    batch_candidate[id] = loss

               top10 = get_top_10(top10, batch_candidate)
          
          top10 = optimize_top_10(target_song, top10)

          score = get_score(target_song, top10)
          total_score = total_score + score

          print("Song: ", idxH+1)
          print("Batch score: ", score)
          print("TMP score: ",total_score/(idxH+1))
     
     total_score = total_score/len(hums_list)
     print(total_score)