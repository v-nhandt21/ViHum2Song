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
import glob, os
from utils import get_chromagram, get_score, get_predict
from dataloader import AudioDataset, AudioCollate

def get_top_10(top10, batch_candidate):
     merge_candidate = {**top10, **batch_candidate}
     if len(merge_candidate) <=10:
          return merge_candidate
     return dict(sorted(merge_candidate.items(), key=lambda item: item[1])[:10])

def getMap():
     targetMap = {}
     with open("/home/nhandt23/Desktop/Hum2Song/data/train/train_meta.csv", "r", encoding="utf-8") as f:
          lines = f.read().splitlines()
          for line in lines:
               id, file_id, _ = line.split(",")
               targetMap[id] = file_id.split("/")[-1].split(".")[0]
     return targetMap

def get_score_public(target_song, top10, targetMap):
     for i, song in enumerate(top10.keys()):
          if target_song == targetMap[song]:
               return 1/(i+1)
     return 0

def optimize_top_10(target, top10, \
     path_hum="/home/nhandt23/Desktop/Hum2Song/data/public_test/hum/", \
     path_song="/home/nhandt23/Desktop/Hum2Song/data/public_test/full_song/", \
     hop_chunk=8, hop_length=128,  gamma = 0.01):

     optimize = {}
     sdtw_trik = SoftDTW(use_cuda=True, gamma=gamma).to("cuda")
     # print(len(top10.keys()))
     # print(top10)
     tg = torch.from_numpy(get_chromagram(path_hum+target+".wav", plot=False, inbatch=True, hop_length=hop_length)).to("cuda")
     for song in top10.keys():
          song = path_song+str(song)+".wav"

          if os.path.isfile(song.replace(".wav","_"+str(hop_length)+".npy")):
               chroma = np.load(song.replace(".wav","_"+str(hop_length)+".npy"))
               sg = torch.from_numpy(chroma)
          else:
               chroma = get_chromagram(song, plot=False, inbatch=True, hop_length=hop_length)
               np.save(song.replace(".wav","_"+str(hop_length)+".npy"), chroma)
               sg = torch.from_numpy(chroma)
          # sg = torch.from_numpy(get_chromagram(song, plot=False, inbatch=True, hop_length=hop_length))

          minDTW = float('inf')
          win_chunk = tg.size(1)
          iter_chunk = win_chunk
          if tg.size(1) > sg.size(1):
               minDTW = sdtw_trik(tg, sg.to("cuda")).tolist()[0]
          while iter_chunk <= sg.size(1):
               cand_chunk = sg[:,iter_chunk-win_chunk:iter_chunk,:].to("cuda")
               loss = sdtw_trik(tg, cand_chunk).tolist()[0]
               if loss < minDTW:
                    minDTW = loss
               iter_chunk += hop_chunk

          optimize[song] = minDTW
     return dict(sorted(optimize.items(), key=lambda item: int(item[1]))[:10])

if __name__ == "__main__":
     
     # Test
     batch_size = 1
     num_workers = 1
     hop_length=1024
     gamma = 0.1 #Acuracy of DTW
     gamma_optimize = 0.01
     hop_chunk = 256
     hop_length_optimize =512
     hop_chunk_optimize = 256
     f = open("predict.csv","w+", encoding="utf-8")

     # Run : 
     # batch_size = 1
     # num_workers = 1
     # hop_length=256
     # gamma = 0.1 #Acuracy of DTW
     # gamma_optimize = 0.01
     # hop_chunk = 8
     # hop_length_optimize =128
     # hop_chunk_optimize = 8
     # f = open("predict_run.csv","w+", encoding="utf-8")

     sdtw = SoftDTW(use_cuda=True, gamma=gamma).to("cuda")

     hums_list = glob.glob("/home/nhandt23/Desktop/Hum2Song/data/public_test/hum/????.wav")
     hums_list.sort()
     top10 = {}

     targetMap = getMap()
     
     total_score = 0

     hums_list = hums_list

     

     for idxH, hum in enumerate(hums_list[:]):
          print(hum)
          target_song = hum.split("/")[-1].split(".")[0]

          hum = torch.from_numpy(get_chromagram(hum, plot=False, inbatch=True, hop_length=hop_length)).to("cuda")
          print(hum.shape)

          songs_list = glob.glob("/home/nhandt23/Desktop/Hum2Song/data/public_test/full_song/??????????.wav")
          songs_list.sort()

          top10 = {} # TODO check

          for song in tqdm(songs_list):
               
               file_id = song.split("/")[-1].split(".")[0]

               if os.path.isfile(song.replace(".wav","_"+str(hop_length)+".npy")):
                    chroma = np.load(song.replace(".wav","_"+str(hop_length)+".npy"))
                    chromagram = torch.from_numpy(chroma)
               else:
                    chroma = get_chromagram(song, plot=False, inbatch=True, hop_length=hop_length)
                    np.save(song.replace(".wav","_"+str(hop_length)+".npy"), chroma)
                    chromagram = torch.from_numpy(chroma)
               
               minDTW = float('inf')
               win_chunk = hum.size(1)
               iter_chunk = win_chunk
               while iter_chunk <= chromagram.size(1):
                    cand_chunk = chromagram[:,iter_chunk-win_chunk:iter_chunk,:].to("cuda")
                    # print(iter_chunk)
                    # print(cand_chunk.size())
                    loss = sdtw(hum, cand_chunk).tolist()[0]
                    if loss < minDTW:
                         minDTW = loss
                    iter_chunk += hop_chunk
               
               batch_candidate = {}
               batch_candidate[file_id] = loss

               top10 = get_top_10(top10, batch_candidate)
          
          top10 = optimize_top_10(target_song, top10, \
               path_hum="/home/nhandt23/Desktop/Hum2Song/data/public_test/hum/", \
               path_song="/home/nhandt23/Desktop/Hum2Song/data/public_test/full_song/", \
               hop_chunk=hop_chunk_optimize, hop_length=hop_length_optimize,  gamma = gamma_optimize)

          # score = get_score_public(target_song, top10, targetMap)
          # total_score = total_score + score
          # print("Hum: ", idxH+1)
          # print("Current core: ", score)
          # print("TMP score: ",total_score/(idxH+1))

          
          predict = get_predict(target_song+".wav", top10)
          f.write(predict+"\n")

     print("Done!")