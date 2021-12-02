
import torch
import numpy as np
import random
import glob
from utils import get_chromagram
import librosa
import os.path
from tqdm import tqdm
class AudioDataset(torch.utils.data.Dataset):
     def __init__(self, training_files, hop_length=256, shuffle=True, train_call=False):
          self.audio_files = training_files
          self.hop_length=hop_length
          self.train_call = train_call
          random.seed(1234)
          if shuffle:
               random.shuffle(self.audio_files)

     def __getitem__(self, index):
          filename = self.audio_files[index]

          # x1, sr = librosa.load(filename, sr=16000)
          # x2, sr = librosa.load(filename.replace("song","hum"), sr=16000)
          # return torch.from_numpy(librosa.stft(x1).T), torch.from_numpy(librosa.stft(x2).T)






          if os.path.isfile(filename.replace(".wav","_"+str(self.hop_length)+".npy")):
               chroma = np.load(filename.replace(".wav","_"+str(self.hop_length)+".npy"))
               chromagram = torch.from_numpy(chroma)
          else:
               chroma = get_chromagram(filename, self.hop_length)
               np.save(filename.replace(".wav","_"+str(self.hop_length)+".npy"), chroma)
               chromagram = torch.from_numpy(chroma)
          
          if not self.train_call:
               file_id = filename.split("/")[-1]
               return file_id, chromagram
          else:
               humnum = filename.replace("song","hum")
               if os.path.isfile(humnum.replace(".wav","_"+str(self.hop_length)+".npy")):
                    chroma_hum = np.load(humnum.replace(".wav","_"+str(self.hop_length)+".npy"))
                    chromagram_hum = torch.from_numpy(chroma_hum)
               else:
                    chroma_hum = get_chromagram(humnum, self.hop_length)
                    np.save(humnum.replace(".wav","_"+str(self.hop_length)+".npy"), chroma_hum)
                    chromagram_hum = torch.from_numpy(chroma_hum)      

               return chromagram_hum, chromagram

     def __len__(self):
          return len(self.audio_files)

class AudioCollateInfer():

     def __init__(self, hum_len, n_pitch, pad_hum):
          self.hum_len = hum_len
          self.n_pitch = n_pitch
          self.pad_hum = pad_hum

     def __call__(self, batch):
          """
          batch: [file_id, chromagram]
          """
          if self.pad_hum:
               choroma_padded = torch.LongTensor(len(batch), self.hum_len, self.n_pitch)
               choroma_padded.zero_()
               file_id = []
               for i, song in enumerate(batch):
                    choroma = batch[i][1]
                    if choroma.size(0) <= choroma_padded.size(1):
                         choroma_padded[i, :choroma.size(0)] = choroma
                    else:
                         choroma_padded[i] = choroma[:choroma_padded.size(1)]
                    file_id.append(batch[i][0])
          else:
               input_lengths, ids_sorted_decreasing = torch.sort( torch.LongTensor([len(x[1]) for x in batch]), dim=0, descending=True)
               max_input_len = input_lengths[0]
               choroma_padded = torch.LongTensor(len(batch), max_input_len, self.n_pitch)
               choroma_padded.zero_()
               file_id = []
               for i in range(len(ids_sorted_decreasing)):
                    choroma = batch[ids_sorted_decreasing[i]][1]
                    choroma_padded[i, :choroma.size(0)] = choroma
                    file_id.append(batch[ids_sorted_decreasing[i]][0])

          return file_id, choroma_padded

class AudioCollateTrain():

     def __init__(self, n_pitch):
          self.n_pitch = n_pitch

     def __call__(self, batch):
          """
          batch: [chromagram_hum, chromagram]
          """
          input_lengths, ids_sorted_decreasing = torch.sort( torch.LongTensor([len(x[1]) for x in batch]), dim=0, descending=True)
          max_input_len = input_lengths[0]
          choroma_padded = torch.FloatTensor(len(batch), max_input_len, self.n_pitch)
          choroma_padded.zero_()

          input_lengths_hum, ids_sorted_decreasing_hum = torch.sort( torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
          max_input_len_hum = input_lengths_hum[0]
          choroma_padded_hum = torch.FloatTensor(len(batch), max_input_len_hum, self.n_pitch)
          choroma_padded_hum.zero_()

          for i in range(len(ids_sorted_decreasing)):
               choroma = batch[ids_sorted_decreasing[i]][1]

               paddingL = int((max_input_len-choroma.size(0))/2)
               paddingU = int((max_input_len-choroma.size(0))/2+choroma.size(0))
               choroma_padded[i, paddingL:paddingU] = choroma

               choroma_hum = batch[ids_sorted_decreasing[i]][0]
               paddingL = int((max_input_len_hum-choroma_hum.size(0))/2)
               paddingU = int((max_input_len_hum-choroma_hum.size(0))/2+choroma_hum.size(0))
               choroma_padded_hum[i, paddingL:paddingU] = choroma_hum

          return choroma_padded_hum, choroma_padded

if __name__ == "__main__":
     # songs_list = glob.glob("/home/nhandt23/Desktop/Hum2Song/data/train/song/????.wav")

     songs_list = glob.glob("/home/nhandt23/Desktop/Hum2Song/data/train/hum/????.wav")
     songs_list.sort()

     # songs_list = glob.glob("/home/nhandt23/Desktop/Hum2Song/data/public_test/hum/????.wav")
     for song in tqdm(songs_list):
          try:
               x, sr = librosa.load(song)
               chroma = get_chromagram(song)
          
          except:
               print(song)

# Song

# /home/nhandt23/Desktop/Hum2Song/data/train/song/1356.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/0806.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/0489.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/2461.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/2560.wav

# Hum

# /home/nhandt23/Desktop/Hum2Song/data/train/song/0371.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/0375.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/0376.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/0497.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/0795.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/1409.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/2297.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/2302.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/2307.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/2527.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/2543.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/2560.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/2836.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/2837.wav