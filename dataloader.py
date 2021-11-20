
import torch
import numpy as np
import random
import glob
from utils import get_chromagram
import librosa
import os.path

class AudioDataset(torch.utils.data.Dataset):
     def __init__(self, training_files, hop_length=256, shuffle=True):
          self.audio_files = training_files
          self.hop_length=hop_length
          random.seed(1234)
          if shuffle:
               random.shuffle(self.audio_files)

     def __getitem__(self, index):
          filename = self.audio_files[index]

          if os.path.isfile(filename.replace(".wav",".npy")):
               chroma = np.load(filename.replace(".wav",".npy"))
               chromagram = torch.from_numpy(chroma)
          else:
               chroma = get_chromagram(filename)
               np.save(filename.replace(".wav",".npy"), chroma)
               chromagram = torch.from_numpy(get_chromagram(filename, plot=False, inbatch=False, hop_length=self.hop_length))
          file_id = filename.split("/")[-1]

          return file_id, chromagram

     def __len__(self):
          return len(self.audio_files)

class AudioCollate():

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

if __name__ == "__main__":
     songs_list = glob.glob("/home/nhandt23/Desktop/Hum2Song/data/train/song/????.wav")
     for song in songs_list:
          try:
               x, sr = librosa.load(song)
          except:
               print(song)

# /home/nhandt23/Desktop/Hum2Song/data/train/song/1356.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/0806.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/0489.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/2461.wav
# /home/nhandt23/Desktop/Hum2Song/data/train/song/2560.wav