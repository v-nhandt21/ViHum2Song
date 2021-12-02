import librosa, torch
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import glob
from tqdm import tqdm
import sys

def norm(nparrary):
     m = np.mean(nparrary, axis=0)
     sd = np.std(nparrary, axis=0)
     return (nparrary - m)/sd

def get_chromagram(path, hop_length, plot=False, inbatch=False):

     try:
          x, sr = librosa.load(path, sr=16000)
          chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)

          if plot:
               print(chromagram.shape)

               fig, ax = plt.subplots()
               img = librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time', ax=ax)
               fig.savefig("chroma.png")
          
          if inbatch:
               chromagram = np.expand_dims(chromagram.T, axis=0)
          else:
               chromagram = chromagram.T

          # chromagram = norm(chromagram)
          return chromagram
     except:
          print(path)
          sys.exit()

def get_top_10(top10, batch_candidate):
     merge_candidate = {**top10, **batch_candidate}
     if len(merge_candidate) <=10:
          return merge_candidate
     return dict(sorted(merge_candidate.items(), key=lambda item: item[1])[:10])

def get_score(target, top10):
     for i, song in enumerate(top10.keys()):
          if target == song:
               return 1/(i+1)
     return 0

def get_predict(target, top10):
     predict = target
     for i, song in enumerate(top10.keys()):
          predict = predict + "," + song.split(".")[0]
     return predict

def get_duration(path="/home/nhandt23/Desktop/Hum2Song/data/public_test/hum/", pattern="????.wav"):
     songs = glob.glob(path+pattern)
     DUR = []
     songs.sort()
     for song in tqdm(songs[:]):
          y, sr = librosa.load(song, sr=16000)
          dur = librosa.get_duration(y=y, sr=sr)
          DUR.append(int(dur))

     # DUR = get_duration("/home/nhandt23/Desktop/Hum2Song/data/public_test/hum/","??????????.wav")
     # print(DUR)
     # plt.hist(np.array(DUR),20)  # density=False would make counts
     # plt.ylabel('Duration(s)')
     # plt.xlabel('Data')
     # plt.savefig("dur_his.png")

     return DUR

def to_gpu(x):
     x = x.contiguous()
     if torch.cuda.is_available():
          x = x.cuda(non_blocking=True)
     return torch.autograd.Variable(x)
     
if __name__ == "__main__":
     aa = 0

     DUR = get_duration("/home/nhandt23/Desktop/Hum2Song/data/public_test/hum/","????.wav")
     print(DUR)
     plt.hist(np.array(DUR),20)  # density=False would make counts
     plt.ylabel('Duration(s)')
     plt.xlabel('Data')
     plt.savefig("dur_his_hum.png")

     # songs = glob.glob("/home/nhandt23/Desktop/Hum2Song/data/public_test/full_song/??????????.wav")
     # #songs = glob.glob("/home/nhandt23/Desktop/Hum2Song/data/train/song/????.wav")
     # Frame = []
     # for song in tqdm(songs):
     #      chroma = get_chromagram(song)
     #      Frame.append(int(chroma.shape[0]))
     # plt.hist(np.array(Frame) , density=False, bins=100)  # density=False would make counts
     # #plt.hist(np.array(Frame)[np.array(Frame)>1024] , density=False, bins=100)
     # #plt.ylim(0, 1024)
     # # plt.xlim(0, 1024)
     # plt.xlabel('Data')
     # plt.savefig("frame_his.png")