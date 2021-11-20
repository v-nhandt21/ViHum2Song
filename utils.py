import librosa
import numpy as np
import matplotlib
import matplotlib.pylab as plt

def get_chromagram(path, plot=False, inbatch=False, hop_length=256):
     x, sr = librosa.load(path)
     hop_length=256
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
     return chromagram

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