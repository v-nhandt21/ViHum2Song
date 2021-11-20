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