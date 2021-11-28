# A mobile-compatible pitch extraction model to recognize the dominant pitch in sung audio. 
# Trained (in a self-supervised way) on the MIR-1k dataset.

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

model = hub.load("https://tfhub.dev/google/spice/2")

def get_pitch(model, fpath, threshold=0.7):
  # Loading audio samples from the wav file:
  sample_rate, audio_samples = wavfile.read(fpath, 'rb')
  audio_samples = audio_samples[: int(5 * 16000)]
  MAX_ABS_INT16 = 32768.0
  audio_samples = audio_samples / float(MAX_ABS_INT16)
  input = tf.constant(audio_samples, tf.float32)
  output = model.signatures["serving_default"](input)
  pitches = output["pitch"]
  confidences = 1.0 - output["uncertainty"]
  indices = range(len(pitches))
  
  pitch_filters = [ 
    (i,p) for i, p, c in zip(indices, pitches, confidences) if  c >= threshold
  ]
  pitch_x, pitch_y = zip(*pitch_filters)
  pitch_y = np.array(pitch_y) - np.average(pitch_y)
  return pitch_x, pitch_y

song_pitch_x, song_pitch_y = get_pitch(model, '/home/tuanpv/workspace/zaloai2021/music/data/train/song/0011.wav')
hum_pitch_x, hum_pitch_y = get_pitch(model, '/home/tuanpv/workspace/zaloai2021/music/data/train/hum/0011.wav')

distance = dtw.distance(song_pitch_y, hum_pitch_y)
print(distance)

path = dtw.warping_path(song_pitch_y, hum_pitch_y)
dtwvis.plot_warping(song_pitch_y, hum_pitch_y, path, filename="warp.png")


print("Len song:", len(song_pitch_x))
print("Len hum:", len(hum_pitch_x))

fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
plt.scatter(song_pitch_x, song_pitch_y, label='song pitch')
plt.scatter(hum_pitch_x, hum_pitch_y, label='hum pitch')
plt.legend(loc="lower right")
plt.show()