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
from utils import get_chromagram, get_score, get_top_10, to_gpu, plot_spectrogram
from dataloader import AudioDataset, AudioCollateInfer, AudioCollateTrain
import sys, os
from model import BiRNN

from model import Hum2Song
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
     batch_size = 8
     num_workers = 8
     hop_length=128
     gamma = 0.1 #Acuracy of DTW
     train_call = True
     pad_hum = True
     n_pitch = 12
     epochs = 1000

     device = "cuda" #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     logs = open("logs.txt","w+", encoding="utf-8")

     sdtw = SoftDTW(use_cuda=True, gamma=gamma).to("cuda")

     for f in glob.glob('/home/nhandt23/Desktop/Hum2Song/Outdir/logdir/*'):
          os.remove(f)

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

     librarySet_train = AudioDataset(songs_list, hop_length=hop_length, shuffle=True, train_call = train_call)
     audio_collate_train = AudioCollateTrain(n_pitch=n_pitch)
     loader_train = DataLoader(librarySet_train, num_workers=num_workers, shuffle=True, sampler=None, batch_size=batch_size, pin_memory=True, drop_last=True, collate_fn=audio_collate_train)

     # model_sru = Song2Hum(12, 12, sru_num_layers = 2, batch_size=batch_size).cuda()
     # model_lstm = BiRNN().to(device)
     # model = model_lstm

     model = Hum2Song()
     model.train()

     optimizer = optim.Adam(model.parameters(), lr=0.0001)
     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

     sw = SummaryWriter("Outdir/logdir")

     iter = 0
     running_loss = []
     emb_loss = []
     recons_loss = []

     for epoch in range(epochs):
          
          train_set = tqdm(loader_train)
          plot = True
          
          for batch in train_set:

               optimizer.zero_grad()

               hums, songs, anchor, mel_hum, mel_song, mel_anchor = batch

               if plot:
                    plot = False
                    plot_spectrogram(hums[7].T,"hum_chroma.png")
                    plot_spectrogram(songs[7].T,"song_chroma.png")

               hums= to_gpu(hums).float()
               songs = to_gpu(songs).float()
               anchor = to_gpu(anchor).float()

               # hums= to_gpu(mel_hum).float()
               # songs = to_gpu(mel_song).float()
               # anchor = to_gpu(mel_anchor).float()

               # try:

               # songs, song_emb = model(songs.permute(0,2,1))
               # hums, hum_emb = model(hums.permute(0,2,1))
               # anchor, anchor_emb = model(anchor.permute(0,2,1))
               
               songs = model(songs)
               hums = model(hums)
               anchor = model(anchor)

               print(songs[0])

               losses = sdtw(songs, hums)
               
               ######################################3
               # loss_chroma_pos = torch.nn.MSELoss()(songs, hums)
               # loss_chroma_neg = torch.nn.MSELoss()(anchor, hums)
               # losses_chroma = torch.relu(loss_chroma_pos - loss_chroma_neg + 1)

               # loss_emb_pos = torch.nn.functional.cosine_similarity(song_emb, hum_emb)
               # loss_emb_neg = torch.nn.functional.cosine_similarity(hum_emb, anchor_emb)
               # losses_emb = torch.relu(loss_emb_pos - loss_emb_neg + 1)

               # losses = losses_chroma + losses_emb

               ##########################################
               # loss_chroma = torch.nn.MSELoss()(songs, hums)
               # loss_emb = torch.nn.functional.cosine_similarity(song_emb, hum_emb)
               # losses = loss_chroma + loss_emb



               losses.mean().backward()
               optimizer.step()

               running_loss.append(losses.cpu().detach().numpy())
               # recons_loss.append(losses_chroma.cpu().detach().numpy())
               # emb_loss.append(losses_emb.cpu().detach().numpy())

               train_set.set_description("Loss %s" % losses.mean().cpu().detach().numpy())

               if iter%10==0:
                    # logs.write("Epoch: {}/{} - Loss: {:.4f}\n".format(epoch, epochs, float(np.mean(running_loss))) )
                    sw.add_scalar("Loss", np.mean(running_loss), iter)
                    sw.add_scalar("Loss_emb", np.mean(emb_loss), iter)
                    sw.add_scalar("Loss_recons", np.mean(recons_loss), iter)
                    sw.add_scalar("Learning_rate", scheduler.get_last_lr()[-1], iter)
                    running_loss = []
                    emb_loss = []
                    recons_loss = []

                    scheduler.step()

               iter+=1

          if epoch%5 == 0:
               model.eval()
               print("====================   Validation   ========================")
               with torch.no_grad():
                    hum_val = glob.glob("/home/nhandt23/Desktop/Hum2Song/data/train/hum/????.wav")[10:12]
                    total_score = 0
                    for idxH, hum in enumerate(hum_val):
                         target_song = hum.split("/")[-1]

                         hum = torch.from_numpy(get_chromagram(hum, hop_length=hop_length, plot=False, inbatch=True))

                         librarySet_val = AudioDataset(songs_list, hop_length=hop_length, shuffle=False, train_call = False)
                         audio_collate_val = AudioCollateInfer(hum_len=hum.shape[1], n_pitch=hum.shape[2], pad_hum=pad_hum)
                         loader_val = DataLoader(librarySet_val, num_workers=num_workers, shuffle=False, sampler=None, batch_size=batch_size, pin_memory=True, drop_last=True, collate_fn=audio_collate_val)
                         
                         top10 = {} # TODO check

                         for batch in tqdm(loader_val):
                              
                              file_id, songs = batch
                              songs = to_gpu(songs).float()
                              hums = to_gpu(hum.repeat(batch_size,1,1)).float()
                              
                              # losses = sdtw(songs, hums).tolist()
                              songs, song_emb = model(songs)
                              hums, hum_emb = model(hums)
                              loss_chroma = torch.nn.MSELoss()(songs, hums)
                              loss_emb = torch.nn.functional.cosine_similarity(song_emb, hum_emb)
                              losses = loss_chroma + loss_emb
                              
                              losses.tolist()

                              batch_candidate = {}
                              for id, loss in zip(file_id, losses):
                                   batch_candidate[id] = loss.item()

                              top10 = get_top_10(top10, batch_candidate)
                         
                         # top10 = optimize_top_10(target_song, top10)

                         score = get_score(target_song, top10)
                         total_score = total_score + score

                         print("------------------------")
                         print("Song: ", idxH+1)
                         print("Batch score: ", score)
                         print("TMP score: ",total_score/(idxH+1))
                    
                    total_score = total_score/10
                    print("Validation score: ",total_score)
               
               model.train()                   

          if epoch%20==0:
               print("Save model >>>>>>>>>>>>>" + "Outdir/"+str(epoch) )
               torch.save({'model': model.state_dict()}, "Outdir/"+str(epoch))

          