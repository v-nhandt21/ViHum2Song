Task need to do:

+ Store cache SoftDTW pair
+ Normalize chroma
+ Handle Cuda max GPU 1024
+ Fine tune : batch-size, gamma, hop-lenght

+ Function to filter background music in songs ? >> hum to music or hum to singer
+ Contraint DTW 

+ Voting inference: search for long duration songs

+ Filter breathe

-------------------------------

Sumary some approach;

+ PitchExtraction: Use a pretrain model from google to extract embedding for pitch -> use DTW to measure distance as score for prediction

+ ChromagramExtraction: Calculate chromagram for song and hum pair then use SoftDTW -> distance -> score for prediction without training

+ Model Training: 