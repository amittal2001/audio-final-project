# Audio
sample_rate = 16000
n_mfcc = 13
n_mels = 40
n_fft = 1024
hop_length = 0.01
win_length = 0.03

low_freq = 20
high_freq = 4000
center = False

# data
split = 0.8
batch_size = 64
num_epochs = 50

# learning
lr = 0.01
patience = 10
momentum = 0.9
weight_decay = 0.0

# seed
seed = 1000
